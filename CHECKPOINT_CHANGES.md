# Deterministic Economy + Checkpointing — Changes & Run Guide

This change set adds two capabilities to CoffeeBench, with **no Valto code yet**:

1. **An active, zero-LLM economy** — all six agents run on deterministic scripted
   policies (no API keys, no tokens, no cost).
2. **Exact-day checkpoint / resume / branching** — stop at the start of a
   simulated day, save a self-contained checkpoint, and resume or fork it into
   independent futures that start from identical state.

For the deeper design/semantics writeup see
[`docs/deterministic_checkpointing.md`](docs/deterministic_checkpointing.md).
This file is the concise "what changed + how to run it" reference.

---

## 1. What was added (new files)

| File | What it is |
|---|---|
| `coffeebench/models/scripted.py` | Deterministic role policies `heuristic_farmer`, `heuristic_roaster`, `heuristic_retailer` (shared `ScriptedModel` base). One tool call/turn, read from the transcript, no randomness, `checkpoint_state()`/`restore_checkpoint_state()`. |
| `coffeebench/ids.py` | `IdGenerator` — a monotonic per-prefix counter that replaces `uuid.uuid4()` so IDs are deterministic and checkpoint-restorable. |
| `coffeebench/checkpoint.py` | `load_checkpoint(...)` — rebuilds a fresh `Environment`, overlays saved state, reconnects callbacks, and validates. Plus `validate_after_load`. |
| `coffeebench/verify_checkpoint.py` | Automated 4-way equivalence harness (uninterrupted vs resumed vs two branches). Runnable as `python -m coffeebench.verify_checkpoint`. |
| `coffeebench/plot_run.py` | matplotlib dashboards for a run/checkpoint (`--run`) and a compare-branches view (`--compare`). |
| `experiments/deterministic_economy.toml` | Config assigning the scripted model to all six agents; no `main_agent` (runs the full horizon). |
| `tests/test_scripted_and_checkpoint.py` | 17 pytest tests covering the 13 required checks. |
| `docs/deterministic_checkpointing.md` | Full documentation (design, schema, semantics, limitations). |

## 2. What was changed (existing files)

| File | Change |
|---|---|
| `coffeebench/environment.py` | Added `save_checkpoint()`; made `run()` resume-aware (skips the day-0 bootstrap / RNG draws when resuming) and stop+save at a requested day boundary; shares the marketplace's `IdGenerator`; replaced `uuid4` for invoices/journal-entries; added `CHECKPOINT_SCHEMA_VERSION`. |
| `coffeebench/marketplace.py` | Uses the shared `IdGenerator` for listing/offer/deal/message IDs (was `uuid4`). |
| `coffeebench/event_loop.py` | Added `peek()`, `get_state()`, `restore_state()` for scheduler serialization. |
| `coffeebench/agent.py` | `init(seed_initial=True)` — on resume, re-attaches the out-of-band system prompt **without** re-seeding the transcript (no duplicated prompts/messages). |
| `coffeebench/models/__init__.py` | Registers the three scripted models; adds `SCRIPTED_MODEL_NAMES` and `is_scripted_model()`. |
| `coffeebench/main.py` | New CLI flags (`--checkpoint-at-day`, `--checkpoint-out`, `--resume-from`, `--branch-name`); resume dispatch in `main()`; end-of-run usage summary (external API calls vs local scripted invocations); populates `env._run_meta` for checkpoints. |

Nothing in the existing LLM run path changed behaviorally — normal configs still
build and run exactly as before when no checkpoint flags are supplied. The old
`coffeebench/models/heuristic_roaster_model.py` is superseded by the generalized
version in `scripted.py` and is no longer imported (kept for reference).

---

## 3. How to run it

All commands run with **no `.env` and no API keys** (the scripted economy is free).
Use `uv run` (the project venv).

### 3.1 Active zero-LLM 90-day economy

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml --seed 0
```

Ends with a usage summary that should read:

```
[usage] total external API calls: 0
[usage] total input tokens: 0
[usage] total output tokens: 0
[usage] total model cost: $0.0
[usage] (local scripted model invocations: N — no external API cost)
```

Trajectory + live event stream are written under
`trajectories/deterministic_economy/seed_0/`.

Add `--max_days 20` for a quick smoke run.

### 3.2 Save a checkpoint at the start of day 45

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml --seed 0 \
  --checkpoint-at-day 45 \
  --checkpoint-out trajectories/checkpoint_demo/day45.checkpoint.json
```

Runs days 0–44, writes the checkpoint at the day-45 boundary, and exits.
("Day 45" = days 0–44 done, day-44 EOD done, day-45 morning not started; the
next event on resume is day-45 `MORNING_OPEN`.) Any day in `[1, max_days-1]` works.

### 3.3 Resume through the original horizon

```bash
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name resumed \
  --output trajectories/checkpoint_demo/resumed.json
```

Continues from day 45 to day 89 (for a 90-day run).

### 3.4 Multiple independent branches from the same checkpoint

```bash
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name branch_a --output trajectories/checkpoint_demo/branch_a.json

uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name branch_b --output trajectories/checkpoint_demo/branch_b.json
```

Under scripted policies all branches produce identical final results.

### 3.5 Override a model on a branch (economic state preserved)

```bash
# roaster_A driven by a different model from day 45 on; everyone else unchanged.
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --models "roaster_A:gpt-5.5" \
  --branch-name gpt_variant --output trajectories/checkpoint_demo/gpt_variant.json
```

(Use `--models "roaster_A:passive"` to test the override path for free — no keys.)
Each branch builds fresh model clients and its own transcript/usage; nothing
mutable is shared between branches.

### 3.6 Automated verification (the proof)

```bash
uv run python -m coffeebench.verify_checkpoint \
  --config experiments/deterministic_economy.toml \
  --seed 0 --checkpoint-day 45
# add --max_days 30 for a faster run
```

Runs the economy four ways (uninterrupted; checkpoint+resume; two branches) and
asserts they are economically identical, branches share no mutable objects, and
external cost is 0. Prints `ALL CHECKS PASSED ✅`.

### 3.7 Visualize results (matplotlib dashboards)

`coffeebench/plot_run.py` turns a run's data into easy-to-read PNGs. It works on
both a finished `run.json` and a `*.checkpoint.json` (both carry the per-day
`daily_stats` it plots), so you can visualize a run at any stage.

Single-run dashboard (8 panels: net worth, cumulative net income, cash,
inventory value, consumer sales/day, spoilage, working capital, and a final-NI
bar) — spring_break is shaded, bankruptcies marked:

```bash
uv run python -m coffeebench.plot_run --run trajectories/deterministic_economy/seed_0/run.json
# state captured AT the checkpoint stage:
uv run python -m coffeebench.plot_run --run trajectories/checkpoint_demo/day45.checkpoint.json
```

Compare mode — overlay several runs/branches/stages, one panel per agent, with a
vertical fork marker (this is the view that will show branches diverging once
they differ, e.g. baseline vs a Valto-governed branch):

```bash
uv run python -m coffeebench.plot_run --compare \
  resumed=trajectories/checkpoint_demo/resumed.json \
  branch_a=trajectories/checkpoint_demo/branch_a.json \
  branch_b=trajectories/checkpoint_demo/branch_b.json \
  --metric equity --mark-day 45
```

`--metric` accepts `equity` (default), `ni_series` (cumulative net income), or
`cash`. PNGs are written to `--out` (default `plots/`).

### 3.8 Tests

```bash
uv run pytest tests/test_scripted_and_checkpoint.py -q     # 18 passed
```

### 3.9 The original passive baseline still works

```bash
uv run python -m coffeebench.main --model passive --max_days 90 --seed 0
```

---

## 4. Quick reference: new CLI flags

| Flag | Meaning |
|---|---|
| `--checkpoint-at-day N` | Run until the start of day N, save a checkpoint, exit. Requires `--checkpoint-out`. |
| `--checkpoint-out PATH` | Where to write that checkpoint. |
| `--resume-from PATH` | Resume/branch from a checkpoint (continues to the original `max_days`). |
| `--branch-name NAME` | Label for a continuation (recorded in `run_meta` + events). |
| `--models "aid:model,…"` | On resume, override selected agents' models (economic state preserved). |

New scripted model names usable anywhere a model name is expected
(`--model`, `--models`, or a config `[models]` block): `heuristic_farmer`,
`heuristic_roaster`, `heuristic_retailer` (plus the existing `passive`).

---

## 5. Verified results (seed 0, day-45 checkpoint, 90 days)

```
[PASS] RunA(uninterrupted) == RunB(resumed)
[PASS] RunC(branch)        == RunD(branch)
[PASS] RunA(uninterrupted) == RunC(branch)
[PASS] branches share no mutable objects
external_api_calls=0 cost=$0.0 → [PASS]
Activity: 325 listings, 542 offers, 350 deals, 236 consumer sales (~$55.6k)
```

Net income (six scripted agents): farmer_A −$160, farmer_B −$279,
roaster_A +$2,106, roaster_B +$1,838, retailer_A +$10,366, retailer_B +$6,609 —
i.e. both positive and negative outcomes, the full supply chain exercised, at $0
external cost.

See [`docs/deterministic_checkpointing.md`](docs/deterministic_checkpointing.md)
§10 for remaining limitations (mainly LLM-branch transcript fidelity) before
Valto integration.
