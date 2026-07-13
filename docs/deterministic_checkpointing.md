# Deterministic Economy, Checkpointing & Branching

This document describes two capabilities added to CoffeeBench as the substrate
for a future controlled CoffeeBench × Valto experiment (see
`VALTO_EXPERIMENT_PLAN.md`):

1. **An active, zero-LLM economy** — the whole six-agent supply chain runs with
   deterministic scripted agents: no API keys, no provider tokens, no cost.
2. **Exact-day checkpoint / resume / branching** — stop a run at the start of a
   specific simulated day, save a self-contained checkpoint, and resume (or fork
   into several independent futures) from it.

No Valto code is added here. This is the foundation Valto will later build on.

---

## 1. Why this exists

The eventual experiment needs to run CoffeeBench through ~day 45, freeze the
complete economy, and fork it into several independent futures (baseline, static
Valto policy, continuously-learning Valto policy) that share an identical
history but then diverge. Before spending on LLM runs or wiring Valto, we need
confidence that:

- the economy can be exercised **for free** and **deterministically**;
- the simulator can stop at an **exact day boundary**;
- **every** piece of state is captured faithfully;
- a checkpoint can be loaded **many times** into **independent** branches;
- deterministic continuations **reproduce** bit-for-bit (economically);
- each future branch can maintain **its own** model/transcript/usage.

All of this is proven by `coffeebench.verify_checkpoint` (see §9) using scripted
agents at $0 cost.

---

## 2. Zero-LLM scripted mode

### The models

`coffeebench/models/scripted.py` adds one deterministic policy per role, all
sharing a common base (`ScriptedModel`):

| Model name           | Role     | Routine (one tool call/turn, read from the transcript) |
|----------------------|----------|--------------------------------------------------------|
| `heuristic_farmer`   | farmer   | produce green to target → list unlisted stock cost-plus → accept roaster bids ≥ floor → pay AP |
| `heuristic_roaster`  | roaster  | buy green when low → roast (specialty first, shared 50 kg/day cap) → list roasted → accept retailer bids → pay AP |
| `heuristic_retailer` | retailer | buy roasted when low → price the storefront below reservation (small inventory nudge) → pay AP |

Both agents of a role use the same policy class. These are **not** meant to be
optimal — just simple, readable, and deterministic enough to keep every economic
channel active.

They conform to the same `Model` protocol as the provider wrappers and plug into
the standard `Agent` harness, so they pay the same 30-min/turn clock cost and
flow through the same tools as an LLM would. Key properties:

- **No network, no keys, no tokens, no cost.** A scripted `query()` is a local,
  in-process function call. Usage reporting counts these as *local scripted
  invocations*, kept strictly separate from *external API calls* (which stay 0).
- **No unseeded randomness.** Every choice is a threshold or a sorted pick with
  an id tiebreak, so runs are reproducible for a given seed.
- **Robust handshake.** Sellers call `view_listings(only_others=False)` so they
  see their **own** open listings (to accept incoming offers by resolving the
  offer's item) and the **market** (to buy from). Sellers only post the
  *unlisted* portion of inventory, so listings don't accumulate without bound.
  Buyers bid at the ask (capped by a fixed max); sellers accept any bid at/above
  a cost-plus floor — so deals close deterministically.
- **Checkpointable.** Each model implements `checkpoint_state()` /
  `restore_checkpoint_state()` (plain dicts, no pickling) so it can resume
  mid-run without repeating completed actions.

### Config

`experiments/deterministic_economy.toml` assigns the correct scripted model to
all six agents and sets **no** `main_agent` (so the run always completes the full
horizon even if an agent goes bankrupt). Run it with **no `.env` and no keys**:

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml --seed 0
```

At the end it prints:

```
[usage] total external API calls: 0
[usage] total input tokens: 0
[usage] total output tokens: 0
[usage] total model cost: $0.0
[usage] (local scripted model invocations: N — no external API cost)
```

Local scripted invocations are still counted in each model's `n_calls`; they are
clearly labelled and never conflated with external API calls.

### Active-economy coverage (seed 0, 90 days)

A single 90-day scripted run exercises the full chain — measured event counts:
farmer production, marketplace **listings**, **offers**, accepted **deals** (350),
**shipments**, **shipment losses** (seeded), **roasting**, roasted listings,
retailer procurement, **consumer sales** (236 sales, ~1,634 kg, ~$55.6k),
**AR/AP**, **invoice payments**, inventory changes, **spoilage**, and both
positive and negative business outcomes (retailers profit ~$6–10k; farmers run
slightly negative under opex + thin margins).

---

## 3. Exact checkpoint-day semantics

"**Checkpoint at day D**" means precisely:

```
Days 0 .. D-1 have completed.
Day D-1 end-of-day mechanics have completed.
Day D morning processing has NOT started.
The next event after loading is day D MORNING_OPEN.
```

Internally this is the point where the previous day's `EOD_MECHANICS` handler has
run (which advanced the clock to `D * 1440` and scheduled day D's `MORNING_OPEN`)
and the run loop is about to pop that `MORNING_OPEN`. The run loop detects this
boundary by peeking the event queue (`next event == MORNING_OPEN(D)`), writes the
checkpoint, and stops. On resume, the first event fired is exactly `MORNING_OPEN(D)`
— nothing from day D-1 is replayed and nothing in day D is skipped.

`--checkpoint-at-day` accepts any day in `[1, max_days-1]`.

---

## 4. Commands

### Complete deterministic run

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml --seed 0
```

### Create a beginning-of-day-45 checkpoint

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml --seed 0 \
  --checkpoint-at-day 45 \
  --checkpoint-out trajectories/checkpoint_demo/day45.checkpoint.json
```

The process runs days 0–44, writes the checkpoint at the day-45 boundary, and
exits (no trajectory is written — the checkpoint is the artifact).

### Resume through the original horizon

```bash
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name resumed \
  --output trajectories/checkpoint_demo/resumed.json
```

Continues from day 45 through the checkpoint's original `max_days` (day 89 for a
90-day run).

### Multiple independent branches

The same checkpoint can be loaded any number of times into independent
environments/processes:

```bash
uv run python -m coffeebench.main --resume-from …/day45.checkpoint.json \
  --branch-name branch_a --output trajectories/checkpoint_demo/branch_a.json

uv run python -m coffeebench.main --resume-from …/day45.checkpoint.json \
  --branch-name branch_b --output trajectories/checkpoint_demo/branch_b.json
```

Under scripted policies, all branches produce identical final results (verified).
Later, branches deliberately differ (different models, or Valto governance).

### Model overrides after resume

Override selected agents' models **without changing the saved economic state**:

```bash
# Continue with a different model on the focal agent (economic state preserved)
uv run python -m coffeebench.main \
  --resume-from …/day45.checkpoint.json \
  --models "roaster_A:gpt-5.5" \
  --branch-name gpt_variant --output …/gpt_variant.json
```

How it works: the checkpoint records each agent's model assignment. On resume the
saved map is the base, and `--models "aid:model,…"` overrides selected agents.
Each continuation constructs **fresh** model clients and its **own** transcript
and usage counters:

- The overridden agent gets a brand-new client of the new type; its transcript
  (restored from the checkpoint) carries forward, but its scripted internal
  routine is not restored (a swapped-in model derives its own behavior).
- Non-overridden agents keep the same model; their cumulative usage counters are
  restored so full-run totals match an uninterrupted run.
- **Nothing mutable is shared between branches** — not model clients, agents,
  message lists, the marketplace, the event loop, RNG, or id generator.

### Visualizing results (matplotlib dashboards)

`coffeebench/plot_run.py` renders easy-to-read PNGs. It accepts either a finished
`run.json` or a `*.checkpoint.json` (both carry per-day `daily_stats`), so a run
can be visualized at any stage.

```bash
# Single-run dashboard: net worth, cumulative net income, cash, inventory,
# consumer sales/day, cumulative spoilage, working capital (AR-AP), and a
# final-NI bar. The spring_break window is shaded; bankruptcies are marked.
uv run python -m coffeebench.plot_run --run trajectories/deterministic_economy/seed_0/run.json
uv run python -m coffeebench.plot_run --run trajectories/checkpoint_demo/day45.checkpoint.json

# Compare mode: overlay branches/stages, one panel per agent, with a fork line
# at the checkpoint day. This is the view that shows branches DIVERGING once
# they differ (e.g. baseline vs a Valto-governed branch).
uv run python -m coffeebench.plot_run --compare \
  resumed=…/resumed.json branch_a=…/branch_a.json branch_b=…/branch_b.json \
  --metric equity --mark-day 45
```

Colors are consistent (one hue per role; instance A solid, B dashed). `--metric`
is `equity` (default), `ni_series`, or `cash`. Output goes to `--out` (default
`plots/`).

---

## 5. Checkpoint schema

The checkpoint is a single self-contained, versioned JSON file written by
`Environment.save_checkpoint` and read by `coffeebench.checkpoint.load_checkpoint`.
`schema_version` is checked on load; a mismatch raises `CheckpointVersionError`.

Top-level keys:

| Section | Contents |
|---|---|
| **identity** | `schema_version`, `coffeebench_version`, `run_meta` (config source/name, seed, model map, economy overrides, kpi, branch lineage), `max_days`, `main_agent_id`, `next_day`, `next_virtual_min`, `saved_at` |
| **clock + scheduler** | `event_loop` (all queued events + insertion counter + current_min), `available_at`, `pending_wake_reason`, `last_active_day_per_agent`, `eod_done_for_day`, `today_nudge`, `step_count` |
| **RNG / exogenous** | `rng_state` (exact `random.getstate()`), `demand_path`, `retailer_loyalty`, `spoilage_carry` |
| **marketplace** | `items`, `listings`, `offers`, `deals`, `messages`, `read_message_ids`, `id_gen` (deterministic counters) |
| **invoices** | shared invoice **pool** keyed by id (each Invoice stored once; agents reference by id so AR/AP aliasing is reconstructed exactly) |
| **agents** | per agent: `cash`, `inventory`, `inventory_total_cost`, `retail_prices`, `accounts_receivable`/`accounts_payable` (invoice ids), `initial_equity`, `model`, `messages` (transcript), `usage`, `scripted_model_state`, `compactor_events` |
| **pending processes** | `pending_shipments` (deal ids), `pending_production`, `pending_roasting`, `production_used_today`, `roast_used_today` |
| **accounting + history** | `truth_ledger`, `daily_stats`, `consumer_sales_log`, `messages_today`, `messages_per_day` |
| **bankruptcy / termination** | `bankrupt_agents`, `bankrupt_day`, `bankrupt_reason`, `bankrupt_equity`, `terminated_early` |

### Object sharing

Two kinds of objects are aliased in the live env and must stay aliased on load:

- **Invoices** live in both the issuer's `accounts_receivable` and the payer's
  `accounts_payable`. They are stored once in a pool and each agent references
  them by id, so a restored invoice is the *same object* on both sides (paying it
  or accruing a late fee updates both views).
- **Deals** live in both `marketplace.deals` and `env._pending_shipments`. Deals
  are the canonical pool; pending shipments reference them by id.

### Reconnected runtime references

`load_checkpoint` rebuilds a fresh world skeleton via `build_run` (which already
wires all callbacks), then overlays saved state by mutating those objects. After
overlay it runs `validate_after_load`, which asserts every required callback and
cross-reference exists: `marketplace._env`, `on_deal_accepted`,
`marketplace.business_apps`, the shared `id_gen`, each `BusinessApp._env` /
`_record_truth`, each agent's bound tools, and that AR/AP invoice sharing is
intact. Any failure raises `CheckpointValidationError`.

### Not calling `init()` twice

On a normal run, `env.run()` calls `agent.init()` to seed the system prompt and
the initial observation. On resume the transcript is restored, so `env.run()`
calls `agent.init(seed_initial=False)`: it (re)attaches only the out-of-band
`model.system_prompt` (which lives on the fresh per-branch model client, not in
the restored transcript) and appends **no** messages. This prevents duplicated
prompts/observations.

---

## 6. Determinism & deterministic IDs

To make an uninterrupted run and a checkpoint-resumed run byte-identical at the
economic level, all IDs are now minted by a single monotonic, checkpoint-restored
counter (`coffeebench/ids.py`, `IdGenerator`) instead of `uuid.uuid4()`. Prefixes:
`lst` (listings), `off` (offers), `dl` (deals), `msg` (messages), `inv`
(invoices), `je` (journal entries). The generator is owned by the marketplace and
shared with the environment (one counter namespace), and its counters are saved
in the checkpoint. Because IDs are a pure function of how many were minted
before, they reproduce exactly across resume and branches — which also gives
future Valto decision-lineage work stable, greppable identifiers.

This was chosen over "canonicalize random IDs before comparison" because
deterministic IDs are simpler and more useful downstream.

---

## 7. Fields excluded from canonical equality

The equality check (`coffeebench.verify_checkpoint`) compares economic state
only. These non-economic metadata fields are deliberately excluded because they
cannot affect simulation behavior:

- **Agent message timestamps** — `datetime.now()` stamped on each turn in
  `Agent.add_message`. Wall-clock only; never read by the simulator.
- **Checkpoint `saved_at`** — the wall-clock time the checkpoint was written.
- **Event-stream `ts_ms`** — wall-clock offsets in the `.events.jsonl` monitor
  stream. The event stream is an analysis artifact, not simulator state.

Everything economically meaningful — cash, inventory, cost basis, AR/AP, retail
prices, the full truth ledger, all listings/offers/deals/messages (including
their deterministic IDs), pending queues, consumer sales, net income, equity,
bankruptcy state, model call counts, and the id-generator counters — **is**
compared and matches exactly.

---

## 8. What the verifier checks

```bash
uv run python -m coffeebench.verify_checkpoint \
  --config experiments/deterministic_economy.toml \
  --seed 0 --checkpoint-day 45
# add --max_days N for a faster horizon
```

It runs the economy four ways and compares canonical summaries:

- **Run A** uninterrupted, days 0..max_days-1.
- **Run B** run to day D, save checkpoint, reload in a **new** env, run D..end.
- **Run C, D** two independent branches loaded from the same checkpoint.

Assertions: `A == B` (resume is faithful), `C == D` (branches equivalent),
`A == C`, branches share **no** mutable objects, and external API cost is 0.

---

## 9. Verification results (seed 0, day-45 checkpoint, 90 days)

```
[PASS] branches share no mutable objects
[PASS] RunA(uninterrupted) == RunB(resumed)      (economically identical)
[PASS] RunC(branch)        == RunD(branch)       (economically identical)
[PASS] RunA(uninterrupted) == RunC(branch)       (economically identical)
external_api_calls=0 cost=$0.0 local_invocations=8259 → [PASS]

Activity (Run A): listings=325 offers=542 deals=350
consumer_sales: 236 sales, 1634 kg, $55,586
net income by agent:
  farmer_A     NI=$   -160.00   farmer_B   NI=$  -279.00
  roaster_A    NI=$  2106.40    roaster_B  NI=$ 1837.58
  retailer_A   NI=$ 10365.90    retailer_B NI=$ 6609.30
ALL CHECKS PASSED ✅
```

The pytest suite (`tests/test_scripted_and_checkpoint.py`, 18 tests) covers:
passive mode unchanged; each scripted role runs without keys; the six-agent
economy completes; it generates listings/offers/deals/deliveries/roasting/consumer
sales; zero external cost; checkpoint at an arbitrary day; no duplicated
prompts/messages on load; preserved pending shipments/production/roasting;
preserved RNG + scheduler; uninterrupted == resumed; two branches equal; model
override preserves economic state; and existing LLM configs still build with no
checkpoint flags.

```bash
uv run pytest tests/test_scripted_and_checkpoint.py -q   # 18 passed
```

---

## 10. Limitations remaining before Valto integration

- **LLM-branch transcript fidelity.** The scripted path round-trips exactly. For
  LLM agents, an assistant turn's raw provider block (`_raw`, e.g. Anthropic
  thinking-block signatures) is serialized best-effort (`json … default=str`) and
  may not perfectly reconstruct provider-native replay structures. Scripted
  branches (the current deliverable) are unaffected. Validate LLM
  checkpoint/resume fidelity before relying on it for paid runs.
- **System-prompt reconstruction on resume** uses the checkpoint's `run_meta`
  (config source + economy + kpi). If the original config file is moved/deleted,
  resume falls back to the stored model map and settings; for LLM agents whose
  behavior depends on KPI score-framing, keep the source config available for
  full fidelity. (Irrelevant to scripted agents, which ignore the prompt.)
- **Compactor state** is recorded as an event trace only; a mid-compaction LLM
  transcript is not a concern for scripted runs and is not exercised here.
- **Determinism guarantee scope.** Reproducibility holds for the scripted economy
  (seeded env RNG + deterministic agents + deterministic serial tool-apply).
  Live LLM branches will legitimately diverge through their own decisions even
  when they start from identical exogenous state — that divergence is the point
  of the eventual experiment, not a bug.
- **No Valto yet.** Tool governance, decision logging, policies, attribution, and
  policy mining are explicitly out of scope here and come next.
