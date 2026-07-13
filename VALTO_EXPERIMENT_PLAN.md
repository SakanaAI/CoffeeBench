# CoffeeBench × Valto: Governed-Agent Experiment Plan

**Goal.** Show that Valto — a decision layer that sits *before* an agent's
economic actions — measurably improves a Claude Sonnet 4.6 agent's performance
in CoffeeBench, using the exact loop Valto was built for: **observe → mine a
policy from the ledger → enforce it → compare, on paired seeds.** The end state
is *online* policy updates: Valto re-mines and hot-swaps the policy mid-run as
the agent operates.

This document is a plan only. No code has been changed. Everything below
describes what to build, in what order, and why — grounded in how both
codebases actually work today.

---

## 1. How CoffeeBench works (the parts that matter here)

- **What it is.** A 90-day, 6-agent supply-chain economy: 2 farmers, 2
  roasters, 2 retailers. Each agent is an LLM driving a business through a
  fixed tool API. Score = `true_net_income` computed by the env's truth ledger
  over `[0, max_days-1]`. (`coffeebench/main.py`, `coffeebench/environment.py`.)
- **The focal subject.** Experiments swap the model on **one** agent and pin the
  rest. `experiments/roaster_focal_sonnet.toml` sets `main_agent = "roaster_A"`
  with everyone on `claude-sonnet-4-6`. `main_agent` means **the run terminates
  early the moment roaster_A goes bankrupt** (`environment.py` `_terminated_early`).
  So *survival is itself a score lever* — a governed agent that avoids the DNF
  captures the whole back half of the horizon.
- **"Policy" today = the system prompt.** There is no separate strategy object.
  Behavior comes from `SYSTEM_PROMPT` + persona + `score_framing`
  (`main.py:50-251`). Valto adds a *second, enforced* control surface that the
  prompt cannot provide (README §"Why not just use a system prompt?").
- **The agent's economic actions are tool calls.** For a roaster the
  capital-committing ones are (verified against `business_app.py`):
  - **`make_offer(listing_id, offered_price, qty, payment_terms_days)`** — this
    is how a roaster **buys green**: it bids on a farmer's green listing and the
    farmer accepts. This is the roaster's procurement spend (its analog of
    taskmarket's `buy_lookup` / vending's `place_order`). *Note: `accept_offer`
    is the roaster accepting a **retailer's** bid to buy its roasted output — a
    revenue event, not procurement.*
  - **`roast(green_item_id, qty_kg)`** — labor spend ($3/kg commodity, $5/kg
    specialty), consumes green + a shared 50 kg/day capacity, 1-day lag.
  - Secondary: **`accept_offer`** (sell roasted — govern to block below-COGS
    sales / bad-credit buyers), **`pay_invoice`** (settle AP — govern for
    solvency).
- **Rich, already-logged outcome data.** Two artifacts per run:
  - `trajectories/<name>/seed_<N>/run.json` — full end state: truth ledger,
    books (cash, inventory, WAVG cost basis, AR/AP), all deals, messages,
    consumer-sales log, bankruptcy info (`environment.py` `save_trajectory`).
  - `trajectories/<name>/seed_<N>/run.events.jsonl` — a **live event stream**:
    `deal_accepted`, `deal_delivered`, `consumer_sale`, `roast_materialized`,
    `production_materialized`, `shipment_lost`, `shipment_delayed`, spoilage,
    opex, late fees, bankruptcy. This is the raw material for outcome
    attribution.
- **Determinism per seed.** Demand noise path and per-retailer loyalty
  multipliers are pre-generated from `env.rng.seed(seed)` at run start
  (`environment.py` `_generate_demand_paths`, `_generate_loyalty_multipliers`).
  **Same seed ⇒ same world**, which is what makes a paired governed-vs-ungoverned
  comparison a clean causal contrast.
- **Hidden structure the agent must discover.** Consumer demand model, the
  per-retailer loyalty multiplier, the spring_break festival **magnitude**
  (days 40–53, 3.0× — window public, size hidden), delivery loss/delay
  probabilities. This is exactly the "unverified claims / hidden accuracy"
  hazard surface Valto's benchmarks exploit: **the ledger recovers what the
  agent cannot see.**
- **⚠️ No mid-run checkpoint exists.** `env.run()` executes day 0 → `max_days`
  in a single `asyncio.run(...)` and only serializes at the very end. There is
  no "save state at day 44 and resume." This shapes the phasing below (§5).

---

## 2. How Valto works (the parts that matter here)

- **The primitive.** `decide(action, amount, payee, intent, context, rail) →
  Decision(verdict ∈ {approve, deny, modify, require_approval}, reason,
  modified)`. Call it *before* the irreversible action. `report(...)` records
  execution; **`evaluate(decision_id, useful, score, note)` attaches a
  DEFERRED usefulness label** once the value is knowable — this is the hook that
  solves CoffeeBench's cross-day attribution (see §4). (`valto/client.py`.)
- **Embedded, no server.** `Valto.local(actor=..., policy="p.yaml",
  db_path="ledger.db")` runs the policy engine + SQLite ledger in-process
  (`_LocalEngine`). One Valto instance **per governed agent**.
- **Non-invasive tool wrapping.** `valto.protect_call(spec, args, env, run)`
  resolves an `EconomicAction` from the tool's args, calls `decide`, runs the
  tool only on approve, and otherwise returns a JSON **block dict**
  (`{"ok": False, "blocked_by": "valto", "verdict", "reason", "suggestion"}`,
  `valto/protected.py:blocked_result`). That dict can be handed straight back to
  the agent as its tool result — the reason steers the next action. Used exactly
  this way in `benchmark-vending/tools.py` and `benchmark-taskmarket/tools.py`.
- **Policy = YAML**, ordered rules, first match wins. Match keys the engine
  supports today (`valto/engine.py`): `actions`, `counterparties`,
  `amount_gt/amount_lt`, `context_gt/context_lt` (dotted-key thresholds against
  the request `context`), `duplicate_within_hours`; verdicts
  `deny|modify|require_approval` with optional `suggest` (replacement args);
  `default: approve`. Every decision records the policy content hash.
- **Policy mining is a real, shipped script.** `benchmark-taskmarket/analyze.py`
  joins each approved decision's **decision-time context** to its **realized
  utility**, evaluates interpretable rule templates, and **keeps only rules
  whose historical matches were net-negative** (`n ≥ 8`, waste precision `≥ 0.6`,
  total utility `< 0`). Each emitted rule's `reason` quotes the measured stats —
  "the distilled ledger speaking to the agent." We replicate this for coffee.
- **The two published results we are echoing.**
  - *taskmarket:* observe → mine → govern lifted mean profit **+19%** and cut
    wasted spend **52%** on 8 paired seeds; lift concentrated in the worst
    episodes (floor $14.14 → $21.43).
  - *vending:* a **stateful** agent with perishable inventory, demand shocks,
    supplier choice, and paid forecasts — much closer to CoffeeBench. Key
    lesson we must respect: a *broad* policy (Policy B) was **harmful** when left
    on outside the regime it was built for; the win came from **narrow,
    regime-scoped** rules and A→B→A switching. CoffeeBench's festival window is
    the natural analog of vending's "campus shutdown" regime.

---

## 3. Where Valto plugs into CoffeeBench

**Principle:** govern **only roaster_A**; leave the other five agents untouched
so the comparison isolates the focal agent (identical to how the focal-model
configs already work).

**Mechanism (small, localized, no harness change):** in `main.py:build_run`,
after constructing the focal agent's `BusinessApp`, wrap its bound
capital-committing methods with `protect_call`:

```
# pseudocode — main.py, focal agent only
valto = Valto.local(actor="roaster_A", policy=policy_path, db_path=ledger_path)
ba.make_offer = govern(valto, ba.make_offer, SPEC_MAKE_OFFER, ba)   # buy green
ba.roast      = govern(valto, ba.roast,      SPEC_ROAST,      ba)   # labor spend
# optional later: ba.accept_offer (sell roasted), ba.pay_invoice (solvency)
```

Why this is safe and clean:

- `_collect_tools` reads bound methods by name and `Agent`/`ToolSpec.from_function`
  build the JSON schema from the signature + docstring. Wrap with
  `functools.wraps` (which sets `__wrapped__`); `inspect.signature` follows it,
  so **the tool schema the model sees is byte-for-byte unchanged.**
- On deny/modify, the wrapper returns a dict shaped like CoffeeBench's own tool
  errors (`{"status": "rejected", "message": <reason>, "suggestion": ...}`) so
  the model reads it as a normal rejected action and re-plans (re-bids lower,
  switches farmers, skips the roast) — exactly the taskmarket/vending pattern.
- The **five non-focal agents never touch Valto**, so their behavior — and thus
  the world dynamics around roaster_A — is identical across arms at a given seed.

**Decision-time context (observable only — never hidden world state):**

| For `make_offer` (buy green) | For `roast` (produce) |
|---|---|
| `amount = offered_price × qty` | `amount = qty × labor_rate` (3 or 5) |
| `payee = seller` (farmer_A/B) | `item = green→roasted recipe` |
| `unit_price` vs **rolling median accepted green price** (from `view_market_aggregate` / `view_deals`) | `days_to_festival` (40 − day, clamped) |
| `days_to_festival`, `day` | `roasted_on_hand + in_flight` vs 120 cap |
| `cash`, cash-runway = cash ÷ known daily burn | `green_on_hand` available |
| `committed_inventory` (on-hand + in-flight) vs 120 cap | `cash`, cash-runway |
| `outstanding_AP`, counterparty late/loss history | expected demand vs current roasted stock |

All of these are computable from the focal `BusinessApp`'s own views + a small
rolling-stats helper — nothing the agent couldn't itself observe.

---

## 4. The attribution problem (the crux — and how we solve it)

taskmarket has **zero** cross-day attribution "by construction": a lookup's
utility is `payout − price`, known immediately. **CoffeeBench does not.** A green
buy on day 5 → roast day 6 → B2B sale day 8 → NET-30 → cash day 38, and its
profitability is entangled with spoilage, festival timing, and downstream
pricing. CoffeeBench also uses **WAVG cost basis, not FIFO lot tracking**, so
exact per-unit provenance is *not* available from the sim as-is. We handle this
in two escalating tiers:

**Tier 1 — proxy waste/utility labels from existing logs (no sim change).**
After each run, an outcome-labeling pass walks `run.events.jsonl` +
`truth_ledger` and attaches a deferred `evaluate()` label to every governed
decision, using signals that *are* cleanly attributable:

- **Spoilage-after-buy/roast:** did the agent book `spoilage_expense` on that
  item within a W-day window of the decision? (over-buying / over-roasting).
- **Bought-above-market:** `offered_price` > rolling median accepted green
  price at decision time → margin erosion (directly measurable from `deals`).
- **Roasted-then-idle:** roasted output that sat and later spoiled or was sold
  below its build cost (COGS from truth ledger vs realized `sale_revenue`).
- **Counterparty quality:** buys from a seller whose delivery history shows
  `shipment_lost` / chronic `shipment_delayed`, or AR that became `bad_debt`.
- **Solvency proximity:** spends that pushed cash-runway below the known
  forward opex+AP obligation (the bankruptcy-DNF precursor).

Utility per decision = a signed score combining these (e.g. realized margin of
the attributable output minus attributable spoilage/write-down/late-fee), or, as
a coarse fallback, the agent's **forward Δtrue_equity over W days** apportioned
to the decision. This is enough to run taskmarket-style mining today.

**Tier 2 — exact decision-tagged provenance (small sim addition, Phase 2).**
Mirror what `benchmark-vending` did ("exact lot provenance: every unit sold,
expired, held, or salvaged tied to the order that purchased it"). Add a
decision-id tag that rides from `make_offer`/`roast` through the pending queues
into `_record_truth` entries, so `sale_revenue`, `inventory_out`,
`spoilage_expense`, and `writedown` can be joined back to the originating
decision exactly. This upgrades utility from proxy to exact and is the honest
basis for a published ROI number. It is the one nontrivial code addition and is
deliberately deferred to Phase 2.

**Highest-legibility, attribution-light win: solvency / survival.** Because
`main_agent` bankruptcy *ends the run*, a Valto rule that `require_approval`s or
`deny`s a spend that would drop cash-runway below known forward obligations
directly prevents the DNF. Its "utility" is unambiguous and needs no lot
tracking: **it keeps the agent alive to earn the back half of the horizon.**
This is the cleanest first product story and should anchor the initial policy.

---

## 5. Experiment design

### 5.1 Conditions (per Valto's own benchmark grammar)

| Arm | Valto | Policy | Purpose |
|---|---|---|---|
| **`observe`** | on | approve-all `observe.yaml` | Behaviorally identical to ungoverned baseline; records the training ledger (taskmarket's `observe == baseline`). |
| **`governed`** | on | mined `learned.yaml` | Rules enforced on roaster_A. |
| *(control)* `baseline` | off | — | Optional; `observe` already doubles as it. |

Everyone except roaster_A stays on `claude-sonnet-4-6`, exactly as
`roaster_focal_sonnet.toml` today.

### 5.2 Phase A — first credible result (no checkpoint needed) ✅ recommended first

This is the paired full-episode design that matches taskmarket/vending exactly
and sidesteps the missing-checkpoint problem.

1. **Collect ledger.** Run `observe` on paired seeds (e.g. `0–7`), 90 days each.
   → `run.json` + `run.events.jsonl` + Valto SQLite ledgers.
2. **Label outcomes** (Tier 1 pass) and mine the policy **on train seeds only**
   (e.g. `0–3`), holding out `4–7`.
3. **Enforce.** Run `governed` on all seeds `0–7` with `learned.yaml`.
4. **Compare within-seed:** `governed_NI[s] − observe_NI[s]`.

> On "first 44 days, then compare with/without": in Phase A the day-44 idea is
> realized as **train/test seed split**, not a mid-run fork — the observe run
> *is* the "data Valto gave us," mined into the improved policy, then the
> governed run is "with Valto" vs the observe run "without." This gives a valid
> causal claim now, with zero new simulator machinery.

### 5.3 Phase B — the "save state at day 44, fork, and online-update" design 🎯 the product vision

This is the user's literal framing and the strongest demo, but it **requires
adding checkpoint/resume to `Environment`** (does not exist today). Flow:

1. Run roaster_A under `observe` to **day 44**, snapshot world state.
2. Mine a policy from the **days 0–43** ledger. Day 44 is a deliberate pick: it
   sits **inside the spring_break window (40–53)**, so by then the agent has
   generated a full normal-regime baseline *plus* the first ~4 days of the
   festival — enough to let the miner **recover the hidden 3× festival
   magnitude** and reprice/restock for the remaining ~10 festival days. (If we'd
   rather govern the *entire* festival, snapshot at day 40 instead; if we want a
   clean post-festival regime, snapshot at 54. Worth A/B-ing the snapshot day.)
3. **Fork two continuations from the identical snapshot**, same seed: one with
   `learned.yaml` enforced (44→90), one ungoverned (44→90). Because they start
   from byte-identical state, the second-half NI difference is a razor-clean
   estimate of the intervention's effect.
4. **Online updates:** re-mine every N days from the growing ledger and
   hot-swap via `engine.load_policy` — Valto's "continuous policy update" story,
   and the analog of vending's A→B→A regime switching.

**Checkpoint spec (what `save_trajectory` already covers vs. what to add).**
`save_trajectory` already serializes cash, inventory, `inventory_total_cost`,
AR/AP, `truth_ledger`, marketplace `listings/offers/deals/messages`, per-agent
`messages` (LLM transcripts), and `models`. A faithful **resume** additionally
needs:

- `rng.getstate()` (determinism after the fork), plus `_demand_path` and
  `_retailer_loyalty` (regenerated from seed today — must be *frozen* instead).
- The in-flight queues: `_pending_shipments`, `_pending_production`,
  `_pending_roasting`; and `_spoilage_carry`.
- Scheduler/agent-clock state: `time_manager.virtual_min`, `_available_at`,
  `_last_active_day_per_agent`, `_messages_today`, `_eod_done_for_day`.
- Bankruptcy state: `_bankrupt_agents/day/reason/equity`.
- Rehydrate each `Agent.messages` and re-`init()` the model system prompt.

This is a self-contained addition (a `save_checkpoint`/`load_checkpoint` pair
next to `save_trajectory`) and is the main engineering investment for Phase B.

### 5.4 Metrics & success criteria

Report a Valto-style paired table (as in vending/taskmarket READMEs):

- **Primary:** `true_net_income` (the leaderboard score), governed vs ungoverned,
  paired-seed mean ± std, **lift**, and **paired win rate** (# seeds where
  governed ≥ ungoverned).
- **Survival:** bankruptcy/DNF rate and mean `final_day` for roaster_A (the
  main_agent early-stop makes this a first-class metric).
- **Waste reduction:** total `spoilage_expense`, `writedown` (lost shipments),
  `bad_debt_expense`, `interest_expense` (late fees); share of green bought
  above rolling-median market price.
- **Festival capture:** roaster_A's B2B roasted `sale_revenue` during days 40–53.
- **Governance activity:** intervention count, rule-hit histogram, and agent
  compliance (did it adopt `modify` suggestions?).

**We can claim success if**, on held-out seeds, governed shows a positive paired
NI lift with a majority win rate, *and* the lift concentrates in the weakest
ungoverned episodes (the taskmarket signature) — ideally driven by fewer
bankruptcies + less spoilage/late-fee waste, each attributable to a named rule.

### 5.5 Candidate seed rules the miner is likely to emit

Grounded in CoffeeBench mechanics — each will only ship if the ledger proves it
net-negative (`n ≥ 8`, precision `≥ 0.6`, total utility `< 0`):

- `green_above_market_deny/modify` — bid `offered_price` above rolling-median
  accepted green → deny, or `modify` down toward the median.
- `overcommit_inventory` — buy/roast when committed holdings already near the
  120 kg cap and recent spoilage > 0 → deny (kills the hoard-then-spoil loss).
- `solvency_buffer` — spend that drops cash-runway below known forward opex+AP →
  `require_approval` / deny (the DNF preventer).
- `festival_underroast` — in the days approaching day 40, roasted stock below
  projected festival demand → (as `require_approval` nudging *more* roast, or an
  informational reason) — the "stock up for the spike" rule the hidden 3× magnitude justifies.
- `bad_counterparty` — buy green from a farmer with a poor delivery-loss/late
  record when the alternate farmer is healthier → `modify` to redirect.
- `below_cogs_sale` (if `accept_offer` governed) — accept a retailer bid whose
  price is under the roasted build cost → deny.

---

## 6. Deliverables & file layout (proposed)

Keep it isolated from core CoffeeBench so upstream stays clean:

```
coffeebench/valto/                    # new integration package
  __init__.py
  govern.py            # protect_call wrappers + ProtectedToolSpecs + context extractors
  labeling.py          # Tier-1 outcome-labeling pass (events.jsonl + truth_ledger → evaluate())
  analyze_coffee.py    # miner: Valto ledger + labels → policies/learned.yaml  (fork of taskmarket/analyze.py)
  run_valto_experiment.py  # orchestrates observe/governed arms over seeds; writes comparison table
  policies/
    observe.yaml       # approve-all
    learned.yaml       # mined (generated)
experiments/
  roaster_focal_sonnet_valto_observe.toml    # focal=roaster_A, [valto] mode="observe"
  roaster_focal_sonnet_valto_governed.toml   # focal=roaster_A, [valto] mode="governed", policy=...
```

Add a small `[valto]` block to `RunConfig` (`config.py`) —
`{ enabled, actor, mode, policy_path, govern_tools = ["make_offer","roast"] }` —
read in `build_run` to decide whether to wrap the focal agent's tools. Default
absent ⇒ Valto off ⇒ existing runs unaffected.

---

## 7. Cost & risk controls

- **Budget.** The README warns ~$200 and ~5 h per 90-day run. Phase A at 8
  seeds × 2 arms = 16 runs is real money/time. Mitigations, in order:
  1. Pin the **five non-focal agents to `claude-haiku-4-5`** (or even
     `heuristic_roaster`/`passive` where acceptable) via the `[models]` map —
     keeps roaster_A on Sonnet where the signal is, slashes cost.
  2. Smoke-test the whole pipeline at `--max_days 20` and 2 seeds first.
  3. Start with fewer seeds (e.g. 4) and widen only if the signal is promising.
- **Don't let a broad policy backfire** (the vending Policy-B lesson): prefer
  `require_approval`/`modify` over hard `deny`; scope festival rules to the
  window; always validate the mined policy on **held-out** seeds before claiming
  a win, and keep the observe arm as the honest control.
- **Fail-open by default** for embedded Valto so a labeling/miner bug never
  crashes a $200 run; protected tools already fail *closed* only when the action
  can't be resolved.
- **Attribution honesty.** Tier-1 proxies are fine for internal iteration and
  the survival/waste story; gate any *published ROI %* on Tier-2 exact
  provenance so the number is defensible.

---

## 8. Milestones

1. **M0 — wiring smoke test.** `[valto] mode="observe"` on roaster_A wraps
   `make_offer`/`roast`; a 20-day, seed-0, non-focal-on-haiku run completes; the
   Valto SQLite ledger fills with decisions + reports. *(Proves the plumbing.)*
2. **M1 — labeling + miner.** Tier-1 labeling pass runs; `analyze_coffee.py`
   emits a `learned.yaml` with quoted stats and a `LEARNED_POLICY_REPORT.md`.
3. **M2 — Phase A result.** Paired observe-vs-governed over 4–8 seeds; produce
   the comparison table (NI lift, win rate, survival, waste). *This is the
   shareable "Valto improves the coffee agent" result.*
4. **M3 — checkpoint/resume.** `save_checkpoint`/`load_checkpoint`; verify a
   day-44 snapshot → resume reproduces an uninterrupted run bit-for-bit at fixed
   seed. *(Unlocks Phase B.)*
5. **M4 — Phase B fork demo.** Day-44 snapshot → mine on 0–43 → fork governed
   vs ungoverned 44→90 on the same seed; report the razor-clean second-half lift.
6. **M5 — online updates.** Re-mine + hot-swap every N days; show the policy
   version advancing in the ledger and NI tracking it — the continuous-update
   product.

---

## 9. Why this ecosystem is a strong Valto testbed (the thesis)

CoffeeBench supplies everything Valto's value proposition needs and that a
toy market can't: **long-horizon compounding**, **hidden economics the agent
must infer** (demand curve, loyalty, festival magnitude, delivery risk), **real
waste channels** (spoilage, lost shipments, late fees, bad debt), a **hard
survival constraint** (bankruptcy ends the run), and **per-seed determinism** for
clean paired causal contrasts. That maps one-to-one onto Valto's loop —
*observe every economic action, attribute its realized outcome, mine an
interpretable policy, enforce it before the next action, and keep updating.* If
Valto lifts a strong Sonnet agent's net income and survival here — especially in
its worst episodes — that is a credible, legible demonstration that a decision
layer beats an ungoverned agent, and the online-update phase is the product.

---

## 10. Integration status & core-package changes (IMPLEMENTED)

This section records what has actually been built (M0 done) and the change made
to the Valto core package, so the decision/outcome data model is captured.

### 10.1 Packaging — install Valto editable from the repo (not PyPI)

CoffeeBench depends on Valto as an **editable path dependency** to the sibling
checkout, because we modify Valto core:

```toml
# CoffeeBench/pyproject.toml
dependencies = [ ..., "valto" ]
[tool.uv.sources]
valto = { path = "../valto", editable = true }
```

`uv sync` installs it editable (`valto==0.1.1 from file:///…/valto`). We use the
embedded engine (`Valto.local()`), so only the base SDK deps (`httpx`, `pyyaml`)
are pulled — no server extras. Any edit in `/valto` takes effect immediately.

### 10.2 Core-package change MADE: arbitrary outcome-metrics logging

**The one gap for "log all kinds of decision + outcome data."** Valto's
*decision* side was already fully flexible — `decisions.context_json` stores an
arbitrary JSON dict, so we log as many decision-time features as we want with no
core change. The *outcome* side was not: the `outcomes` table had only fixed
columns (`executed`, `usdc_paid`, `result_summary` string, `useful`,
`usefulness` scalar, note) — nowhere to put the many realized metrics a rich sim
produces.

Change made in `/valto` (backward-compatible, additive):

- `ledger.py`: new nullable `metrics_json` column on `outcomes` (+ the same
  ALTER-TABLE migration pattern as the usefulness columns). `record_outcome` and
  `record_usefulness` now accept a `metrics: dict` and **merge** it into the
  stored JSON (so the two-phase `report()` → deferred `evaluate()` flow
  accumulates: execution-time metrics at report, realized metrics at evaluate).
  `list_decisions` unpacks `metrics_json` → `metrics` dict.
- `client.py`: `report(..., metrics=None)` and `evaluate(..., metrics=None)`
  thread through to the embedded engine and the HTTP body; the
  `outcome_evaluator` callback may now return a `"metrics"` dict that
  `protect_call`/`guard` log automatically.
- `server.py`: `metrics` field added to the HTTP outcome/usefulness models for
  parity.
- Tests added in `/valto/tests/test_sdk.py` (round-trip + merge + optional).
  All 17 Valto tests pass.

This makes Valto's ledger **self-sufficient for outcome attribution / policy
mining** — no need to re-join to external run files — which is exactly Valto's
product thesis ("the ledger is what Valto sells").

### 10.3 Integration built: `coffeebench/valto/`

- `govern.py` — `make_valto(actor, mode, policy, db_path)` builds the embedded
  client; `govern_business_app(valto, ba, tools)` wraps the focal agent's bound
  `make_offer` / `roast` with `protect_call` **before** `_collect_tools`, using
  `functools.wraps` so the **tool schema the model sees is byte-for-byte
  unchanged** (asserted by a test). On deny/modify the Valto block is reshaped
  into CoffeeBench's `{"status":"rejected","message":…}` so the agent re-plans.
- `policies/observe.yaml` — approve-all (log-only) observe policy.
- `[valto]` config block on `RunConfig` (`enabled`, `actor`, `mode`,
  `policy`, `govern_tools`, `db_path`); read in `build_run`. Absent ⇒ Valto off
  ⇒ existing runs unaffected. Ledger defaults to
  `trajectories/<name>/seed_<N>/valto_ledger.db`.
- `env._valto` / `env._valto_actor` expose the client for later analysis. It is
  **not** serialized into checkpoints — each branch builds its own client, so no
  Valto state is shared across branches (satisfies the checkpoint non-sharing
  rule).

### 10.4 The decision + outcome data model (what we log)

**Decision-time context (observable-only, per governed action):**

| `buy_green` (make_offer) | `roast` |
|---|---|
| day, days_to_festival, in_festival | day, days_to_festival, in_festival |
| item, seller, offered_price, qty, spend, payment_terms_days | input_item, output_item, qty_kg, labor_spend |
| market_median_price, price_to_median | green_on_hand, roasted_on_hand |
| green_on_hand, roasted_on_hand | committed_kg, cap_headroom_kg |
| committed_kg, cap_headroom_kg | roast_cap_used_today, roast_cap_remaining_today |
| cash, cash_runway_days, outstanding_ap | cash, cash_runway_days |
| counterparty_prior_deals | |

**Execution-time outcome metrics (from the tool result, via `outcome_evaluator`):**
`executed_ok`, `action`, and per-action facts — `offer_id`/`seller` for buys;
`roasted_output_kg`/`labor_cost`/`ready_day`/`roasted_unit_cost_estimate` for
roasts; `reject_reason` on env rejections. All merged into `metrics_json`.

**Deferred realized-outcome metrics (Tier-1 labeling pass — next milestone).**
A run-end pass will `evaluate()` each decision with realized economics joined
from the truth ledger + event stream (attributed spoilage, realized margin,
days-to-sale, late fees, downstream Δequity, was-sold, shipment-lost). These
merge into the same `metrics_json` — the ledger then holds the full
decision→outcome story per decision, ready for mining.

### 10.5 M0 result (free scripted economy, $0)

`experiments/roaster_valto_observe_scripted.toml` (all six agents scripted,
observe on roaster_A, 20 days): **56 decisions logged** (37 `buy_green`, 19
`roast`), all `approve`, **all with rich context (18/15 fields) + structured
outcome metrics**, at **zero external API cost**. Integration tests
(`tests/test_valto_integration.py`) assert the ledger fills, cost is zero, and
the governed tool schema is unchanged. Plumbing proven.

### 10.6 Likely future core-package changes (noted, not yet needed)

- **Policy match vocabulary.** The engine's `KNOWN_MATCH_KEYS` currently covers
  `actions`, `counterparties`, `amount_gt/lt`, `context_gt/lt` (numeric
  dotted-key thresholds), `duplicate_within_hours`. CoffeeBench governed rules
  will likely want **`context_eq`** (exact match, e.g. `item ==
  "green_specialty_kg"`, `in_festival == 1`) and possibly **`context_in`**
  (membership). Both are small additive functions in `engine.py` +
  `KNOWN_MATCH_KEYS`. Add when the miner emits rules that need them.
- **Branch-specific ledgers.** On a resumed *governed* branch, the ledger
  db_path should carry the branch name so parallel branches don't share a DB.
  Handled at the CoffeeBench integration layer (pass a branch suffix), not a
  Valto core change.

### 10.7 M1 built: attribution (labeling) + policy miner (free, done)

- **`labeling.py` — outcome attribution (M1a).** Runs at run-end (auto-wired in
  `main()`); joins every governed decision to its realized consequences from the
  **env truth ledger** (ground truth, not the agent's report), by explicit keys
  (buy→deal via `offer_id`; spoilage→originating buys/roasts, windowed to
  `(d−W, d]`, default W=14). Computes structured outcome metrics + a waste-based
  utility and writes them back via `evaluate(metrics=…)`. Emits a
  **reconciliation** line: Σ attributed spoilage vs realized spoilage, with the
  unattributed residual — the "is the outcome signal trustworthy" gauge.
  *Verified on the free scripted economy: 56 decisions labeled, spoilage
  attribution **coverage = 1.0** ($13.98/$13.98, $0 unattributed).*
- **`analyze_coffee.py` — policy miner (M1b).** The **generator/validator split**
  from the design discussion: pluggable `GENERATORS` (interpretable templates
  today — overpay, overcommit, solvency, bad-counterparty — with grid-searched
  thresholds; a tree-learner or LLM proposer drops in later behind the same
  contract) and a fixed strict `backtest` (ship only if matches are net-negative
  utility on the eval rows: n≥8, precision≥0.6, total_utility<0). Held-out split
  by ledger/seed when multiple are given. *On the clean scripted economy it
  honestly emits **0 rules** — no hallucinated governance on well-behaved data;
  real rules will appear from the Claude run's context-dependent mistakes.*
- Utility is **logged as components** (overpay, attributed_spoilage, waste,
  delivered, …) with a default scalar combination, so W and the utility formula
  are swappable offline and ML on the same table stays open ("log raw, model
  later").
- Tests: `tests/test_valto_integration.py` now covers labeling reconciliation and
  that the miner always emits a validator-passing policy. Full suite green (23),
  ruff clean.

**Status: the entire observe → label → mine → govern loop runs end-to-end at $0
on the scripted economy.**

### 10.8 Paid smoke (Haiku) + Phase B orchestrator (built)

- **Cheaper model.** `claude-haiku-4-5` is a valid model string with pricing
  already wired (~3× cheaper than Sonnet). It's also a *better* experimental
  subject: a weaker agent makes more context-dependent mistakes → more mineable
  waste → a clearer Valto lift (matches why taskmarket/vending used a small
  model). Haiku is the primary experiment model; Sonnet is an optional
  robustness arm.
- **Smoke A (Haiku observe, 12 days, ~$0.67):** loop confirmed with a real LLM —
  22 decisions logged with rich context, labeling coverage **1.0**, Haiku traded
  coherently (NI +$261, survived). The miner found its **first real rule** from
  Haiku's own mistakes: `overcommit_roast` (8 roasts into thin cap headroom, 88%
  wasteful, −$13.91 → require_approval) — while correctly **rejecting** the
  non-wasteful `overcommit_buy_green` candidate (0% precision). The
  generator/validator split works.
- **Branch-specific Valto ledgers.** `load_checkpoint(..., valto={...})` governs
  a resumed fork independently of the checkpoint's original run, with a
  branch-suffixed `valto_ledger_<branch>.db` so forks never share a DB. Governance
  can differ per fork (governed vs baseline) — the whole point of the experiment.
- **`run_experiment.py` orchestrator.** One command runs Phase B in stages:
  STAGE 1 observe→checkpoint@D→label+mine (with `--stop-after-mine` to inspect
  the policy before spending on forks); STAGE 2 forks governed vs baseline from
  the identical checkpoint and prints a paired NI/survival comparison. Validated
  end-to-end at $0 on the scripted economy (correctly yields governed==baseline
  when 0 rules are mined — no fabricated lift).

**Phase B Haiku result (seed 0, revenue_max pressure):** governed **underperformed**
baseline by **−$1,181** (NI $463 vs $1,644). The mined `deny cap_headroom<40` rule
blocked 8 of 22 roasts → production halved → revenue $3.3k vs $7.2k → spoilage
*rose*. This is the documented vending failure mode ("a broad policy hurt the
agent; the ledger attributed the damage"). **Root cause: the utility metric
measured waste (spoilage), not net contribution (profit)** — so it flagged
profitable-but-slightly-spoiling roasts as bad and denied the profit with the
waste. The full loop worked flawlessly (attribution coverage 1.0, harm detected
and attributed to the exact rule); the *policy* was wrong because we optimized the
wrong objective. Valto made policy testable and reversible — but the placement +
objective were off.

### 10.9 Reframe: govern the margin engine, not just the leaks (Option 1, built)

The Phase-B post-mortem exposed the structural issue: we governed **volume**
decisions (`roast` qty, `make_offer` qty) and scored by **waste**, but roaster_A's
money is in **pricing** — Haiku sold roasted at $10/kg when retailers pay ~$14–16
(~$2,400 left on the table in 45 days, ~10× all spoilage). Valto's approve/deny/
modify-at-a-price primitive maps most naturally onto pricing, and that's exactly
what we hadn't been governing. Rebuilt the integration to **Option 1** — instrument
the full margin engine in **observe (log-only)** mode under the **original prompt**
(no KPI pressure), to produce a clean raw dataset for offline analysis:

- **Four governed/logged decisions:** `make_offer` (buy green), `roast` (produce
  + mix), **`post_listing`** (set the sell ASK — the margin lever), **`accept_offer`**
  (execute a sale). Each with pricing-aware context (ask/bid vs market median,
  margin-at-price over cost basis, days-of-inventory, festival proximity).
- **Real agent intent** — the wrapper captures the agent's *actual* stated
  rationale (assistant content + extended-thinking) at tool-call time into Valto's
  `intent` field, instead of a template. (Verified it reads real reasoning.)
- **Margin-centric outcome labeling** — per decision: realized margin, days-to-sale,
  fill rate, price-vs-market, margin-left-on-table (sell side); delivered/lost,
  overpay, attributed spoilage (buy/produce side); buyer-paid / bad-debt on sales.
- **Raw-data exporter** (`coffeebench/valto/export.py`) — flattens the labeled
  ledger to `valto_decisions.csv` (one row/decision, `ctx_*` + `out_*` columns) +
  `.jsonl`, auto-written next to the ledger after every governed run. **This is
  the artifact for offline analysis.**
- Validated free on the scripted economy end-to-end: 102 decisions (37 buy / 19
  roast / 19 list / 27 sell), **91 columns**, all with realized outcomes, sell-side
  realized margin $2,212, spoilage reconciliation coverage 1.0. Full suite green
  (23), ruff clean.
- Config: `experiments/roaster_valto_full_haiku.toml` (original prompt, observe,
  all four tools). **This is the next paid run** — produces the raw dataset for
  offline analysis of where governance should live to be economically positive.

**Verdicts:** `require_approval` is not usable in a fully-automated run (no human
approver), so the miner now emits only `deny`/`modify`; enforcement comes after
the offline analysis identifies economically-positive rules.

---

## 11. Experiment campaign — how it works, findings, and next options

This section is the running record of the CoffeeBench × Valto experiment
campaign: the end-to-end system as it actually works now, what we learned from
each run, the code that was changed (and whether to keep it), and concrete
options to get better data/experimentation.

### 11.1 How the whole pipeline works (end-to-end)

```
[config.toml]                         one config drives everything
   ├─ [models]  which model per agent (gpt-5.4-mini / claude-haiku-4-5 / gemini-* / scripted)
   ├─ [economy] env difficulty overrides (spoilage, p_res_commodity, demand_sigma, …)
   └─ [valto]   enable, actor=roaster_A, mode=observe|governed, govern_tools, policy

RUN (coffeebench.valto.run_experiment  OR  coffeebench.main)
 1. build_run(): wires 6 agents; if [valto].enabled, wraps the FOCAL agent's
    make_offer/roast/post_listing/accept_offer with valto.protect_call BEFORE the
    tools are registered (functools.wraps → tool schema unchanged to the model).
 2. env.run(): the live economy. Each governed tool call →
       decide()  logs decision-time CONTEXT (39 ctx_* features) + the agent's REAL
                 stated intent (assistant message content+thinking) + verdict.
       run()     executes the tool (observe mode = always approve).
       report()  logs execution-time OUTCOME facts (offer_id, roasted_kg, fill…).
 3. checkpoint@day D (optional): saves a full, restorable snapshot; run stops.
 4. label_run(env): joins every decision to its REALIZED outcome from the ENV
    truth ledger (ground truth, not the agent's word), by explicit keys
    (deal_id / listing_id / roast-batch), over a W=14d window. Writes margin-
    centric metrics + a waste-based utility back via evaluate(). Prints a
    reconciliation coverage (Σ attributed spoilage / total) — the "is the signal
    trustworthy" gauge.
 5. export_ledger(): flattens the labeled ledger to valto_decisions.csv (one row
    per decision, ctx_* + out_* columns) + .jsonl. THE artifact for offline analysis.
 6. mine (analyze_coffee): GENERATORS propose interpretable rules over ctx_*
    features; a fixed VALIDATOR keeps only rules whose historical matches were
    net-negative utility on held-out rows (n≥8, precision≥0.6). Emits learned.yaml.
 7. fork (stage 2): resume the SAME checkpoint twice — governed (learned.yaml) vs
    baseline (ungoverned) — 45→90, and print the paired NI/survival comparison.
    Branch-specific Valto ledgers so forks don't collide.
 8. plot_run: dashboards (per-agent metrics over time) + compare (governed vs
    baseline diverging at the fork).
```

Key properties: the tool schema the model sees is unchanged; the focal agent is
isolated (only it is governed); ground-truth outcomes come from the env, never
the agent; checkpoints are branch-independent (no shared Valto/model state);
everything except the paid LLM calls is free and deterministic-friendly.

### 11.2 Findings across the campaign

| Run | Setup | Result | Lesson |
|---|---|---|---|
| Deterministic economy | 6 scripted, checkpoint/branch | verified reproducible, coverage 1.0 | substrate works |
| Haiku, unpressured | 1 Haiku + 5 scripted, observe | competent agent; 1 tiny rule ($28 spoilage) | capable models waste little |
| Haiku, `revenue_max` | pressure induces mistakes | mined `overcommit_roast` (n=21, 90%, −$46) | pressure creates minable waste |
| Haiku, governed fork | enforce that rule 45→90 | **governed LOST −$1,181** (revenue halved) | **waste-only utility ≠ profit** — denying slightly-wasteful-but-profitable roasts destroys NI |
| gpt-5.4-mini, easy | all-6 live economy, observe | disciplined; prices $14.5 (fill 1.0); **0 rules** | strong models run the task well → little to govern |
| gpt-5.4-mini, HARD (A+B) | thin margin ($20) + 3% spoilage | real pricing mistakes appear (unsold listings → spoil), but **whole economy underwater** (all 6 agents −$0.9k…−$1.7k); **0 auto-rules** | **A+B overshot**: a losing game for everyone, not "viable-but-hard"; roaster_A was the *least-bad* agent |

**Cross-cutting lessons:**
1. **Valto's value = agent fallibility × task hazard.** Capable modern LLMs
   (Haiku, gpt-5.4-mini) run this roaster task well, so waste-reduction governance
   has little to grab unless the environment is genuinely hard or the agent is
   pressured/weaker.
2. **Optimize net contribution, not waste.** The Haiku fork proved that denying
   "wasteful" actions destroys profit when the waste is small relative to the
   margin. Utility must be `realized_margin − attributed_waste`, not `−waste`.
3. **The miner needs pricing templates.** We instrumented + labeled the sell side
   (post_listing / accept_offer), but `analyze_coffee.py` only has buy/roast rule
   GENERATORS. The pricing signal is now IN the data (fill-rate → spoilage) but
   the auto-miner can't propose rules for it yet.
4. **Environment calibration is delicate.** Easy → no mistakes (0 rules);
   too-hard (A+B) → everyone loses, so the roaster's mistakes are indistinguishable
   from a losing world. Target: **viable-but-hard** — agents can profit with good
   decisions, so bad decisions stand out.
5. **The pipeline works and is cheap.** Full 45-day live 6-agent observe on
   gpt-5.4-mini = **~$2.3** (OpenAI prompt-caching gives ~99% cache hits on the
   growing context). Multi-seed experiments are now affordable (~$12–15 for a full
   3-arm run).
6. **Provider notes:** gpt-5.4-mini is the sweet spot (cheap, capable, no
   throttling, proven on Valto's own benchmarks). Haiku works (~3× pricier).
   Gemini free-tier is unusable here — constant `429 RESOURCE_EXHAUSTED` under the
   6-agent parallel dispatch (needs a paid tier).

### 11.3 Code changed for these runs — and whether to revert

**None of it needs reverting.** Every change is either (a) additive and gated by
config (off by default), or (b) a strict robustness/pricing improvement. The
experiment-specific *difficulty* lives entirely in **config files** (`[economy]`
overrides), not in code — so default runs are unaffected.

| File | Change | Keep? |
|---|---|---|
| `/valto` `ledger.py`, `client.py`, `server.py` | `metrics_json` outcome column + merge; `metrics=` on report/evaluate | **Keep** — genuine Valto improvement (rich outcome logging) |
| `coffeebench/valto/*` (new package) | govern / labeling / analyze_coffee / export / run_experiment / observe.yaml | **Keep** — the integration; inert unless `[valto].enabled` |
| `coffeebench/config.py`, `main.py` | `[valto]` block + wiring; `valto_override`; export hook | **Keep** — gated by `[valto].enabled` |
| `coffeebench/checkpoint.py` | governance + branch-ledger on resume | **Keep** — only active when a fork passes `valto=` |
| `coffeebench/main.py`, `config.py` | **per-item reservation** (`p_res_commodity` / `p_res_specialty`) | **Keep** — general env feature; no effect unless a config sets it |
| `coffeebench/models/openai_model.py` | `gpt-5.4-mini` pricing + graceful fallback; **`timeout=180, max_retries=0`** (hang fix) | **Keep** — strictly better; the hang fix prevents 15-min silent stalls (our `call_with_retry` handles retries with logging) |
| `coffeebench/models/gemini_model.py` | Flash/Flash-Lite pricing + graceful fallback | **Keep** — additive |
| `pyproject.toml` | `valto` editable path dep | **Keep** — needed for the integration |

The one change worth a conscious note: **`openai_model.py` `max_retries=0`** — this
disables the OpenAI SDK's *own* silent internal retries (which multiplied a hung
call to ~15 min); our `call_with_retry` still retries transient errors with
backoff + visible logging. This is the correct behavior and should stay. The
**hard-mode `[economy]` values** (`p_res_commodity=20`, `spoilage=0.03`) are NOT
code — they live only in `experiments/all6_gpt_valto_hard.toml`; the module
defaults (`INVENTORY_SPOILAGE_PER_DAY=0.005`, reservation $30/$80) are unchanged.

### 11.4 Options to get better data / experimentation

Ordered roughly by leverage:

1. **Recalibrate to viable-but-hard (immediate).** Back off one lever so the
   economy is profitable-with-good-play: e.g. `p_res_commodity=26` + `spoilage=0.02`.
   Keeps the perishability that creates real pricing/inventory mistakes without
   making it a losing game for everyone. Re-run 45-day observe (~$2.5).
2. **Add pricing rule generators to the miner.** `list_roasted` / `sell_roasted`
   templates (e.g. "ask_to_median high → low fill → spoilage → modify ask down";
   "sell below cost/ to a shaky buyer → deny"). The signal is already in the
   labeled data; this lets the auto-miner surface it. Free.
3. **Net-margin utility (fixes the fork failure).** Change the labeling target
   from `−waste` to `realized_margin − attributed_waste`. Best done with Tier-2
   exact provenance (decision-id lot tags through the pending queues) so each
   decision's true margin is exact, not proxied. Then the miner only denies
   decisions that actually lost money.
4. **Offline ML on the exported dataset.** Hand `valto_decisions.csv` + the
   analysis prompt (see chat) to an advanced ML agent: construct a per-decision
   net-contribution target, find interpretable rules, backtest on held-out rows.
   The generator/validator split means whatever it proposes is validated the same
   rigorous way, then confirmed by an actual fork.
5. **More seeds + a pressured condition.** Cheap now (~$2.5/seed). `revenue_max`
   pressure + net-margin utility together is the most likely path to a *positive*
   governed lift (pressure creates the mistakes; net-margin utility only prunes the
   money-losing ones). Multi-seed gives the held-out validation the miner wants.
6. **A more fallible subject.** If the goal is a clean "governance rescues the
   agent" story, an older/weaker/cheaper model (or a heavier KPI pressure) makes
   more systematic mistakes for Valto to catch — Valto's own benchmarks used a
   small model deliberately.
7. **Structural env levers (code, for maximum bite):** tighter green supply
   (farmer daily cap) to make the roaster A/B competition *binding*; a demand-regime
   schedule (slump/boom) so willingness-to-pay shifts and static pricing misfires;
   a "festival fizzle" (spike smaller than the public calendar implies) to punish
   over-stocking. Each turns a currently-easy dimension into a learnable one.

**Recommended next step:** (1) recalibrate to viable-but-hard **and** (2) add
pricing generators, then re-run observe → the miner should finally surface a
*net-positive* pricing/clearance rule; validate it with a fork. Keep (3) net-margin
utility as the principled follow-up once we want a defensible ROI number.

---

We need to build the infrastructure for a controlled CoffeeBench × Valto experiment.

Do **not** integrate Valto yet. Valto will come afterward.

The immediate goal is to make CoffeeBench support:

1. A useful, active economy that can run with **zero external LLM/API calls**.
2. A restartable checkpoint at an exact simulated day.
3. Multiple independent continuations loaded from the same checkpoint.
4. Different continuation variants that can each use their own Claude, GPT, scripted, or later Valto-controlled agent state without affecting the others.

## Overall experiment we are preparing for

Eventually, we want to run CoffeeBench normally through approximately day 45, save the complete state of the economy, and fork it into several independent futures:

```text
                        Shared history through day 44
                                   │
                             Day-45 checkpoint
                                   │
             ┌─────────────────────┼─────────────────────┐
             │                     │                     │
       Baseline branch       Static-Valto branch   Learning-Valto branch
       normal LLM agent      fixed Valto policy    continuously updated policy
```

Each branch must begin from the exact same:

* cash;
* inventory;
* accounting state;
* listings and offers;
* pending shipments;
* pending roasting and production;
* invoices;
* market history;
* agent transcript;
* simulator clock;
* environment randomness.

After branching, the continuations are independent.

For example, one branch might use Claude, another GPT, and another Claude with Valto. Each branch will make separate API calls and develop its own agent transcript and economic state after the checkpoint.

The checkpoint must not contain a shared live model object or shared mutable state between branches.

Before adding Valto or paying for LLM runs, we first need to prove this entire mechanism using deterministic scripted agents at zero API cost.

# Scope of this task

This task has two major deliverables:

## Deliverable A: active zero-LLM mode

CoffeeBench already has a `passive` model that performs zero external LLM calls, but it produces an inactive economy.

We verified this command works:

```bash
uv run python -m coffeebench.main \
  --model passive \
  --max_days 90 \
  --seed 0
```

It completes 90 days with zero input tokens, zero output tokens, and zero API cost.

Do not build a new no-LLM execution framework.

Instead, extend the existing scripted-model pathway so the entire six-agent economy can operate actively without API calls.

Implement deterministic scripted models for all three roles:

```text
heuristic_farmer
heuristic_roaster
heuristic_retailer
```

There are two agents for each role, and both instances may use the same role policy.

The scripted agents must:

* make no network calls;
* require no API keys;
* consume zero provider tokens;
* have zero API cost;
* use no unseeded randomness;
* interact through the same CoffeeBench tool interface as LLM agents;
* use only information available to that agent;
* maintain serializable internal state;
* produce a functioning supply-chain economy.

The goal is not to create optimal agents. The goal is to create simple, readable, deterministic agents that keep the economy active.

### Scripted farmer behavior

The farmer should deterministically:

1. Inspect its books, inventory, listings, offers, receivables and payables.
2. Produce commodity and specialty green coffee when inventory is below fixed target levels.
3. Post available green coffee for sale at simple fixed markups above production cost.
4. Accept reasonable offers from roasters.
5. Pay invoices when due.
6. Wait for the next day after completing its deterministic daily routine.

### Scripted roaster behavior

The roaster should deterministically:

1. Inspect its books, inventory, listings, offers, receivables and payables.
2. Buy green coffee when green inventory is below fixed reorder levels.
3. Use fixed maximum purchase prices for commodity and specialty green coffee.
4. Roast available green inventory within the daily capacity limit.
5. Post roasted output at fixed markups over estimated cost.
6. Accept retailer offers only above deterministic minimum sale prices.
7. Pay invoices when due.
8. Wait for the next day when finished.

Reuse the existing heuristic roaster implementation where possible, but improve or generalize it as necessary.

### Scripted retailer behavior

The retailer should deterministically:

1. Inspect its books, inventory, listings, offers, consumer sales and payables.
2. Replenish roasted commodity and specialty coffee below fixed inventory thresholds.
3. Make offers to roasters using fixed maximum wholesale prices.
4. Set deterministic consumer retail prices.
5. Optionally adjust those prices using a simple deterministic inventory rule.
6. Pay invoices when due.
7. Wait for the next day when finished.

### Active-economy acceptance criteria

A 90-day scripted run should create meaningful examples of:

* farmer production;
* marketplace listings;
* offers;
* accepted deals;
* shipments;
* shipment delays or losses when generated by the seeded environment;
* roasting;
* roasted-coffee listings;
* retailer procurement;
* consumer sales;
* receivables and payables;
* invoice payments;
* inventory changes;
* spoilage;
* positive and negative business outcomes.

Create a configuration such as:

```text
experiments/deterministic_economy.toml
```

that assigns the correct scripted model to all six agents.

This run must work without a `.env` file and must report:

```text
total external API calls: 0
total input tokens: 0
total output tokens: 0
total model cost: $0
```

CoffeeBench may still count local scripted-model invocations in `n_calls`. That is fine. Clearly distinguish local model invocations from external API calls.

## Deliverable B: exact-day checkpoint, resume and branching

Add a real checkpoint/resume system.

Do not treat `save_trajectory()` as a restartable checkpoint. It is an analysis artifact and does not currently preserve everything necessary to resume execution.

Implement a versioned checkpoint format and APIs conceptually equivalent to:

```python
env.save_checkpoint(path)
env = load_checkpoint(path)
```

The actual API can differ if reconstruction fits better at the `build_run()` level.

Prefer explicit serialization and reconstruction rather than pickling the entire Python environment.

The checkpoint must be self-contained and portable across processes.

## Exact checkpoint semantics

We need to be able to request:

```text
checkpoint at day 45
```

That must mean:

```text
Days 0 through 44 have completed.
Day 44 end-of-day mechanics have completed.
Day 45 morning processing has not started.
The next event after loading is day 45 MORNING_OPEN.
```

Add a command similar to:

```bash
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml \
  --seed 0 \
  --checkpoint-at-day 45 \
  --checkpoint-out trajectories/checkpoint_demo/day45.checkpoint.json
```

The process should save the checkpoint at the requested boundary and exit successfully.

Add a resume command similar to:

```bash
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --output trajectories/checkpoint_demo/resumed.json
```

It should continue from day 45 through the checkpoint’s original `max_days`, which will normally be day 89 for a 90-day run.

## Multiple independent branches

The same checkpoint must be loadable multiple times:

```bash
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name baseline \
  --output trajectories/checkpoint_demo/baseline.json

uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name variant_a \
  --output trajectories/checkpoint_demo/variant_a.json

uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name variant_b \
  --output trajectories/checkpoint_demo/variant_b.json
```

For the deterministic test, all three should initially produce equivalent results.

Later, the branches will deliberately differ.

The resume interface must support overriding selected post-checkpoint model assignments without changing the saved economic state.

Conceptually, we should eventually be able to run:

```bash
# Continue with Claude on the focal agent
uv run python -m coffeebench.main \
  --resume-from day45.checkpoint.json \
  --models "roaster_A:claude-sonnet-4-6" \
  --branch-name claude_baseline

# Continue with GPT on the focal agent
uv run python -m coffeebench.main \
  --resume-from day45.checkpoint.json \
  --models "roaster_A:gpt-5.5" \
  --branch-name gpt_variant
```

Each continuation must create a new model client and its own independent post-checkpoint transcript and usage counters.

Do not share:

* model-client objects;
* agent objects;
* mutable message lists;
* marketplace objects;
* event-loop objects;
* RNG objects;
* Valto state, once Valto is added later;
* any other mutable state between branches.

## Checkpoint contents

Preserve all state required for faithful continuation.

### Run identity and compatibility

* checkpoint schema version;
* CoffeeBench code or schema version if available;
* source experiment configuration;
* original seed;
* original maximum day;
* next day and next virtual time;
* model assignments active before the checkpoint;
* branch-parent metadata.

Fail loudly on incompatible checkpoint versions.

### Clock and event scheduler

* `time_manager.virtual_min`;
* all queued events;
* event insertion order or sequence counter;
* `_available_at`;
* `_pending_wake_reason`;
* `_last_active_day_per_agent`;
* `_eod_done_for_day`;
* `_today_nudge`;
* `step_count`.

The event queue must resume at the correct next event without replaying day-44 mechanics or skipping day-45 mechanics.

### Environment randomness

* `env.rng.getstate()`;
* pre-generated demand paths;
* retailer loyalty multipliers;
* shipment/delivery randomness state;
* any additional stochastic paths or counters.

Restore the RNG exactly with `setstate()`.

Different deterministic branches loaded from the same checkpoint should see the same exogenous future.

When LLM branches later diverge through their decisions, their marketplace futures may naturally diverge even though they started with the same exogenous state.

### Marketplace

* items;
* listings;
* offers;
* deals;
* direct messages;
* all IDs;
* statuses;
* timestamps;
* marketplace counters and indexes.

### Business state

For every agent:

* cash;
* inventory quantities;
* inventory total cost;
* weighted-average cost basis;
* retail prices;
* accounts receivable;
* accounts payable;
* initial equity;
* identity;
* role;
* persona;
* any business-level counters.

### Pending processes

* pending shipments;
* pending production;
* pending roasting;
* production used today;
* roasting capacity used today;
* spoilage carry;
* delivery state;
* any reserved inventory state.

### Accounting and history

* truth ledger;
* daily statistics;
* consumer-sales log;
* message counts;
* bankruptcy agents;
* bankruptcy day, reason and frozen equity;
* `_terminated_early`;
* main/focal-agent configuration.

### Agent state

For every agent:

* conversation transcript/messages;
* initial/system prompt state;
* scripted-model internal state;
* local action queues or daily-phase state;
* provider usage counters;
* compactor state, where applicable.

Do not call `agent.init()` after restore in a way that duplicates the system prompt or initial observation.

When replacing a model after checkpoint load, preserve the agent’s transcript but construct a fresh model client for that branch.

### Reconnected runtime references

After deserialization, reconnect all non-serializable callbacks and references:

* `BusinessApp._env`;
* `BusinessApp._record_truth`;
* marketplace environment reference;
* marketplace business-app map;
* marketplace deal hooks;
* agent tool bindings;
* event logger;
* any model or scheduler callbacks.

Add validation after load to confirm all required callbacks and cross-references exist.

## Scripted-model state interface

Add a clean state interface for scripted models, such as:

```python
def checkpoint_state(self) -> dict:
    ...

def restore_checkpoint_state(self, state: dict) -> None:
    ...
```

Do not rely on arbitrary object pickling.

Every scripted model should be able to resume in the middle of its deterministic routine without repeating completed actions.

## Verification requirements

Create an automated verification command such as:

```bash
uv run python -m coffeebench.verify_checkpoint \
  --config experiments/deterministic_economy.toml \
  --seed 0 \
  --checkpoint-day 45
```

It should perform the following runs.

### Run A: uninterrupted

```text
Seed 0
Scripted economy
Days 0–89
```

### Run B: checkpoint plus resume

```text
Seed 0
Scripted economy
Days 0–44
Save checkpoint
Load checkpoint in a new environment
Run days 45–89
```

Compare final economic results.

### Run C and Run D: two independent branches

Load the same day-45 checkpoint twice into separate environments or processes and run both to day 89.

They should produce equivalent final economic results under scripted policies.

This proves the checkpoint is suitable for future experiment branches.

## Equality checks

Compare at least:

* final cash by agent;
* inventory quantities;
* inventory total cost;
* AR/AP;
* retail prices;
* all listings, offers and deals;
* pending queues at completion;
* truth-ledger totals;
* daily snapshots;
* consumer-sales history;
* final net income;
* final equity;
* bankruptcy state;
* tool/local model call counts;
* final simulated day;
* RNG state where meaningful.

The simulator currently uses fields such as wall-clock timestamps or random UUIDs that may prevent literal JSON equality.

Either:

1. replace execution-relevant random IDs with a deterministic environment-owned ID generator; or
2. canonicalize non-economic metadata before comparison.

Document every excluded field and explain why it cannot affect simulation behavior.

Prefer deterministic IDs where practical because they simplify future decision-lineage work for Valto.

## Tests

Add tests showing:

1. Passive mode still runs unchanged.
2. Each new scripted role works without API keys.
3. A six-agent scripted economy completes a short run.
4. The economy generates listings, offers, deals, deliveries, roasting and consumer sales.
5. External token usage and API cost remain zero.
6. A checkpoint can be created at the beginning of an arbitrary day, not just day 45.
7. Loading a checkpoint does not duplicate agent prompts or messages.
8. Loading preserves pending shipments, production and roasting.
9. Loading preserves RNG and scheduler state.
10. Uninterrupted and checkpoint-resumed deterministic runs are economically equivalent.
11. Two independent continuations from the same checkpoint are economically equivalent.
12. A continuation can override a selected model assignment while preserving the checkpoint’s economic state.
13. Existing normal LLM configurations still work when no checkpoint flags are supplied.

## Documentation

Create:

```text
docs/deterministic_checkpointing.md
```

Include:

* why this functionality is being added;
* how zero-LLM scripted mode works;
* exact checkpoint-day semantics;
* checkpoint schema;
* commands for a complete deterministic run;
* commands to create a day-45 checkpoint;
* commands to resume;
* commands to create multiple branches;
* how model overrides work after resume;
* verification results;
* fields excluded from canonical equality;
* limitations that remain before Valto integration.

## Explicit non-goals

Do not implement any of the following yet:

* Valto tool wrappers;
* Valto decision logging;
* static Valto policies;
* continuous policy learning;
* decision-outcome attribution;
* policy mining;
* batch or lot provenance;
* official paid Claude/GPT comparison runs.

This milestone is only the substrate required before those features can be added safely.

## Why we are doing this first

The final Valto experiment needs to compare multiple governed and ungoverned futures from one shared economic history.

Before Valto enters the system, we need confidence that:

* the economy can be exercised for free;
* the simulator can stop at an exact day;
* every piece of state is faithfully captured;
* a checkpoint can be loaded multiple times;
* branches are independent;
* deterministic continuations reproduce;
* future Claude/GPT branches can each maintain their own state.

Once this foundation is proven, the next milestone will integrate Valto around `roaster_A` and create the actual three-arm experiment:

```text
baseline
static Valto
continuously learning Valto
```

Do not start that integration until the deterministic checkpoint and branch verification passes.

## Completion criteria

The milestone is complete only when all of the following work:

```bash
# 1. Active zero-LLM 90-day economy
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml \
  --seed 0

# 2. Stop after day 44 and save the beginning-of-day-45 checkpoint
uv run python -m coffeebench.main \
  --config experiments/deterministic_economy.toml \
  --seed 0 \
  --checkpoint-at-day 45 \
  --checkpoint-out trajectories/checkpoint_demo/day45.checkpoint.json

# 3. Resume the checkpoint through day 89
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name resumed \
  --output trajectories/checkpoint_demo/resumed.json

# 4. Load the same checkpoint into two independent deterministic branches
uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name branch_a \
  --output trajectories/checkpoint_demo/branch_a.json

uv run python -m coffeebench.main \
  --resume-from trajectories/checkpoint_demo/day45.checkpoint.json \
  --branch-name branch_b \
  --output trajectories/checkpoint_demo/branch_b.json

# 5. Verify equivalence automatically
uv run python -m coffeebench.verify_checkpoint \
  --config experiments/deterministic_economy.toml \
  --seed 0 \
  --checkpoint-day 45
```

At completion, report:

* files created and changed;
* architecture decisions;
* commands executed;
* test results;
* activity counts for the deterministic economy;
* net income for all six scripted agents;
* confirmation of zero external tokens and zero API cost;
* whether uninterrupted versus resumed runs matched;
* whether both checkpoint branches matched;
* any canonicalized nondeterministic metadata;
* how post-checkpoint model replacement is supported;
* any state that remains unsafe or incomplete for future branching.
