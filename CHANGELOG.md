# Changelog

## v2.0.0 — NeurIPS 2026 camera-ready (2026-11)

Version 2 is the code behind the camera-ready paper. The environment, the agent harness and the evaluated models all
changed, so **v1 and v2 results are not comparable**. v1 remains available as the tag `v1.0.0`.

### Environment

- **Inventory**: lot-based with per-lot expiry (green coffee 60 days, roasted 21 days) and FIFO costing; expired lots
  are written off. v1 applied a flat 0.5%/day decay.
- **Trade credit**: invoices fall due after the agreed terms (0–90 days, net-30 by default); overdue invoices accrue a
  late fee, invoices more than 30 days overdue are collected from the debtor's cash, and a debtor that cannot pay is
  declared bankrupt. Returns within 14 days of delivery (lots with enough shelf life).
- **Suppliers**: the two farms differ in ways agents are not told but can learn: one ships reliably, the other is
  cheaper but often late or short.
- **Consumer demand**: weekday pattern, a slow upward trend, weather events (heatwaves, cold snaps) forecast three days
  ahead, and a festival whose dates are announced but whose size is not. Customers who find a shop sold out buy at the
  other shop; a shop that keeps an item in stock gains regular customers.
- **Supply shock**: an announced seven-day event cuts farm output to 40% and raises farm costs by 50%.
- **Capacity upgrade**: a one-time $2,500 investment (expensed) raises the roast cap from 50 to 90 kg/day and the
  inventory cap from 120 to 180 kg.
- **Accounting**: revenue on delivery, purchases carried as inventory until used (accrual accounting); at the end of
  the run inventory is written down to the lower of cost and a market reference (median arm's-length price over the
  last 14 days, reciprocal pairs excluded, capped), and receivables from insolvent debtors are written down
  (`[mechanics.end_lcnrv]`, `[mechanics.end_credit_loss]`). Interest on cash (5%/yr). Net income =
  revenue − COGS − OpEx − interest expense + interest revenue.
- **Other**: counterparties can go bankrupt and leave the market; price and terms validation; the run stops early if
  the evaluated firm goes bankrupt; initial cash $7,000 (v1: $15,000); `cancel_listing`, `reject_offer`,
  `view_dashboard` and a capacity-upgrade tool.
- **Rule-based agents**: `heuristic_roaster_v2` (prices from lot costs and observed market prices, festival stocking,
  invoice payment), negotiating / adversarial farmer and retailer profiles.

### Harness (Harbor)

- Every firm, evaluated or background, is a **coding agent** in its own container (no network) running on
  [Harbor](https://harborframework.com). The `coffee` CLI is the only way to act in the market; it talks to one shared
  environment server over a bind-mounted Unix socket with per-agent HMAC tokens and a strict endpoint allowlist.
- Agents use the **mini-SWE-agent action format** (one short note and exactly one shell command per turn) with a
  uniform 1M-token context budget and summarization at 85%. Each `coffee` command costs 30 simulated minutes, shell
  commands are free. The prompt states only the rules (no strategy advice).
- Daily checkpoints and **exact resume** (same LLM context, workspace restored) after a crash or walltime limit;
  per-run admin / harness / verifier tokens; terminal audit (`harbor_adapter/audit_terminal.py`).
- v1's in-process provider wrappers and `ContextCompactor` are no longer used for reported results.

### Evaluation protocol

- Panel of eight models through OpenRouter, each at its **highest reasoning effort**: Claude Sonnet 5.5, Claude Opus
  5.5, Claude Haiku 5.5, GPT-6 Astra, Gemini 3.8 Flash, Kimi K3.1 Pro, DeepSeek V4.1 Flash, GLM-5.3. Background model:
  Claude Sonnet 5.5 at the API default (robustness check with Qwen3.8-Flash, reasoning off). Three seeds per setting.
- Rule-based baselines: `heuristic_roaster_v2` and `passive`.
- **KPI-pressure track**: a revenue or net-income target with a replacement threat for both roasters, rival-model
  swaps, mitigation prompts, and rule-based detection of circular trades at inflated prices (`scripts/mf_sham_detect.py`).
- Ablations: lowest reasoning effort, background model, focal role (farmer / retailer), harness (terminus-2 vs
  mini-SWE-agent format).

### Repository

- `experiments/`: v2 configs only (`*_v21.toml`); the v1 `roaster_focal_*.toml` configs are gone (tag `v1.0.0`).
- `scripts/`: paper tables and figures from the released runs, sham-trade detection, $0 tests.
- `harbor_adapter/cluster/`: PBS job script and rootless-podman setup for cluster runs.
- `CHANGELOG.md`, `CITATION.cff`.

### Trajectory site

- The trajectory site is being rebuilt for v2 (agent terminals, inter-firm messages, KPI-pressure cases); the released
  runs will be distributed as a dataset that the site loads.

## v1.0.0 — arXiv v1 (2026-06)

Initial release: in-process LLM wrappers (OpenAI, Anthropic, Gemini, OpenRouter), 90-day runs with Claude Sonnet 4.6 as
the background model, Streamlit viewer.
