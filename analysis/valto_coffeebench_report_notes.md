# Valto × CoffeeBench report build notes

## Audience and structure

- Audience: technical founder/product owner and benchmark implementer.
- Decision: whether CoffeeBench should be the primary proof of Valto efficacy and what experiment to run next.
- Narrative: recommendation → evidence boundary → measurement failure → benchmark fit → replacement design → KPIs and gates.

## Source inventory

- CoffeeBench simulation, prompts, tool surface, truth accounting, checkpoint/resume, and Valto integration code.
- Existing experiment plan and generated learned-policy reports.
- Six Valto decision ledgers, five used for observe-label QA and one used for governed intervention timing.
- Complete 90-day deterministic run, one complete live-LLM baseline/governed pair, several incomplete smoke runs, and day-45 checkpoints.

## Chart map

| Dataset | Chart | Intent | Encoding | Aggregation | Missing values | Denominator | Notes |
|---|---|---|---|---|---|---|---|
| `fork_equity` | Line | Trend | day × true equity, series by arm | None; one row per simulated day | None | N/A | Days 44–89 only; x-reference at first Valto denial, day 66. Tooltip dataset also carries cash, deal counts, equity gap, and treatment flag. |

## Omitted visuals

- A bar chart of label-error counts was omitted because the categories can overlap and the table communicates definitions and benchmark risk more precisely.
- A benchmark-fit radar or score chart was omitted because the ratings are qualitative judgments, not measured values.
- No treatment-effect chart is shown because one live-LLM branch pair is not a defensible effect estimate.

## Evidence boundaries

- The report distinguishes exact observed values from design recommendations.
- “First world divergence” compares complete daily world records, while the displayed line chart shows focal true equity; focal equity separates later.
- The final net-income gap is reported as an observed diagnostic, not a causal Valto effect.
