-- Reproduces the report's daily baseline/governed equity comparison and
-- intervention timing when run from the CoffeeBench repository root with
-- the sqlite3 CLI (JSON1 and readfile() enabled).

ATTACH DATABASE
  'trajectories/roaster_valto_revmax_haiku/seed_0/valto_experiment/valto_ledger_governed.db'
  AS valto;

WITH
baseline_document AS (
  SELECT json(CAST(readfile(
    'trajectories/roaster_valto_revmax_haiku/seed_0/valto_experiment/baseline.json'
  ) AS TEXT)) AS document
),
governed_document AS (
  SELECT json(CAST(readfile(
    'trajectories/roaster_valto_revmax_haiku/seed_0/valto_experiment/governed.json'
  ) AS TEXT)) AS document
),
baseline_daily AS (
  SELECT
    CAST(json_extract(day.value, '$.day') AS INTEGER) AS day,
    ROUND(json_extract(day.value, '$.per_agent.roaster_A.true_equity'), 2) AS baseline_equity,
    ROUND(json_extract(day.value, '$.per_agent.roaster_A.cash'), 2) AS baseline_cash,
    CAST(json_extract(day.value, '$.per_agent.roaster_A.deals_count') AS INTEGER) AS baseline_deals
  FROM baseline_document, json_each(baseline_document.document, '$.daily_stats') AS day
),
governed_daily AS (
  SELECT
    CAST(json_extract(day.value, '$.day') AS INTEGER) AS day,
    ROUND(json_extract(day.value, '$.per_agent.roaster_A.true_equity'), 2) AS governed_equity,
    ROUND(json_extract(day.value, '$.per_agent.roaster_A.cash'), 2) AS governed_cash,
    CAST(json_extract(day.value, '$.per_agent.roaster_A.deals_count') AS INTEGER) AS governed_deals
  FROM governed_document, json_each(governed_document.document, '$.daily_stats') AS day
),
intervention AS (
  SELECT MIN(CAST(json_extract(context_json, '$.day') AS INTEGER)) AS first_intervention
  FROM valto.decisions
  WHERE verdict <> 'approve'
)
SELECT
  baseline_daily.day,
  baseline_equity,
  governed_equity,
  ROUND(governed_equity - baseline_equity, 2) AS equity_gap,
  baseline_cash,
  governed_cash,
  baseline_deals,
  governed_deals,
  baseline_daily.day >= intervention.first_intervention AS treatment_started
FROM baseline_daily
JOIN governed_daily USING (day)
CROSS JOIN intervention
WHERE baseline_daily.day BETWEEN 44 AND 89
ORDER BY baseline_daily.day;
