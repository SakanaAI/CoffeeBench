-- Reproduces the four label-quality counts in the report table when run from
-- the CoffeeBench repository root with the sqlite3 CLI (JSON1 enabled).

ATTACH DATABASE 'trajectories/roaster_valto_observe_haiku/seed_0/valto_ledger.db' AS haiku_observe;
ATTACH DATABASE 'trajectories/roaster_valto_revmax_haiku/seed_0/valto_ledger.db' AS haiku_revmax;
ATTACH DATABASE 'trajectories/all6_gpt_valto/seed_0/valto_ledger.db' AS gpt_easy;
ATTACH DATABASE 'trajectories/all6_gpt_valto_hard/seed_0/valto_ledger.db' AS gpt_hard;
ATTACH DATABASE 'trajectories/roaster_valto_full_scripted/seed_0/valto_ledger.db' AS scripted_full;

WITH rows AS (
  SELECT d.action, o.useful, o.metrics_json
  FROM haiku_observe.decisions d LEFT JOIN haiku_observe.outcomes o ON o.decision_id = d.id
  UNION ALL
  SELECT d.action, o.useful, o.metrics_json
  FROM haiku_revmax.decisions d LEFT JOIN haiku_revmax.outcomes o ON o.decision_id = d.id
  UNION ALL
  SELECT d.action, o.useful, o.metrics_json
  FROM gpt_easy.decisions d LEFT JOIN gpt_easy.outcomes o ON o.decision_id = d.id
  UNION ALL
  SELECT d.action, o.useful, o.metrics_json
  FROM gpt_hard.decisions d LEFT JOIN gpt_hard.outcomes o ON o.decision_id = d.id
  UNION ALL
  SELECT d.action, o.useful, o.metrics_json
  FROM scripted_full.decisions d LEFT JOIN scripted_full.outcomes o ON o.decision_id = d.id
),
issues AS (
  SELECT
    'Roasts negative only because spoilage was attributed' AS issue,
    COUNT(*) AS count,
    'Blocks productive transformation without crediting downstream margin.' AS benchmark_risk
  FROM rows
  WHERE action = 'roast'
    AND json_extract(metrics_json, '$.attributed_spoilage_w') > 0
    AND json_extract(metrics_json, '$.utility') < 0
  UNION ALL
  SELECT
    'Sell outcomes missing miner utility', COUNT(*),
    'The miner cannot learn sell-side rules from logged outcomes.'
  FROM rows
  WHERE action = 'sell_roasted'
    AND json_type(metrics_json, '$.utility') IS NULL
  UNION ALL
  SELECT
    'Non-dealt buys marked useful', COUNT(*),
    'Credits offers that never changed inventory or economics.'
  FROM rows
  WHERE action = 'buy_green'
    AND json_extract(metrics_json, '$.dealt') = 0
    AND useful = 1
  UNION ALL
  SELECT
    'Zero-fill listings marked useful', COUNT(*),
    'Rewards listings that sold nothing.'
  FROM rows
  WHERE action = 'list_roasted'
    AND json_extract(metrics_json, '$.fill_rate') = 0
    AND useful = 1
)
SELECT issue, count, benchmark_risk
FROM issues
ORDER BY count DESC;
