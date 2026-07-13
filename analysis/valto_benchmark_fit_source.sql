-- Decision rubric used in the benchmark-search report.
-- Scores are ordinal analyst judgments on a 1–5 scale, based on the public
-- benchmark sources reviewed on 13 July 2026. They are not Valto performance.

WITH rubric(
  candidate,
  overall_fit,
  exact_outcomes,
  valto_headroom,
  implementation_ease,
  external_credibility,
  primary_use_case,
  decision
) AS (
  VALUES
    ('τ³-bench Airline/Retail', 5, 5, 5, 4, 5, 'Policy-bound customer-service mutations', 'Build first'),
    ('AgentDojo', 4, 5, 5, 4, 5, 'Prompt-injection-resistant authorization', 'Build second'),
    ('Instrumental Choices', 3, 5, 2, 4, 3, 'Rule-breaking shortcuts in cloud operations', 'Compact demo'),
    ('AppWorld', 3, 5, 3, 2, 5, 'Collateral-damage control across apps', 'Later generality study'),
    ('Agent-SafetyBench', 2, 3, 4, 2, 4, 'Broad agent safety', 'Use enforceable subset only'),
    ('RetailBench', 2, 3, 4, 1, 4, 'Long-horizon store operations', 'Avoid now')
)
SELECT *
FROM rubric
ORDER BY overall_fit DESC, candidate;
