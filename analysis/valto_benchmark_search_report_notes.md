# Valto benchmark-search report notes

## Reporting job

- Decision: which external benchmark/use case should receive the next Valto engineering and experiment cycle.
- Audience: product stakeholders.
- Success criterion: a credible benchmark with exact outcomes, sufficient error headroom, a natural Valto enforcement boundary, manageable replication cost, and an honest comparison against prompt-only and existing defenses.
- Time basis: public sources reviewed on 13 July 2026.

## Executive-report structure mapping

1. Title → `Where Valto Can Win`
2. Executive summary → explicit `Executive Summary`
3. Key findings with visual evidence → τ³ recommendation, experiment design, AgentDojo finding, and comparison table
4. Recommended next steps → six-step implementation/evaluation sequence
5. Further questions → product-position and baseline questions
6. Caveats and assumptions → non-guarantee, verifier tax, and qualitative-rating limits

## Evidence boundaries

- Published safeguard results establish benchmark headroom, not expected Valto effect size.
- AgentDojo result rows combine different models, attacks, and defenses; they are examples, not a ranked apples-to-apples leaderboard.
- Candidate ratings are qualitative synthesis, not measured scores.
- The report distinguishes safe action prevention from end-to-end safe completion.

## Chart map

| Section | Question | Family/type | Fields | Supported takeaway | Palette | Artifact |
|---|---|---|---|---|---|---|
| Benchmark Landscape | Which benchmark should Valto build first? | Ranked comparison / horizontal bar | candidate × overall_fit; component rubric fields in tooltip | τ³-bench is the strongest first implementation target | Single blue root, ordinal ranking | `benchmark_fit_chart` |

The chart is explicitly labeled as an ordinal analyst rubric rather than empirical performance. A markdown evidence table follows because the central evidence mixes quantitative benchmark facts with qualitative implementation judgment; published success rates are not plotted together because models, attacks, datasets, and versions are not comparable.

## Source authority

- Primary benchmark papers, official repositories, official policies, and official result registries control.
- Third-party commercial and community sources discovered during search were not used for core conclusions.

## Validation assessment

- **Share with caveats.** Benchmark counts and published results were checked against primary papers or official registries; the recommended experiment remains prospective.
- The τ³ recommendation is supported by independent safeguard lift, but those interventions are not identical to Valto and should not be treated as an expected effect size.
- AgentDojo result rows are explicitly not a leaderboard because model, defense, and attack combinations differ.
- The artifact passed canonical validation and structural HTML packaging. Browser-level chart/source interaction QA did not run because no compatible Chromium executable was installed; the semantic chart table remains in the report.
