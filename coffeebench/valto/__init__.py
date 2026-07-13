"""CoffeeBench × Valto integration.

Governs ONLY the focal agent's capital-committing tools (`make_offer` = buy
green, `roast` = production) via `valto.protect_call`, logging every decision
with rich observable-only context and every execution with structured outcome
metrics. Observe mode approves everything (log-only); governed mode enforces a
policy. See `coffeebench/valto/govern.py` and `VALTO_EXPERIMENT_PLAN.md`.
"""

from coffeebench.valto.govern import govern_business_app, make_valto

__all__ = ["make_valto", "govern_business_app"]

# Packaged approve-all policy path (observe mode default).
from pathlib import Path

OBSERVE_POLICY = str(Path(__file__).parent / "policies" / "observe.yaml")
