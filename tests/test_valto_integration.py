"""Valto integration tests — the observe-mode wiring on the free scripted
economy. No API keys, no cost. Proves the decision + outcome ledger fills with
rich context and structured outcome metrics, and that governance only touches
the focal agent's tool schema without changing it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from valto import Valto

from coffeebench.main import build_run


def _args(config, seed=0, max_days=12, models=None):
    return SimpleNamespace(
        config=config, model=None, models=models, max_days=max_days,
        seed=seed, main_agent=None, output=None,
    )


def _run_observe(tmp_path, govern_tools=None):
    """Run the free scripted economy with Valto observe on roaster_A, writing
    the ledger under tmp_path. Returns (env, ledger_path)."""
    import tomllib

    # Build an on-the-fly config dict → write a temp toml so build_run reads it.
    ledger = tmp_path / "valto_ledger.db"
    toml = f"""
[experiment]
name = "valto_test"
[run]
seeds = [0]
max_days = 12
[models]
default = "passive"
farmer_A = "heuristic_farmer"
farmer_B = "heuristic_farmer"
roaster_A = "heuristic_roaster"
roaster_B = "heuristic_roaster"
retailer_A = "heuristic_retailer"
retailer_B = "heuristic_retailer"
[valto]
enabled = true
actor = "roaster_A"
mode = "observe"
db_path = "{ledger}"
govern_tools = {govern_tools or ["make_offer", "roast"]!r}
"""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(toml)
    # sanity: the toml parses
    tomllib.loads(toml)
    env, _o, _e = build_run(_args(str(cfg)))
    env.verbose = False
    env.event_logger = None
    asyncio.run(env.run())
    return env, str(ledger)


def test_observe_ledger_fills_with_context_and_metrics(tmp_path):
    env, ledger_path = _run_observe(tmp_path)
    v = Valto.local(actor="roaster_A", db_path=ledger_path)
    rows = v.decisions(limit=2000)
    assert rows, "no decisions logged"
    actions = {r["action"] for r in rows}
    assert actions <= {"buy_green", "roast"}
    assert "buy_green" in actions and "roast" in actions
    # Observe mode approves everything.
    assert all(r["verdict"] == "approve" for r in rows)
    # Every decision carries rich context AND structured outcome metrics.
    for r in rows:
        assert isinstance(r["context"], dict) and "day" in r["context"]
        assert isinstance(r["metrics"], dict) and "executed_ok" in r["metrics"]
    # buy_green context is genuinely rich (observable-only decision features).
    buy = next(r for r in rows if r["action"] == "buy_green")
    for key in ("days_to_festival", "price_to_median", "cash_runway_days",
                "cap_headroom_kg", "counterparty_prior_deals"):
        assert key in buy["context"], f"missing context key {key}"
    assert buy["amount"] is not None and buy["counterparty"] is not None


def test_valto_zero_external_cost(tmp_path):
    env, _ = _run_observe(tmp_path)
    from coffeebench.main import _usage_summary

    u = _usage_summary(env)
    assert u["external_api_calls"] == 0
    assert u["external_model_cost"] == 0.0


def test_labeling_reconciles_and_writes_utility(tmp_path):
    """The attribution pass labels every governed decision with a utility, and
    attributed spoilage reconciles against realized spoilage (coverage ≤ 1,
    and exactly 1.0 when every spoilage event has an in-window origin)."""
    from coffeebench.valto.labeling import label_run

    env, ledger_path = _run_observe(tmp_path)
    report = label_run(env, verbose=False)
    assert report["labeled_decisions"] > 0
    # Reconciliation: we never attribute MORE than the realized spoilage.
    assert report["attributed_spoilage"] <= report["total_spoilage"] + 1e-6
    cov = report["spoilage_attribution_coverage"]
    assert cov is None or 0.0 <= cov <= 1.0
    # Utility is now on the ledger rows.
    from valto.ledger import Ledger

    rows = Ledger(ledger_path).list_decisions(agent_id=None, limit=10000)
    labeled = [r for r in rows if (r.get("metrics") or {}).get("utility") is not None]
    assert len(labeled) == report["labeled_decisions"]


def test_miner_emits_valid_policy(tmp_path):
    """The miner runs the propose→backtest→emit loop and always produces a
    policy that passes Valto's own validator (0 rules on clean data is fine)."""
    from valto.engine import validate_policy

    from coffeebench.valto.analyze_coffee import load_rows, mine
    from coffeebench.valto.labeling import label_run

    env, ledger_path = _run_observe(tmp_path)
    label_run(env, verbose=False)
    rows = load_rows([ledger_path])
    assert rows and all("utility" in r for r in rows)
    rules, report = mine(rows, rows)
    policy = {
        "version": "test", "budgets": {},
        "categories": {"buy_green": "procurement", "roast": "production"},
        "rules": rules, "default": "approve",
    }
    validate_policy(policy)  # raises on a malformed rule
    assert isinstance(rules, list)


def test_governance_does_not_change_tool_schema(tmp_path):
    """The wrapped tools must present the SAME JSON schema as the originals, so
    the model sees no difference."""
    from coffeebench.util import function_to_json

    # Ungoverned build (no valto) → reference schemas.
    plain_toml = """
[experiment]
name = "plain"
[run]
seeds = [0]
max_days = 4
[models]
default = "passive"
roaster_A = "heuristic_roaster"
"""
    p = tmp_path / "plain.toml"
    p.write_text(plain_toml)
    env_plain, _, _ = build_run(_args(str(p), max_days=4))
    ba_plain = env_plain.business_apps["roaster_A"]
    ref = {n: function_to_json(getattr(ba_plain, n)) for n in ("make_offer", "roast")}

    # Governed build.
    env_gov, ledger = _run_observe(tmp_path)  # noqa: F841 (runs it)
    ba_gov = env_gov.business_apps["roaster_A"]
    for n in ("make_offer", "roast"):
        got = function_to_json(getattr(ba_gov, n))
        assert got == ref[n], f"schema for {n} changed under governance"
