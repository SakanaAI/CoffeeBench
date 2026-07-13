"""Tests for the zero-LLM scripted economy and the checkpoint/resume/branch
system. Everything here runs with deterministic scripted agents — no API keys,
no network, no provider cost — so the suite is fast and hermetic.

Covers the 13 required checks in the task spec (see docstring per test).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from coffeebench.checkpoint import load_checkpoint
from coffeebench.main import _usage_summary, build_run
from coffeebench.models import get_model, is_scripted_model
from coffeebench.verify_checkpoint import canonical_summary

CONFIG = "experiments/deterministic_economy.toml"


def _args(**kw):
    base = dict(
        config=None, model=None, models=None, max_days=None, seed=0,
        main_agent=None, output=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _fresh(config=CONFIG, seed=0, max_days=12, model=None):
    env, _o, _e = build_run(_args(config=config, seed=seed, max_days=max_days, model=model))
    env.verbose = False
    env.event_logger = None
    return env


def _run(env):
    return asyncio.run(env.run())


# --- 1. Passive mode still runs unchanged --------------------------------
def test_passive_mode_runs_and_costs_nothing():
    env = _fresh(config=None, model="passive", max_days=6)
    _run(env)
    assert env.time_manager.get_current_day() == 5  # last in-window day (max_days-1)
    usage = _usage_summary(env)
    assert usage["external_api_calls"] == 0
    assert usage["external_model_cost"] == 0.0


# --- 2. Each scripted role works without API keys ------------------------
@pytest.mark.parametrize("name", ["heuristic_farmer", "heuristic_roaster", "heuristic_retailer"])
def test_scripted_role_emits_toolcall_no_network(name):
    assert is_scripted_model(name)
    model = get_model(name)
    obs = [{"role": "user", "content": "Observation: Day 0. You are X (roaster_A)."}]
    resp = model.query(obs)
    assert resp.cost == 0.0
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name  # a real tool name


# --- 3. Six-agent scripted economy completes a short run -----------------
def test_six_agent_economy_completes():
    env = _fresh(max_days=8)
    _run(env)
    assert env.time_manager.get_current_day() == 7  # max_days-1
    assert len(env.business_apps) == 6


# --- 4. The economy generates the full range of activity -----------------
def test_economy_generates_activity():
    env = _fresh(max_days=14)
    _run(env)
    assert len(env.marketplace.listings) > 0, "no listings"
    assert len(env.marketplace.offers) > 0, "no offers"
    assert len(env.marketplace.deals) > 0, "no deals"
    assert len(env.consumer_sales_log) > 0, "no consumer sales"
    # Roasting happened (roast output landed in some roaster's ledger).
    roasted = any(
        e.entry_type == "inventory_in" and "roast output" in (e.memo or "")
        for entries in env.truth_ledger.values()
        for e in entries
    )
    assert roasted, "no roasting occurred"
    # Deliveries happened (buyer-side inventory_in from a deal).
    delivered = any(
        e.entry_type == "inventory_in" and (e.counterparty or "") not in ("", None)
        for entries in env.truth_ledger.values()
        for e in entries
    )
    assert delivered, "no deliveries occurred"


# --- 5. External token usage and API cost remain zero --------------------
def test_zero_external_cost():
    env = _fresh(max_days=10)
    _run(env)
    u = _usage_summary(env)
    assert u["external_api_calls"] == 0
    assert u["external_input_tokens"] == 0
    assert u["external_output_tokens"] == 0
    assert u["external_model_cost"] == 0.0
    assert u["local_scripted_invocations"] > 0  # they DID run locally


# --- 6. A checkpoint can be created at an arbitrary day -------------------
@pytest.mark.parametrize("day", [3, 5, 9])
def test_checkpoint_at_arbitrary_day(tmp_path, day):
    ckpt = str(tmp_path / f"d{day}.json")
    env = _fresh(max_days=12)
    env._checkpoint_at_day = day
    env._checkpoint_out = ckpt
    result = _run(env)
    assert result.get("checkpoint_saved_at_day") == day
    import json

    data = json.loads(open(ckpt).read())
    assert data["next_day"] == day
    assert data["schema_version"] == 1


# --- 7. Loading does not duplicate agent prompts / messages --------------
def test_load_does_not_duplicate_messages(tmp_path):
    ckpt = str(tmp_path / "c.json")
    env = _fresh(max_days=12)
    env._checkpoint_at_day = 4
    env._checkpoint_out = ckpt
    _run(env)
    import json

    saved = json.loads(open(ckpt).read())
    saved_len = {aid: len(st["messages"]) for aid, st in saved["agents"].items()}
    env2 = load_checkpoint(ckpt, verbose=False)
    for aid, agent in env2.agents.items():
        assert len(agent.messages) == saved_len[aid], f"{aid} messages duplicated"


# --- 8. Loading preserves pending shipments / production / roasting -------
def test_load_preserves_pending_processes(tmp_path):
    ckpt = str(tmp_path / "c.json")
    env = _fresh(max_days=12)
    env._checkpoint_at_day = 4
    env._checkpoint_out = ckpt
    _run(env)
    import json

    saved = json.loads(open(ckpt).read())
    # By day 4 the pipeline has in-flight work in at least one queue.
    assert (
        saved["pending_shipments"]
        or saved["pending_production"]
        or saved["pending_roasting"]
    ), "expected some in-flight process at day 4"
    env2 = load_checkpoint(ckpt, verbose=False)
    assert [d.id for d in env2._pending_shipments] == saved["pending_shipments"]
    assert len(env2._pending_production) == len(saved["pending_production"])
    assert len(env2._pending_roasting) == len(saved["pending_roasting"])


# --- 9. Loading preserves RNG + scheduler state --------------------------
def test_load_preserves_rng_and_scheduler(tmp_path):
    from coffeebench.event_loop import EVENT_MORNING_OPEN

    ckpt = str(tmp_path / "c.json")
    env = _fresh(max_days=12)
    env._checkpoint_at_day = 5
    env._checkpoint_out = ckpt
    _run(env)
    env2 = load_checkpoint(ckpt, verbose=False)
    # Next event is day-5 MORNING_OPEN; nothing replayed or skipped.
    nxt = env2.event_loop.peek()
    assert nxt is not None and nxt.kind == EVENT_MORNING_OPEN
    assert int(nxt.payload["day"]) == 5
    # RNG + exogenous paths restored.
    assert env2._demand_path, "demand path not restored"
    assert env2._retailer_loyalty, "loyalty not restored"
    assert env2.time_manager.get_current_day() == 5


# --- 10. Uninterrupted == checkpoint-resumed (economically) --------------
def test_uninterrupted_equals_resumed(tmp_path):
    ckpt = str(tmp_path / "c.json")
    env_a = _fresh(max_days=15)
    _run(env_a)
    sum_a = canonical_summary(env_a)

    env_b0 = _fresh(max_days=15)
    env_b0._checkpoint_at_day = 6
    env_b0._checkpoint_out = ckpt
    _run(env_b0)
    env_b = load_checkpoint(ckpt, verbose=False)
    _run(env_b)
    sum_b = canonical_summary(env_b)
    assert sum_a == sum_b


# --- 11. Two independent branches are equivalent -------------------------
def test_two_branches_equivalent(tmp_path):
    ckpt = str(tmp_path / "c.json")
    env0 = _fresh(max_days=15)
    env0._checkpoint_at_day = 6
    env0._checkpoint_out = ckpt
    _run(env0)

    env_c = load_checkpoint(ckpt, branch_name="c", verbose=False)
    env_d = load_checkpoint(ckpt, branch_name="d", verbose=False)
    # No shared mutable objects.
    assert env_c is not env_d
    assert env_c.marketplace is not env_d.marketplace
    assert env_c.rng is not env_d.rng
    for aid in env_c.agents:
        assert env_c.agents[aid] is not env_d.agents[aid]
        assert env_c.agents[aid].model is not env_d.agents[aid].model
        assert env_c.agents[aid].messages is not env_d.agents[aid].messages
    _run(env_c)
    _run(env_d)
    assert canonical_summary(env_c) == canonical_summary(env_d)


# --- 12. A continuation can override a model, preserving economic state ---
def test_model_override_preserves_economic_state(tmp_path):
    ckpt = str(tmp_path / "c.json")
    env0 = _fresh(max_days=12)
    env0._checkpoint_at_day = 5
    env0._checkpoint_out = ckpt
    _run(env0)
    import json

    saved = json.loads(open(ckpt).read())

    env2 = load_checkpoint(
        ckpt, model_overrides={"roaster_A": "passive"}, verbose=False
    )
    # The overridden agent's MODEL changed…
    assert env2.agents["roaster_A"].model.model == "passive"
    # …but the economic starting state is exactly the checkpoint's, for every
    # agent (cash + inventory), including the overridden one.
    for aid, st in saved["agents"].items():
        assert env2.business_apps[aid].cash == pytest.approx(st["cash"])
        assert env2.business_apps[aid].inventory == {
            k: int(v) for k, v in st["inventory"].items()
        }
    # And it still runs to completion.
    _run(env2)
    assert env2.time_manager.get_current_day() == 11  # max_days-1


# --- Plotting smoke test: dashboard renders a PNG from a run ------------
def test_plot_dashboard_smoke(tmp_path):
    from coffeebench.plot_run import load_run, plot_dashboard

    env = _fresh(max_days=8)
    _run(env)
    run_json = tmp_path / "run.json"
    env.save_trajectory(str(run_json))
    out = tmp_path / "dash.png"
    plot_dashboard(load_run(str(run_json)), out)
    assert out.exists() and out.stat().st_size > 0


# --- 13. Existing LLM configs still build with no checkpoint flags -------
def test_normal_llm_config_builds():
    # Construction only (no run → no API calls). Proves the normal path is
    # intact and that no checkpoint/resume machinery is engaged by default.
    env, _o, _e = build_run(_args(config="experiments/roaster_focal_sonnet.toml", max_days=5))
    assert len(env.agents) == 6
    assert env._is_resume is False
    assert env._checkpoint_at_day is None
    assert env.agents["roaster_A"].model.model == "claude-sonnet-4-6"
