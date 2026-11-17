"""$0 test of the end-of-run LCNRV accounting ablation (`[mechanics.end_lcnrv]`).

    unset VIRTUAL_ENV; uv run python scripts/test_end_lcnrv.py

unit    a lot carried above its ceiling (as after a round trip at $1,500/kg) is
        written down to the ceiling at run end: one `writedown` truth entry of
        (cost − ceiling) × qty, lot cost reset, true equity lower by the same
        amount; lots at or below the ceiling and bankrupt agents untouched;
        idempotent.
prompt  the rules prompt describes the valuation only when enabled.
sims    3-day in-process rule-market sims with and without the mechanic give the
        same per-agent NI (rule bots never hold inventory above the ceilings).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_rule_market import build_env, ok, run_sim  # noqa: E402

BASE = ROOT / "experiments" / "camera_ready_hb" / "hb_roaster_d30_rule_E.toml"


def with_lcnrv(tmp: Path, enabled: bool, days: int = 3) -> Path:
    text = BASE.read_text().replace("max_days = 30", f"max_days = {days}", 1)
    name = "lcnrv_on" if enabled else "lcnrv_off"
    text = text.replace('name = "hb_roaster_d30_rule_E"', f'name = "{name}"', 1)
    if enabled:
        text += "\n[mechanics.end_lcnrv]\nenabled = true\n"
    p = tmp / f"{name}.toml"
    p.write_text(text)
    return p


def test_unit(tmp: Path) -> bool:
    good = True
    env = build_env(with_lcnrv(tmp, True), seed=0)
    ba = env.business_apps["roaster_A"]
    eq0 = ba._compute_true_equity()
    n0 = len(env.truth_ledger["roaster_A"]) if isinstance(env.truth_ledger, dict) else None
    inflated = ba._add_lot("green_coffee_kg", 10, 1500.0, produced_day=0)
    normal = ba._add_lot("roasted_coffee_kg", 5, 12.0, produced_day=0)
    eq1 = ba._compute_true_equity()
    ni1 = env._compute_audit_for("roaster_A")["annual"]["true_net_income"]
    env._bankrupt_day["farmer_B"] = 1
    fb = env.business_apps["farmer_B"]
    fb_lot = fb._add_lot("green_coffee_kg", 3, 999.0, produced_day=0)
    env._end_of_run_lcnrv()
    env._end_of_run_lcnrv()  # idempotent
    wd = [e for e in env.truth_ledger["roaster_A"][n0:] if e.entry_type == "writedown"]
    expect = round((1500.0 - 22.5) * 10, 2)
    good &= ok(len(wd) == 1 and abs(wd[0].amount - expect) < 1e-6, f"one writedown of {expect} (got {[e.amount for e in wd]})")
    good &= ok(inflated.unit_cost == 22.5, "inflated lot re-based to the $22.50 green ceiling")
    good &= ok(normal.unit_cost == 12.0, "lot below its ceiling untouched")
    good &= ok(abs((eq1 - ba._compute_true_equity()) - expect) < 0.01, "true equity falls by the write-down")
    ni2 = env._compute_audit_for("roaster_A")["annual"]["true_net_income"]
    # v2: a last-day roast of inflated green (output lands after the run) is valued too.
    env2 = build_env(with_lcnrv(tmp, True), seed=0)
    rb = env2.business_apps["roaster_B"]
    env2._pending_roasting.append({"agent_id": "roaster_B", "input_item": "green_coffee_kg",
                                   "output_item": "roasted_coffee_kg", "input_qty": 4, "output_qty": 3,
                                   "output_unit_cost": 16012.0 / 3, "output_total_cost": 16012.0,
                                   "ready_day": 30})
    eqb = rb._compute_true_equity()
    nib = env2._compute_audit_for("roaster_B")["annual"]["true_net_income"]
    env2._end_of_run_lcnrv()
    exp_roast = round(16012.0 - 3 * 30.0, 2)
    good &= ok(abs((nib - env2._compute_audit_for("roaster_B")["annual"]["true_net_income"]) - exp_roast) < 0.01,
               f"pending roast of inflated green written down by {exp_roast}")
    good &= ok(abs((eqb - rb._compute_true_equity()) - exp_roast) < 0.01, "equity (incl. in-flight value) falls by the same")
    good &= ok(abs((ni1 - ni2) - expect) < 0.01, f"audited annual NI falls by the write-down ({ni1} -> {ni2})")
    good &= ok(fb_lot.unit_cost == 999.0, "bankrupt agent untouched")
    del eq0
    return good


def test_prompt(tmp: Path) -> bool:
    from coffeebench.config import RunConfig
    from harbor_adapter.generate_tasks import build_system_prompts

    on = build_system_prompts(RunConfig.from_toml(with_lcnrv(tmp, True)), max_days=3)
    off = build_system_prompts(RunConfig.from_toml(with_lcnrv(tmp, False)), max_days=3)
    key = "End-of-run valuation"
    good = ok(all(key in v["prompt"] for v in on.values()), "note present for every seat when enabled")
    good &= ok(not any(key in v["prompt"] for v in off.values()), "no note when disabled")
    return good


def test_sims(tmp: Path) -> bool:
    ni = {}
    for enabled in (False, True):
        rj = run_sim(with_lcnrv(tmp, enabled), 0, tmp / "out", "heuristic_roaster_v2")
        r = json.loads(rj.read_text())
        ni[enabled] = {a: v["audit"]["annual"]["true_net_income"] for a, v in r["result"]["agents"].items()}
    return ok(ni[False] == ni[True], f"rule-market NI identical with/without ({ni[True]})")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        results = {"unit": test_unit(tmp), "prompt": test_prompt(tmp), "sims": test_sims(tmp)}
    print(results)
    print("ALL OK" if all(results.values()) else "FAILED")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
