"""$0 tests for the rule-based background market and the optional [mechanics].

    unset VIRTUAL_ENV; uv run python scripts/test_rule_market.py [--only unit,e_unit,persona,off,sims,determinism,resume]
        [--seeds 0,1,2] [--configs hb_roaster_d30_rule,hb_roaster_d30_rule_D,hb_roaster_d30_rule_E]
    # after the three scripted Harbor runs (see test_harbor's docstring):
    uv run python scripts/test_rule_market.py --only harbor

unit         mechanics: parse/realize determinism, supply-shock factors in the
             window only, advisory K days ahead (none when K=0), festival from
             the config (env multiplier + rules prompt), demand-shift factor
             from its day on; rule agents parse their own id / role / costs
             from the rules prompt (also as roaster_B / farmer_B).
off          a config without [mechanics] → env.mechanics == {}, the module
             FESTIVALS, and a rules prompt identical to the no-mechanics one.
sims         30-day in-process sims (rule background; focal roaster_A = passive,
             heuristic_roaster, heuristic_roaster_v2) for each config × seed:
             no crash, active trading, per-agent NI table; v2 focal solvent and
             mean NI ordered passive < heuristic < heuristic_v2; with D2 the
             shock days show reduced production caps / higher unit cost and
             the advisory appears exactly advisory_days before the shock.
e_unit       E1-E4: traits (delay/short-ship thresholds, farm cost factor),
             E2 bounded deterministic concessions (farmer counter-quotes,
             retailer bid raises), E3 weekday/trend/weather factor + published
             forecast (observation + view_market_aggregate), E4a upgrade tool
             (capex expensed, caps after the build delay, once only, tool only
             registered when enabled), E4b availability floor growth/reset.
persona      generate_tasks with --bg Persona:<m>: persona block appended to the
             non-focal seats' instruction.md only (focal and shared text
             unchanged), dataset name encodes the profile set.
harbor       checks the scripted Harbor runs (rule E, all-persona E, mixed
             persona E): mechanics apply to remote (LLM-style) seats.
determinism  the same config/seed twice → identical run.json + events.
resume       stop at day 12 (checkpoint) → resume in a new process → identical
             to the uninterrupted run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG_DIR = ROOT / "experiments" / "camera_ready_hb"
FOCALS = ("passive", "heuristic_roaster", "heuristic_roaster_v2")


def ok(cond: bool, msg: str) -> bool:
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    return bool(cond)


def run_sim(cfg: Path, seed: int, out: Path, focal: str | None = None, *extra: str) -> Path:
    cmd = [sys.executable, "-m", "coffeebench.main", "--config", str(cfg), "--seed", str(seed),
           "--output-root", str(out), *extra]
    if focal:
        cmd += ["--models", f"roaster_A:{focal}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"sim failed: {cmd}")
    from coffeebench.config import RunConfig

    return out / RunConfig.from_toml(cfg).name / f"seed_{seed}" / "run.json"


def normalized(run_json: Path) -> tuple[str, list[str]]:
    d = json.loads(run_json.read_text())
    d.pop("saved_at", None)
    for msgs in d.get("messages_per_agent", {}).values():
        for m in msgs:
            m.pop("timestamp", None)
    ev = []
    for line in run_json.with_name("run.events.jsonl").read_text().splitlines():
        e = json.loads(line)
        for k in ("ts_ms", "ts", "timestamp"):
            e.pop(k, None)
        ev.append(json.dumps(e, sort_keys=True))
    return json.dumps(d, sort_keys=True), ev


# ---------------------------------------------------------------- unit


def build_env(cfg: Path, seed: int = 0, models: str | None = None):
    from coffeebench.main import build_run

    with tempfile.TemporaryDirectory() as td:
        args = Namespace(config=str(cfg), model=None, models=models, max_days=None, seed=seed,
                         main_agent=None, output=None, output_root=td)
        env, _, _ = build_run(args, open_events=False)
    return env


def test_unit(_tmp: Path) -> bool:
    from coffeebench import environment as env_mod
    from coffeebench import mechanics as M

    res = []
    spec = M.parse({
        "announce_advisories": True,
        "festivals": [{"name": "x", "start": 5, "end": 7, "multiplier": 4.0}],
        "demand_shift": {"enabled": True, "window": [8, 20], "factors": [0.6, 1.4]},
        "supply_shock": {"enabled": True, "window": [6, 14], "days": 5, "capacity_factor": 0.4,
                         "cost_factor": 1.5, "advisory_days": 3},
    })
    items = ["roasted_coffee_kg", "roasted_specialty_kg"]
    a, b = M.realize(spec, 7, items), M.realize(spec, 7, items)
    res.append(ok(a == b, f"realize deterministic for a seed: {a}"))
    draws = {json.dumps(M.realize(spec, s, items), sort_keys=True) for s in range(20)}
    res.append(ok(len(draws) > 1, f"schedule varies across seeds ({len(draws)} distinct / 20)"))
    ss = a["supply_shock"]
    res.append(ok(6 <= ss["start"] <= 14 and ss["end"] == ss["start"] + 4, "shock window inside config"))
    inside = [M.supply_factors(a, "green_coffee_kg", d) for d in range(ss["start"], ss["end"] + 1)]
    outside = [M.supply_factors(a, "green_coffee_kg", d) for d in (ss["start"] - 1, ss["end"] + 1)]
    res.append(ok(all(x == (0.4, 1.5) for x in inside) and all(x == (1.0, 1.0) for x in outside),
                  "supply factors only on shock days"))
    adv = {d for d in range(40) if M.advisory_line(a, d)}
    res.append(ok(adv == set(range(ss["start"] - 3, ss["end"] + 1)),
                  f"advisory from day {ss['start'] - 3} (K=3) to the shock end"))
    no_adv = dict(a, supply_shock=dict(ss, advisory_days=0))
    res.append(ok(not any(M.advisory_line(no_adv, d) for d in range(40)), "K=0 → no advisory (unannounced)"))
    ds = a["demand_shift"]
    res.append(ok(M.demand_factor(a, "roasted_coffee_kg", ds["day"] - 1) == 1.0
                  and M.demand_factor(a, "roasted_coffee_kg", ds["day"]) == ds["factors"]["roasted_coffee_kg"]
                  and set(ds["factors"].values()) <= {0.6, 1.4}, f"demand shift from day {ds['day']}: {ds['factors']}"))
    try:
        M.parse({"bogus": 1})
        res.append(ok(False, "unknown [mechanics] key rejected"))
    except ValueError:
        res.append(ok(True, "unknown [mechanics] key rejected"))

    env = build_env(CFG_DIR / "hb_roaster_d30_rule_D.toml", seed=0)
    res.append(ok(env._festival_multiplier(15) == 4.0 and env._festival_multiplier(14) == 1.0
                  and env._festival_multiplier(22) == 1.0, "festival ×4 days 15-21 from the D config"))
    prompt = env.agents["roaster_A"].system_prompt
    res.append(ok("`spring_break` days 15–21" in prompt and "weather advisories" in prompt.lower()
                  and "shift over time" not in prompt, "rules prompt: festival dates + advisory note, no D1 note"))
    sched = env.mechanics["supply_shock"]
    obs = env._format_observation("roaster_A", sched["start"] - 3, None)
    obs_before = env._format_observation("roaster_A", sched["start"] - 4, None)
    res.append(ok("Weather advisory (public)" in obs and "Weather advisory" not in obs_before,
                  "observation carries the advisory exactly 3 days ahead"))
    base = build_env(CFG_DIR / "hb_roaster_d30_rule.toml", seed=0)
    res.append(ok(base.agents["roaster_A"].system_prompt == prompt, "base and D configs give the same prompt"))
    res.append(ok(base.mechanics == {} or "supply_shock" not in base.mechanics, "base: no supply shock"))
    # Rule agents parse their seat from the prompt (works as either id).
    from coffeebench.models import get_model

    for aid, model in (("roaster_B", "heuristic_roaster_v2"), ("farmer_B", "heuristic_farmer"),
                       ("retailer_A", "heuristic_retailer")):
        m = get_model(model)
        m.system_prompt = env.agents[aid].system_prompt
        m._parse_rules(m.system_prompt)
        res.append(ok(m.me == aid and m.festivals == [(15, 21)], f"{model} as {aid}: me={m.me}, festivals={m.festivals}"))
    f = get_model("heuristic_farmer")
    f._parse_rules(env.agents["farmer_A"].system_prompt)
    res.append(ok(f.prod_cost == {"green_coffee_kg": 2.0, "green_specialty_kg": 10.0}
                  and f.opex == env_mod.DAILY_OPEX_BY_ROLE["farmer"] and f.inv_cap == 120 and f.max_days == 30,
                  "farmer reads costs / opex / cap / horizon from the prompt"))
    return all(res)


def test_e_unit(_tmp: Path) -> bool:
    from coffeebench import mechanics as M
    from coffeebench.main import tool_names_for
    from coffeebench.models.heuristic_farmer_model import HeuristicFarmerModel
    from coffeebench.models.heuristic_retailer_model import HeuristicRetailerModel

    res = []
    cfg = CFG_DIR / "hb_roaster_d30_rule_E.toml"
    env = build_env(cfg, seed=0)
    sch = env.mechanics
    res.append(ok(M.trait(sch, "farmer_B", "short_ship_prob", 0) == 0.30
                  and M.trait(sch, "farmer_A", "short_ship_prob", 0) == 0
                  and M.trait(sch, "roaster_B", "delivery_delay_prob", 0.07) == 0.07, "E1 traits per seat, defaults elsewhere"))
    # E3
    w = sch["demand_patterns"]["weather"]
    f0, f5 = M.pattern_factor(sch, 0, 30), M.pattern_factor(sch, 5, 30)
    clear = [d for d in range(30) if not any(a <= d <= b for a, b, *_ in w)]
    d = clear[0]
    exp = sch["demand_patterns"]["weekday"][d % 7] * (1 + 0.2 * d / 29)
    res.append(ok(abs(M.pattern_factor(sch, d, 30) - exp) < 1e-12 and f0 != f5,
                  f"E3 weekday × trend factor (day {d}: {exp:.3f}); weather {w}"))
    a, b, name, fac = w[0]
    res.append(ok(abs(M.pattern_factor(sch, a, 30) / (sch["demand_patterns"]["weekday"][a % 7] * (1 + 0.2 * a / 29)) - fac) < 1e-9,
                  f"E3 weather '{name}' ×{fac} on days {a}-{b}"))
    vis = [dd for dd in range(30) if any(e["start_day"] == a for e in M.weather_forecast(sch, dd))]
    res.append(ok(vis[0] == a - 3, f"E3 forecast published 3 days ahead (from day {vis[0]})"))
    obs = env._format_observation("retailer_A", a - 3, None)
    res.append(ok("Weather forecast (public" in obs and f"days {a}–{b}" in obs, "E3 forecast line in the observation"))
    # E4a
    ba = env.business_apps["roaster_A"]
    env.time_manager.virtual_min = 0 * 1440 + 9 * 60
    cash0 = ba.cash
    r1 = ba.buy_roaster_upgrade()
    r2 = ba.buy_roaster_upgrade()
    res.append(ok(r1["status"] == "success" and ba.cash == cash0 - 2500 and r2["status"] == "error"
                  and any(e.entry_type == "operating_expense" and "upgrade" in (e.memo or "")
                          for e in env.truth_ledger["roaster_A"]), "E4a capex paid + expensed once"))
    res.append(ok(env.roast_cap_for("roaster_A", 2) == 50 and env.roast_cap_for("roaster_A", 3) == 90
                  and env.roast_cap_for("roaster_B", 3) == 50, "E4a roast cap 50 → 90 from day 3 (buyer only)"))
    env.time_manager.virtual_min = 3 * 1440 + 9 * 60
    res.append(ok(ba._inventory_cap_kg() == 180 and env.business_apps["roaster_B"]._inventory_cap_kg() == 120,
                  "E4a inventory cap +60 after the build delay"))
    res.append(ok("buy_roaster_upgrade" in tool_names_for("roaster", M.parse({"roaster_upgrade": {"enabled": True}}))
                  and "buy_roaster_upgrade" not in tool_names_for("roaster", {})
                  and "buy_roaster_upgrade" not in tool_names_for("farmer", M.parse({"roaster_upgrade": {"enabled": True}})),
                  "E4a tool registered only when enabled, roasters only"))
    # E4b
    for _ in range(4):
        env._note_availability("retailer_A", "roasted_coffee_kg", True)
    b4 = env._availability_bonus("retailer_A", "roasted_coffee_kg")
    env._note_availability("retailer_A", "roasted_coffee_kg", False)
    res.append(ok(b4 == 2.0 and env._availability_bonus("retailer_A", "roasted_coffee_kg") == 0.0,
                  "E4b floor +0.5 kg/day per in-stock day, reset on stock-out"))
    for _ in range(20):
        env._note_availability("retailer_B", "roasted_coffee_kg", True)
    res.append(ok(env._availability_bonus("retailer_B", "roasted_coffee_kg") == 5.0, "E4b capped at +5 kg/day"))
    # E2 farmer: counter-quotes concede half the gap per closer offer, never below reservation.
    fm = HeuristicFarmerModel(params={"negotiate": True})
    seq = []
    for price in (2.0, 2.2, 2.2, 2.35):
        seq.append(fm._negotiate("roaster_A", "green_coffee_kg", price, thr=2.88, floor=2.3, list_price=3.2))
    quotes = [q for _, q in seq]
    res.append(ok(quotes == [2.44, 2.4, 2.4, 2.4] and not any(a for a, _ in seq),
                  f"E2 farmer quotes {quotes} (list 3.20, reservation 2.40 = 75% of list)"))
    acc = fm._negotiate("roaster_A", "green_coffee_kg", 2.4, thr=2.88, floor=2.3, list_price=3.2)
    res.append(ok(acc == (True, None), "E2 farmer accepts at the current quote"))
    # E2 retailer: bid raises after rejections, capped at the ask and ceiling.
    rt = HeuristicRetailerModel(params={"negotiate": True})
    rt.me, rt.day = "retailer_A", 5
    lim = {"roasted_coffee_kg": 12.0, "roasted_specialty_kg": 35.0}
    lst = {"seller_id": "roaster_A", "item_id": "roasted_coffee_kg", "asking_price": 12.0}
    bids = [rt._bid_price(lst, lim)]
    for i in range(4):
        rt._last_call = {"name": "view_offers", "args": {"direction": "all"}}
        rt._on_result("view_offers", {"status": "success", "offers": [
            {"id": f"o{i}", "buyer_id": "retailer_A", "seller_id": "roaster_A", "item_id": "roasted_coffee_kg",
             "status": "rejected", "posted_day": 5, "listing_id": "L"}]})
        bids.append(rt._bid_price(lst, lim))
    res.append(ok(bids == [10.2, 10.8, 11.4, 12.0, 12.0], f"E2 retailer bids {bids} (85% → ask, 5%/rejection)"))
    hi = dict(lst, asking_price=20.0)
    res.append(ok(rt._bid_price(hi, lim) is None, "E2 retailer never bids above its ceiling region"))
    return all(res)


def test_persona(tmp: Path) -> bool:
    sys.path.insert(0, str(ROOT / "harbor_adapter"))
    from generate_tasks import generate

    res = []
    cfg = CFG_DIR / "hb_roaster_d30_persona_E.toml"
    plain = generate(cfg, tmp / "ds", "none", None, None)
    pers = generate(cfg, tmp / "ds", "none", None, "Persona:Sonnet55")
    res.append(ok(plain.name != pers.name and "persona" in pers.name, f"dataset names {plain.name} / {pers.name}"))
    for aid in ("farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"):
        a = (plain / aid / "instruction.md").read_text()
        b = (pers / aid / "instruction.md").read_text()
        if aid == "roaster_A":
            res.append(ok(a == b and "Private business profile" not in b, "focal roaster_A: no persona"))
            continue
        res.append(ok(b.startswith(a) and "Private business profile" in b[len(a):], f"{aid}: persona appended only"))
    fb = (pers / "farmer_B" / "instruction.md").read_text()
    fa = (pers / "farmer_A" / "instruction.md").read_text()
    res.append(ok("profile `adversarial`" in fb and "supply is short" in fb and "supply is short" not in fa
                  and "counter-quote" in fa, "farmer_B adversarial / farmer_A negotiator personas"))
    return all(res)


def test_harbor(_tmp: Path) -> bool:
    """Checks the three scripted Harbor runs (all $0, local Docker):
        HARBOR_CMD=... SCRIPTED_MECH=1 uv run python harbor_adapter/launch_sim.py \
            --config experiments/camera_ready_hb/hb_roaster_d30_rule_E.toml --seed 0 --scripted \
            --bg Rule|Persona:Sonnet55 --max-days 8 --network relay
        (+ hb_roaster_d30_persona_E.toml --bg Persona:Sonnet55: rule roaster_B, persona rest)."""
    res = []
    base = ROOT / "trajectories" / "harbor"
    runs = {"rule": base / "Rule/scripted/hb_roaster_d30_rule_E/seed_0/run.json",
            "persona_all": base / "PersonaSonnet55/scripted/hb_roaster_d30_rule_E/seed_0/run.json",
            "persona_mixed": base / "PersonaSonnet55/scripted/hb_roaster_d30_persona_E/seed_0/run.json"}
    for name, p in runs.items():
        if not p.exists():
            res.append(ok(False, f"{name}: {p} missing"))
            continue
        d = json.loads(p.read_text())
        ev = [json.loads(x) for x in p.with_name("run.events.jsonl").read_text().splitlines()]
        up = [e for e in ev if (e.get("type") or e.get("event")) == "roaster_upgrade_bought"]
        res.append(ok(any(e.get("agent_id") == "roaster_A" for e in up), f"{name}: remote focal bought the upgrade"))
        models = d["models"]
        print(f"    {name}: models {models}")
        if name == "persona_mixed":
            res.append(ok(models["roaster_B"] == "heuristic_roaster_v2"
                          and all(models[a] != "heuristic_roaster_v2" for a in ("farmer_A", "retailer_A")),
                          "mixed: roaster_B rule, other seats remote"))
        if name.startswith("persona"):
            prod = [e for e in d["truth_ledger"]["farmer_B"] if (e.get("memo") or "").startswith("produce ")]
            res.append(ok(any("$1.70/unit" in e["memo"] for e in prod),
                          f"{name}: remote farmer_B produces at the trait cost ($1.70): {[e['memo'] for e in prod][:1]}"))
        if name == "rule":
            shorts = [e for e in ev if (e.get("type") or e.get("event")) == "shipment_short"]
            res.append(ok(all(e["seller"] == "farmer_B" for e in shorts), f"rule: short shipments only from farmer_B ({len(shorts)})"))
            fl = [s for s in d["consumer_sales_log"] if s["floor_qty"] > 5]
            res.append(ok(len(fl) > 0, f"rule: availability growth raised retail floors ({len(fl)} sales)"))
    return all(res)


def test_off(_tmp: Path) -> bool:
    from coffeebench import environment as env_mod
    from coffeebench.config import RunConfig
    from coffeebench.server.prompts import build_system_prompts

    res = []
    cfg = CFG_DIR / "hb_roaster.toml"
    env = build_env(cfg, seed=0, models=",".join(f"{a}:passive" for a in (
        "farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B")))
    res.append(ok(env.mechanics == {} and env.festivals == env_mod.FESTIVALS and env.mechanics_summary is None,
                  "no [mechanics] → nothing enabled, module FESTIVALS"))
    c = RunConfig.from_toml(cfg)
    p1 = build_system_prompts(c)
    c.mechanics = {}
    from coffeebench.main import _build_score_framing, _seed_world
    from coffeebench.rules_prompt import build_rules_prompt

    items, ends = _seed_world()
    e = ends[2]
    legacy = build_rules_prompt(e, ends, items, c.max_days, _build_score_framing(None, c.max_days))
    res.append(ok(p1["roaster_A"]["prompt"] == legacy, "rules prompt unchanged without [mechanics]"))
    return all(res)


# ---------------------------------------------------------------- sims


def summarize(run_json: Path) -> dict:
    d = json.loads(run_json.read_text())
    ni = {a: round(v["audit"]["annual"]["true_net_income"]) for a, v in d["result"]["agents"].items()}
    deals = [x for x in d["marketplace"]["deals"] if x.get("cancelled_day") is None]
    return {"ni": ni, "deals": len(deals), "bankrupt": d["bankrupt_day"],
            "consumer_kg": sum(s["qty"] for s in d["consumer_sales_log"]), "mechanics": d.get("mechanics"),
            "data": d}


def check_shock(s: dict) -> bool:
    d = s["data"]
    sched = ((s["mechanics"] or {}).get("schedule") or {}).get("supply_shock")
    if not sched:
        return True
    good = True
    per_day: dict = {}
    for aid in ("farmer_A", "farmer_B"):
        for e in d["truth_ledger"][aid]:
            m = re.match(r"produce (\d+) (\w+) at \$([\d.]+)/unit \(ready day (\d+)\)", e.get("memo") or "")
            if m:
                day = int(m.group(4)) - 2
                key = (aid, m.group(2), day)
                per_day[key] = per_day.get(key, 0) + int(m.group(1))
                in_shock = sched["start"] <= day <= sched["end"]
                cost = {"green_coffee_kg": 2.0, "green_specialty_kg": 10.0}[m.group(2)]
                tf = (s["mechanics"]["schedule"].get("traits") or {}).get(aid, {}).get("production_cost_factor", 1.0)
                good &= abs(float(m.group(3)) - round(cost * (1.5 if in_shock else 1.0) * tf, 4)) < 0.006
    for (aid, item, day), q in per_day.items():
        if sched["start"] <= day <= sched["end"]:
            good &= q <= {"green_coffee_kg": 12, "green_specialty_kg": 4}[item]
    if not good:
        print("    shock check failed:", sched, sorted(per_day.items())[:12])
    adv_days = set()
    for m in d["messages_per_agent"]["roaster_A"]:
        if m["role"] == "user" and "Weather advisory (public)" in m["content"]:
            dm = re.search(r"Day (\d+)", m["content"])
            adv_days.add(int(dm.group(1)))
    k = sched["advisory_days"]
    good &= min(adv_days, default=-1) == sched["start"] - k
    return good


def test_sims(tmp: Path, configs: list[str], seeds: list[int]) -> bool:
    from concurrent.futures import ThreadPoolExecutor

    res = []
    jobs = [(c, f, s) for c in configs for f in FOCALS for s in seeds]
    with ThreadPoolExecutor(max_workers=6) as ex:
        paths = list(ex.map(lambda j: run_sim(CFG_DIR / f"{j[0]}.toml", j[2], tmp / "sims" / j[1], j[1]), jobs))
    table: dict = {}
    for (c, f, s), p in zip(jobs, paths):
        table[(c, f, s)] = summarize(p)
    agents = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]
    for c in configs:
        print(f"\n  == {c}: true_net_income per agent (deals, consumer kg)")
        print("  " + f"{'focal':22s} seed " + " ".join(f"{a:>10s}" for a in agents) + "  deals  cons_kg")
        for f in FOCALS:
            for s in seeds:
                t = table[(c, f, s)]
                print(f"  {f:22s} {s:4d} " + " ".join(f"{t['ni'][a]:10d}" for a in agents)
                      + f"  {t['deals']:5d}  {t['consumer_kg']:7d}")
        means = {f: mean(table[(c, f, s)]["ni"]["roaster_A"] for s in seeds) for f in FOCALS}
        print("  focal mean NI: " + ", ".join(f"{f}={v:.0f}" for f, v in means.items()))
        res.append(ok(means["passive"] < means["heuristic_roaster"] < means["heuristic_roaster_v2"],
                      f"{c}: passive < heuristic < heuristic_v2"))
        res.append(ok(all(table[(c, "heuristic_roaster_v2", s)]["bankrupt"] == {} for s in seeds),
                      f"{c}: no bankruptcy with the v2 focal"))
        res.append(ok(all(table[(c, f, s)]["deals"] >= 40 for f in FOCALS for s in seeds),
                      f"{c}: active trading (≥ 40 deals in every run)"))
        res.append(ok(all(check_shock(table[(c, f, s)]) for f in FOCALS for s in seeds),
                      f"{c}: supply shock caps/costs + advisory timing as configured"))
    return all(res)


def test_determinism(tmp: Path) -> bool:
    cfg = CFG_DIR / "hb_roaster_d30_rule_E.toml"  # every mechanic on
    a = run_sim(cfg, 1, tmp / "det_a")
    b = run_sim(cfg, 1, tmp / "det_b")
    return ok(normalized(a) == normalized(b), "same seed twice → identical run.json + events")


def test_resume(tmp: Path) -> bool:
    cfg = CFG_DIR / "hb_roaster_d30_rule_E.toml"  # every mechanic on
    full = run_sim(cfg, 2, tmp / "res_full")
    ck = tmp / "ckpts"
    subprocess.run([sys.executable, "-m", "coffeebench.main", "--config", str(cfg), "--seed", "2",
                    "--output-root", str(tmp / "res_seg"), "--checkpoint-dir", str(ck), "--stop-day", "12"],
                   cwd=ROOT, check=True, capture_output=True)
    seg = run_sim(cfg, 2, tmp / "res_seg", None, "--resume", str(ck / "day_012.pkl"))
    return ok(normalized(full) == normalized(seg), "stop at day 12 + resume → identical run.json + events")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="unit,e_unit,persona,off,sims,determinism,resume")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default="hb_roaster_d30_rule,hb_roaster_d30_rule_D,hb_roaster_d30_rule_E")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = args.configs.split(",")
    results = {}
    with tempfile.TemporaryDirectory(prefix="cb_rule_") as td:
        tmp = Path(td)
        for name in args.only.split(","):
            print(f"[{name}]")
            fn = {"unit": test_unit, "e_unit": test_e_unit, "persona": test_persona, "off": test_off,
                  "determinism": test_determinism, "resume": test_resume, "harbor": test_harbor}.get(name)
            results[name] = fn(tmp) if fn else test_sims(tmp, configs, seeds)
            print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
