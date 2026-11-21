"""$0 offline tests for the `heuristic_roaster_v2` baseline (no LLM, no network).

    unset VIRTUAL_ENV; uv run python scripts/test_heuristic_roaster_v2.py [--only unit,passive,market]

unit     decision functions (pricing step, clearance, max bid, festival phase,
         invoice timing, terms premium, anchor) + a replay of the policy on
         canned tool results (mid-day wake does not restart the routine).
passive  90-day in-process sim, rest passive (`python -m coffeebench.main`).
market   90-day in-process sims in a scripted TEST-ONLY "liquid market": both
         farmers produce and list green at cost × 1.5 every day, both retailers
         bid on roasted listings priced ≤ a fixed willingness to pay (more kg
         during the public festival window), roaster_B passive. Runs v2 and the
         old `heuristic_roaster` on the same market and checks that v2 buys
         both greens, roasts, sells, uses clearance + festival pricing, pays
         its invoices and keeps cash ≥ 0. The NI numbers describe this toy
         market only, not the benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffeebench.models import heuristic_roaster_v2_model as hv2  # noqa: E402
from coffeebench.models.types import ModelResponse, ToolCall  # noqa: E402

CONFIG = """\
[experiment]
name = "{name}"
description = "heuristic_roaster_v2 offline test ($0)."

[run]
seeds = [0]
max_days = {days}
main_agent = "roaster_A"

[models]
default = "passive"
roaster_A = "{roaster}"
"""


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------

def test_unit() -> bool:
    ok = True

    def check(cond: bool, label: str) -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")

    np = hv2.next_price
    check(np(10.0, True, False, 7.0, 30.0) == 10.5, "sell-out → +5%")
    check(np(10.0, False, True, 7.0, 30.0) == 9.5, "unsold day → −5%")
    check(np(10.0, False, False, 7.0, 30.0) == 10.0, "partial sale → hold")
    check(np(7.2, False, True, 7.0, 30.0) == 7.0, "cut clamped at floor")
    check(np(29.0, True, False, 7.0, 30.0) == 30.0, "raise clamped at reservation cap")
    check(np(10.0, False, True, 7.0, 30.0, allow_cut=False) == 10.0, "no cuts during festival ramp")
    check(hv2.clearance_price(10.0, 6.0, 5) == 9.5, "clearance 5 d left → −5%")
    check(hv2.clearance_price(10.0, 6.0, 3) == 8.5, "clearance 3 d left → −15%")
    check(hv2.clearance_price(10.0, 8.0, 2) == 8.0, "clearance floored at lot cost (≥ 2 d)")
    check(hv2.clearance_price(10.0, 8.0, 0) == 7.0, "clearance 0 d → −30% (floor ½ cost)")
    check(abs(hv2.max_bid(12.0, 0.85, 3.0) - 5.5) < 1e-9, "max bid commodity = 12×0.85/1.2 − 3")
    check(abs(hv2.max_bid(30.0, 0.82, 5.0) - 15.5) < 1e-9, "max bid specialty = 30×0.82/1.2 − 5")
    check(hv2.max_bid(3.0, 0.85, 3.0) == 0.0, "max bid never negative")
    w = [(40, 53)]
    check([hv2.festival_phase(d, w) for d in (34, 35, 39, 40, 53, 54)]
          == [None, "ramp", "ramp", "festival", "festival", None], "festival ramp = 5 days before")
    ia = hv2.invoice_action
    check(ia(10, 10, 100, 500, 400) == "pay", "invoice due today → pay")
    check(ia(8, 10, 100, 150, 400) == "pay", "overdue → pay even inside buffer")
    check(ia(11, 10, 100, 600, 400) == "pay", "due tomorrow, buffer intact → pay")
    check(ia(11, 10, 100, 450, 400) == "defer", "due tomorrow, would breach buffer → defer")
    check(ia(15, 10, 100, 9999, 400) == "defer", "not due yet → defer")
    check(ia(10, 10, 100, 50, 0) == "defer", "cannot cover → defer (never negative cash)")
    check(hv2.required_price(10.0, 30) == 10.0 and abs(hv2.required_price(10.0, 60) - 10.6) < 1e-9,
          "terms premium 0.2%/day beyond 30 d")
    check(hv2.anchor_price(5.88, start_retail=18.0) == 10.8, "anchor = start retail × 0.6")
    check(hv2.anchor_price(5.88, consumer_avg=20.0, start_retail=18.0) == 12.0, "consumer avg beats start retail")
    check(hv2.anchor_price(5.88, own_last=11.0, consumer_avg=20.0) == 11.0, "own deal price first")
    check(hv2.anchor_price(5.88, start_retail=18.0, competitor_ask=9.0) == 9.0, "match competitor ask")
    check(hv2.anchor_price(5.88, start_retail=18.0, competitor_ask=6.0) == 7.06, "never below 1.2× cost")

    # Policy replay on canned results: the routine survives a mid-day wake.
    m = hv2.HeuristicRoasterV2Model()
    m.system_prompt = ("# CoffeeBench — Roaster A (`roaster_A`, roaster)\nThe other five: farmer_A (farmer), "
                       "retailer_A (retailer), retailer_B (retailer). consumers never pay more than the "
                       "reservation price (roasted_coffee_kg $30/kg, roasted_specialty_kg $80/kg). "
                       "starting retail prices roasted_coffee_kg $18/kg, roasted_specialty_kg $50/kg\n"
                       "demand is higher during the public festival window(s): `spring_break` days 40–53")
    hist: list[dict] = [{"role": "user", "content": "Observation: Day 3, 09:00 (x). Cash $5000.00 | ..."}]

    def step(result: dict | None = None, obs: str | None = None) -> ToolCall:
        if result is not None:
            last = hist[-1]["tool_calls"][0]
            hist.append({"role": "tool", "name": last.name, "content": json.dumps(result)})
        if obs is not None:
            hist.append({"role": "user", "content": obs})
        call = m.query(hist).tool_calls[0]
        hist.append({"role": "assistant", "tool_calls": [call]})
        return call

    c = step()
    check(c.name == "view_dashboard", "morning starts with view_dashboard")
    check(m.festivals == [(40, 53)] and m.me == "roaster_A" and m.retailers == ["retailer_A", "retailer_B"]
          and m.start_retail["roasted_coffee_kg"] == 18.0, "rules prompt parsed (festival, ids, prices)")
    dash = {"status": "success", "headline": {"cash": 5000.0, "ap": 120.0},
            "inventory": {"on_hand_kg": 70, "in_flight_inbound_kg": 0, "cap_kg": 120},
            "pending_roasts": [], "my_open_listings": {"rows": []},
            "payables_due": {"rows": [{"id": "inv_1", "outstanding": 120.0, "due_day": 3}]}}
    c = step(dash)
    check(c.name == "view_inventory", "then view_inventory")
    inv = {"status": "success", "lots": [
        {"item_id": "green_specialty_kg", "qty": 20, "unit_cost": 15.0, "days_left": 50},
        {"item_id": "green_coffee_kg", "qty": 40, "unit_cost": 3.0, "days_left": 55},
        {"item_id": "roasted_coffee_kg", "qty": 4, "unit_cost": 6.0, "days_left": 2},
        {"item_id": "roasted_coffee_kg", "qty": 6, "unit_cost": 7.0, "days_left": 20}]}
    c = step(inv)
    check(c.name == "view_offers", "then view_offers")
    c = step({"status": "success", "offers": [
        {"id": "off_1", "listing_id": "lst_x", "seller_id": "roaster_A", "buyer_id": "retailer_A", "item_id": "roasted_coffee_kg",
         "offered_price": 5.0, "qty": 3, "payment_terms_days": 7, "status": "pending"}]})
    check(c.name == "view_listings", "then view_listings")
    c = step({"status": "success", "listings": [
        {"id": "lst_g", "seller_id": "farmer_A", "item_id": "green_coffee_kg", "qty": 30,
         "asking_price": 3.0, "payment_terms_days": 14, "delivery_eta_days": 1}]},
             obs="Observation: Day 3, 11:00 (x).\n[Wake reason: offer_received: retailer_B offered 2 kg]")
    check(c.name == "view_offers" and c.input == {"direction": "incoming"},
          "mid-day offer wake → view_offers inserted, routine not restarted")
    c = step({"status": "success", "offers": [
        {"id": "off_2", "listing_id": "lst_y", "seller_id": "roaster_A", "buyer_id": "retailer_B", "item_id": "roasted_coffee_kg",
         "offered_price": 12.0, "qty": 2, "payment_terms_days": 7, "status": "pending"}]})
    check(c.name == "accept_offer" and c.input["offer_id"] == "off_2", "accept offer above floor (best first)")
    def fake_result(call: ToolCall) -> dict:
        """Plausible success payloads for the calls the policy makes."""
        a = call.input
        if call.name == "accept_offer":
            return {"status": "success", "item_id": "roasted_coffee_kg", "qty": 2, "unit_price": 12.0}
        if call.name == "roast":
            y = 0.85 if a["green_item_id"] == "green_coffee_kg" else 0.82
            return {"status": "success", "input_item": a["green_item_id"],
                    "output_item": a["green_item_id"].replace("green", "roasted"),
                    "green_consumed_kg": a["qty_kg"], "roasted_output_kg": int(a["qty_kg"] * y),
                    "remaining_cash": 4000.0, "shared_roast_capacity_used_today": 0}
        if call.name == "post_listing":
            return {"status": "success", "listing_id": f"lst_{a['item_id'][:4]}_{a['qty']}"}
        return {"status": "success"}

    names = [(c.name, c.input)]
    for _ in range(25):
        c = step(fake_result(c))
        names.append((c.name, c.input))
        if c.name == "wait_for_next_day":
            break
    seq = [n for n, _ in names]
    check(seq.count("view_dashboard") == 0, "no restart of the morning views after the wake")
    check(("reject_offer", {"offer_id": "off_1"}) in names, "rejects the sub-floor morning offer")
    check(("pay_invoice", {"invoice_id": "inv_1"}) in names, "pays the invoice due today")
    roasts = [a for n, a in names if n == "roast"]
    check(roasts and roasts[0]["green_item_id"] == "green_specialty_kg"
          and sum(r["qty_kg"] for r in roasts) == 50, "roasts specialty first, 50 kg total")
    posts = [a for n, a in names if n == "post_listing"]
    check(any(p["qty"] == 2 and p["asking_price"] < 10.8 for p in posts) and any(p["qty"] == 6 for p in posts),
          "clearance listing for the 2-day lot (4 kg − 2 kg shipped FIFO)")
    check(any(n == "make_offer" for n in seq), "bids on green")
    check(seq[-1] == "wait_for_next_day" and len(seq) + 6 <= hv2.ACTION_BUDGET + 1, "ends the day within budget")
    return ok


# ---------------------------------------------------------------------------
# TEST-ONLY scripted counterparts (liquid market)
# ---------------------------------------------------------------------------

class _Scripted:
    """Minimal scripted policy: daily plan + a wake hook; reads every message
    after its last assistant turn (same contract as the real baselines)."""

    checkpointable = True

    def __init__(self, agent_id: str):
        self.model = f"test/{type(self).__name__}"
        self.agent_id = agent_id
        self.cost, self.n_calls = 0.0, 0
        self.total_input_tokens = self.total_output_tokens = self.last_input_tokens = 0
        self.max_input_tokens = 1_000_000
        self.system_prompt = ""
        self.day, self.plan, self.seq = -1, [], 0

    def get_usage_stats(self) -> dict:
        return {"model": self.model, "n_calls": self.n_calls, "cost": 0.0}

    def summarize(self, instructions, content, max_tokens=4096):  # noqa: ARG002
        return content

    def query(self, messages, tools=None):  # noqa: ARG002
        self.n_calls += 1
        tail = []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                break
            tail.append(msg)
        for msg in reversed(tail):
            if msg.get("role") == "user":
                txt = msg.get("content") or ""
                d = hv2._extract_day(txt)
                if d is not None and d != self.day:
                    self.day, self.plan = d, [["view_dashboard", {}]]
                elif "[Wake reason: offer_received" in txt:
                    self.plan.insert(0, ["view_offers", {"direction": "incoming"}])
            elif msg.get("role") == "tool":
                try:
                    data = json.loads(msg.get("content") or "{}")
                except ValueError:
                    data = {}
                if isinstance(data, dict) and data.get("status") == "success":
                    self.ingest(msg.get("name"), data)
        tool, args = self.plan.pop(0) if self.plan else ["wait_for_next_day", {}]
        self.seq += 1
        return ModelResponse(content="(test)", stop_reason="tool_use", tool_calls=[
            ToolCall(id=f"t_{self.agent_id}_{self.seq}", name=tool, input=args)])

    def ingest(self, name, data):  # pragma: no cover - overridden
        pass


class LiquidFarmer(_Scripted):
    """Produces both greens at the daily cap (within its inventory cap), lists
    newly landed stock at cost × 1.5 (never cancels); accepts any offer ≥ cost."""

    COST = {"green_specialty_kg": 10.0, "green_coffee_kg": 2.0}  # specialty produced first
    CAP = {"green_specialty_kg": 10, "green_coffee_kg": 30}

    def ingest(self, name, data):
        if name == "view_dashboard":
            inv = {r["item_id"]: r["qty"] for r in data["inventory"]["by_item"]}
            held = data["inventory"]["on_hand_kg"] + sum(p["qty"] for p in data.get("pending_production", []))
            plan = [["view_offers", {"direction": "incoming"}]]
            plan += [["pay_invoice", {"invoice_id": r["id"]}] for r in data["payables_due"]["rows"]]
            # Never cancel (a buyer may be bidding on it): list only stock not
            # yet covered by an open listing.
            listed: dict = {}
            for r in data["my_open_listings"]["rows"]:
                listed[r["item_id"]] = listed.get(r["item_id"], 0) + r["qty"]
            for it, c in self.COST.items():
                new = inv.get(it, 0) - listed.get(it, 0)
                if new > 0 and not data["my_open_listings"].get("omitted"):
                    plan.insert(0, ["post_listing", {"item_id": it, "qty": new, "asking_price": c * 1.5,
                                                     "payment_terms_days": 14}])
                q = min(self.CAP[it], 120 - held)
                if q > 0:
                    held += q
                    plan.append(["produce_item", {"item_id": it, "quantity": q}])
            self.plan = plan + self.plan
        elif name == "view_offers":
            acc = [["accept_offer", {"offer_id": o["id"]}] for o in data["offers"]
                   if o["status"] == "pending" and o["offered_price"] >= self.COST.get(o.get("item_id"), 1e9)]
            self.plan = acc + self.plan


class LiquidRetailer(_Scripted):
    """Bids the ask on roasted listings priced ≤ a fixed willingness to pay
    (WTP), wanting 2.5× the kg at 1.25× the WTP during the public festival
    window; pays AP on the due day."""

    WTP = {"roasted_coffee_kg": 12.0, "roasted_specialty_kg": 38.0}
    WANT = {"roasted_coffee_kg": 18, "roasted_specialty_kg": 6}

    def ingest(self, name, data):
        if name == "view_dashboard":
            self.headroom = data["inventory"]["cap_kg"] - data["inventory"]["on_hand_kg"] - data[
                "inventory"]["in_flight_inbound_kg"]
            plan = [["pay_invoice", {"invoice_id": r["id"]}] for r in data["payables_due"]["rows"]
                    if r["due_day"] <= self.day]
            plan.append(["view_listings", {}])
            self.plan = plan + self.plan
        elif name == "view_listings":
            fest = 2.5 if 40 <= self.day <= 53 else 1.0  # festival: more kg, WTP × 1.25
            plan = []
            for r in sorted(data["listings"], key=lambda r: r["asking_price"]):
                it = r["item_id"]
                wtp = self.WTP.get(it, 0.0) * (1.25 if fest > 1 else 1.0)
                if it in self.WTP and r["asking_price"] <= wtp and self.headroom > 0:
                    q = min(int(self.WANT[it] * fest), r["qty"], self.headroom)
                    if q > 0:
                        self.headroom -= q
                        plan.append(["make_offer", {"listing_id": r["id"], "offered_price": r["asking_price"],
                                                    "qty": q, "payment_terms_days": 14}])
            self.plan = plan + self.plan


HELPERS = {"farmer_A": LiquidFarmer, "farmer_B": LiquidFarmer,
           "retailer_A": LiquidRetailer, "retailer_B": LiquidRetailer}


def run_market(cfg: Path) -> Path:
    from coffeebench.agent import Agent
    from coffeebench.main import INITIAL_OBSERVATION, build_run

    def factory(aid, model_id, tools, prompt):  # noqa: ARG001
        if aid in HELPERS:
            return Agent(model=HELPERS[aid](aid), tools=tools, instruct_prompt=prompt,
                         initial_observation=INITIAL_OBSERVATION, name=aid)
        return None

    args = argparse.Namespace(config=str(cfg), model=None, models=None, max_days=None, seed=0,
                              main_agent=None, output=None, output_root="trajectories")
    env, out, _ = build_run(args, agent_factory=factory)
    env.verbose = False
    asyncio.run(env.run())
    env.save_trajectory(out)
    env.event_logger.close()
    return Path(out).parent


def summarize(run_dir: Path, aid: str = "roaster_A") -> dict:
    data = json.loads((run_dir / "run.json").read_text())
    a = data["result"]["agents"][aid]
    ann = a["audit"]["annual"]
    ev = [json.loads(ln) for ln in (run_dir / "run.events.jsonl").read_text().splitlines() if ln.strip()]
    st = [e for e in ev if e["type"] == "agent_step" and e["agent_id"] == aid]
    deals = data["marketplace"]["deals"] if isinstance(data.get("marketplace"), dict) else []
    bought = {}
    sold = {}
    for d in deals:
        if d.get("cancelled_day") is not None:
            continue
        if d["buyer_id"] == aid:
            bought[d["item_id"]] = bought.get(d["item_id"], 0) + d["qty"]
        if d["seller_id"] == aid:
            sold[d["item_id"]] = sold.get(d["item_id"], 0) + d["qty"]
    cash = [ds["per_agent"][aid]["cash"] for ds in data.get("daily_stats") or []]
    fest_sales = [(d["unit_price"], d["item_id"]) for d in deals
                  if d["seller_id"] == aid and 40 <= d["deal_at"] // 1440 <= 53]
    pre_sales = [(d["unit_price"], d["item_id"]) for d in deals
                 if d["seller_id"] == aid and 30 <= d["deal_at"] // 1440 < 35]
    errors = [e for e in st if e["observation"].get("status") == "error"]
    per_day = {}
    for e in st:
        if e["action"] != "wait_for_next_day":
            per_day[e["day"]] = per_day.get(e["day"], 0) + 1
    roast = a["audit"].get("roast_metrics") or {}
    return {
        "ni": round(ann["true_net_income"], 2),
        "revenue": round(ann["true_revenue"], 2),
        "spoilage": round(ann["true_opex_buckets"]["spoilage_expense"], 2),
        "interest": round(ann["true_interest_net"], 2),
        "bought": bought, "sold": sold,
        "roasted_green_kg": roast.get("green_consumed_kg"),
        "min_eod_cash": min(cash) if cash else None,
        "bankrupt": a["bankrupt_day"],
        "errors": len(errors),
        "error_kinds": sorted({(e["action"], str(e["observation"].get("message"))[:60]) for e in errors})[:6],
        "max_actions_day": max(per_day.values()) if per_day else 0,
        "avg_commodity_price_fest": _avg(fest_sales, "roasted_coffee_kg"),
        "avg_commodity_price_d30_34": _avg(pre_sales, "roasted_coffee_kg"),
        "clearance_days": _clearance_days(st),
        "messages": sum(1 for e in st if e["action"] == "send_message"),
        "invoices_paid": sum(1 for e in st if e["action"] == "pay_invoice"),
    }


def _clearance_days(steps: list) -> int:
    """Days with a listing priced below 1.2 × the item's cheapest lot cost
    (from the preceding view_inventory) — only the near-expiry clearance rule
    may go below the regular floor."""
    days, low = set(), {}
    for e in steps:
        obs = e["observation"]
        if e["action"] == "view_inventory" and obs.get("status") == "success":
            low = {}
            for lt in obs["lots"]:
                low[lt["item_id"]] = min(low.get(lt["item_id"], 1e9), lt["unit_cost"])
        if e["action"] == "post_listing" and obs.get("status") == "success":
            it = e["action_input"]["item_id"]
            if e["action_input"]["asking_price"] < 1.2 * low.get(it, 0) - 1e-6:
                days.add(e["day"])
    return len(days)


def _avg(rows: list, item: str) -> float | None:
    xs = [p for p, it in rows if it == item]
    return round(sum(xs) / len(xs), 2) if xs else None


def test_passive(tmp: Path) -> bool:
    cfg = tmp / "hv2test_passive.toml"
    cfg.write_text(CONFIG.format(name="hv2test_passive", days=90, roaster="heuristic_roaster_v2"))
    subprocess.run([sys.executable, "-m", "coffeebench.main", "--config", str(cfg), "--seed", "0"],
                   cwd=ROOT, check=True, capture_output=True)
    s = summarize(ROOT / "trajectories/hv2test_passive/seed_0")
    print(f"  passive world: {s}")
    return (s["bankrupt"] is None and s["errors"] == 0 and s["max_actions_day"] <= 20
            and s["clearance_days"] > 0)


def test_market(tmp: Path) -> bool:
    res = {}
    for tag, model in (("v2", "heuristic_roaster_v2"), ("v1", "heuristic_roaster")):
        name = f"hv2test_market_{tag}"
        cfg = tmp / f"{name}.toml"
        cfg.write_text(CONFIG.format(name=name, days=90, roaster=model))
        res[tag] = summarize(run_market(cfg))
        res[tag]["dir"] = f"trajectories/{name}/seed_0"
        print(f"  {tag} ({model}): {json.dumps(res[tag])}")
    v2 = res["v2"]
    ok = (v2["bankrupt"] is None and (v2["min_eod_cash"] or 0) >= 0
          and v2["bought"].get("green_coffee_kg", 0) > 0 and v2["bought"].get("green_specialty_kg", 0) > 0
          and v2["sold"].get("roasted_coffee_kg", 0) > 0 and v2["sold"].get("roasted_specialty_kg", 0) > 0
          and (v2["roasted_green_kg"] or 0) > 0 and v2["max_actions_day"] <= 20)
    print(f"  NI in the liquid test market: v2 {v2['ni']:+.2f} vs old heuristic {res['v1']['ni']:+.2f}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="unit,passive,market")
    a = ap.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="hv2_") as td:
        for name in a.only.split(","):
            print(f"[{name}]")
            results[name] = {"unit": lambda _t: test_unit(), "passive": test_passive,
                             "market": test_market}[name](Path(td))
            print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
