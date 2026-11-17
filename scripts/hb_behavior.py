"""Behaviour metrics of the focal roaster in synced Harbor runs.

    uv run python scripts/hb_behavior.py [--root results/hb_f1] [--csv out.csv]

One row per (roster, focal model): P&L (revenue, COGS, gross margin, opex,
spoilage, NI), production (roasted kg, specialty share), B2B selling (kg, mean
unit price per item, payment terms, buyers), B2B buying (green kg, mean price,
suppliers), receivables at the end (and whether the debtor is bankrupt),
festival share of sales (days 40-53), capex (roaster upgrade), activity
(agent steps by tool, messages) and market context (rival roaster NI,
retail consumer prices, all-six NI).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCAL = "roaster_A"
FESTIVAL = range(40, 54)
VIEW_TOOLS = {"view_listings", "view_offers", "view_deals", "view_dashboard", "view_inventory",
              "view_consumer_sales", "view_market_aggregate", "view_messages", "view_payables",
              "view_receivables", "view_trial_balance", "read_message"}


def wmean(pairs):
    q = sum(n for n, _ in pairs)
    return sum(n * p for n, p in pairs) / q if q else float("nan")


def analyse(run_dir: Path) -> dict:
    r = json.loads((run_dir / "run.json").read_text())
    res = r["result"]
    ag = res["agents"]
    ann = ag[FOCAL]["audit"]["annual"]
    bs = ag[FOCAL]["audit"]["balance_sheet"]
    rm = ag[FOCAL]["audit"]["roast_metrics"]
    deals = [d for d in r["marketplace"]["deals"] if d.get("cancelled_day") is None]
    sell = [d for d in deals if d["seller_id"] == FOCAL]
    buy = [d for d in deals if d["buyer_id"] == FOCAL]
    by_item = collections.defaultdict(list)
    for d in sell:
        by_item[d["item_id"]].append((d["qty"] - d.get("returned_qty", 0), d["unit_price"]))
    fest = sum(d["total_price"] for d in sell if d["deal_at"] // 1440 in FESTIVAL)
    tot_sell = sum(d["total_price"] for d in sell) or float("nan")
    steps = collections.Counter()
    upgrade_day = None
    for line in open(run_dir / "run.events.jsonl"):
        if FOCAL not in line:
            continue
        ev = json.loads(line)
        if ev.get("agent_id") != FOCAL:
            continue
        if ev.get("type") == "agent_step":
            steps[ev.get("action")] += 1
        elif ev.get("type") == "roaster_upgrade_bought":
            upgrade_day = ev.get("day")
    bankrupt = set(r.get("bankrupt_agents") or [])
    ar = {}
    for a, amt in (r["agent_books"][FOCAL].get("accounts_receivable") or {}).items() if isinstance(
            r["agent_books"][FOCAL].get("accounts_receivable"), dict) else []:
        ar[a] = amt
    retail_px = collections.defaultdict(list)
    for s in r["consumer_sales_log"]:
        retail_px[s["item_id"]].append((s["qty"], s["unit_price"]))
    return {
        "ni": ann["true_net_income"],
        "revenue": ann["true_revenue"],
        "cogs": ann["true_cogs"],
        "gross_margin": 1 - ann["true_cogs"] / ann["true_revenue"] if ann["true_revenue"] else float("nan"),
        "opex": ann["true_opex"],
        "spoilage": ann["true_opex_buckets"].get("spoilage_expense", 0.0),
        "roasted_kg": rm["roasted_produced_kg"],
        "specialty_share": rm["specialty_capacity_share"],
        "sell_kg_com": sum(q for q, _ in by_item["roasted_coffee_kg"]),
        "sell_px_com": wmean(by_item["roasted_coffee_kg"]),
        "sell_kg_spec": sum(q for q, _ in by_item["roasted_specialty_kg"]),
        "sell_px_spec": wmean(by_item["roasted_specialty_kg"]),
        "sell_terms": statistics.mean(d["payment_terms_days"] for d in sell) if sell else float("nan"),
        "sell_deals": len(sell),
        "buyers": dict(collections.Counter(d["buyer_id"] for d in sell)),
        "buy_kg": sum(d["qty"] for d in buy),
        "buy_px": wmean([(d["qty"], d["unit_price"]) for d in buy]),
        "suppliers": dict(collections.Counter(d["seller_id"] for d in buy)),
        "ar_end": bs["true_accounts_receivable"],
        "ar_share_rev": bs["true_accounts_receivable"] / ann["true_revenue"] if ann["true_revenue"] else float("nan"),
        "cash_end": bs["true_cash"],
        "festival_share": fest / tot_sell,
        "upgrade_day": upgrade_day,
        "steps": sum(steps.values()),
        "views": sum(v for k, v in steps.items() if k in VIEW_TOOLS),
        "msgs": steps["send_message"],
        "top_tools": steps.most_common(5),
        "rival_ni": ag["roaster_B"]["audit"]["annual"]["true_net_income"],
        "all6_ni": sum(a["audit"]["annual"]["true_net_income"] for a in ag.values()),
        "retail_px_com": wmean(retail_px["roasted_coffee_kg"]),
        "retail_px_spec": wmean(retail_px["roasted_specialty_kg"]),
        "bankrupt": sorted(bankrupt),
        "final_day": res.get("actual_final_day"),
        "ni_by_agent": {a: round(v["audit"]["annual"]["true_net_income"]) for a, v in ag.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results" / "hb_f1")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for rj in sorted(args.root.glob("*/*/*/seed_0/run.json")):
        roster, model = rj.parts[-5].replace("Persona", ""), rj.parts[-4]
        rows.append({"roster": roster, "model": model, **analyse(rj.parent)})
    rows.sort(key=lambda x: (x["roster"], -x["ni"]))
    cols = ["roster", "model", "ni", "revenue", "gross_margin", "roasted_kg", "specialty_share", "sell_px_com",
            "sell_px_spec", "sell_terms", "buy_px", "ar_end", "festival_share", "upgrade_day", "steps", "views",
            "msgs", "rival_ni", "all6_ni", "retail_px_com", "retail_px_spec"]
    print(" ".join(f"{c[:11]:>11s}" for c in cols))
    for x in rows:
        out = []
        for c in cols:
            v = x[c]
            out.append(f"{v:11.2f}" if isinstance(v, float) and abs(v) < 10 else
                       f"{v:11.0f}" if isinstance(v, (int, float)) and v is not None else f"{str(v)[:11]:>11s}")
        print(" ".join(out))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
