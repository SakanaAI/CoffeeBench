"""Further statistics of Roaster A (main 90-day runs, mini-SWE-agent, background Claude Sonnet 5.5, highest effort).

    uv run python scripts/mf_extra_stats.py

Pooled over three runs (sales/purchase shares by revenue / spend; acceptance rates over offers):
  sales to retailers / rival / farmers        share of Roaster A's sales revenue by buyer type (%)
  purchases from farmers / rival / retailers  share of Roaster A's purchase spend by seller type (%)
  own offers accepted                          share of Roaster A's offers (any item) that the seller accepted (%)
  offers received accepted                     share of offers on Roaster A's listings that it accepted (%)
  end stock                                    inventory on hand at the end of the run (kg, mean per run)
  other expenses                               other operating expense ($ per run): roaster-upgrade capex, goods lost in
                                               transit, end-of-run valuation write-down and credit-loss allowance
  pre-shock buying                             green coffee bought in the 3 announced days before the supply shock,
                                               relative to the average 3-day purchase before the advisory (x)
Writes paper/tables_cr/extra.tex.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
MODELS = [("Sonnet55", "Claude~Sonnet~5.5"), ("Astra", "GPT-6~Astra"), ("Opus55", "Claude~Opus~5.5"),
          ("Haiku55", "Claude~Haiku~5.5"), ("K31P", "Kimi~K3.1~Pro"), ("GLM53", "GLM-5.3"),
          ("DSv41F", "DeepSeek~V4.1~Flash"), ("Gemini38F", "Gemini~3.8~Flash"), ("heuristicv2", "HeuristicRoaster")]


def kind(a: str) -> str:
    return "rival" if a == "roaster_B" else a.split("_")[0]


def main() -> None:
    lines = []
    for m, label in MODELS:
        sales, buys, c = Counter(), Counter(), Counter()
        end_kg, wd, shock = [], [], []
        for f in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*/run.json"))):
            R = json.load(open(f))
            mk = R["marketplace"]
            sched = R["mechanics"]["schedule"]["supply_shock"]
            s0, adv = int(sched["start"]), int(sched.get("advisory_days", 3))
            green = Counter()
            for x in mk["deals"]:
                if x.get("cancelled_day") is not None:
                    continue
                if x["seller_id"] == "roaster_A":
                    sales[kind(x["buyer_id"])] += x["total_price"]
                if x["buyer_id"] == "roaster_A":
                    buys[kind(x["seller_id"])] += x["total_price"]
                    if x["item_id"].startswith("green"):
                        green[x["deal_at"] // 1440] += x["qty"]
            pre = sum(green[d] for d in range(s0 - adv, s0))
            base_days = max(1, s0 - adv)
            base = 3 * sum(green[d] for d in range(0, s0 - adv)) / base_days
            if base > 0:
                shock.append(pre / base)
            for o in mk["offers"]:
                if o["buyer_id"] == "roaster_A":
                    c["own"] += 1
                    c["own_acc"] += o["status"] == "accepted"
                if o["seller_id"] == "roaster_A":
                    c["recv"] += 1
                    c["recv_acc"] += o["status"] == "accepted"
            a = R["result"]["agents"]["roaster_A"]
            wd.append(a["audit"]["annual"]["true_opex_buckets"].get("other_operating_expense", 0.0))
            last = None
            for line in open(Path(f).with_name("run.events.jsonl")):
                if '"day_end"' in line:
                    last = line
            if last:
                e = json.loads(last)
                end_kg.append(sum((e["snapshot"]["per_agent"]["roaster_A"].get("inventory_by_item") or {}).values()))
        ts, tb = sum(sales.values()) or 1.0, sum(buys.values()) or 1.0
        v = dict(s_ret=100 * sales["retailer"] / ts, s_riv=100 * sales["rival"] / ts, s_far=100 * sales["farmer"] / ts,
                 b_far=100 * buys["farmer"] / tb, b_riv=100 * buys["rival"] / tb, b_ret=100 * buys["retailer"] / tb,
                 own=100 * c["own_acc"] / max(1, c["own"]), recv=100 * c["recv_acc"] / max(1, c["recv"]),
                 end=st.mean(end_kg) if end_kg else float("nan"), wd=st.mean(wd), shock=st.mean(shock) if shock else float("nan"))
        print(f"{m:12s} " + " ".join(f"{k} {x:6.1f}" for k, x in v.items()))
        if m == "heuristicv2":
            lines.append("\\midrule")
        # counterparty shares are printed but not tabulated: in the main runs every model sells only to retailers
        # and buys only from farmers (100 % / 100 %)
        lines.append(f"{label} & {v['own']:.0f} & {v['recv']:.0f} & {v['end']:.0f} & {v['wd']:.0f} & {v['shock']:.1f} \\\\")
    out = ROOT / "paper" / "tables_cr" / "extra.tex"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
