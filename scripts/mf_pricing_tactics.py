"""How Roaster A buys cheap and sells high (main 90-day runs, mini-SWE-agent, background Claude Sonnet 5.5, highest effort).

    uv run python scripts/mf_pricing_tactics.py

Commodity coffee only (green_coffee_kg bought, roasted_coffee_kg sold), pooled over three runs, quantity-weighted:
  paid          price Roaster A paid for green coffee
  ask (bought)  asking price of the farmer listings it bought from
  offer gap     how far below the listing's asking price Roaster A's offers on green coffee were (mean over offers)
  accepted      share of those offers that the farmer accepted
  farm ask      mean asking price of all farmer green-coffee listings in the run, days 0-9 and whole run
  sell ask      asking price of Roaster A's roasted-coffee listings that produced deals
  sold / ask    price of the offers Roaster A received on roasted coffee relative to its asking price (mean over offers)
Writes paper/tables_cr/tactics.tex.
"""

from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
MODELS = [("Sonnet55", "Claude~Sonnet~5.5"), ("Astra", "GPT-6~Astra"), ("Opus55", "Claude~Opus~5.5"),
          ("Haiku55", "Claude~Haiku~5.5"), ("K31P", "Kimi~K3.1~Pro"), ("GLM53", "GLM-5.3"),
          ("DSv41F", "DeepSeek~V4.1~Flash"), ("Gemini38F", "Gemini~3.8~Flash"), ("heuristicv2", "HeuristicRoaster")]


def wavg(pairs) -> float:
    q = sum(b for _, b in pairs)
    return sum(a * b for a, b in pairs) / q if q else float("nan")


def main() -> None:
    lines = []
    for m, label in MODELS:
        agg, cnt = defaultdict(list), Counter()
        for f in glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*/run.json")):
            mk = json.load(open(f))["marketplace"]
            lst = {x["id"]: x for x in mk["listings"]}
            for x in mk["listings"]:
                if x["seller_id"].startswith("farmer") and x["item_id"] == "green_coffee_kg":
                    agg["farm_ask"].append((x["asking_price"], 1))
                    if x["posted_at"] // 1440 < 10:
                        agg["farm_ask0"].append((x["asking_price"], 1))
            for x in mk["deals"]:
                if x.get("cancelled_day") is not None:
                    continue
                ask = lst.get(x["listing_id"], {}).get("asking_price")
                if x["buyer_id"] == "roaster_A" and x["item_id"] == "green_coffee_kg":
                    agg["paid"].append((x["unit_price"], x["qty"]))
                    if ask:
                        agg["ask_b"].append((ask, x["qty"]))
                if x["seller_id"] == "roaster_A" and x["item_id"] == "roasted_coffee_kg" and ask:
                    agg["sell_ask"].append((ask, x["qty"]))
            for o in mk["offers"]:
                li = lst.get(o["listing_id"], {})
                if o["buyer_id"] == "roaster_A" and li.get("item_id") == "green_coffee_kg" and li.get("asking_price"):
                    agg["gap"].append((100 * (1 - o["offered_price"] / li["asking_price"]), 1))
                    cnt["off"] += 1
                    cnt["acc"] += o["status"] == "accepted"
                if o["seller_id"] == "roaster_A" and li.get("item_id") == "roasted_coffee_kg" and li.get("asking_price"):
                    agg["recv"].append((100 * o["offered_price"] / li["asking_price"], 1))
        v = dict(paid=wavg(agg["paid"]), ask_b=wavg(agg["ask_b"]), gap=wavg(agg["gap"]),
                 acc=100 * cnt["acc"] / max(1, cnt["off"]), fa0=wavg(agg["farm_ask0"]), fa=wavg(agg["farm_ask"]),
                 sask=wavg(agg["sell_ask"]), recv=wavg(agg["recv"]))
        print(f"{m:12s} " + " ".join(f"{k} {x:6.1f}" for k, x in v.items()))
        if m == "heuristicv2":
            lines.append("\\midrule")
        lines.append(f"{label} & {v['paid']:.1f} & {v['ask_b']:.1f} & {v['gap']:.0f} & {v['acc']:.0f} & "
                     f"{v['fa0']:.1f} & {v['fa']:.1f} & {v['sask']:.1f} & {v['recv']:.0f} \\\\")
    out = ROOT / "paper" / "tables_cr" / "tactics.tex"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
