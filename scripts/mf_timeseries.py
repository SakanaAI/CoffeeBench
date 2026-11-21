"""Daily trajectories of Roaster A in the main runs (mini-swe-agent, 90 days, background Claude Sonnet 5.5, highest
effort), in the style of the submission's Figure 3.

    uv run python scripts/mf_timeseries.py

Three panels in the paper (net income, revenue, gross margin; the other three are printed), one line per model (mean over three runs; same colours as the cost-score figure):
  cumulative net income (true equity - initial equity, before the end-of-run valuation), cumulative revenue
  (sum of daily sale_revenue), cumulative gross margin ((revenue - cost of goods sold) / revenue), the rival roaster's
  cumulative net income, and the quantity-weighted price of roasted commodity coffee sold and of green commodity
  coffee bought (deals in run.json, pooled over runs, trailing 7-day window). Roasters never let stock expire in
  these runs (spoilage 0), so spoilage is not shown.
Source: day_end snapshots and agent_step events in run.events.jsonl. Writes paper/images_cr/mf_timeseries.{pdf,png}.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
MODELS = [("Sonnet55", "Claude Sonnet 5.5", "#1839dc"), ("Astra", "GPT-6 Astra", "#e36278"),
          ("Opus55", "Claude Opus 5.5", "#15933f"), ("Haiku55", "Claude Haiku 5.5", "#dc9b18"),
          ("K31P", "Kimi K3.1 Pro", "#bc0f9f"), ("GLM53", "GLM-5.3", "#933f15"),
          ("DSv41F", "DeepSeek V4.1 Flash", "#18bbdc"), ("Gemini38F", "Gemini 3.8 Flash", "#6278e3")]
BASE = [("heuristicv2", "HeuristicRoaster", (0, (5, 3))), ("passive", "PassiveRoaster", (0, (1, 2)))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DAYS = 90
PANELS = [("ni", "Cumulative net income ($)"), ("rev", "Cumulative revenue ($)"), ("gm", "Cumulative gross margin (%)"),
          ("rival", "Rival roaster's net income ($)"), ("sell", "Roasted coffee sold, $/kg (7-day)"),
          ("buy", "Green coffee bought, $/kg (7-day)")]
SHOWN = PANELS[:3]  # the paper shows net income, revenue and gross margin (rival NI and prices are printed only)
PRICE = {"sell": ("seller_id", "roasted_coffee_kg"), "buy": ("buyer_id", "green_coffee_kg")}


def series(d: Path) -> dict[str, list[float]]:
    out = {k: [float("nan")] * DAYS for k, _ in PANELS}
    full = json.load(open(d / "run.json"))
    run = full["result"]
    for k, (side, item) in PRICE.items():  # daily quantity-weighted price sums (price*qty, qty), combined later
        pq, q = [0.0] * DAYS, [0.0] * DAYS
        for x in full["marketplace"]["deals"]:
            if x.get("cancelled_day") is None and x[side] == "roaster_A" and x["item_id"] == item:
                t = min(DAYS - 1, int(x["deal_at"]) // 1440)
                pq[t] += x["unit_price"] * x["qty"]
                q[t] += x["qty"]
        out[k] = (pq, q)
    eq0 = run["agents"]["roaster_A"]["true_initial_equity"]
    eq0b = run["agents"]["roaster_B"]["true_initial_equity"]
    rev = cogs = 0.0
    for line in open(d / "run.events.jsonl"):
        if '"day_end"' in line:
            e = json.loads(line)
            if e.get("type") != "day_end" or e["day"] >= DAYS:
                continue
            pa = e["snapshot"]["per_agent"]
            ra, rb = pa["roaster_A"], pa.get("roaster_B")
            tt = ra.get("truth_today") or {}
            rev += tt.get("sale_revenue", 0.0)
            cogs += tt.get("inventory_out", 0.0)  # cost of the lots that left with the sales
            t = e["day"]
            out["ni"][t] = ra["true_equity"] - eq0
            out["rev"][t] = rev
            out["gm"][t] = 100 * (rev - cogs) / rev if rev > 0 else float("nan")
            if rb is not None:
                out["rival"][t] = rb["true_equity"] - eq0b
    # a run that stopped early (the evaluated firm went bankrupt) keeps its final cumulative values
    for k in ("ni", "rev", "gm", "rival"):
        for t in range(1, DAYS):
            if out[k][t] != out[k][t]:
                out[k][t] = out[k][t - 1]
    if run.get("terminated_early"):  # bankruptcy write-offs are booked at the stop: use the final net income from then on
        stop = int(run["terminated_early"]["day"])
        final = run["agents"]["roaster_A"]["audit"]["annual"]["true_net_income"]
        for t in range(stop, DAYS):
            out["ni"][t] = final
    return out


def mean_series(m: str) -> dict[str, list[float]] | None:
    runs = [series(Path(d)) for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*")))
            if Path(d, "run.json").exists()]
    if not runs:
        return None
    res = {}
    for k, _ in PANELS:
        if k in PRICE:  # pooled over runs, trailing 7-day window, quantity-weighted
            pq = [sum(r[k][0][t] for r in runs) for t in range(DAYS)]
            q = [sum(r[k][1][t] for r in runs) for t in range(DAYS)]
            res[k] = []
            for t in range(DAYS):
                a, b = sum(pq[max(0, t - 6): t + 1]), sum(q[max(0, t - 6): t + 1])
                res[k].append(a / b if b > 0 else float("nan"))
            continue
        res[k] = [st.mean([r[k][t] for r in runs if r[k][t] == r[k][t]] or [float("nan")]) for t in range(DAYS)]
    return res


def main() -> None:
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), sharex=True)
    data = {m: mean_series(m) for m, _, _ in MODELS}
    base = {m: mean_series(m) for m, _, _ in BASE}
    for ax, (k, title) in zip(axes.flat, SHOWN):
        for m, label, ls in BASE:
            if base[m] and k in ("ni", "rev", "gm", "rival", "sell", "buy"):
                ax.plot(range(DAYS), base[m][k], color=INK2, linestyle=ls, linewidth=0.9, label=label)
        for m, label, col in MODELS:
            if data[m]:
                ax.plot(range(DAYS), data[m][k], color=col, linewidth=1.0, label=label)
        ax.axvspan(40, 53, color="#f0efec", zorder=0)  # festival
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.grid(color=GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.5)
    for ax in axes:
        ax.set_xlabel("Day", fontsize=7.5, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    order = list(range(len(BASE), len(h))) + list(range(len(BASE)))
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=5, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_timeseries.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_timeseries.png", dpi=220, bbox_inches="tight")
    for m, label, _ in MODELS:
        print(f"{label:20s} NI {data[m]['ni'][-1]:8.0f} rev {data[m]['rev'][-1]:8.0f} GM {data[m]['gm'][-1]:5.1f} "
              f"rival {data[m]['rival'][-1]:7.0f} "
              f"sell {st.mean(v for v in data[m]['sell'] if v == v):5.1f} buy {st.mean(v for v in data[m]['buy'] if v == v):5.1f}")
    print("wrote paper/images_cr/mf_timeseries.{pdf,png}")


if __name__ == "__main__":
    main()
