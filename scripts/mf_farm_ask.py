"""Prices at which Roaster A buys green coffee and sells roasted coffee over the run (main 90-day runs,
mini-SWE-agent, background Claude Sonnet 5.5, highest effort).

    uv run python scripts/mf_farm_ask.py

Left: quantity-weighted price Roaster A paid for green_coffee_kg. Right: quantity-weighted price at which it sold
roasted_coffee_kg. Both commodity grade, trailing 7-day window, pooled over the three runs of the model running
Roaster A. Writes paper/images_cr/mf_farm_ask.{pdf,png}.
"""

from __future__ import annotations

import glob
import json
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
BASE = [("heuristicv2", "HeuristicRoaster", (0, (5, 3)))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DAYS, WIN = 90, 7


def series(m: str) -> tuple[list[float], list[float]]:
    """(price Roaster A pays for green coffee, price it sells roasted coffee at), commodity grade, $/kg, 7-day."""
    buy_s, buy_q = [0.0] * DAYS, [0.0] * DAYS
    sell_s, sell_q = [0.0] * DAYS, [0.0] * DAYS
    for f in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*/run.json"))):
        for x in json.load(open(f))["marketplace"]["deals"]:
            if x.get("cancelled_day") is not None:
                continue
            t = min(DAYS - 1, int(x["deal_at"]) // 1440)
            if x["buyer_id"] == "roaster_A" and x["item_id"] == "green_coffee_kg":
                buy_s[t] += x["unit_price"] * x["qty"]
                buy_q[t] += x["qty"]
            if x["seller_id"] == "roaster_A" and x["item_id"] == "roasted_coffee_kg":
                sell_s[t] += x["unit_price"] * x["qty"]
                sell_q[t] += x["qty"]

    def roll(s, n):
        out = []
        for t in range(DAYS):
            a, b = sum(s[max(0, t - WIN + 1): t + 1]), sum(n[max(0, t - WIN + 1): t + 1])
            out.append(a / b if b else float("nan"))
        return out
    return roll(buy_s, buy_q), roll(sell_s, sell_q)


def main() -> None:
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    for m, label, ls in BASE:
        ask, pay = series(m)
        for ax, v in zip(axes, (ask, pay)):
            ax.plot(range(DAYS), v, color=INK2, linestyle=ls, linewidth=0.9, label=label)
    for m, label, col in MODELS:
        ask, pay = series(m)
        lw = 1.6 if m == "Sonnet55" else 0.9
        for ax, v in zip(axes, (ask, pay)):
            ax.plot(range(DAYS), v, color=col, linewidth=lw, label=label, zorder=3 if m == "Sonnet55" else 2)
        print(f"{label:20s} buy d80-89 {sum(ask[80:]) / 10:5.2f}  sell d80-89 {sum(pay[80:]) / 10:5.2f}")
    for ax, title in zip(axes, ("Buying green coffee", "Selling roasted coffee")):
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlabel("Day", fontsize=7.5, color=INK)
        ax.grid(color=GRID, linewidth=0.5)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Price paid, $/kg", fontsize=7.5, color=INK)
    axes[1].set_ylabel("Price received, $/kg", fontsize=7.5, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    order = list(range(len(BASE), len(h))) + list(range(len(BASE)))
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=5, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out = ROOT / "paper" / "images_cr" / "mf_farm_ask"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=220)
    print("wrote", f"{out}.pdf")


if __name__ == "__main__":
    main()
