"""Cost–score plot for the v2.1 main result (90 days, background Sonnet 5.5, 3 seeds).

    uv run python scripts/v21_pareto.py

x = mean API cost of the evaluated (focal) agent per 90-day run (log scale); y = mean net income ± sd over
seeds (sd in results/v21_tables.md). Each point is the provider logo in a circle with the reasoning-effort setting written under it:
solid circle = highest effort (launch_sim.FOCAL_EFFORT), dashed circle = lowest effort (FOCAL_MIN_EFFORT,
runs tagged `<model>-min`; drawn only when those 90-day runs exist), the two joined by a line. The Pareto
frontier over all points (no other point is both cheaper and better) is drawn behind; rule-based baselines
(no API cost) are reference lines. Writes paper/images_cr/v21_cost_ni_pareto.{pdf,png} and prints the table.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGOS = ROOT / "paper" / "assets" / "logos"
CFG = "hb_roaster_none_E_llmrival_v21"
# reasoning-off runs saved under the -min tag before min = effort low (2026-10-26); never plotted as min
RENAMED = {("results/hb_v21b", "DSv41F-min")}
# key: (display name, logo file in paper/assets/logos (lobe-icons PNGs), highest-effort label, lowest-effort label)
PANEL = {"Sonnet55": ("Claude Sonnet 5.5", "claude.png", "max", "low"),
         "Opus55": ("Claude Opus 5.5", "claude.png", "max", "low"),
         "Fable5": ("Claude Fable 5", "claude.png", "max", "low"),
         "Astra": ("GPT-6 Astra", "gpt.png", "max", "low"),
         "Gemini38F": ("Gemini 3.8 Flash", "gemini.png", "high", "low"),
         "K31P": ("Kimi K3.1 Pro", "kimi.png", "max", "low"),
         "DSv41F": ("DeepSeek V4.1 Flash", "deepseek.png", "max", "low"),
         "GLM53": ("GLM-5.3", "glm.png", "max", "low")}
# rule baselines: (legend label, line style)
BASE = {"heuristicv2": ("rule-based heuristic (v2), no API cost", (0, (5, 3))),
        "passive": ("passive baseline, no API cost", (0, (1, 2)))}
R, L, A, B = ((12, 0), "left", "center"), ((-12, 0), "right", "center"), ((0, 11), "center", "bottom"), \
    ((0, -11), "center", "top")
# model name (at the highest-effort point): offset in points, ha, va
NAME_POS = {"DSv41F": ((0, 22), "center", "bottom"), "Gemini38F": L, "GLM53": R, "K31P": R, "Fable5": R,
            "Opus55": R, "Astra": L, "Sonnet55": R}
# effort labels: (model, lowest?) -> placement; default = under the marker
EFFORT_POS = {("K31P", False): L, ("Fable5", True): A, ("K31P", True): L, ("Gemini38F", True): L, ("Opus55", True): L,
              ("GLM53", True): A}
HALO = [pe.withStroke(linewidth=3, foreground="white")]  # keeps labels legible over lines
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def collect(model: str) -> tuple[list[float], list[float]]:
    seen, ni, cost = set(), [], []
    for root in ("results/hb_v21a", "results/hb_v21b", "results/hb_v21c"):
        if (root, model) in RENAMED:
            continue
        for d in sorted(glob.glob(str(ROOT / root / "PersonaSonnet55" / model / CFG / "seed_*"))):
            seed = d.rsplit("/", 1)[-1]
            if seed in seen or not Path(d, "run.json").exists():
                continue
            seen.add(seed)
            r = json.load(open(Path(d, "run.json")))
            a = r["result"]["agents"]["roaster_A"]
            ni.append(a["audit"]["annual"]["true_net_income"])
            cost.append((a.get("usage") or {}).get("cost", 0.0))
    return ni, cost


def marker(ax, key: str, x: float, y: float, dashed: bool) -> None:
    ax.scatter([x], [y], s=260, facecolor="white", edgecolor=INK2, linewidth=1.2,
               linestyle=(0, (2.5, 1.5)) if dashed else "solid", zorder=4)
    img = Image.open(LOGOS / PANEL[key][1]).convert("RGBA")
    img.thumbnail((80, 80))
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.135), (x, y), frameon=False, zorder=5))

def main() -> None:
    pts = []  # (key, effort label, dashed, cost, ni mean, ni sd, n)
    for m, (_, _, hi, lo) in PANEL.items():
        for tag, lab, dashed in ((m, hi, False), (f"{m}-min", lo, True)):
            ni, cost = collect(tag)
            if ni:
                pts.append((m, lab, dashed, st.mean(cost), st.mean(ni), st.pstdev(ni), len(ni)))
    base = {m: st.mean(collect(m)[0]) for m in BASE}
    pts.sort(key=lambda p: p[3])
    frontier, best = [], float("-inf")
    for p in pts:
        if p[4] > best:
            frontier.append(p)
            best = p[4]
    print(f"{'model':22s} {'effort':6s} {'focal $/run':>11s} {'NI mean':>9s} {'sd':>7s}  n  frontier")
    for p in pts:
        print(f"{PANEL[p[0]][0]:22s} {p[1]:6s} {p[3]:11.1f} {p[4]:9.0f} {p[5]:7.0f}  {p[6]}  "
              f"{'yes' if p in frontier else ''}")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 400)
    for m, (label, ls) in BASE.items():
        ax.axhline(base[m], color=INK2, linewidth=1.0, linestyle=ls, zorder=1, label=label)
    for m in PANEL:  # join each model's highest and lowest effort
        mp = [p for p in pts if p[0] == m]
        if len(mp) == 2:
            ax.plot([p[3] for p in mp], [p[4] for p in mp], color="#a3a19b", linewidth=1.2, zorder=2)
    ax.plot([p[3] for p in frontier], [p[4] for p in frontier], color=BLUE, linewidth=2, alpha=0.7, zorder=2.5,
            label="Pareto frontier")
    for p in pts:  # sd per point: see results/v21_tables.md (error bars cluttered the overlapping markers)
        m, lab, dashed, x, y = p[:5]
        marker(ax, m, x, y, dashed)
        off, ha, va = EFFORT_POS.get((m, dashed), B)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=7, color=INK2, style="italic", path_effects=HALO, zorder=6)
        if not dashed:
            off, ha, va = NAME_POS.get(m, R)
            ax.annotate(PANEL[m][0], (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=8, color=INK, path_effects=HALO, zorder=6)
    ax.scatter([], [], s=120, facecolor="white", edgecolor=INK2, label="highest effort")
    if any(p[2] for p in pts):
        ax.scatter([], [], s=120, facecolor="white", edgecolor=INK2, linestyle=(0, (2.5, 1.5)),
                   label="lowest effort")
    ax.set_xlabel("API cost of the evaluated agent per 90-day run (USD, log scale)", color=INK, fontsize=9)
    ax.set_ylabel("Net income over 90 days (USD)", color=INK, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v):,.0f}"))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Cost vs. score at the highest and lowest reasoning effort (mean of 3 seeds; background Sonnet 5.5)", fontsize=9.5, color=INK)
    fig.tight_layout()
    out = ROOT / "paper" / "images_cr" / "v21_cost_ni_pareto"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=200)
    print("wrote", f"{out}.pdf")


if __name__ == "__main__":
    main()
