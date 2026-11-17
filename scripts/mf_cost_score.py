"""Cost–score plot for the camera-ready main result: mini-swe-agent action format, env v2.1, 90 days,
background Sonnet 5.5, 3 seeds (results/hb_mf1, results/hb_mf2).

    uv run python scripts/mf_cost_score.py

x = mean API cost of the evaluated (focal) agent per 90-day run (log scale); y = mean net income. Each model has
its own colour and marker shape; a line joins its lowest-effort point (runs tagged `<model>-min`) and its
highest-effort point (launch_sim.FOCAL_EFFORT), with the effort setting written next to each point and the model
name in the model's colour. No error bars (std over seeds: Table 2 / appendix). Rule-based baselines (no API cost) are
labelled reference lines. Drawn at its printed size (linewidth = 5.5 in); baseline labels sit outside the axes. Writes
paper/images_cr/mf_cost_ni.{pdf,png} and prints the table.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.ticker
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
ROSTER = "PersonaSonnet55-mswe"
ROOTS = ("results/hb_mf1", "results/hb_mf2")
# key: display name, colour (8 colours validated as distinct for all pairs: normal dE >= 17, protan/deutan >= 9), marker, highest / lowest effort
PANEL = {"Sonnet55": ("Claude Sonnet 5.5", "#1839dc", "o", "max", "low"),
         "Astra": ("GPT-6 Astra", "#e36278", "D", "max", "low"),
         "Opus55": ("Claude Opus 5.5", "#15933f", "s", "max", "low"),
         "Haiku55": ("Claude Haiku 5.5", "#dc9b18", "^", "max", "low"),
         "K31P": ("Kimi K3.1 Pro", "#bc0f9f", "P", "max", "low"),
         "GLM53": ("GLM-5.3", "#933f15", "v", "max", "low"),
         "DSv41F": ("DeepSeek V4.1 Flash", "#18bbdc", "X", "max", "low"),
         "Gemini38F": ("Gemini 3.8 Flash", "#6278e3", "*", "high", "low")}
BASE = {"heuristicv2": ("HeuristicRoaster", (0, (5, 3))), "passive": ("PassiveRoaster", (0, (1, 2)))}
# model name placement relative to the highest-effort point: offset (points), ha, va
NAME_POS = {"Sonnet55": ((7, 0), "left", "center"), "Astra": ((7, 0), "left", "center"),
            "Opus55": ((0, 7), "center", "bottom"), "Haiku55": ((-6, -3), "right", "top"),
            "K31P": ((7, -2), "left", "top"), "GLM53": ((4, -7), "left", "top"),
            "DSv41F": ((0, -8), "center", "top"), "Gemini38F": ((6, -4), "left", "top")}
# effort tag placement: (model, lowest?) -> offset, ha, va (default: below the point)
EFFORT_POS = {("Sonnet55", False): ((-6, 0), "right", "center"), ("Astra", False): ((0, -6), "center", "top"),
              ("Opus55", False): ((6, -2), "left", "top"), ("Haiku55", False): ((5, 3), "left", "bottom"),
              ("K31P", False): ((0, 6), "center", "bottom"), ("GLM53", False): ((-6, -2), "right", "top"),
              ("DSv41F", False): ((6, 0), "left", "center"), ("Gemini38F", False): ((-6, 0), "right", "center"),
              ("K31P", True): ((-6, 2), "right", "bottom"), ("Gemini38F", True): ((6, 0), "left", "center"),
              ("GLM53", True): ((-6, 0), "right", "center"), ("Astra", True): ((7, 0), "left", "center"),
              ("Opus55", True): ((6, 0), "left", "center"), ("Haiku55", True): ((6, 0), "left", "center"),
              ("Sonnet55", True): ((-6, 0), "right", "center"), ("DSv41F", True): ((-6, 0), "right", "center")}
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def collect(model: str) -> tuple[list[float], list[float]]:
    seen, ni, cost = set(), [], []
    for root in ROOTS:
        for d in sorted(glob.glob(str(ROOT / root / ROSTER / model / CFG / "seed_*"))):
            seed = d.rsplit("/", 1)[-1]
            if seed in seen or not Path(d, "run.json").exists():
                continue
            seen.add(seed)
            a = json.load(open(Path(d, "run.json")))["result"]["agents"]["roaster_A"]
            ni.append(a["audit"]["annual"]["true_net_income"])
            cost.append((a.get("usage") or {}).get("cost", 0.0))
    return ni, cost


def main() -> None:
    pts = {}  # (model, lowest?) -> (cost, ni mean, ni sd, cost sd)
    for m in PANEL:
        for tag, low in ((m, False), (f"{m}-min", True)):
            ni, cost = collect(tag)
            if ni:
                pts[m, low] = (st.mean(cost), st.mean(ni), st.pstdev(ni), st.pstdev(cost))
    base = {m: st.mean(collect(m)[0]) for m in BASE}
    print(f"{'model':22s} {'effort':6s} {'focal $/run':>11s} {'NI mean':>9s} {'sd':>7s}")
    for (m, low), p in sorted(pts.items(), key=lambda kv: kv[1][0]):
        print(f"{PANEL[m][0]:22s} {PANEL[m][4 if low else 3]:6s} {p[0]:11.1f} {p[1]:9.0f} {p[2]:7.0f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 400)
    for m, (label, ls) in BASE.items():
        ax.axhline(base[m], color=INK2, linewidth=0.8, linestyle=ls, zorder=1)
        ax.text(1.01, base[m], label, transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="center", fontsize=7, color=INK2, clip_on=False)  # outside the axes, at the line
    for m, (name, col, mk, hi, lo) in PANEL.items():
        if (m, True) in pts and (m, False) in pts:
            (x0, y0), (x1, y1) = pts[m, True][:2], pts[m, False][:2]
            ax.plot([x0, x1], [y0, y1], color=col, linewidth=1.2, alpha=0.8, zorder=2)  # low <-> highest effort
        for low in (True, False):
            if (m, low) not in pts:
                continue
            x, y = pts[m, low][:2]
            # no error bars (cluttered); the std over seeds is in Table 2 and the appendix effort table
            # same filled marker at both efforts (shape = model); the effort is annotated next to the point
            ax.scatter([x], [y], s=34 if mk != "*" else 60, marker=mk, color=col, edgecolor="white", linewidth=0.6,
                       zorder=4)
            off, ha, va = EFFORT_POS.get((m, low), ((0, -6), "center", "top"))
            ax.annotate(lo if low else hi, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=6.5, color=col, style="italic", path_effects=HALO, zorder=5)
            if not low:
                off, ha, va = NAME_POS[m]
                lead = False
                ax.annotate(name, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                            color=col, path_effects=HALO, zorder=5,
                            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.6, "shrinkB": 4} if lead else None)
    ax.set_xlabel("API Cost (USD, log scale)", color=INK, fontsize=8)
    ax.set_ylabel("Net Income", color=INK, fontsize=8)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v):,.0f}"))
    fig.tight_layout()
    out = ROOT / "paper" / "images_cr" / "mf_cost_ni"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=250)
    print("wrote", f"{out}.pdf")
    compact(pts, base)


def compact(pts, base) -> None:
    """Half-width version for a side-by-side layout (2.75 in). Highest effort = filled marker with a solid edge, low
    effort = open marker with a dotted edge; a model-name label is placed at the middle of each model's line where it
    does not overlap another label or marker (otherwise skipped); all models, baselines and the two effort styles are in
    the legend below."""
    fig, ax = plt.subplots(figsize=(2.75, 2.7))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 300)
    handles = []
    for m, (label, ls) in BASE.items():
        handles.append(ax.axhline(base[m], color=INK2, linewidth=0.7, linestyle=ls, zorder=1, label=label))
    model_h, markers = [], []
    for m, (name, col, mk, hi, lo) in PANEL.items():
        if (m, True) in pts and (m, False) in pts:
            (x0, y0), (x1, y1) = pts[m, True][:2], pts[m, False][:2]
            ax.plot([x0, x1], [y0, y1], color=col, linewidth=0.8, alpha=0.7, zorder=2)
        size = 26 if mk != "*" else 44
        for low in (True, False):
            if (m, low) in pts:
                x, y, sd, csd = pts[m, low]
                # std over seeds (net income and cost) as faint bars behind the marker
                ax.errorbar([x], [y], yerr=[sd], xerr=[csd], fmt="none", ecolor=col, elinewidth=0.6, alpha=0.35,
                            capsize=0, zorder=3)
                if low:
                    sc = ax.scatter([x], [y], s=size, marker=mk, facecolor="white", edgecolor=col, linewidth=0.9,
                                    linestyle=(0, (1, 1)), zorder=4)
                else:
                    sc = ax.scatter([x], [y], s=size, marker=mk, color=col, edgecolor=col, linewidth=0.6, zorder=4)
                markers.append(sc)
        model_h.append(ax.scatter([], [], s=22 if mk != "*" else 36, marker=mk, color=col, label=name))
    eff_h = [ax.scatter([], [], s=22, marker="o", color=INK2, edgecolor=INK2, linewidth=0.6, label="highest effort"),
             ax.scatter([], [], s=22, marker="o", facecolor="white", edgecolor=INK2, linewidth=0.9,
                        linestyle=(0, (1, 1)), label="low effort")]
    ax.set_xlabel("API Cost (USD, log scale)", color=INK, fontsize=6.5)
    ax.set_ylabel("Net Income", color=INK, fontsize=6.5)
    ax.tick_params(colors=INK2, labelsize=6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v) / 1000:g}K"))
    ax.legend(handles=eff_h, loc="upper left", fontsize=5.0, frameon=False, handletextpad=0.3, borderaxespad=0.3)
    fig.legend(handles=model_h + handles, loc="lower center", ncol=3, fontsize=5.0, frameon=False,
               handlelength=1.4, columnspacing=0.6, handletextpad=0.3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    # model names along each model's line, greedily, skipping any label that would overlap
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken = []
    for sc in markers:  # marker footprints (pixel boxes around each point)
        for xy in sc.get_offsets():
            px, py = ax.transData.transform(xy)
            r = 3.2 * fig.dpi / 72
            taken.append(matplotlib.transforms.Bbox([[px - r, py - r], [px + r, py + r]]))
    axbox = ax.get_window_extent(rend)
    taken.append(ax.get_legend().get_window_extent(rend))  # effort legend, upper left
    # labelled models (sparse regions first); Haiku, GLM and Kimi sit in the dense cluster around $20-60, where a label
    # cannot be placed unambiguously, and are identified by the legend only
    order = ["Sonnet55", "Astra", "Opus55", "DSv41F", "Gemini38F"]
    for m in order:
        if (m, False) not in pts:
            continue
        col = PANEL[m][1]
        p1 = ax.transData.transform(pts[m, False][:2])
        p0 = ax.transData.transform(pts[m, True][:2]) if (m, True) in pts else p1
        # positions along the line joining the two efforts (display space, log x), middle first
        cands = [(f, off, ha, va) for f in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8)
                 for off, ha, va in (((0, 0), "center", "center"), ((4, 0), "left", "center"),
                                     ((-4, 0), "right", "center"))]
        cands += [(f, off, ha, va) for f in (0.5, 0.3, 0.7) for off, ha, va in (((0, -6), "center", "top"),
                                                                                ((0, 6), "center", "bottom"))]
        cands += [(f, off, ha, va) for f in (0.5, 0.7, 0.3) for off, ha, va in (((-4, 4), "right", "bottom"),
                                                                                ((4, -4), "left", "top"))]
        # then diagonally off the highest-effort end of the line
        cands += [(1.0, off, ha, va) for off, ha, va in (((-3, 5), "right", "bottom"), ((3, 5), "left", "bottom"),
                                                         ((-3, -5), "right", "top"), ((3, -5), "left", "top"))]
        for f, off, ha, va in cands:
            x, y = ax.transData.inverted().transform(p0 + f * (p1 - p0))
            t = ax.annotate(PANEL[m][0], (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=5.3,
                            color=col, path_effects=HALO, zorder=5)
            bb = t.get_window_extent(rend)
            inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
            if inside and not any(bb.overlaps(o) for o in taken):
                taken.append(bb)
                break
            t.remove()
        else:
            print("skipped label:", PANEL[m][0])
    out = ROOT / "paper" / "images_cr" / "mf_cost_ni_small"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=400)
    print("wrote", f"{out}.pdf")



if __name__ == "__main__":
    main()
