"""Per-tool and per-recipient figures for the main result (mini-swe-agent, 90 days, background Claude Sonnet 5.5,
highest effort), in the style of the submission's Figures 4-5.

    uv run python scripts/mf_detail_figs.py

  paper/images_cr/mf_messages_network.pdf  one panel per model: Roaster A -> each other firm, edge width = mean
                                           messages per run, label = mean ± std (over three runs)
  paper/images_cr/mf_tool_calls.pdf        heatmap of mean `coffee` tool calls per run (tools grouped by function x models)
Counts come from run.events.jsonl agent_step events of roaster_A (send_message recipient from action_input).
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
MODELS = [("Sonnet55", "Claude Sonnet 5.5"), ("Astra", "GPT-6 Astra"), ("Opus55", "Claude Opus 5.5"),
          ("Haiku55", "Claude Haiku 5.5"), ("K31P", "Kimi K3.1 Pro"), ("GLM53", "GLM-5.3"),
          ("DSv41F", "DeepSeek V4.1 Flash"), ("Gemini38F", "Gemini 3.8 Flash")]
# 8-colour palette validated with all pairs distinct (normal vision dE >= 17, protan/deutan >= 9); model order above
COL = ["#1839dc", "#e36278", "#15933f", "#dc9b18", "#bc0f9f", "#933f15", "#18bbdc", "#6278e3"]
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
OTHERS = ["farmer_A", "farmer_B", "roaster_B", "retailer_A", "retailer_B"]
NICE = {"roaster_A": "Roaster A", "roaster_B": "Roaster B", "farmer_A": "Farmer A", "farmer_B": "Farmer B",
        "retailer_A": "Retailer A", "retailer_B": "Retailer B"}
CATS = [("Overview", ["view_dashboard"]),
        ("Marketplace", ["post_listing", "view_listings", "make_offer", "withdraw_offer", "accept_offer", "reject_offer",
                         "cancel_listing", "view_offers", "view_deals", "view_inventory"]),
        ("Production", ["roast", "buy_roaster_upgrade"]),
        ("Financial", ["view_payables", "view_receivables", "pay_invoice", "return_shipment", "view_trial_balance"]),
        ("Demand", ["view_market_aggregate"]),
        ("Communication", ["send_message", "view_messages", "read_message"]),
        ("Wait", ["wait_for_next_day"])]


def runs(m: str) -> list[Path]:
    return [Path(d) for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*")))
            if Path(d, "run.json").exists()]


def steps(d: Path):
    for line in open(d / "run.events.jsonl"):
        if '"agent_step"' in line and '"roaster_A"' in line:
            e = json.loads(line)
            if e.get("type") == "agent_step" and e.get("agent_id") == "roaster_A":
                yield e


def collect():
    tools, msgs = {}, {}
    for m, _ in MODELS:
        tc, mc = [], []
        for d in runs(m):
            t, r = Counter(), Counter()
            for e in steps(d):
                t[e.get("action")] += 1
                if e.get("action") == "send_message":
                    ai = e.get("action_input") or {}
                    if isinstance(ai, str):
                        try:
                            ai = json.loads(ai)
                        except json.JSONDecodeError:
                            ai = {}
                    r[ai.get("recipient", "?")] += 1
            tc.append(t)
            mc.append(r)
        tools[m], msgs[m] = tc, mc
    return tools, msgs


def network(msgs) -> None:
    def ms(v):
        return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)
    vmax = max(st.mean([c[o] for c in msgs[m]]) for m, _ in MODELS for o in OTHERS)
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 8.4))  # printed at \\linewidth: text ~ caption size
    for ax, (k, (m, label)) in zip(axes.flat, enumerate(MODELS)):
        ys = [2, 1, 0, -1, -2]
        tot_m, tot_s = ms([sum(c.values()) for c in msgs[m]])
        for o, y in zip(OTHERS, ys):
            mu, sd = ms([c[o] for c in msgs[m]])
            if mu > 0:
                ax.plot([0, 1], [0, y], color=COL[k], linewidth=0.6 + 6.0 * mu / vmax, alpha=0.85, zorder=1,
                        solid_capstyle="round")
                t = ax.annotate(f"{mu:.0f} ± {sd:.0f}", xy=(0.62, 0.62 * y), fontsize=7.5, ha="center", va="center",
                                color=INK, zorder=3)
                t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
            ax.plot(1, y, "o", ms=8, mfc="white", mec=INK, mew=0.9, zorder=2)
            ax.annotate(NICE[o], xy=(1, y), xytext=(7, 0), textcoords="offset points", fontsize=8, va="center",
                        color=INK)
        ax.plot(0, 0, "o", ms=8, mfc="white", mec=INK, mew=0.9, zorder=2)
        ax.annotate("Roaster A", xy=(0, 0), xytext=(-7, 0), textcoords="offset points", fontsize=8, ha="right",
                    va="center", color=INK)
        ax.set_xlim(-0.55, 1.6)
        ax.set_ylim(-2.4, 2.4)
        ax.set_title(f"{label} ({tot_m:.0f} ± {tot_s:.0f})", fontsize=9, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_messages_network.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_messages_network.png", dpi=200, bbox_inches="tight")


def tool_bars(tools) -> None:
    """Heatmap: tools (rows, grouped by function) x models (columns), cell = mean calls per run."""
    used = {t for m, _ in MODELS for c in tools[m] for t in c}
    rows = [(cat, t) for cat, ts in CATS for t in ts if t in used]
    data = [[st.mean(c[t] for c in tools[m]) for m, _ in MODELS] for _, t in rows]
    vmax = max(max(r) for r in data)
    fig, ax = plt.subplots(figsize=(5.5, 7.2))  # printed at \\linewidth: text ~ caption size
    ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(len(MODELS)), [lab.replace(" ", "\n", 1) for _, lab in MODELS], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [t for _, t in rows], fontsize=8, family="monospace")
    prev = None
    for i, (cat, _) in enumerate(rows):  # category separators and labels
        if cat != prev:
            if prev is not None:
                ax.axhline(i - 0.5, color="white", linewidth=2.5)
            n = sum(1 for c, _ in rows if c == cat)
            ax.text(len(MODELS) - 0.35, i - 0.5 + n / 2, cat, va="center", ha="left", fontsize=8,
                    color=INK2, style="italic")
            prev = cat
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_tool_calls.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_tool_calls.png", dpi=200, bbox_inches="tight")


def main() -> None:
    tools, msgs = collect()
    for m, label in MODELS:
        rec = {o: round(st.mean(c[o] for c in msgs[m]), 1) for o in OTHERS}
        print(f"{label:20s} msgs {rec}  wait {st.mean(c['wait_for_next_day'] for c in tools[m]):.0f}")
    network(msgs)
    tool_bars(tools)
    print("wrote paper/images_cr/mf_messages_network.{pdf,png}, mf_tool_calls.{pdf,png}")


if __name__ == "__main__":
    main()
