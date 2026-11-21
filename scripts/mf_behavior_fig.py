"""Behaviour figure for the main result (mini-swe-agent, 90 days, background Claude Sonnet 5.5, highest effort).

    uv run python scripts/mf_behavior_fig.py

One panel (models in main-table order); the CSV also holds actions by function and messages by recipient:
  (a) composition of the evaluated agent's shell commands (LLM-classified; results/mf_shell_llm.csv from
      scripts/mf_shell_classify.py), % of commands;
  (b) simulation actions (`coffee` tools) per run by function, from run.events.jsonl agent_step events;
  (messages by recipient: the network figure of scripts/mf_detail_figs.py; still written to the CSV)
Writes paper/images_cr/mf_behavior.{pdf,png} and results/mf_behavior.csv (the table view of the figure).
"""

from __future__ import annotations

import csv
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
MODELS = [("Sonnet55", "Claude Sonnet 5.5"), ("Astra", "GPT-6 Astra"), ("Opus55", "Claude Opus 5.5"),
          ("Haiku55", "Claude Haiku 5.5"), ("K31P", "Kimi K3.1 Pro"), ("GLM53", "GLM-5.3"),
          ("DSv41F", "DeepSeek V4.1 Flash"), ("Gemini38F", "Gemini 3.8 Flash")]
# validated categorical slots (dataviz reference palette, light): blue, orange, aqua, yellow, magenta, green
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SHELL = [("coffee", ["coffee_cli"]), ("analysis scripts", ["analysis"]), ("automation scripts", ["automation"]),
         ("memos", ["memo_write", "memo_read"]), ("other", ["file_management", "environment_probe", "other"])]
TOOLS = [("trade", {"post_listing", "make_offer", "accept_offer", "reject_offer", "cancel_listing", "withdraw_offer"}),
         ("messages", {"send_message", "view_messages", "read_message"}),
         ("market & state views", {"view_dashboard", "view_listings", "view_offers", "view_deals", "view_inventory",
                                   "view_market_aggregate", "view_trial_balance"}),
         ("invoices & returns", {"pay_invoice", "view_payables", "view_receivables", "return_shipment"}),
         ("roasting & capex", {"roast", "buy_roaster_upgrade"})]
RECIP = [("farmers", ("farmer_A", "farmer_B")), ("retailers", ("retailer_A", "retailer_B")), ("rival roaster", ("roaster_B",))]


def runs(m: str) -> list[Path]:
    return [Path(d) for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*")))
            if Path(d, "run.json").exists()]


def tool_counts(d: Path) -> Counter:
    c: Counter = Counter()
    for line in open(d / "run.events.jsonl"):
        if '"agent_step"' not in line or '"roaster_A"' not in line:
            continue
        e = json.loads(line)
        if e.get("type") == "agent_step" and e.get("agent_id") == "roaster_A":
            c[e.get("action")] += 1
    return c


def main() -> None:
    shell = {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / "mf_shell_llm.csv"))}
    rows = []
    for m, label in MODELS:
        s = shell[m]
        tot = float(s["cmds"])
        a = [100 * sum(float(s[k]) for k in ks) / tot for _, ks in SHELL]
        tcs = [tool_counts(d) for d in runs(m)]
        b = [st.mean(sum(tc[t] for t in ts) for tc in tcs) for _, ts in TOOLS]
        msgs = []
        for d in runs(m):
            ms = [x for x in json.load(open(d / "run.json"))["marketplace"]["messages"] if x["sender_id"] == "roaster_A"]
            msgs.append([sum(x["recipient_id"] in ids for x in ms) for _, ids in RECIP])
        c = [st.mean(v) for v in zip(*msgs)]
        rows.append((label, a, b, c))
    with open(ROOT / "results" / "mf_behavior.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"shell_%_{n}" for n, _ in SHELL] + [f"actions_per_run_{n}" for n, _ in TOOLS]
                   + [f"messages_per_run_to_{n}" for n, _ in RECIP])
        for label, a, b, c in rows:
            w.writerow([label] + [round(x, 1) for x in a + b + c])
            print(f"{label:20s} shell% {[round(x) for x in a]} actions {[round(x) for x in b]} msgs {[round(x) for x in c]}")

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 1, figsize=(5.5, 2.5))  # printed at \\linewidth: text ~ caption size
    axes = [axes]
    y = list(range(len(rows)))[::-1]
    # the rival roaster never receives a message from Roaster A in these runs (0 in every run): stated in the caption
    assert all(r[3][2] == 0 for r in rows)
    # messages by recipient are shown in a separate network figure (scripts/mf_detail_figs.py)
    # actions by function are similar across models and the per-recipient messages are shown in the appendix
    # (scripts/mf_detail_figs.py); both stay in the CSV
    panels = [(SHELL, 1, "Shell commands of the evaluated agent (% of commands)")]
    for ax, (cats, k, title) in zip(axes, panels):
        left = [0.0] * len(rows)
        for i, (name, _) in enumerate(cats):
            v = [r[k][i] for r in rows]
            ax.barh(y, v, left=left, height=0.62, color=C[i], edgecolor=SURF, linewidth=1.0, label=name)
            left = [p + q for p, q in zip(left, v)]
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
                  handlelength=1.0, columnspacing=0.8)
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks(y, [r[0] for r in rows])
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_behavior.pdf", bbox_inches="tight")
    # half-width version for a side-by-side layout (2.75 in)
    fig2, ax2 = plt.subplots(figsize=(2.75, 2.7))
    left = [0.0] * len(rows)
    for i, (name, _) in enumerate(SHELL):
        v = [r[1][i] for r in rows]
        ax2.barh(y, v, left=left, height=0.62, color=C[i], edgecolor=SURF, linewidth=0.8,
                 label="coffee command" if name == "coffee" else name)
        left = [p + q for p, q in zip(left, v)]
    ax2.set_xlim(0, 100)
    ax2.set_yticks(y, [r[0] for r in rows], fontsize=6)
    ax2.tick_params(axis="x", labelsize=6)
    ax2.set_xlabel("Share of shell commands (%)", fontsize=6.5, color=INK)
    ax2.grid(axis="x", color=GRID, linewidth=0.5)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig2.legend(loc="lower center", ncol=3, fontsize=5.6, frameon=False, handlelength=1.0, columnspacing=0.8,
                handletextpad=0.4, bbox_to_anchor=(0.5, 0.045), handles=[ax2.get_legend_handles_labels()[0][i] for i in (0, 3, 1, 4, 2)])
    # legend columns fill top-down: this order reads row by row in bar order (coffee, analysis, automation / memos,
    # other)
    fig2.tight_layout(rect=(0, 0.16, 1, 1))  # same plot area as mf_cost_ni_small beside it
    fig2.savefig(ROOT / "paper" / "images_cr" / "mf_behavior_small.pdf")
    fig2.savefig(ROOT / "paper" / "images_cr" / "mf_behavior_small.png", dpi=300)
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_behavior.png", dpi=200, bbox_inches="tight")
    print("wrote paper/images_cr/mf_behavior.{pdf,png}")


if __name__ == "__main__":
    main()
