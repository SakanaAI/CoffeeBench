"""Financial position of Roaster A over time and additional operating statistics (main 90-day runs, mini-SWE-agent,
background Claude Sonnet 5.5, highest effort).

    uv run python scripts/mf_finance.py

Figure paper/images_cr/mf_finance.{pdf,png}: cash, accounts receivable, accounts payable and inventory value (at cost)
of Roaster A from the day_end snapshots (mean over three runs; a run that stops early keeps its last values).
Table paper/tables_cr/finance.tex (pooled over three runs):
  specialty share   share of B2B sales revenue from roasted specialty coffee
  terms granted     quantity-weighted payment terms on Roaster A's sales (days)
  terms received    quantity-weighted payment terms on Roaster A's purchases (days)
  pre-festival kg   inventory on hand (kg) averaged over days 35-39
  top buyer         share of Roaster A's sales revenue from its largest buyer
Late fees are not tabulated: Roaster A pays none in these runs (interest_expense entries in the truth ledger are 0
for every model except Gemini 3.8 Flash, $0.29 per run). Note that audit.annual.true_interest_net is interest expense
minus interest revenue, so its negative values are net interest income on cash, not late fees.
"""

from __future__ import annotations

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
MODELS = [("Sonnet55", "Claude Sonnet 5.5", "Claude~Sonnet~5.5", "#1839dc"), ("Astra", "GPT-6 Astra", "GPT-6~Astra", "#e36278"),
          ("Opus55", "Claude Opus 5.5", "Claude~Opus~5.5", "#15933f"), ("Haiku55", "Claude Haiku 5.5", "Claude~Haiku~5.5", "#dc9b18"),
          ("K31P", "Kimi K3.1 Pro", "Kimi~K3.1~Pro", "#bc0f9f"), ("GLM53", "GLM-5.3", "GLM-5.3", "#933f15"),
          ("DSv41F", "DeepSeek V4.1 Flash", "DeepSeek~V4.1~Flash", "#18bbdc"),
          ("Gemini38F", "Gemini 3.8 Flash", "Gemini~3.8~Flash", "#6278e3")]
BASE = [("heuristicv2", "HeuristicRoaster", "HeuristicRoaster", (0, (5, 3)))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DAYS = 90
PANELS = [("cash", "Cash ($)"), ("ar_total", "Accounts receivable ($)"), ("ap_total", "Accounts payable ($)"),
          ("inventory_value", "Inventory at cost ($)")]


def runs(m: str) -> list[Path]:
    return [Path(d) for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_*")))
            if Path(d, "run.json").exists()]


def daily(d: Path) -> dict[str, list[float]]:
    out = {k: [float("nan")] * DAYS for k, _ in PANELS}
    out["kg"] = [float("nan")] * DAYS
    for line in open(d / "run.events.jsonl"):
        if '"day_end"' not in line:
            continue
        e = json.loads(line)
        if e.get("type") != "day_end" or e["day"] >= DAYS:
            continue
        ra = e["snapshot"]["per_agent"]["roaster_A"]
        for k, _ in PANELS:
            out[k][e["day"]] = ra.get(k, 0.0) or 0.0
        out["kg"][e["day"]] = sum((ra.get("inventory_by_item") or {}).values())
    for k in out:
        for t in range(1, DAYS):
            if out[k][t] != out[k][t]:
                out[k][t] = out[k][t - 1]
    return out


def stats(m: str) -> dict[str, float]:
    spec = tot = 0.0
    gt, rt, buyers = [], [], Counter()
    prefest = []
    for d in runs(m):
        mk = json.load(open(d / "run.json"))["marketplace"]
        for x in mk["deals"]:
            if x.get("cancelled_day") is not None:
                continue
            if x["seller_id"] == "roaster_A":
                tot += x["total_price"]
                spec += x["total_price"] if x["item_id"] == "roasted_specialty_kg" else 0.0
                gt.append((x["payment_terms_days"], x["qty"]))
                buyers[(d, x["buyer_id"])] += x["total_price"]
            if x["buyer_id"] == "roaster_A":
                rt.append((x["payment_terms_days"], x["qty"]))
        dd = daily(d)
        prefest.append(st.mean(dd["kg"][35:40]))
    top = []
    for d in runs(m):
        per = {b: v for (dd, b), v in buyers.items() if dd == d}
        if per:
            top.append(max(per.values()) / sum(per.values()))

    def w(p):
        return sum(a * b for a, b in p) / sum(b for _, b in p) if p else float("nan")
    return dict(spec=100 * spec / tot if tot else float("nan"), gt=w(gt), rt=w(rt),
                prefest=st.mean(prefest), top=100 * st.mean(top) if top else float("nan"))


def main() -> None:
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.9), sharex=True)
    series = {m: [daily(d) for d in runs(m)] for m, *_ in MODELS + BASE}
    for ax, (k, title) in zip(axes.flat, PANELS):
        for m, label, _, ls in BASE:
            v = [st.mean(r[k][t] for r in series[m]) for t in range(DAYS)]
            ax.plot(range(DAYS), v, color=INK2, linestyle=ls, linewidth=0.9, label=label)
        for m, label, _, col in MODELS:
            v = [st.mean(r[k][t] for r in series[m]) for t in range(DAYS)]
            ax.plot(range(DAYS), v, color=col, linewidth=1.0, label=label)
        ax.axvspan(40, 53, color="#f0efec", zorder=0)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.grid(color=GRID, linewidth=0.5)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Day", fontsize=7.5, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    order = list(range(len(BASE), len(h))) + list(range(len(BASE)))
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=5, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_finance.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "images_cr" / "mf_finance.png", dpi=220, bbox_inches="tight")
    lines = []
    for m, label, tex, _ in MODELS + [(b[0], b[1], b[2], None) for b in BASE]:
        v = stats(m)
        print(f"{label:20s} " + " ".join(f"{k} {x:7.1f}" for k, x in v.items()))
        if m == "heuristicv2":
            lines.append("\\midrule")
        lines.append(f"{tex} & {v['spec']:.0f} & {v['gt']:.0f} & {v['rt']:.0f} & {v['prefest']:.0f} & {v['top']:.0f} \\\\")
    (ROOT / "paper" / "tables_cr" / "finance.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/images_cr/mf_finance.{pdf,png} and paper/tables_cr/finance.tex")


if __name__ == "__main__":
    main()
