"""Render clear matplotlib dashboards from CoffeeBench runs and checkpoints.

Two modes:

1. Single-run dashboard (an 8-panel overview of every agent's metrics over time):

       uv run python -m coffeebench.plot_run --run trajectories/.../run.json
       uv run python -m coffeebench.plot_run --run trajectories/.../day45.checkpoint.json

   Works on BOTH a finished `run.json` and a `*.checkpoint.json` — both carry the
   per-day `daily_stats` this script plots, so you can visualize a run at any
   stage (e.g. the state captured at the day-45 checkpoint).

2. Compare mode (overlay several runs/branches/stages, one panel per agent):

       uv run python -m coffeebench.plot_run --compare \
           baseline=trajectories/.../resumed.json \
           branch_a=trajectories/.../branch_a.json \
           gpt=trajectories/.../gpt_variant.json \
           --mark-day 45

   Use it to see how branches diverge after a checkpoint fork (draw a vertical
   line at the fork with `--mark-day`).

Output: PNG(s) under `--out` (default `plots/`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

try:
    from coffeebench.environment import FESTIVALS
except Exception:  # pragma: no cover — keep the plotter importable standalone
    FESTIVALS = [(40, 53, 3.0, "spring_break")]

# Consistent colors: one hue per role, A = solid, B = dashed.
ROLE_COLOR = {"farmer": "#2ca02c", "roaster": "#d17a22", "retailer": "#1f77b4"}
AGENT_ORDER = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]


def _dollars(x, _pos=None) -> str:
    if abs(x) < 0.5:
        return "$0"
    neg = x < 0
    ax = abs(x)
    if ax >= 1000:
        s = f"${ax / 1000:.1f}k".replace(".0k", "k")
    else:
        s = f"${ax:.0f}"
    return ("-" + s) if neg else s


_DOLLAR_FMT = FuncFormatter(_dollars)


def _agent_style(aid: str) -> dict:
    role = aid.split("_")[0]
    color = ROLE_COLOR.get(role, "#555555")
    dashed = aid.endswith("_B")
    return {"color": color, "linestyle": "--" if dashed else "-", "linewidth": 1.9}


# ---------------------------------------------------------------------------
# Loading / normalization (handles both run.json and checkpoint.json)
# ---------------------------------------------------------------------------
def load_run(path: str, label: str | None = None) -> dict:
    data = json.loads(Path(path).read_text())
    daily = data.get("daily_stats") or []
    if not daily:
        raise SystemExit(f"{path}: no daily_stats to plot (empty or too-early stop).")

    # Discover agents + roles from the snapshots.
    agents_seen: dict[str, str] = {}
    for snap in daily:
        for aid, per in (snap.get("per_agent") or {}).items():
            agents_seen.setdefault(aid, per.get("role", aid.split("_")[0]))
    order = [a for a in AGENT_ORDER if a in agents_seen] + [
        a for a in agents_seen if a not in AGENT_ORDER
    ]

    days = [int(s.get("day", i)) for i, s in enumerate(daily)]

    def series(aid, key):
        return [float((s.get("per_agent") or {}).get(aid, {}).get(key, 0.0) or 0.0) for s in daily]

    # initial equity: prefer explicit fields, else first snapshot's equity.
    def initial_equity(aid):
        res = (data.get("result") or {}).get("agents", {})
        if aid in res and res[aid].get("true_initial_equity") is not None:
            return float(res[aid]["true_initial_equity"])
        ck = (data.get("agents") or {})
        if aid in ck and ck[aid].get("initial_equity") is not None:
            return float(ck[aid]["initial_equity"])
        eq = series(aid, "true_equity")
        return eq[0] if eq else 15000.0

    agents = {}
    for aid in order:
        eq = series(aid, "true_equity")
        init = initial_equity(aid)
        spoil = series(aid, "spoiled_value_today")
        spoil_cum = []
        run = 0.0
        for v in spoil:
            run += v
            spoil_cum.append(run)
        agents[aid] = {
            "role": agents_seen[aid],
            "equity": eq,
            "cash": series(aid, "cash"),
            "inv_value": series(aid, "inventory_value"),
            "ar": series(aid, "ar_total"),
            "ap": series(aid, "ap_total"),
            "spoil_cum": spoil_cum,
            "initial_equity": init,
            "ni_series": [v - init for v in eq],
            "final_ni": (eq[-1] - init) if eq else 0.0,
            "bankrupt_day": (data.get("bankrupt_day") or {}).get(aid),
        }

    # consumer sales per day (from the log, aggregated).
    consumer_by_day: dict[int, float] = {}
    consumer_by_shop_day: dict[str, dict[int, float]] = {}
    for s in data.get("consumer_sales_log") or []:
        d = int(s.get("day", -1))
        rev = float(s.get("total_price", 0.0) or 0.0)
        consumer_by_day[d] = consumer_by_day.get(d, 0.0) + rev
        shop = s.get("shop_id")
        consumer_by_shop_day.setdefault(shop, {})
        consumer_by_shop_day[shop][d] = consumer_by_shop_day[shop].get(d, 0.0) + rev

    return {
        "label": label or Path(path).stem,
        "path": path,
        "days": days,
        "agents": agents,
        "order": order,
        "consumer_by_day": consumer_by_day,
        "consumer_by_shop_day": consumer_by_shop_day,
        "max_days": data.get("max_days"),
        "final_day": data.get("final_day") or data.get("next_day"),
        "is_checkpoint": "next_day" in data and "result" not in data,
    }


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------
def _shade_festivals(ax, xmax: int) -> None:
    for start, end, _mult, name in FESTIVALS:
        if start > xmax:
            continue
        ax.axvspan(start, min(end, xmax), color="#ffd966", alpha=0.18, zorder=0)
        ax.text(
            (start + min(end, xmax)) / 2,
            0.97,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#8a6d00",
        )


def _mark_bankruptcies(ax, run: dict) -> None:
    for aid, a in run["agents"].items():
        bd = a.get("bankrupt_day")
        if bd is not None:
            ax.axvline(bd, color=_agent_style(aid)["color"], alpha=0.35, ls=":", lw=1)


def _line_panel(ax, run, key, title, ylabel, dollar=True):
    days = run["days"]
    for aid in run["order"]:
        ax.plot(days, run["agents"][aid][key], label=aid, **_agent_style(aid))
    _shade_festivals(ax, max(days) if days else 0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    if dollar:
        ax.yaxis.set_major_formatter(_DOLLAR_FMT)
    ax.axhline(0, color="#999", lw=0.8, alpha=0.6)


# ---------------------------------------------------------------------------
# Single-run dashboard
# ---------------------------------------------------------------------------
def plot_dashboard(run: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    days = run["days"]

    _line_panel(axes[0][0], run, "equity", "Net worth (true equity)", "equity")
    _line_panel(axes[0][1], run, "ni_series", "Cumulative net income (vs start)", "net income")
    _line_panel(axes[1][0], run, "cash", "Cash", "cash")
    _line_panel(axes[1][1], run, "inv_value", "Inventory value", "inventory $")

    # Consumer sales per day (total + per shop).
    axc = axes[2][0]
    if run["consumer_by_day"]:
        xs = sorted(run["consumer_by_day"])
        axc.bar(xs, [run["consumer_by_day"][d] for d in xs], color="#bbbbbb",
                width=0.9, label="market total", zorder=1)
        for shop, byday in sorted(run["consumer_by_shop_day"].items()):
            sx = sorted(byday)
            axc.plot(sx, [byday[d] for d in sx], **_agent_style(shop), label=shop)
    _shade_festivals(axc, max(days) if days else 0)
    axc.set_title("Consumer sales revenue / day", fontsize=11, fontweight="bold")
    axc.set_ylabel("revenue", fontsize=9)
    axc.yaxis.set_major_formatter(_DOLLAR_FMT)
    axc.grid(True, alpha=0.25)
    axc.legend(fontsize=7, ncol=2)

    _line_panel(axes[2][1], run, "spoil_cum", "Cumulative spoilage (waste)", "spoiled $")

    # Working capital: AR − AP.
    axw = axes[3][0]
    max_abs = 0.0
    for aid in run["order"]:
        a = run["agents"][aid]
        net = [ar - ap for ar, ap in zip(a["ar"], a["ap"])]
        max_abs = max(max_abs, max((abs(v) for v in net), default=0.0))
        axw.plot(days, net, label=aid, **_agent_style(aid))
    if max_abs < 1.0:
        # Scripted agents settle invoices promptly → AR ≈ AP. Give the panel a
        # readable range and say so, so the flat line isn't mistaken for missing
        # data (LLM branches on NET-30 terms will show real swings here).
        axw.set_ylim(-1, 1)
        axw.text(0.5, 0.5, "≈ $0 all run (prompt settlement)",
                 transform=axw.transAxes, ha="center", va="center",
                 fontsize=10, color="#888")
    _shade_festivals(axw, max(days) if days else 0)
    axw.axhline(0, color="#999", lw=0.8, alpha=0.6)
    axw.set_title("Working capital (receivables − payables)", fontsize=11, fontweight="bold")
    axw.set_ylabel("AR − AP", fontsize=9)
    axw.yaxis.set_major_formatter(_DOLLAR_FMT)
    axw.grid(True, alpha=0.25)

    # Final net income bar (summary).
    axb = axes[3][1]
    order = run["order"]
    vals = [run["agents"][a]["final_ni"] for a in order]
    colors = [_agent_style(a)["color"] for a in order]
    bars = axb.barh(range(len(order)), vals, color=colors)
    for bar, a in zip(bars, order):
        bar.set_alpha(0.6 if a.endswith("_B") else 1.0)
    axb.set_yticks(range(len(order)))
    axb.set_yticklabels(order, fontsize=9)
    axb.invert_yaxis()
    axb.axvline(0, color="#333", lw=1)
    axb.xaxis.set_major_formatter(_DOLLAR_FMT)
    axb.set_title("Final net income by agent", fontsize=11, fontweight="bold")
    axb.grid(True, axis="x", alpha=0.25)
    for i, (b, v) in enumerate(zip(bars, vals)):
        axb.text(v, i, f" {_dollars(v)}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8)

    for row in axes[2:]:
        for ax in row:
            ax.set_xlabel("day", fontsize=9)

    # One shared agent legend at the top.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.995))

    kind = "checkpoint" if run["is_checkpoint"] else "run"
    fig.suptitle(
        f"CoffeeBench {kind}: {run['label']}   "
        f"(through day {run['final_day']}"
        + (f" / {run['max_days']}" if run["max_days"] else "")
        + ")",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Compare mode (overlay several runs, one panel per agent)
# ---------------------------------------------------------------------------
def plot_compare(runs: list[dict], out_path: Path, metric: str, mark_day: int | None) -> None:
    # Union of agents across the runs, in canonical order.
    all_agents: list[str] = []
    for r in runs:
        for a in r["order"]:
            if a not in all_agents:
                all_agents.append(a)
    all_agents = [a for a in AGENT_ORDER if a in all_agents] + [
        a for a in all_agents if a not in AGENT_ORDER
    ]

    run_colors = plt.cm.tab10.colors
    n = len(all_agents)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4.2 * rows), squeeze=False)

    metric_title = {
        "equity": "Net worth (true equity)",
        "ni_series": "Cumulative net income",
        "cash": "Cash",
    }.get(metric, metric)

    for idx, aid in enumerate(all_agents):
        ax = axes[idx // 2][idx % 2]
        xmax = 0
        for ri, r in enumerate(runs):
            if aid not in r["agents"]:
                continue
            ax.plot(r["days"], r["agents"][aid][metric],
                    label=r["label"], color=run_colors[ri % len(run_colors)], lw=2)
            xmax = max(xmax, max(r["days"]) if r["days"] else 0)
        _shade_festivals(ax, xmax)
        if mark_day is not None:
            ax.axvline(mark_day, color="#333", ls="--", lw=1.2, alpha=0.7)
            ax.text(mark_day, 0.02, f" fork@{mark_day}", transform=ax.get_xaxis_transform(),
                    fontsize=7, color="#333", va="bottom")
        ax.axhline(0, color="#999", lw=0.8, alpha=0.6)
        ax.set_title(aid, fontsize=11, fontweight="bold")
        ax.set_xlabel("day", fontsize=9)
        ax.yaxis.set_major_formatter(_DOLLAR_FMT)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    # hide any unused axis
    for j in range(n, rows * 2):
        axes[j // 2][j % 2].axis("off")

    fig.suptitle(f"CoffeeBench compare — {metric_title}", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="Path to a run.json OR a *.checkpoint.json for a single dashboard.")
    p.add_argument("--compare", nargs="+", metavar="LABEL=PATH",
                   help="Overlay multiple runs, e.g. baseline=a.json branch=b.json")
    p.add_argument("--metric", default="equity", choices=["equity", "ni_series", "cash"],
                   help="Metric to overlay in --compare mode (default: equity).")
    p.add_argument("--mark-day", type=int, default=None,
                   help="Draw a vertical line at this day (e.g. the checkpoint/fork day).")
    p.add_argument("--out", default="plots", help="Output directory (default: plots/).")
    args = p.parse_args()

    out_dir = Path(args.out)

    if args.run:
        run = load_run(args.run)
        plot_dashboard(run, out_dir / f"dashboard_{run['label']}.png")
        return

    if args.compare:
        runs = []
        for spec in args.compare:
            if "=" in spec:
                label, path = spec.split("=", 1)
            else:
                label, path = Path(spec).stem, spec
            runs.append(load_run(path, label=label))
        name = "_vs_".join(r["label"] for r in runs)[:60]
        plot_compare(runs, out_dir / f"compare_{args.metric}_{name}.png",
                     metric=args.metric, mark_day=args.mark_day)
        return

    p.error("pass either --run PATH or --compare LABEL=PATH ...")


if __name__ == "__main__":
    main()
