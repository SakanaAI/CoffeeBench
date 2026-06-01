"""Post-hoc viewer for a saved CoffeeBench trajectory.         # specific"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any


TRAJ_DIR = "trajectories"


# ---------- discovery ----------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("path", nargs="?", default=None)
    p.add_argument("--latest", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--messages", action="store_true", help="Dump every public+DM message in full."
    )
    return p.parse_args()


def _resolve(args: argparse.Namespace) -> str:
    if args.path:
        return args.path
    cands = sorted(
        glob.glob(os.path.join(TRAJ_DIR, "*.json"))
        + glob.glob(os.path.join(TRAJ_DIR, "*", "seed_*", "run.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    cands = [c for c in cands if not c.endswith(".events.jsonl")]
    if not cands:
        sys.exit(f"No trajectories under {TRAJ_DIR}/.")
    return cands[0]


# ---------- formatting helpers ----------


def _money(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)


def _hdr(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def _sub(s: str) -> None:
    print()
    print("-- " + s + " " + "-" * max(1, 72 - len(s)))


# ---------- text sections ----------


def _print_run_summary(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    _hdr(
        f"CoffeeBench run summary  "
        f"(max_days={traj.get('max_days')}, final_day={traj.get('final_day')})"
    )
    print(f"  models: {traj.get('models')}")
    if res.get("main_agent"):
        print(f"  main_agent: {res['main_agent']}")
    te = res.get("terminated_early")
    if te:
        print(f"  terminated_early: day {te.get('day')} reason={te.get('reason')}")
    _sub("Per-agent score (true net income)")
    print(
        f"  {'agent':<14} {'status':<32} {'true_NI':>12} {'true_rev':>12} {'true_eq':>12}"
    )
    for aid, ag in agents.items():
        audit = ag.get("audit") or {}
        ann = audit.get("annual") or {}
        bs = audit.get("balance_sheet") or {}
        if not ag.get("completed", True):
            status = (
                f"DNF@{ag.get('bankrupt_day')} ({ag.get('bankrupt_reason') or '?'})"
            )
        else:
            status = "completed"
        print(
            f"  {aid:<14} {status:<32} "
            f"{_money(ann.get('true_net_income')):>12} "
            f"{_money(ann.get('true_revenue')):>12} "
            f"{_money(bs.get('true_equity')):>12}"
        )


def _print_usage(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    _sub("API spend & token usage (per agent)")
    print(
        f"  {'agent':<14} {'model':<26} {'$cost':>9} {'calls':>6} "
        f"{'in_tok':>10} {'out_tok':>10}"
    )
    total_cost = 0.0
    for aid, ag in agents.items():
        u = ag.get("usage") or {}
        cost = u.get("cost") or 0.0
        total_cost += float(cost)
        print(
            f"  {aid:<14} {str(u.get('model') or '—'):<26} "
            f"{_money(cost):>9} "
            f"{int(u.get('n_calls') or 0):>6} "
            f"{int(u.get('total_input_tokens') or 0):>10,d} "
            f"{int(u.get('total_output_tokens') or 0):>10,d}"
        )
    print(f"  {'TOTAL':<14} {'':<26} {_money(total_cost):>9}")


def _print_message_volume(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    rows = []
    for aid, ag in agents.items():
        mv = (ag.get("audit") or {}).get("message_volume") or {}
        if mv:
            rows.append((aid, mv))
    if not rows:
        return
    _sub("Message volume (per agent)")
    print(f"  {'agent':<14} {'sent':>6} {'cap_hits':>9} {'max/day':>9}")
    for aid, mv in rows:
        print(
            f"  {aid:<14} "
            f"{int(mv.get('total_sent') or 0):>6} "
            f"{len(mv.get('cap_hit_days') or []):>9} "
            f"{int(mv.get('max_per_day') or 0):>9}"
        )


def _print_per_quarter(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    any_q = any((ag.get("audit") or {}).get("per_quarter") for ag in agents.values())
    if not any_q:
        return
    _sub("Per-quarter TRUE P&L (per agent)")
    print(
        f"  {'agent':<14} {'Q':>2}  {'window':>10}  "
        f"{'true_rev':>10} {'true_cogs':>10} {'true_opex':>10} "
        f"{'true_NI':>10}"
    )
    for aid, ag in agents.items():
        for q in (ag.get("audit") or {}).get("per_quarter", []):
            win = q.get("window") or [None, None]
            print(
                f"  {aid:<14} Q{q.get('quarter')}  {f'{win[0]}-{win[1]}':>10}  "
                f"{_money(q.get('true_revenue')):>10} "
                f"{_money(q.get('true_cogs')):>10} "
                f"{_money(q.get('true_opex')):>10} "
                f"{_money(q.get('true_net_income')):>10}"
            )


def _print_balance_sheets(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    _sub("Run-end balance sheet (per agent, env truth)")
    print(
        f"  {'agent':<14} "
        f"{'cash':>11} {'inventory':>11} {'AR':>11} {'AP':>11} {'equity':>11}"
    )
    for aid, ag in agents.items():
        bs = (ag.get("audit") or {}).get("balance_sheet") or {}
        print(
            f"  {aid:<14} "
            f"{_money(bs.get('true_cash')):>11} "
            f"{_money(bs.get('true_inventory_value')):>11} "
            f"{_money(bs.get('true_accounts_receivable')):>11} "
            f"{_money(bs.get('true_accounts_payable')):>11} "
            f"{_money(bs.get('true_equity')):>11}"
        )


def _print_roast_metrics(traj: dict) -> None:
    res = traj.get("result") or {}
    agents = res.get("agents") or {}
    rows = [
        (aid, (ag.get("audit") or {}).get("roast_metrics"))
        for aid, ag in agents.items()
        if (ag.get("audit") or {}).get("roast_metrics")
    ]
    if not rows:
        return
    _sub("Roaster yield audit")
    print(
        f"  {'agent':<14} {'green_kg':>9} {'roasted_kg':>11} "
        f"{'yield':>7} {'b2b_qty':>8} {'$/kg':>9} {'labor':>10}"
    )
    for aid, m in rows:
        ay = m.get("actual_yield")
        ycol = f"{ay:.3f}" if ay is not None else "—"
        cogs_per = m.get("truth_cogs_per_kg_roasted")
        ccol = f"${cogs_per:.2f}" if cogs_per is not None else "—"
        print(
            f"  {aid:<14} "
            f"{int(m.get('green_consumed_kg') or 0):>9} "
            f"{int(m.get('roasted_produced_kg') or 0):>11} "
            f"{ycol:>7} "
            f"{int(m.get('roasted_b2b_qty_sold') or 0):>8} "
            f"{ccol:>9} "
            f"{_money(m.get('roast_labor_paid')):>10}"
        )


def _print_marketplace(traj: dict) -> None:
    mp = traj.get("marketplace") or {}
    _sub("Marketplace activity")
    print(f"  items:    {len(mp.get('items') or [])}")
    print(f"  listings: {len(mp.get('listings') or [])}")
    print(f"  offers:   {len(mp.get('offers') or [])}")
    print(f"  deals:    {len(mp.get('deals') or [])}")
    print(f"  messages: {len(mp.get('messages') or [])}")


def _print_deals(traj: dict) -> None:
    deals = (traj.get("marketplace") or {}).get("deals") or []
    if not deals:
        return
    _sub(f"Deals ({len(deals)} total, showing first 30)")
    for d in deals[:30]:
        print(
            f"  day {d.get('deal_at', 0) // 1440:>3}  "
            f"{d.get('seller_id'):<14} → {d.get('buyer_id'):<14}  "
            f"{d.get('item_id'):<24} x{d.get('qty'):<6} "
            f"@{_money(d.get('unit_price'))}  "
            f"= {_money(d.get('total_price'))}"
        )


def _print_messages(traj: dict) -> None:
    msgs = (traj.get("marketplace") or {}).get("messages") or []
    if not msgs:
        return
    _hdr(f"All messages ({len(msgs)} total)")
    for m in msgs:
        day = (m.get("sent_at") or 0) // 1440
        recipient = m.get("recipient_id")
        title = m.get("title") or ""
        body = (m.get("body") or "").strip()
        print(f"  [day {day:>3}] {m.get('sender_id'):<14} → {recipient}  📧 {title}")
        if body:
            print(f"                  {body}")


# ---------- plots ----------


def _cumulative_truth_pl_per_day(traj: dict, agent_id: str) -> dict:
    """Build per-day cumulative true revenue / cogs / NI for one agent
    from its truth ledger entries."""
    truth = (traj.get("truth_ledger") or {}).get(agent_id) or []
    max_day = int(traj.get("max_days") or 0)
    if max_day == 0:
        max_day = max((e.get("day", 0) for e in truth), default=0)
    daily_rev = [0.0] * (max_day + 1)
    daily_returns = [0.0] * (max_day + 1)
    daily_cogs = [0.0] * (max_day + 1)
    daily_opex = [0.0] * (max_day + 1)
    for e in truth:
        d = int(e.get("day") or 0)
        if d > max_day:
            continue
        et = e.get("entry_type")
        amt = float(e.get("amount") or 0)
        if et == "sale_revenue":
            daily_rev[d] += amt
        elif et == "sale_reversal":
            daily_returns[d] += amt
        elif et == "inventory_out":
            daily_cogs[d] += amt
        elif et in (
            "operating_expense",
            "spoilage_expense",
            "bad_debt_expense",
            "writedown",
        ):
            daily_opex[d] += amt
    cum_rev: list[float] = []
    cum_net: list[float] = []
    r = ret = c = o = 0.0
    for i in range(max_day + 1):
        r += daily_rev[i]
        ret += daily_returns[i]
        c += daily_cogs[i]
        o += daily_opex[i]
        cum_rev.append(r - ret)
        cum_net.append((r - ret) - c - o)
    return {
        "days": list(range(max_day + 1)),
        "cum_rev": cum_rev,
        "cum_net": cum_net,
    }


def _save_plot(traj: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[show] matplotlib not available — skipping plot.")
        return
    daily = traj.get("daily_stats") or []
    if not daily:
        print("[show] no daily_stats — skipping plot.")
        return
    days = [d["day"] for d in daily]
    agents = list((daily[0].get("per_agent") or {}).keys())
    color_for = {aid: f"C{i}" for i, aid in enumerate(agents)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"CoffeeBench performance\nmodels={traj.get('models')}",
        fontsize=11,
    )

    # Cash
    ax = axes[0, 0]
    for aid in agents:
        cash = [d.get("per_agent", {}).get(aid, {}).get("cash") for d in daily]
        ax.plot(days, cash, label=aid, color=color_for[aid])
    ax.set_title("Cash")
    ax.set_xlabel("day")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # True equity
    ax = axes[0, 1]
    for aid in agents:
        eq = [d.get("per_agent", {}).get(aid, {}).get("true_equity") for d in daily]
        ax.plot(days, eq, label=aid, color=color_for[aid])
    ax.set_title("True equity")
    ax.set_xlabel("day")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Cumulative true revenue + NI
    pl = {aid: _cumulative_truth_pl_per_day(traj, aid) for aid in agents}

    ax = axes[1, 0]
    for aid in agents:
        s = pl[aid]
        ax.plot(s["days"], s["cum_rev"], label=aid, color=color_for[aid])
    ax.set_title("Cumulative true revenue (net of returns)")
    ax.set_xlabel("day")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for aid in agents:
        s = pl[aid]
        ax.plot(s["days"], s["cum_net"], label=aid, color=color_for[aid])
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("Cumulative true net income")
    ax.set_xlabel("day")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Quarter-end markers
    for qe in (89, 180, 271, 364):
        if qe <= max(days, default=0):
            for ax in axes.flat:
                ax.axvline(qe, color="grey", alpha=0.4, linestyle="--", linewidth=0.7)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = path.replace(".json", ".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[show] plot saved to {out}")


def _save_stats_plot(traj: dict, path: str, events: list[dict]) -> None:
    """Action distribution + marketplace activity in one figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    daily = traj.get("daily_stats") or []
    if not daily:
        return
    days = [d["day"] for d in daily]
    agents = list((daily[0].get("per_agent") or {}).keys())

    action_counts: dict[str, Counter] = defaultdict(Counter)
    for ev in events:
        if ev.get("type") == "agent_step":
            aid = ev.get("agent_id")
            act = ev.get("action")
            if aid and act:
                action_counts[aid][act] += 1
    overall = Counter()
    for c in action_counts.values():
        overall.update(c)
    action_names = [a for a, _ in overall.most_common()]

    deals = (traj.get("marketplace") or {}).get("deals") or []
    deals_per_day = Counter((d.get("deal_at", 0) // 1440) for d in deals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"CoffeeBench statistics\nmodels={traj.get('models')}", fontsize=11)

    # action heatmap
    ax = axes[0, 0]
    if agents and action_names:
        import numpy as np

        m = np.array(
            [[action_counts[a].get(name, 0) for name in action_names] for a in agents]
        )
        ax.imshow(m, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha="right", fontsize=7)
        ax.set_title("Action counts (per agent × action)")
        for i in range(len(agents)):
            for j in range(len(action_names)):
                v = m[i, j]
                if v > 0:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        color=("white" if v > m.max() / 2 else "black"),
                        fontsize=6,
                    )

    # deals per day
    ax = axes[0, 1]
    deal_y = [deals_per_day.get(d, 0) for d in days]
    ax.plot(days, deal_y, color="C0")
    ax.fill_between(days, deal_y, alpha=0.2)
    ax.set_title("Deals accepted per day")
    ax.grid(alpha=0.3)
    ax.set_xlabel("day")
    ax.set_ylabel("count")

    # consumer revenue per day
    ax = axes[1, 0]
    rev_y = [d.get("consumer_sales_revenue", 0) for d in daily]
    ax.plot(days, rev_y, color="C1")
    ax.fill_between(days, rev_y, alpha=0.2, color="C1")
    ax.set_title("Consumer-sales revenue per day")
    ax.grid(alpha=0.3)
    ax.set_xlabel("day")
    ax.set_ylabel("$")

    # opex per day
    ax = axes[1, 1]
    opex_y = [d.get("opex_total", 0) for d in daily]
    ax.plot(days, opex_y, color="C2")
    ax.fill_between(days, opex_y, alpha=0.2, color="C2")
    ax.set_title("Total daily opex (across agents)")
    ax.grid(alpha=0.3)
    ax.set_xlabel("day")
    ax.set_ylabel("$")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = path.replace(".json", ".stats.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[show] stats plot saved to {out}")


def _load_events_alongside(traj_path: str) -> list[dict]:
    events_path = traj_path.replace(".json", ".events.jsonl")
    if not os.path.exists(events_path):
        return []
    out: list[dict] = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------- main ----------


def main() -> None:
    args = _parse_args()
    path = _resolve(args)
    print(f"[show] loading {path}")
    with open(path) as f:
        traj = json.load(f)

    _print_run_summary(traj)
    _print_per_quarter(traj)
    _print_balance_sheets(traj)
    _print_roast_metrics(traj)
    _print_marketplace(traj)
    _print_deals(traj)
    _print_message_volume(traj)
    _print_usage(traj)

    if args.messages:
        _print_messages(traj)

    if args.plot:
        events = _load_events_alongside(path)
        _save_plot(traj, path)
        _save_stats_plot(traj, path, events)


if __name__ == "__main__":
    main()
