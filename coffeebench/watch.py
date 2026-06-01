"""Live terminal watcher for a CoffeeBench run."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any


REFRESH_SECONDS = 1.0
EVENTS_DIR = "trajectories"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "events_path",
        nargs="?",
        default=None,
        help="Path to .events.jsonl file. If omitted, use --latest.",
    )
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use newest .events.jsonl in trajectories/.",
    )
    p.add_argument("--once", action="store_true", help="Render snapshot and exit.")
    p.add_argument(
        "--actions",
        action="store_true",
        help="Stream every agent action as it happens (verbose).",
    )
    p.add_argument(
        "--messages",
        action="store_true",
        help="Stream every send_message action as it happens.",
    )
    return p.parse_args()


def _resolve_path(args: argparse.Namespace) -> str:
    if args.events_path:
        return args.events_path
    # Both flat (`<name>.events.jsonl`) and hierarchical (`<experiment>/seed_<N>/run.events.jsonl`).
    cands = sorted(
        glob.glob(os.path.join(EVENTS_DIR, "*.events.jsonl"))
        + glob.glob(os.path.join(EVENTS_DIR, "*", "seed_*", "run.events.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not cands:
        sys.exit(f"No .events.jsonl in {EVENTS_DIR}/.")
    return cands[0]


def _read_events(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "models": {},
        "max_days": None,
        "agent_ids": [],
        "current_day": 0,
        "step_count": 0,
        "last_action_per_agent": {},  # agent_id -> {action, action_input, day, step}
        "last_deal": None,
        "deals_total": 0,
        # Latest accepted-deal price per item (any seller), so the rolling
        # status can show "what's the going rate for roasted_coffee_kg?"
        # at a glance without scrolling deal history.
        "latest_deal_per_item": {},  # item_id -> deal-event dict
        # Latest asking-price per (seller, item) extracted from post_listing
        # agent_step events. Captures pricing intent (what's currently
        # advertised) regardless of whether a deal has cleared yet.
        "latest_ask_per_seller_item": {},  # (seller_id, item_id) -> {ask, qty, day}
        # Last few B2B deals, newest-last, so the rolling status can
        # show recent price trajectory rather than just the single
        # last deal.
        "recent_deals": [],
        "consumer_sales_total": 0,
        "consumer_revenue_total": 0.0,
        "last_consumer_sale": None,
        "last_message": None,
        "latest_snapshot": None,
        "final": None,
        # Per-agent cumulative API cost + LLM call count, refreshed on every
        # agent_step event (models report cost monotonically).
        "cost_per_agent": {},  # agent_id -> latest cost_so_far ($)
        "calls_per_agent": {},  # agent_id -> latest n_calls_so_far
    }
    for ev in events:
        t = ev.get("type")
        if t == "run_start":
            state["models"] = ev.get("models", {})
            state["max_days"] = ev.get("max_days")
            state["agent_ids"] = ev.get("agent_ids", [])
        elif t == "agent_step":
            state["step_count"] = ev.get("step", state["step_count"])
            state["current_day"] = max(state["current_day"], ev.get("day", 0))
            aid = ev.get("agent_id")
            state["last_action_per_agent"][aid] = {
                "action": ev.get("action"),
                "action_input": ev.get("action_input"),
                "day": ev.get("day"),
                "step": ev.get("step"),
            }
            if "cost_so_far" in ev:
                state["cost_per_agent"][aid] = float(ev.get("cost_so_far") or 0.0)
            if "n_calls_so_far" in ev:
                state["calls_per_agent"][aid] = int(ev.get("n_calls_so_far") or 0)
            ai = ev.get("action_input") or {}
            if ev.get("action") == "send_message" and ai.get("recipient"):
                state["last_message"] = {
                    "day": ev.get("day"),
                    "sender": ev.get("agent_id"),
                    "recipient": ai.get("recipient"),
                    "body": ai.get("body", ""),
                }
            if ev.get("action") == "post_listing":
                seller = ev.get("agent_id")
                item_id = ai.get("item_id")
                ask = ai.get("asking_price")
                qty = ai.get("qty")
                if seller and item_id and ask is not None:
                    state["latest_ask_per_seller_item"][(seller, item_id)] = {
                        "ask": float(ask),
                        "qty": int(qty) if qty is not None else None,
                        "day": ev.get("day"),
                    }
        elif t == "deal_accepted":
            state["deals_total"] += 1
            # `deal_accepted` carries `deal_at` (minutes-since-start),
            # not `day`. Backfill so downstream rendering can use
            # ev.get("day") uniformly.
            if ev.get("day") is None and ev.get("deal_at") is not None:
                ev = {**ev, "day": ev.get("deal_at") // 1440}
            state["last_deal"] = ev
            item_id = ev.get("item_id")
            if item_id:
                state["latest_deal_per_item"][item_id] = ev
            state["recent_deals"].append(ev)
            # Cap at 8 — enough to spot a trend without wrapping.
            if len(state["recent_deals"]) > 8:
                state["recent_deals"] = state["recent_deals"][-8:]
        elif t == "consumer_sale":
            state["consumer_sales_total"] += 1
            state["consumer_revenue_total"] += float(ev.get("total_price") or 0.0)
            state["last_consumer_sale"] = ev
        elif t == "day_end":
            snap = ev.get("snapshot")
            if snap is not None:
                state["latest_snapshot"] = snap
            state["current_day"] = max(state["current_day"], ev.get("day", 0))
        elif t == "run_end":
            state["final"] = ev
    return state


def _render(state: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    head = f"CoffeeBench | day {state['current_day']}/{state['max_days'] or '?'} | step {state['step_count']}"
    total_cost = sum(state["cost_per_agent"].values())
    total_calls = sum(state["calls_per_agent"].values())
    if total_calls:
        head += f" | API cost {_money(total_cost)} ({total_calls} calls)"
    if state["models"]:
        head += "  | " + ", ".join(f"{a}:{m}" for a, m in state["models"].items())
    lines.append(head)
    lines.append("-" * 78)

    if state["final"] is not None:
        f = state["final"]
        lines.append("STATUS: FINISHED")
        per = f.get("agents", {})
        lines.append(f"  {'agent':<14} {'status':<22} {'true_NI':>10} {'true_rev':>10}")
        for aid, ag in per.items():
            audit = ag.get("audit") or {}
            ann = audit.get("annual") or {}
            completed = ag.get("completed", True)
            if not completed:
                status = (
                    f"DNF@{ag.get('bankrupt_day')} ({ag.get('bankrupt_reason') or '?'})"
                )
            else:
                status = "completed"
            true_rev = ann.get("true_revenue") if ann else None
            true_ni = ann.get("true_net_income") if ann else None
            lines.append(
                f"  {aid:<14} {status:<22} "
                f"{(_money(true_ni) if true_ni is not None else '—'):>10} "
                f"{(_money(true_rev) if true_rev is not None else '—'):>10}"
            )
    else:
        snap = state["latest_snapshot"]
        if snap is not None:
            per = snap.get("per_agent", {})
            lines.append(
                f"  {'agent':<14} {'role':<11} {'cash':>10} {'inv$':>10} {'AR':>9} {'AP':>9} "
                f"{'true_eq':>10} {'deals':>6} {'API$':>8} {'calls':>6}"
            )
            for aid, ag in per.items():
                cost = state["cost_per_agent"].get(aid, 0.0)
                calls = state["calls_per_agent"].get(aid, 0)
                lines.append(
                    f"  {aid:<14} {ag.get('role', ''):<11} {_money(ag.get('cash')):>10} "
                    f"{_money(ag.get('inventory_value')):>10} {_money(ag.get('ar_total')):>9} "
                    f"{_money(ag.get('ap_total')):>9} {_money(ag.get('true_equity')):>10} "
                    f"{ag.get('deals_count', 0):>6} {_money(cost):>8} {calls:>6}"
                )
        lines.append("")
        lines.append(
            f"  marketplace: {state['deals_total']} deals total, "
            f"{state['consumer_sales_total']} retail sales (cum revenue {_money(state['consumer_revenue_total'])})"
        )

        if state["last_deal"]:
            d = state["last_deal"]
            lines.append(
                f"  last deal:    [day {d.get('day')}] {d.get('seller')} -> {d.get('buyer')}  "
                f"{d.get('item_id')} x{d.get('qty')} @ {_money(d.get('unit_price'))}/unit "
                f"= {_money(d.get('total_price'))}"
            )
        # Per-item latest cleared price (the "going rate" right now).
        if state["latest_deal_per_item"]:
            lines.append("")
            lines.append("  latest B2B price per item:")
            for item_id, d in state["latest_deal_per_item"].items():
                qty = d.get("qty") if d.get("qty") is not None else "?"
                day = d.get("day") if d.get("day") is not None else "?"
                seller = d.get("seller") or "?"
                buyer = d.get("buyer") or "?"
                lines.append(
                    f"    {item_id:<24} {_money(d.get('unit_price'))}/unit  "
                    f"x{qty} on day {day}  ({seller}→{buyer})"
                )
        # Recent B2B deals — last 8, newest-first — so price drift is visible.
        if state["recent_deals"]:
            lines.append("")
            lines.append("  recent B2B deals (newest first):")
            for d in reversed(state["recent_deals"]):
                day_s = str(d.get("day")) if d.get("day") is not None else "?"
                qty_s = str(d.get("qty")) if d.get("qty") is not None else "?"
                seller = d.get("seller") or "?"
                buyer = d.get("buyer") or "?"
                item = d.get("item_id") or "?"
                lines.append(
                    f"    [day {day_s:>3}] {seller:<11}→{buyer:<11} "
                    f"{item:<24} x{qty_s:>3} @ {_money(d.get('unit_price'))}/unit"
                )
        # Currently advertised asking prices per (seller, item). These
        # are post_listing-derived and may include closed/expired listings;
        # the goal is to surface pricing INTENT, not a clean open-listing
        # registry (the trajectory has that for post-hoc).
        if state["latest_ask_per_seller_item"]:
            lines.append("")
            lines.append("  latest asking price per (seller, item):")
            for (seller, item_id), info in sorted(
                state["latest_ask_per_seller_item"].items()
            ):
                qty_s = f"x{info.get('qty')}" if info.get("qty") is not None else ""
                day_s = info.get("day") if info.get("day") is not None else "?"
                lines.append(
                    f"    {seller:<12} {item_id:<24} ask {_money(info.get('ask'))}/unit  "
                    f"{qty_s}  posted day {day_s}"
                )
        if state["last_consumer_sale"]:
            s = state["last_consumer_sale"]
            boost = s.get("boost_multiplier")
            boost_s = f"{boost:.2f}x" if isinstance(boost, (int, float)) else "—"
            lines.append(
                f"  last retail:  [day {s.get('day')}] {s.get('shop_id')} sold {s.get('item_id')} x{s.get('qty')} "
                f"@ {_money(s.get('unit_price'))}/unit (boost {boost_s})"
            )
        if state["last_message"]:
            m = state["last_message"]
            body = (m.get("body") or "")[:120]
            lines.append(
                f"  last msg:     [day {m.get('day')}] {m.get('sender')}→{m.get('recipient')}: {body!r}"
            )

        # Per-agent last action
        if state["last_action_per_agent"]:
            lines.append("")
            lines.append("  last action per agent:")
            for aid, info in state["last_action_per_agent"].items():
                ai = info.get("action_input") or {}
                ai_str = (
                    ", ".join(f"{k}={v}" for k, v in list(ai.items())[:3])
                    if isinstance(ai, dict)
                    else ""
                )
                lines.append(
                    f"    {aid:<14} day {info.get('day')}  step {info.get('step')}  "
                    f"{info.get('action')}({ai_str})"
                )

    lines.append("=" * 78)
    return "\n".join(lines)


def _stream(events: list[dict[str, Any]], actions: bool, messages: bool) -> None:
    seen = _stream.seen if hasattr(_stream, "seen") else 0
    for ev in events[seen:]:
        t = ev.get("type")
        if t == "agent_step":
            ai = ev.get("action_input") or {}
            if actions:
                ai_str = ", ".join(f"{k}={v}" for k, v in list(ai.items())[:5])
                print(
                    f"  [day {ev.get('day')} step {ev.get('step')}] {ev.get('agent_id'):14s} "
                    f"{ev.get('action')}({ai_str})"
                )
            elif messages and ev.get("action") == "send_message":
                recipient = ai.get("recipient")
                title = (ai.get("title") or "(no title)").strip() or "(no title)"
                body = (ai.get("body") or "")[:160]
                print(
                    f"  [day {ev.get('day')}] {ev.get('agent_id'):14s} → {recipient}  📧 {title}"
                )
                if body:
                    print(f"                      {body}")
        elif t == "deal_accepted":
            print(
                f"  [day {ev.get('day')}] DEAL: {ev.get('seller')} -> {ev.get('buyer')}  "
                f"{ev.get('item_id')} x{ev.get('qty')} @ {_money(ev.get('unit_price'))}"
            )
        elif t == "consumer_sale":
            print(
                f"  [day {ev.get('day')}] RETAIL: {ev.get('shop_id')} sold {ev.get('item_id')} "
                f"x{ev.get('qty')} @ {_money(ev.get('unit_price'))} "
                f"(boost {ev.get('boost_multiplier'):.2f}x)"
            )
    _stream.seen = len(events)


def main() -> None:
    args = _parse_args()
    path = _resolve_path(args)
    print(f"[coffeebench.watch] tailing {path}")

    if args.once:
        events = _read_events(path)
        state = _summarize(events)
        if args.actions or args.messages:
            _stream(events, args.actions, args.messages)
        print(_render(state))
        return

    last_count = 0
    try:
        while True:
            events = _read_events(path)
            if len(events) != last_count:
                state = _summarize(events)
                if args.actions or args.messages:
                    _stream(events, args.actions, args.messages)
                else:
                    sys.stdout.write("\033[2J\033[H")
                    print(_render(state))
                last_count = len(events)
                if state["final"] is not None:
                    print("[coffeebench.watch] run finished — exiting.")
                    return
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\n[coffeebench.watch] stopped.")


if __name__ == "__main__":
    main()
