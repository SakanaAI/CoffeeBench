"""Streamlit live dashboard for CoffeeBench."""

from __future__ import annotations

import glob
import json
import os
import time
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from coffeebench.environment import (
    FESTIVALS,
    MONTH_END_DAYS,
    MONTH_NAMES,
    date_label,
)
from coffeebench.event_loop import format_min


EVENTS_DIR = "trajectories"
REFRESH_OPTIONS = (2.5, 5.0, 15.0, 30.0, 60.0, 120.0)
REFRESH_DEFAULT = 15.0


# ---------- demand-calendar helpers (used in sidebar context panel) ----------


def _month_index_for_day(day: int) -> int:
    d = int(day) % 365
    for idx, end in enumerate(MONTH_END_DAYS):
        if d <= end:
            return idx
    return len(MONTH_NAMES) - 1


def _festival_for_day(day: int):
    best = None
    for start, end, mult, name in FESTIVALS:
        if start <= day <= end and (best is None or mult > best[2]):
            best = (start, end, mult, name)
    return best


# ---------- run discovery ----------


def _list_runs() -> list[str]:
    """Stable alphabetical sort. Streamlit's selectbox keeps the user's
    selection across reruns when the option list ordering is stable —
    using mtime caused the list to re-shuffle on every auto-refresh
    tick, snapping the cursor to whichever run was most-recently
    written."""
    return sorted(
        glob.glob(os.path.join(EVENTS_DIR, "*.events.jsonl"))
        + glob.glob(os.path.join(EVENTS_DIR, "*", "seed_*", "run.events.jsonl"))
    )


def _run_label(path: str, root: str = EVENTS_DIR) -> str:
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        rel = path
    parts = rel.split(os.sep)
    if len(parts) >= 3 and parts[-1] == "run.events.jsonl":
        return " / ".join(parts[:-1])
    return rel.replace(".events.jsonl", "")


# ---------- event reader ----------


@st.cache_data(show_spinner=False)
def _read_events(path: str, mtime: float) -> list[dict]:
    """Cached on (path, mtime) so Streamlit only re-reads when the file
    actually changes."""
    del mtime  # cache key only
    out: list[dict] = []
    if not os.path.exists(path):
        return out
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


def _reduce_state(events: list[dict]) -> dict:
    """Walk events into a snapshot the dashboard renders. Last-write-wins
    for per-agent + marketplace state."""
    state: dict[str, Any] = {
        "run_start": None,
        "current_day": 0,
        "models": {},
        "agent_ids": [],
        "snapshots": [],  # list[day_end snapshot dict]
        "deals": [],  # list[deal_accepted event]
        "messages": [],  # list[send_message event]
        "agent_steps": [],  # list[agent_step event]
        "consumer_sales": [],  # list[consumer_sale event]
        "bankrupt": [],  # list[agent_bankrupt event]
        "final": None,  # run_end payload
        "last_ts_ms": 0,  # ts_ms of the most recent event
        "day_end_ts_ms": {},  # {day: ts_ms_at_day_end} for wall-clock-per-day chart
    }
    for ev in events:
        ts = ev.get("ts_ms")
        if isinstance(ts, (int, float)) and ts > state["last_ts_ms"]:
            state["last_ts_ms"] = int(ts)
        t = ev.get("type")
        if t == "run_start":
            state["run_start"] = ev
            state["models"] = ev.get("models") or {}
            state["agent_ids"] = ev.get("agent_ids") or []
        elif t == "agent_step":
            state["agent_steps"].append(ev)
            state["current_day"] = max(state["current_day"], int(ev.get("day", 0)))
            ai = ev.get("action_input") or {}
            if ev.get("action") == "send_message" and ai.get("recipient"):
                state["messages"].append(
                    {
                        "day": ev.get("day"),
                        "at": ev.get("at"),
                        "sender": ev.get("agent_id"),
                        "recipient": ai.get("recipient"),
                        "title": ai.get("title", ""),
                        "body": ai.get("body", ""),
                    }
                )
        elif t == "deal_accepted":
            state["deals"].append(ev)
        elif t == "consumer_sale":
            state["consumer_sales"].append(ev)
        elif t == "day_end":
            snap = ev.get("snapshot") or {}
            if snap:
                state["snapshots"].append(snap)
            day_idx = int(ev.get("day", 0))
            if isinstance(ts, (int, float)):
                state["day_end_ts_ms"][day_idx] = int(ts)
        elif t == "agent_bankrupt":
            state["bankrupt"].append(ev)
        elif t == "run_end":
            state["final"] = ev
    return state


# ---------- rendering primitives ----------


def _money(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)


def _short(s: str, n: int = 80) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_wall(ms: int) -> str:
    """Render milliseconds as `Hh Mm Ss` / `Mm Ss` / `Ss`."""
    if ms <= 0:
        return "0s"
    s = int(ms) // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# ---------- header / status ----------


def _render_header(state: dict, path: str) -> None:
    rs = state["run_start"]
    final = state["final"]
    cur_day = state["current_day"]
    max_days = (rs or {}).get("max_days") if rs else None
    status = "running" if final is None else "FINISHED"

    cols = st.columns([3, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(f"### {_run_label(path)}")
        st.caption(f"`{path}`")
    with cols[1]:
        st.metric("Status", status)
    with cols[2]:
        st.metric("Day", f"{cur_day} / {max_days or '?'}")
    with cols[3]:
        st.metric("Steps", f"{len(state['agent_steps']):,}")
    with cols[4]:
        st.metric("Wall time", _fmt_wall(state.get("last_ts_ms", 0)))
    with cols[5]:
        # Sum cost across all agent_step events (latest cost_so_far per agent).
        latest_cost: dict[str, float] = {}
        for ev in state["agent_steps"]:
            aid = ev.get("agent_id")
            if aid is None:
                continue
            c = ev.get("cost_so_far")
            if c is not None:
                latest_cost[aid] = float(c)
        st.metric("Spend", _money(sum(latest_cost.values())))

    if state["models"]:
        st.caption(
            "models: "
            + ", ".join(f"{aid}={mid}" for aid, mid in (state["models"] or {}).items())
        )

    today_obs = []
    if cur_day is not None:
        festival = _festival_for_day(cur_day)
        label = date_label(cur_day) if cur_day is not None else ""
        today_obs.append(
            f"day {cur_day} ({label}) · "
            f"month {_month_index_for_day(cur_day) + 1} {MONTH_NAMES[_month_index_for_day(cur_day)]}"
        )
        if festival:
            today_obs.append(
                f"festival window: **{festival[3]}** (days {festival[0]}-{festival[1]}, ×{festival[2]:.1f})"
            )
    if today_obs:
        st.info(" · ".join(today_obs))


# ---------- final-results section ----------


def _render_final_results(state: dict) -> None:
    final = state["final"]
    if final is None:
        return
    st.markdown("## Final results")
    agents = final.get("agents") or {}

    # Leaderboard
    rows = []
    for aid, ag in agents.items():
        audit = ag.get("audit") or {}
        ann = audit.get("annual") or {}
        bs = audit.get("balance_sheet") or {}
        completed = ag.get("completed", True)
        status = (
            f"DNF@{ag.get('bankrupt_day')} ({ag.get('bankrupt_reason') or '?'})"
            if not completed
            else "completed"
        )
        rows.append(
            {
                "agent": aid,
                "model": (ag.get("usage") or {}).get("model"),
                "status": status,
                "true_NI": ann.get("true_net_income"),
                "true_revenue": ann.get("true_revenue"),
                "true_returns": ann.get("true_returns"),
                "return_rate": audit.get("return_rate"),
                "true_equity": bs.get("true_equity"),
                "$ spend": (ag.get("usage") or {}).get("cost"),
            }
        )
    if rows:
        df = pd.DataFrame(rows).sort_values(
            "true_NI", ascending=False, na_position="last"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Roast metrics
    rm_rows = []
    for aid, ag in agents.items():
        rm = (ag.get("audit") or {}).get("roast_metrics")
        if rm:
            rm_rows.append(
                {
                    "agent": aid,
                    "green_consumed_kg": rm.get("green_consumed_kg"),
                    "roasted_produced_kg": rm.get("roasted_produced_kg"),
                    "actual_yield": rm.get("actual_yield"),
                    "expected_yield": rm.get("expected_yield"),
                    "roasted_b2b_qty_sold": rm.get("roasted_b2b_qty_sold"),
                    "truth_cogs_per_kg": rm.get("truth_cogs_per_kg_roasted"),
                    "roast_labor_paid": rm.get("roast_labor_paid"),
                }
            )
    if rm_rows:
        with st.expander("Roaster yield audit", expanded=False):
            st.dataframe(
                pd.DataFrame(rm_rows), use_container_width=True, hide_index=True
            )


# ---------- live operational sections ----------


def _latest_per_agent(state: dict) -> dict[str, dict]:
    """Latest per-agent dict from the most-recent day_end snapshot."""
    if not state["snapshots"]:
        return {}
    return state["snapshots"][-1].get("per_agent") or {}


def _render_kpi_cards(state: dict) -> None:
    per = _latest_per_agent(state)
    if not per:
        return
    st.markdown("#### Per-agent state (latest day)")
    aids = list(per.keys())
    cols = st.columns(len(aids))
    for col, aid in zip(cols, aids):
        d = per[aid] or {}
        eq = d.get("true_equity")
        cash = d.get("cash")
        with col:
            st.metric(
                label=aid,
                value=_money(eq),
                delta=f"cash {_money(cash)}",
                delta_color="off",
            )
            st.caption(
                f"inv {_money(d.get('inventory_value'))} · "
                f"AR {_money(d.get('ar_total'))} · "
                f"AP {_money(d.get('ap_total'))}"
            )


def _build_timeseries_frames(state: dict):
    """Return (days, aids, frames) where frames is a dict of pd.DataFrame
    keyed by metric name (cash, inv, eq, cum_rev, cum_ni, cost, ctx)."""
    snaps = state["snapshots"]
    if not snaps:
        return [], [], {}
    days = [s.get("day") for s in snaps]
    aids = list((snaps[0].get("per_agent") or {}).keys())
    if not aids:
        return days, aids, {}

    # Snapshot-derived: cash, inventory $, true equity per day.
    cash_df = pd.DataFrame({"day": days})
    inv_df = pd.DataFrame({"day": days})
    eq_df = pd.DataFrame({"day": days})
    for aid in aids:
        cash_df[aid] = [s.get("per_agent", {}).get(aid, {}).get("cash") for s in snaps]
        inv_df[aid] = [
            s.get("per_agent", {}).get(aid, {}).get("inventory_value") for s in snaps
        ]
        eq_df[aid] = [
            s.get("per_agent", {}).get(aid, {}).get("true_equity") for s in snaps
        ]

    # Truth-ledger derived: cumulative revenue (gross-net-of-returns) and NI
    # walked from `truth_today` per-day amounts on each snapshot.
    cum_rev: dict[str, list[float]] = {aid: [] for aid in aids}
    cum_ni: dict[str, list[float]] = {aid: [] for aid in aids}
    running_rev = {aid: 0.0 for aid in aids}
    running_ni = {aid: 0.0 for aid in aids}
    for s in snaps:
        for aid in aids:
            t = (s.get("per_agent", {}).get(aid, {}) or {}).get("truth_today") or {}
            day_rev = float(t.get("sale_revenue", 0.0)) - float(
                t.get("sale_reversal", 0.0)
            )
            day_cogs = float(t.get("inventory_out", 0.0))
            day_opex = (
                float(t.get("operating_expense", 0.0))
                + float(t.get("spoilage_expense", 0.0))
                + float(t.get("bad_debt_expense", 0.0))
                + float(t.get("writedown", 0.0))
            )
            day_int_net = float(t.get("interest_expense", 0.0)) - float(
                t.get("interest_revenue", 0.0)
            )
            running_rev[aid] += day_rev
            running_ni[aid] += day_rev - day_cogs - day_opex - day_int_net
            cum_rev[aid].append(running_rev[aid])
            cum_ni[aid].append(running_ni[aid])
    rev_df = pd.DataFrame({"day": days, **cum_rev})
    ni_df = pd.DataFrame({"day": days, **cum_ni})

    # Event-derived: latest cost_so_far + last_input_tokens per agent per day.
    cost_by_day: dict[int, dict[str, float]] = defaultdict(dict)
    ctx_by_day: dict[int, dict[str, int]] = defaultdict(dict)
    for ev in state["agent_steps"]:
        d = int(ev.get("day", 0))
        aid = ev.get("agent_id")
        if aid is None:
            continue
        c = ev.get("cost_so_far")
        if c is not None:
            cost_by_day[d][aid] = float(c)
        ctx = ev.get("last_input_tokens")
        if ctx is not None:
            ctx_by_day[d][aid] = int(ctx)

    # Forward-fill latest cost / context per agent across the day axis.
    def _forward_fill(source: dict, init_value):
        last = {aid: init_value for aid in aids}
        cols: dict[str, list] = {aid: [] for aid in aids}
        for d in days:
            if d in source:
                for aid, v in source[d].items():
                    last[aid] = v
            for aid in aids:
                cols[aid].append(last[aid])
        return cols

    cost_cols = _forward_fill(cost_by_day, 0.0)
    ctx_cols = _forward_fill(ctx_by_day, 0)
    cost_df = pd.DataFrame({"day": days, **cost_cols})
    ctx_df = pd.DataFrame({"day": days, **ctx_cols})

    # Per-agent per-item inventory qty over time. {aid: DataFrame} where
    # each DataFrame has a "day" column plus one column per item the
    # agent ever held a non-zero qty of.
    inv_by_item: dict[str, pd.DataFrame] = {}
    for aid in aids:
        items_seen: set[str] = set()
        for s in snaps:
            d = (s.get("per_agent", {}).get(aid, {}) or {}).get(
                "inventory_by_item"
            ) or {}
            for k in d.keys():
                items_seen.add(k)
        if not items_seen:
            inv_by_item[aid] = pd.DataFrame({"day": days})
            continue
        cols: dict[str, list] = {it: [] for it in sorted(items_seen)}
        for s in snaps:
            d = (s.get("per_agent", {}).get(aid, {}) or {}).get(
                "inventory_by_item"
            ) or {}
            for it in cols:
                cols[it].append(int(d.get(it, 0) or 0))
        inv_by_item[aid] = pd.DataFrame({"day": days, **cols})

    # Daily units-sold per (agent, item), summed across B2B deals and
    # B2C consumer sales. Walked from the deals + consumer_sales event
    # streams (cheap; both lists are bounded by sim length).
    sold_qty: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )  # sold_qty[seller][item][day] = qty
    items_sold: set[str] = set()
    for d in state.get("deals") or []:
        seller = d.get("seller")
        if seller is None:
            continue
        deal_at = d.get("deal_at")
        day = int(deal_at) // 1440 if deal_at is not None else int(d.get("day", 0))
        item = d.get("item_id") or "?"
        sold_qty[seller][item][day] += int(d.get("qty") or 0)
        items_sold.add(item)
    for s in state.get("consumer_sales") or []:
        seller = s.get("shop_id")
        if seller is None:
            continue
        day = int(s.get("day", 0))
        item = s.get("item_id") or "?"
        sold_qty[seller][item][day] += int(s.get("qty") or 0)
        items_sold.add(item)

    sold_per_agent: dict[str, pd.DataFrame] = {}
    for aid in aids:
        agent_items = sorted(sold_qty.get(aid, {}).keys())
        if not agent_items:
            sold_per_agent[aid] = pd.DataFrame({"day": days})
            continue
        cols: dict[str, list] = {it: [] for it in agent_items}
        for d in days:
            for it in agent_items:
                cols[it].append(int(sold_qty[aid][it].get(d, 0)))
        sold_per_agent[aid] = pd.DataFrame({"day": days, **cols})

    sold_global_by_item: dict[str, pd.DataFrame] = {}
    for it in sorted(items_sold):
        cols: dict[str, list] = {aid: [] for aid in aids}
        for d in days:
            for aid in aids:
                cols[aid].append(int(sold_qty.get(aid, {}).get(it, {}).get(d, 0)))
        sold_global_by_item[it] = pd.DataFrame({"day": days, **cols})

    # Per-item, per-agent inventory qty — for the Overview tab to show
    # how stock is distributed across agents per item over time.
    items_global: set[str] = set()
    for df in inv_by_item.values():
        for c in df.columns:
            if c != "day":
                items_global.add(c)
    inv_by_item_global: dict[str, pd.DataFrame] = {}
    for it in sorted(items_global):
        cols: dict[str, list] = {aid: [] for aid in aids}
        for s in snaps:
            for aid in aids:
                d = (s.get("per_agent", {}).get(aid, {}) or {}).get(
                    "inventory_by_item"
                ) or {}
                cols[aid].append(int(d.get(it, 0) or 0))
        inv_by_item_global[it] = pd.DataFrame({"day": days, **cols})

    return (
        days,
        aids,
        {
            "cash": cash_df,
            "inv": inv_df,
            "eq": eq_df,
            "cum_rev": rev_df,
            "cum_ni": ni_df,
            "cost": cost_df,
            "ctx": ctx_df,
            "inv_by_item": inv_by_item,
            "inv_by_item_global": inv_by_item_global,
            "sold_per_agent": sold_per_agent,
            "sold_global_by_item": sold_global_by_item,
        },
    )


def _render_wall_time_per_day(state: dict) -> None:
    """Plot how many wall-clock seconds each simulation day took (a
    diagnostic for run-pacing and provider latency: a slow day means
    LLM calls or network were heavy that day)."""
    day_ts = state.get("day_end_ts_ms") or {}
    if len(day_ts) < 2:
        return
    rows = []
    days = sorted(day_ts.keys())
    prev_ts = None
    for d in days:
        ts_s = float(day_ts[d]) / 1000.0
        if prev_ts is not None:
            rows.append({"day": d, "wall_seconds": round(ts_s - prev_ts, 2)})
        prev_ts = ts_s
    if not rows:
        return
    df = pd.DataFrame(rows)
    avg = df["wall_seconds"].mean()
    total = float(day_ts[days[-1]] - day_ts[days[0]]) / 1000.0
    st.markdown("### Wall-clock seconds per simulation day")
    st.line_chart(df.set_index("day"), height=200)
    st.caption(
        f"avg {avg:.1f}s / day · total elapsed across days "
        f"{int(days[0])}–{int(days[-1])}: {total:.0f}s "
        f"({total / 60:.1f} min)"
    )


def _render_overview_charts(days, aids, frames) -> None:
    """All-agent overlay grid: 3×2 panel + a 7th context-length panel."""
    if not frames:
        return
    r1 = st.columns(3)
    with r1[0]:
        st.caption("Cash")
        st.line_chart(frames["cash"].set_index("day"), height=200)
    with r1[1]:
        st.caption("Inventory $")
        st.line_chart(frames["inv"].set_index("day"), height=200)
    with r1[2]:
        st.caption("True equity")
        st.line_chart(frames["eq"].set_index("day"), height=200)

    r2 = st.columns(3)
    with r2[0]:
        st.caption("Cumulative true revenue (net of returns)")
        st.line_chart(frames["cum_rev"].set_index("day"), height=200)
    with r2[1]:
        st.caption("Cumulative true net income")
        st.line_chart(frames["cum_ni"].set_index("day"), height=200)
    with r2[2]:
        st.caption("API spend ($)")
        st.line_chart(frames["cost"].set_index("day"), height=200)

    st.caption("Context length (most recent prompt tokens, per agent)")
    st.line_chart(frames["ctx"].set_index("day"), height=200)

    inv_global = frames.get("inv_by_item_global") or {}
    if inv_global:
        items = sorted(inv_global.keys())
        cols = st.columns(len(items))
        for col, it in zip(cols, items):
            with col:
                st.caption(f"Inventory qty: {it}")
                st.line_chart(inv_global[it].set_index("day"), height=200)

    sold_global = frames.get("sold_global_by_item") or {}
    if sold_global:
        items = sorted(sold_global.keys())
        cols = st.columns(len(items))
        for col, it in zip(cols, items):
            with col:
                st.caption(f"Units sold per day: {it}")
                st.line_chart(sold_global[it].set_index("day"), height=200)


def _render_per_agent_panels(days, aids, frames) -> None:
    """Per-agent tabs, each with the agent's 6 curves."""
    if not frames or not aids:
        return
    tabs = st.tabs(aids)
    for tab, aid in zip(tabs, aids):
        with tab:
            r1 = st.columns(3)
            with r1[0]:
                st.caption("Cash")
                st.line_chart(
                    frames["cash"].set_index("day")[[aid]],
                    height=200,
                )
            with r1[1]:
                st.caption("Cumulative true revenue")
                st.line_chart(
                    frames["cum_rev"].set_index("day")[[aid]],
                    height=200,
                )
            with r1[2]:
                st.caption("Inventory $")
                st.line_chart(
                    frames["inv"].set_index("day")[[aid]],
                    height=200,
                )
            r2 = st.columns(3)
            with r2[0]:
                st.caption("Net asset (true equity)")
                st.line_chart(
                    frames["eq"].set_index("day")[[aid]],
                    height=200,
                )
            with r2[1]:
                st.caption("Cumulative true net income")
                st.line_chart(
                    frames["cum_ni"].set_index("day")[[aid]],
                    height=200,
                )
            with r2[2]:
                st.caption("API spend ($)")
                st.line_chart(
                    frames["cost"].set_index("day")[[aid]],
                    height=200,
                )

            # Inventory qty per item — full row width, one line per item.
            inv_df_agent = (frames.get("inv_by_item") or {}).get(aid)
            if inv_df_agent is not None and len(inv_df_agent.columns) > 1:
                st.caption("Inventory qty by item")
                st.line_chart(inv_df_agent.set_index("day"), height=220)

            sold_df_agent = (frames.get("sold_per_agent") or {}).get(aid)
            if sold_df_agent is not None and len(sold_df_agent.columns) > 1:
                st.caption("Units sold per day, by item (B2B + consumer)")
                st.line_chart(sold_df_agent.set_index("day"), height=220)


def _render_deals(state: dict, n: int = 50) -> None:
    deals = state["deals"]
    if not deals:
        return
    st.markdown(f"### Recent deals (latest {min(n, len(deals))} of {len(deals)})")
    rows = []
    for d in deals[-n:][::-1]:
        deal_at = d.get("deal_at")
        day = int(deal_at) // 1440 if deal_at is not None else d.get("day")
        rows.append(
            {
                "day": day,
                "seller": d.get("seller"),
                "buyer": d.get("buyer"),
                "item": d.get("item_id"),
                "qty": d.get("qty"),
                "unit_price": d.get("unit_price"),
                "total": d.get("total_price"),
                "terms": f"NET-{d.get('payment_terms_days', 30)}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_retail_prices(state: dict) -> None:
    """Per-retailer retail price over time with per-item reference
    lines (p_res demand cliff). One faceted panel per consumer-facing
    item. Sales-volume bubbles overlay the price line so a reader can
    spot whether high-price days actually moved units."""
    sales = state["consumer_sales"]
    if not sales:
        return
    rows = [
        {
            "day": int(s.get("day") or 0),
            "shop_id": s.get("shop_id"),
            "item_id": s.get("item_id"),
            "unit_price": float(s.get("unit_price") or 0.0),
            "qty": int(s.get("qty") or 0),
        }
        for s in sales
        if s.get("unit_price") is not None
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.markdown("### Retail price per retailer (with reference lines)")

    # Per-item reservation prices read from the run's item catalog
    # snapshot (state["items"]). Falls back to the canonical defaults
    # if the snapshot is missing for any reason.
    items_meta = state.get("items") or []
    p_res_by_item: dict[str, float] = {}
    cost_basis_by_item: dict[str, float] = {
        # Retailer starter cost-basis values mirrored from main.py
        # _seed_world. Used only for the monopoly-optimal reference.
        "roasted_coffee_kg": 10.0,
        "roasted_specialty_kg": 18.0,
    }
    for it in items_meta:
        p = it.get("retail_reservation_price") if isinstance(it, dict) else None
        if p is not None:
            p_res_by_item[it["id"]] = float(p)
    # Sensible defaults if the catalog isn't snapshotted in state.
    p_res_by_item.setdefault("roasted_coffee_kg", 30.0)
    p_res_by_item.setdefault("roasted_specialty_kg", 80.0)

    try:
        import altair as alt
    except ImportError:
        for item_id, sub in df.groupby("item_id"):
            st.markdown(f"#### {item_id}")
            pivot = sub.pivot_table(
                index="day",
                columns="shop_id",
                values="unit_price",
                aggfunc="mean",
            )
            st.line_chart(pivot, height=240)
        st.caption("(altair not available — install `altair` for reference lines)")
        return

    daily = df.groupby(["day", "shop_id", "item_id"], as_index=False).agg(
        unit_price=("unit_price", "mean"),
        qty=("qty", "sum"),
    )
    for item_id, sub in daily.groupby("item_id"):
        st.markdown(f"#### {item_id}")
        p_res = float(p_res_by_item.get(item_id, sub["unit_price"].max() * 1.1))
        cost_basis = float(cost_basis_by_item.get(item_id, 0.0))
        monopoly_opt = (p_res + cost_basis) / 2.0
        price_lines = (
            alt.Chart(sub)
            .mark_line(point=True)
            .encode(
                x=alt.X("day:Q", title="day"),
                y=alt.Y("unit_price:Q", title="$/kg", scale=alt.Scale(zero=False)),
                color=alt.Color("shop_id:N", title="retailer"),
                tooltip=["day", "shop_id", "unit_price", "qty"],
            )
        )
        qty_marks = (
            alt.Chart(sub)
            .mark_circle(opacity=0.35)
            .encode(
                x="day:Q",
                y="unit_price:Q",
                size=alt.Size("qty:Q", legend=None, scale=alt.Scale(range=[10, 220])),
                color="shop_id:N",
                tooltip=["day", "shop_id", "unit_price", "qty"],
            )
        )
        refs = pd.DataFrame(
            [
                {"y": p_res, "label": f"p_res = ${p_res:.0f} (demand cliff)"},
                {"y": monopoly_opt, "label": f"monopoly-optimal ≈ ${monopoly_opt:.2f}"},
            ]
        )
        ref_lines = (
            alt.Chart(refs)
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y="y:Q",
                color=alt.Color(
                    "label:N",
                    title=None,
                    scale=alt.Scale(range=["#d62728", "#2ca02c"]),
                    legend=alt.Legend(orient="top"),
                ),
            )
        )
        chart = (
            (price_lines + qty_marks + ref_lines).properties(height=280).interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        cap_parts = []
        for sid, ss in sub.groupby("shop_id"):
            avg_p = ss["unit_price"].mean()
            days_above = int((ss["unit_price"] > p_res).sum())
            cap_parts.append(
                f"**{sid}**: avg ${avg_p:.2f}, {days_above} day(s) above p_res"
            )
        st.caption(
            f"Bubble size = sales qty. Red dashed = `p_res` (${p_res:.0f}), "
            f"green dashed = monopoly-optimal reference. " + " · ".join(cap_parts)
        )


def _render_consumer_sales(state: dict) -> None:
    sales = state["consumer_sales"]
    if not sales:
        return
    by_day_shop: dict = defaultdict(lambda: defaultdict(float))
    for s in sales:
        by_day_shop[s.get("day")][s.get("shop_id")] += float(s.get("total_price") or 0)
    if not by_day_shop:
        return
    days = sorted(by_day_shop.keys())
    shops = sorted({sid for d in by_day_shop.values() for sid in d.keys()})
    df = pd.DataFrame({"day": days})
    for sid in shops:
        df[sid] = [by_day_shop[d].get(sid, 0.0) for d in days]
    st.markdown("### Consumer-sales revenue per retailer")
    st.line_chart(df.set_index("day"), height=240)


def _render_b2b_prices(state: dict) -> None:
    deals = state["deals"]
    if not deals:
        return
    st.markdown("### B2B deal prices over time")
    rows = []
    for d in deals:
        deal_at = d.get("deal_at")
        day = int(deal_at) // 1440 if deal_at is not None else d.get("day")
        rows.append(
            {
                "day": day,
                "item": d.get("item_id"),
                "unit_price": d.get("unit_price"),
                "seller": d.get("seller"),
                "buyer": d.get("buyer"),
                "qty": d.get("qty"),
                "pair": f"{d.get('seller')}→{d.get('buyer')}",
            }
        )
    df = pd.DataFrame(rows)
    items = sorted(df["item"].unique())
    tabs = st.tabs([f"{it} ({len(df[df['item'] == it])})" for it in items])
    for tab, item in zip(tabs, items):
        with tab:
            sub = df[df["item"] == item].copy()
            # Scatter of deal unit_price by day, colored by seller→buyer pair.
            try:
                st.scatter_chart(
                    sub,
                    x="day",
                    y="unit_price",
                    color="pair",
                    size="qty",
                    height=260,
                )
            except (TypeError, AttributeError):
                # Streamlit < 1.30 fallback
                pivot = sub.pivot_table(
                    index="day",
                    columns="pair",
                    values="unit_price",
                    aggfunc="mean",
                )
                st.line_chart(pivot, height=260)
            st.caption(
                f"{len(sub)} deals · "
                f"min ${sub['unit_price'].min():.2f} · "
                f"max ${sub['unit_price'].max():.2f} · "
                f"mean ${sub['unit_price'].mean():.2f}"
            )


def _render_messages(state: dict, n: int = 100) -> None:
    msgs = state["messages"]
    if not msgs:
        return
    st.markdown(f"### Messages ({len(msgs)} total)")

    senders = sorted({m["sender"] for m in msgs})
    cols = st.columns([3, 1])
    with cols[0]:
        sender_filter = st.multiselect(
            "Filter by sender",
            options=senders,
            default=[],
            key="msg_sender_filter",
        )
    with cols[1]:
        n_show = st.slider("Show latest", 20, 500, 100, 20, key="msg_n_slider")

    shown = msgs[::-1]
    if sender_filter:
        shown = [m for m in shown if m["sender"] in sender_filter]
    shown = shown[:n_show]

    # Table for scanning. Body is shown truncated here only as a preview;
    # selecting a row below reveals the full body untruncated.
    rows = [
        {
            "time": format_min(m["at"])
            if m.get("at") is not None
            else f"day {m['day']}",
            "sender": m["sender"],
            "recipient": m["recipient"],
            "title": _short(m.get("title", ""), 60),
            "body_preview": _short((m.get("body") or "").strip(), 200),
        }
        for m in shown
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        key="msg_table",
    )
    selected = event.selection.rows if event and event.selection else []
    # Streamlit dataframe selection state can persist across reruns even
    # when `shown` shrinks (e.g. user adjusts the sender filter or slider
    # after selecting a row). Guard against the now-stale index.
    if selected and 0 <= selected[0] < len(shown):
        idx = selected[0]
        m = shown[idx]
        st.markdown("---")
        ts = format_min(m["at"]) if m.get("at") is not None else f"day {m['day']}"
        st.markdown(
            f"**{ts}** · `{m['sender']}` → `{m['recipient']}`  \n"
            f"**Title:** {m.get('title', '') or '(empty)'}"
        )
        st.text(m.get("body") or "(empty body)")
    else:
        st.caption("Select a row to read the full message body.")


def _render_recent_actions(state: dict, default_n: int = 30) -> None:
    steps = state["agent_steps"]
    if not steps:
        return
    st.markdown(f"### Tool calls ({len(steps)} total)")

    # Group by agent so each gets its own tab — easier than scrolling
    # through one mixed feed when 5+ agents are active.
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for ev in steps:
        by_agent[ev.get("agent_id") or "?"].append(ev)
    aids = sorted(by_agent.keys())

    n_show = st.slider(
        "Show latest per agent",
        10,
        500,
        default_n,
        10,
        key="action_n_slider",
    )

    tabs = st.tabs([f"{aid} ({len(by_agent[aid])})" for aid in aids])
    for tab, aid in zip(tabs, aids):
        with tab:
            shown = by_agent[aid][-n_show:][::-1]
            for ev in shown:
                action = ev.get("action") or "?"
                day = ev.get("day")
                ai = ev.get("action_input") or {}
                obs = ev.get("observation")
                thought = ev.get("thought") or ""
                # One-line summary: day, action, brief args, status icon.
                # Full thought / input / observation live inside the
                # expander — never duplicated in the summary, so the
                # observation is only ever shown once (when expanded).
                if isinstance(obs, dict):
                    status = obs.get("status") or "?"
                    icon = "✓" if status == "success" else "✗"
                else:
                    icon = "·"
                ai_brief = _short(
                    ", ".join(f"{k}={v}" for k, v in ai.items()) if ai else "",
                    80,
                )
                at = ev.get("at")
                time_label = format_min(at) if at is not None else f"day {day}"
                summary = (
                    f"{time_label} · `{action}`"
                    + (f"({ai_brief})" if ai_brief else "()")
                    + f" {icon}"
                )
                with st.expander(summary, expanded=False):
                    if thought:
                        st.caption("thought")
                        st.text(_short(thought, 1500))
                    st.caption("action_input")
                    st.json(ai, expanded=False)
                    st.caption("observation")
                    st.json(obs, expanded=False)


# ---------- sidebar ----------


def _sidebar() -> tuple[str, float]:
    runs = _list_runs()
    if not runs:
        st.sidebar.error(f"No event files in {EVENTS_DIR}/")
        st.stop()
    # Use the path itself as the selectbox value so the user's
    # selection survives auto-refresh reruns even if the file list
    # changes (new run added, mtime shifts, etc.).
    selected_path = st.sidebar.selectbox(
        "Run",
        options=runs,
        format_func=_run_label,
        key="run_selector",
    )
    refresh = st.sidebar.selectbox(
        "Refresh interval (s)",
        options=list(REFRESH_OPTIONS),
        index=REFRESH_OPTIONS.index(REFRESH_DEFAULT),
        key="refresh_interval",
    )
    st.sidebar.caption(f"Auto-refresh every {refresh}s while watching.")
    return selected_path, float(refresh)


# ---------- main ----------


def main() -> None:
    st.set_page_config(page_title="CoffeeBench dashboard", layout="wide")
    path, refresh = _sidebar()
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    events = _read_events(path, mtime)
    state = _reduce_state(events)

    _render_header(state, path)

    days, aids, frames = _build_timeseries_frames(state)

    tab_overview, tab_per_agent, tab_market, tab_msgs, tab_tools = st.tabs(
        ["Overview", "Per-agent", "Marketplace", "Messages", "Tool calls"]
    )

    with tab_overview:
        _render_final_results(state)
        _render_kpi_cards(state)
        _render_wall_time_per_day(state)
        st.markdown("#### Timeseries (all agents)")
        _render_overview_charts(days, aids, frames)

    with tab_per_agent:
        _render_per_agent_panels(days, aids, frames)

    with tab_market:
        _render_retail_prices(state)
        _render_consumer_sales(state)
        _render_b2b_prices(state)
        _render_deals(state)

    with tab_msgs:
        _render_messages(state)

    with tab_tools:
        _render_recent_actions(state)

    # Auto-refresh while the run isn't done.
    if state["final"] is None:
        time.sleep(refresh)
        st.rerun()


if __name__ == "__main__":
    main()
