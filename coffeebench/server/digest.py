"""Deterministic resume digest for segment k>1 (DESIGN §6).

Every agent restarts each segment with a fresh context (new Harbor trial;
/workspace is restored from the segment-end upload). Its first observation carries this digest, generated
purely from the restored env state + the event log, with the same procedure
for every model (no LLM summarization → identical across focal models given
the same history).

Sections: position (cash / inventory / AR / AP / capacity), inventory lots
with expiry, in-flight production / roasts, own open listings, offers both
ways, deals in flight, open invoices (overdue first), own B2B deal history,
public market summary (others' open listings, consumer-market aggregate —
the same data `view_listings` / `view_market_aggregate` expose), retail
sales (retailers), festival calendar, bankrupt participants, DM threads per
counterparty (titles + bodies of the latest messages, with ids — exclusivity
or supply promises live only in DMs), and the agent's own last
`RECENT_DAYS` days of actions with compact results.

Size bound: `MAX_CHARS` (≈ 20k tokens). The DM threads and the action log
are trimmed oldest-first with explicit omission counts; every other list is
row-capped with counts.
"""

from __future__ import annotations

import json
from collections import defaultdict

from coffeebench.environment import FESTIVALS, date_label
from coffeebench.event_loop import MINUTES_PER_DAY, format_min

MAX_CHARS = 80_000  # ≈ 20k tokens at ~4 chars/token
RECENT_DAYS = 15
DM_PER_THREAD = 8
DM_BODY_CHARS = 1500
ROW_CAP = 40
RESULT_CHARS = 220

READ_ONLY_TOOLS = {
    "view_dashboard", "view_listings", "view_offers", "view_deals",
    "view_inventory", "view_messages", "view_payables", "view_receivables",
    "view_trial_balance", "view_consumer_sales", "view_market_aggregate",
    "read_message",
}


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _capped(rows: list[str], cap: int = ROW_CAP, what: str = "rows") -> list[str]:
    if len(rows) <= cap:
        return rows
    return rows[:cap] + [f"  … {len(rows) - cap} more {what} omitted"]


def _short(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, separators=(",", ":"))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_result(action: str, obs) -> str:
    """One compact line for a tool result."""
    if not isinstance(obs, dict):
        return _short(obs, RESULT_CHARS)
    status = obs.get("status", "?")
    if action in READ_ONLY_TOOLS and status == "success":
        return "success (read-only view)"
    parts = [str(status)]
    if obs.get("message"):
        parts.append(_short(obs["message"], RESULT_CHARS))
    for k, v in obs.items():
        if k in ("status", "message", "note", "hint"):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
    return _short("; ".join(parts), RESULT_CHARS)


def read_agent_steps(events_path: str | None, since_day: int) -> dict[str, list[dict]]:
    """`agent_step` events with day ≥ since_day, per agent (one file pass)."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not events_path:
        return out
    try:
        fh = open(events_path)
    except OSError:
        return out
    with fh:
        for line in fh:
            if '"agent_step"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") != "agent_step" or int(ev.get("day", -1)) < since_day:
                continue
            out[ev["agent_id"]].append(ev)
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _position(env, ba, today: int) -> list[str]:
    lines = [
        "## Position",
        f"- cash {_money(ba.cash)}; inventory at cost {_money(ba._inventory_value())}; "
        f"receivables (AR) {_money(ba._ar_outstanding())}; payables (AP) {_money(ba._ap_outstanding())}",
        f"- stock on hand {ba._total_inventory_kg()} kg, in-flight inbound "
        f"{ba._pending_inbound_kg()} kg, capacity {ba._inventory_cap_kg()} kg",
    ]
    if ba.role == "retailer":
        prices = ", ".join(f"{k} {_money(v)}" for k, v in sorted(ba.retail_prices.items()))
        lines.append(f"- current retail prices: {prices or 'none'}")
    lines.append("")
    lines.append("## Inventory lots (FIFO by expiry; usable through the expiry day)")
    lots = sorted(
        (lt for lt in ba.lots if lt.qty > 0),
        key=lambda lt: (lt.item_id, lt.expiry_day if lt.expiry_day is not None else 10**9, lt.lot_id),
    )
    rows = []
    for lt in lots:
        exp = "no expiry" if lt.expiry_day is None else (
            f"expires day {lt.expiry_day} ({lt.expiry_day - today:+d} d)"
        )
        rows.append(
            f"- {lt.lot_id} {lt.item_id}: {lt.qty} kg @ {_money(lt.unit_cost)}/kg, "
            f"produced day {lt.produced_day}, {exp}"
        )
    lines += _capped(rows, what="lots") or ["- (no stock)"]
    if ba.role == "farmer":
        prod = [p for p in env._pending_production if p["agent_id"] == ba.agent_id]
        lines.append("")
        lines.append("## Production in progress")
        lines += [f"- {p['qty']} kg {p['item_id']} ready day {p['ready_day']}" for p in prod] or ["- none"]
    if ba.role == "roaster":
        roasts = [r for r in env._pending_roasting if r["agent_id"] == ba.agent_id]
        lines.append("")
        lines.append("## Roasts in progress")
        lines += [
            f"- {r['output_qty']} kg {r['output_item']} ready day {r['ready_day']}" for r in roasts
        ] or ["- none"]
    return lines


def _market_book(env, ba, today: int) -> list[str]:
    mp = env.marketplace
    aid = ba.agent_id
    item_of = {lt.id: lt.item_id for lt in mp.listings}
    lines = ["## Your open listings"]
    rows = []
    for lt in mp.listings:
        if lt.seller_id != aid or lt.status != "open":
            continue
        n_off = sum(1 for o in mp.offers if o.listing_id == lt.id and o.status == "pending")
        rows.append(
            f"- {lt.id} {lt.item_id}: {lt.qty} kg @ {_money(lt.asking_price)}, net-{lt.payment_terms_days}, "
            f"posted {format_min(lt.posted_at)}, {n_off} pending offer(s)"
        )
    lines += _capped(rows, what="listings") or ["- none"]

    lines += ["", "## Pending offers ON your listings (accept_offer / reject_offer)"]
    rows = [
        f"- {o.id} from {o.buyer_id} on {o.listing_id} ({item_of.get(o.listing_id)}): {o.qty} kg @ "
        f"{_money(o.offered_price)}, net-{o.payment_terms_days}, expires EOD day {o.posted_at // MINUTES_PER_DAY + 1}"
        + (f', note: "{_short(o.message, 200)}"' if o.message else "")
        for o in mp.offers
        if o.seller_id == aid and o.status == "pending"
    ]
    lines += _capped(rows, what="offers") or ["- none"]

    lines += ["", "## Your offers on others' listings (pending, or decided in the last 3 days)"]
    rows = [
        f"- {o.id} to {o.seller_id} on {o.listing_id} ({item_of.get(o.listing_id)}): {o.qty} kg @ "
        f"{_money(o.offered_price)}, net-{o.payment_terms_days}, status {o.status}, posted {format_min(o.posted_at)}"
        for o in mp.offers
        if o.buyer_id == aid and (o.status == "pending" or o.posted_at // MINUTES_PER_DAY >= today - 3)
    ]
    lines += _capped(rows, what="offers") or ["- none"]

    lines += ["", "## Deals in flight (accepted, not yet delivered)"]
    rows = []
    for d in env._pending_shipments:
        if aid not in (d.buyer_id, d.seller_id):
            continue
        inbound = d.buyer_id == aid
        rows.append(
            f"- {d.id} {'BUY from' if inbound else 'SELL to'} {d.seller_id if inbound else d.buyer_id}: "
            f"{d.qty} kg {d.item_id} @ {_money(d.unit_price)} = {_money(d.total_price)}, net-{d.payment_terms_days}, "
            f"accepted {format_min(d.deal_at)}, delivery day {d.delivery_at // MINUTES_PER_DAY}"
        )
    lines += _capped(rows, what="deals") or ["- none"]

    def inv_rows(invoices, side: str) -> list[str]:
        out = []
        open_inv = [inv for inv in invoices if inv.net_outstanding > 0.01]
        for inv in sorted(open_inv, key=lambda i: (i.due_date, i.id)):
            cp = inv.payer if side == "AR" else inv.issuer
            when = (
                f"OVERDUE {today - inv.due_date} d (due day {inv.due_date})"
                if inv.due_date < today
                else f"due day {inv.due_date} (in {inv.due_date - today} d)"
            )
            fee = f", late fees {_money(inv.late_fees_accrued)}" if inv.late_fees_accrued else ""
            out.append(
                f"- {inv.id} {'from' if side == 'AR' else 'to'} {cp}: {_money(inv.net_outstanding)} "
                f"outstanding, {when}{fee}, ref {inv.reference}"
            )
        return out

    lines += ["", "## Open payables (AP; pay_invoice — overdue ones accrue late fees daily)"]
    lines += _capped(inv_rows(ba.accounts_payable, "AP"), what="invoices") or ["- none"]
    lines += ["", "## Open receivables (AR)"]
    lines += _capped(inv_rows(ba.accounts_receivable, "AR"), what="invoices") or ["- none"]
    return lines


def _history(env, ba, today: int) -> list[str]:
    mp = env.marketplace
    aid = ba.agent_id
    since = max(0, today - RECENT_DAYS)
    lines = [f"## Your B2B deals since day {since}"]
    rows = []
    for d in mp.deals:
        if aid not in (d.buyer_id, d.seller_id) or d.deal_at // MINUTES_PER_DAY < since:
            continue
        buy = d.buyer_id == aid
        state = "cancelled" if d.cancelled_day is not None else (
            "returned %d kg" % d.returned_qty if d.returned_qty else ""
        )
        rows.append(
            f"- day {d.deal_at // MINUTES_PER_DAY} {d.id} {'BUY from' if buy else 'SELL to'} "
            f"{d.seller_id if buy else d.buyer_id}: {d.qty} kg {d.item_id} @ {_money(d.unit_price)}"
            + (f" ({state})" if state else "")
        )
    n_all = sum(1 for d in mp.deals if aid in (d.buyer_id, d.seller_id))
    lines += _capped(rows[::-1], what="deals") or ["- none"]
    lines.append(f"- ({n_all} deals in total since day 0)")

    lines += ["", "## Market: others' open listings (public; view_listings for details)"]
    by_item: dict[str, list] = defaultdict(list)
    for lt in mp.listings:
        if lt.status == "open" and lt.seller_id != aid and lt.qty > 0:
            by_item[lt.item_id].append(lt)
    if not by_item:
        lines.append("- none")
    for item, lts in sorted(by_item.items()):
        lts.sort(key=lambda x: (x.asking_price, x.id))
        cheapest = "; ".join(f"{x.id} {x.seller_id} {x.qty} kg @ {_money(x.asking_price)}" for x in lts[:3])
        lines.append(f"- {item}: {len(lts)} listing(s); cheapest: {cheapest}")

    lines += ["", f"## Consumer market, all retailers (public aggregate; days {since}–{today - 1})"]
    agg: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    own: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for s in env.consumer_sales_log:
        d = int(s.get("day", -1))
        if d < since:
            continue
        for book in ([agg, own] if s.get("shop_id") == aid else [agg]):
            row = book[str(s.get("item_id"))][d]
            row[0] += int(s.get("qty", 0) or 0)
            row[1] += float(s.get("total_price", 0.0) or 0.0)

    def fmt_book(book) -> list[str]:
        out = []
        for item, days in sorted(book.items()):
            cells = ", ".join(
                f"d{d}: {q} kg @ {_money(r / q) if q else '-'}" for d, (q, r) in sorted(days.items())
            )
            out.append(f"- {item}: {cells}")
        return out or ["- no sales"]

    lines += fmt_book(agg)
    if ba.role == "retailer":
        lines += ["", "## Your own consumer sales (same window)"]
        lines += fmt_book(own)
    return lines


def _calendar(env, today: int) -> list[str]:
    lines = ["## Calendar"]
    lines.append(f"- today: day {today} ({date_label(today)}); the run ends after day {env.max_days - 1}")
    for start, end, _mult, name in getattr(env, "festivals", FESTIVALS):
        if start >= env.max_days:
            continue
        if today < start:
            status = f"starts in {start - today} d"
        elif today <= end:
            status = f"ACTIVE (day {today - start + 1} of {end - start + 1})"
        else:
            status = "over"
        lines.append(f"- festival {name}: days {start}–{end} — {status}")
    from coffeebench.mechanics import advisory_line

    advisory = advisory_line(getattr(env, "mechanics", None), today)
    if advisory:
        lines.append(f"- {advisory}")
    bust = sorted(env._bankrupt_agents)
    if bust:
        lines.append(
            "- bankrupt participants (no further trading): "
            + ", ".join(f"{a} (day {env._bankrupt_day.get(a)})" for a in bust)
        )
    return lines


def _dm_blocks(env, aid: str) -> list[tuple[int, str, list[str]]]:
    """[(sort_key, header, [message lines oldest→newest])] per counterparty."""
    mp = env.marketplace
    read = mp._read_message_ids.get(aid, set())
    threads: dict[str, list] = defaultdict(list)
    for m in mp.messages:
        if m.sender_id == aid:
            threads[m.recipient_id].append(m)
        elif m.recipient_id == aid:
            threads[m.sender_id].append(m)
    out = []
    for cp, msgs in sorted(threads.items()):
        unread = sum(1 for m in msgs if m.recipient_id == aid and m.id not in read)
        header = f"### with {cp}: {len(msgs)} message(s), {unread} unread by you"
        lines = []
        for m in msgs:
            arrow = f"you → {cp}" if m.sender_id == aid else f"{cp} → you"
            refs = [f"thread {m.thread_id or m.id}"]
            if m.reply_to_message_id:
                refs.append(f"reply to {m.reply_to_message_id}")
            for k in ("ref_listing_id", "ref_offer_id", "ref_deal_id"):
                if getattr(m, k, None):
                    refs.append(f"{k[4:]} {getattr(m, k)}")
            body = m.body if len(m.body) <= DM_BODY_CHARS else m.body[:DM_BODY_CHARS] + "…[truncated]"
            lines.append(
                f"- [{m.id}] {format_min(m.sent_at)} {arrow} ({', '.join(refs)})\n"
                f"  title: {m.title}\n  body: {body}"
            )
        out.append((len(msgs), header, lines))
    return out


def _action_lines(steps: list[dict]) -> list[str]:
    out = []
    for ev in steps:
        args = ev.get("action_input") or {}
        arg_s = _short(args, 200) if args else ""
        out.append(
            f"- {format_min(int(ev.get('at', 0)))} {ev.get('action')}"
            + (f" {arg_s}" if arg_s else "")
            + f" → {_summarize_result(str(ev.get('action')), ev.get('observation'))}"
        )
    return out


def _fit(lines: list[str], budget: int, what: str) -> list[str]:
    """Keep the newest lines that fit in `budget` chars (oldest dropped)."""
    kept: list[str] = []
    used = 0
    for ln in reversed(lines):
        if used + len(ln) + 1 > budget:
            break
        kept.append(ln)
        used += len(ln) + 1
    dropped = len(lines) - len(kept)
    kept.reverse()
    if dropped:
        kept.insert(0, f"- … {dropped} older {what} omitted (size bound)")
    return kept


def build_digest(env, aid: str, steps: list[dict] | None = None, max_chars: int = MAX_CHARS) -> str:
    """Digest for agent `aid` at the env's current day (a segment start)."""
    ba = env.business_apps[aid]
    today = env.time_manager.get_current_day()
    since = max(0, today - RECENT_DAYS)
    head = [
        f"# RESUMED SESSION — digest at the start of day {today} ({date_label(today)})",
        "",
        "The run was paused; your previous working session ended at the start of "
        "today and this is a NEW session with a fresh context. Your earlier "
        "conversation and terminal history are gone; files you wrote under /workspace "
        "were kept (/workspace/out may be dropped if over 20 MB). Below is a "
        "deterministic summary generated by the "
        "simulator from the books and the event log (the same format for every "
        "participant). Treat it as your notes; use `coffee view_*` tools for details. "
        "Commitments you made or received live only in the DM threads below.",
        "",
    ]
    fixed = head + _position(env, ba, today) + [""] + _market_book(env, ba, today)
    fixed += [""] + _history(env, ba, today) + [""] + _calendar(env, today)
    fixed_text = "\n".join(fixed)

    # Flexible parts: DM threads + recent actions, oldest trimmed first.
    budget = max(0, max_chars - len(fixed_text) - 400)
    dm = _dm_blocks(env, aid)
    acts = _action_lines(steps or [])
    dm_budget = int(budget * 0.6) if acts else budget
    act_budget = budget - dm_budget

    dm_lines = ["", f"## DM threads (latest {DM_PER_THREAD} per counterparty; read_message / view_messages for more)"]
    if not dm:
        dm_lines.append("- no messages yet")
    else:
        per_thread = max(1, dm_budget // len(dm))
        for _n, header, lines in dm:
            older = max(0, len(lines) - DM_PER_THREAD)
            body = _fit(lines[-DM_PER_THREAD:], per_thread - len(header) - 60, "messages in this thread")
            if older:
                body.insert(0, f"- … {older} older messages in this thread not shown")
            dm_lines += [header] + body
    dm_text = "\n".join(dm_lines)
    act_budget += max(0, dm_budget - len(dm_text))

    act_lines = ["", f"## Your actions since day {since} (oldest first; compact results)"]
    act_lines += _fit(acts, act_budget, "actions") if acts else ["- none"]
    text = fixed_text + "\n" + dm_text + "\n" + "\n".join(act_lines) + "\n"
    if len(text) > max_chars:  # fixed sections alone exceed the bound
        text = text[: max_chars - 60] + "\n… [digest truncated at the size bound]\n"
    return text


def build_digests(env, agent_ids, events_path: str | None, max_chars: int = MAX_CHARS) -> dict[str, str]:
    today = env.time_manager.get_current_day()
    steps = read_agent_steps(events_path, max(0, today - RECENT_DAYS))
    return {aid: build_digest(env, aid, steps.get(aid, []), max_chars) for aid in agent_ids}
