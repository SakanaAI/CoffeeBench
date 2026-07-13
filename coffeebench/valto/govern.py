"""Wire Valto in front of the focal roaster's capital-committing tools.

This governs ONLY the focal agent (e.g. roaster_A); the other five agents never
touch Valto, so a governed-vs-ungoverned comparison isolates the focal agent.

The roaster's economic decisions are wrapped via `valto.protect_call`. Which
tools are wrapped is configurable (`[valto].govern_tools`); the full Option-1
set covers both sides of the margin engine:

  - `make_offer`   → **buy green** (procurement price/qty). action = "buy_green".
  - `roast`        → **produce** (volume + commodity/specialty mix). action = "roast".
  - `post_listing` → **list roasted at an ASK price** (sets the sell price).
                     action = "list_roasted". THE margin lever.
  - `accept_offer` → **accept a retailer bid = execute a SALE**. action =
                     "sell_roasted". Realizes the margin.

In **observe** mode the policy approves everything, so behavior is identical to
an ungoverned run — but every decision is logged with a RICH, observable-only
decision-time `context`, the agent's actual stated `intent`, and structured
outcome `metrics`. That ledger is the raw data for offline analysis / policy
mining. In **governed** mode a learned policy can deny / modify, and the block is
handed back to the agent as an ordinary rejected-tool result so it re-plans.

Nothing here changes the tool schema the model sees: the wrapper uses
`functools.wraps`, so `inspect.signature` (and thus `ToolSpec.from_function`)
still reads the original bound method's signature + docstring.
"""

from __future__ import annotations

import functools
import statistics
from pathlib import Path
from typing import Any

from valto import ProtectedToolSpec, Valto

# Roaster economics (kept in sync with environment.ROAST_RECIPES).
_ROAST = {
    "green_coffee_kg": {"output": "roasted_coffee_kg", "labor": 3.0, "yield": 0.85},
    "green_specialty_kg": {"output": "roasted_specialty_kg", "labor": 5.0, "yield": 0.82},
}
# spring_break festival window (environment.FESTIVALS).
_FESTIVAL_START, _FESTIVAL_END = 40, 53
_ROASTER_DAILY_OPEX = 30.0
_MEDIAN_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Small observable-only helpers over the focal BusinessApp (`ba`)
# ---------------------------------------------------------------------------
def _day(ba) -> int:
    return int(ba.time_manager.get_current_day())


def _days_to_festival(day: int) -> int:
    return max(0, _FESTIVAL_START - day)


def _in_festival(day: int) -> int:
    return 1 if _FESTIVAL_START <= day <= _FESTIVAL_END else 0


def _cash_runway_days(ba) -> float:
    # Days of known fixed opex the current cash covers (a crude solvency gauge).
    return round(float(ba.cash) / _ROASTER_DAILY_OPEX, 1) if _ROASTER_DAILY_OPEX else 0.0


def _rolling_median_price(ba, item_id: str, day: int) -> float | None:
    """Median accepted unit price for `item_id` over the last N days across the
    whole market — the benchmark a bid/ask should be measured against. Uses only
    closed deals (public, observable via view_deals / view_market_aggregate).
    Works for green (procurement benchmark) AND roasted (sell-price benchmark)."""
    prices = [
        float(d.unit_price)
        for d in ba.marketplace.deals
        if d.item_id == item_id and (day - (d.deal_at // 1440)) <= _MEDIAN_WINDOW_DAYS
    ]
    return round(statistics.median(prices), 4) if prices else None


def _counterparty_buy_count(ba, seller_id: str) -> int:
    """How many times WE (as buyer) have dealt with this seller."""
    return sum(
        1
        for d in ba.marketplace.deals
        if d.seller_id == seller_id and d.buyer_id == ba.agent_id
    )


def _counterparty_sell_count(ba, buyer_id: str) -> int:
    """How many times WE (as seller) have dealt with this buyer."""
    return sum(
        1
        for d in ba.marketplace.deals
        if d.buyer_id == buyer_id and d.seller_id == ba.agent_id
    )


def _listing(ba, listing_id: str):
    return next((lt for lt in ba.marketplace.listings if lt.id == listing_id), None)


def _offer(ba, offer_id: str):
    return next((o for o in ba.marketplace.offers if o.id == offer_id), None)


def _held(ba, item_id: str) -> int:
    return int(ba.inventory.get(item_id, 0) or 0)


def _cost_basis(ba, item_id: str) -> float:
    """WAVG unit cost of an item currently held (the sell-margin reference)."""
    return round(float(ba.cost_basis.get(item_id, 0.0) or 0.0), 4)


def _committed_kg(ba) -> int:
    return ba._total_inventory_kg() + ba._pending_inbound_kg()


def _agent_intent(ba) -> str:
    """The focal agent's ACTUAL stated rationale for the action about to run.

    The CoffeeBench prompt requires the agent to state one sentence of intent
    before each tool call. By the time a protected tool executes, that assistant
    turn is `agent.messages[-1]` (step_apply appends it before running the tool),
    so we can capture the real reasoning (content + any extended-thinking summary)
    instead of a generic template. Empty for scripted models (no reasoning)."""
    env = getattr(ba, "_env", None)
    if env is None:
        return ""
    agent = getattr(env, "agents", {}).get(ba.agent_id)
    if agent is None or not getattr(agent, "messages", None):
        return ""
    last = agent.messages[-1]
    if last.get("role") != "assistant":
        return ""
    parts: list[str] = []
    think = (last.get("thinking") or "").strip()
    content = (last.get("content") or "").strip()
    if think:
        parts.append("[thinking] " + think[:600])
    if content:
        parts.append(content[:600])
    return " | ".join(parts)[:1200]


# ---------------------------------------------------------------------------
# Decision-time context extractors (the RICH, observable-only metrics)
# ---------------------------------------------------------------------------
def _buy_green_context(args: dict, ba) -> dict[str, Any]:
    day = _day(ba)
    lst = _listing(ba, args.get("listing_id", ""))
    item_id = lst.item_id if lst is not None else None
    seller = lst.seller_id if lst is not None else None
    price = float(args.get("offered_price", 0.0) or 0.0)
    qty = int(args.get("qty", 0) or 0)
    median = _rolling_median_price(ba, item_id, day) if item_id else None
    cap = ba._inventory_cap_kg()
    committed = _committed_kg(ba)
    return {
        # env-day bucket — Valto uses context["day"] for budget/duplicate windows
        "day": day,
        "days_to_festival": _days_to_festival(day),
        "in_festival": _in_festival(day),
        "item": item_id,
        "seller": seller,
        "offered_price": price,
        "qty": qty,
        "spend": round(price * qty, 2),
        "payment_terms_days": int(args.get("payment_terms_days", 30) or 30),
        "market_median_price": median,
        "price_to_median": round(price / median, 3) if median else None,
        "green_on_hand": _held(ba, item_id) if item_id else 0,
        "roasted_on_hand": _held(ba, "roasted_coffee_kg") + _held(ba, "roasted_specialty_kg"),
        "committed_kg": committed,
        "cap_headroom_kg": max(0, cap - committed),
        "cash": round(float(ba.cash), 2),
        "cash_runway_days": _cash_runway_days(ba),
        "outstanding_ap": round(ba._ap_outstanding(), 2),
        "counterparty_prior_deals": _counterparty_buy_count(ba, seller) if seller else 0,
    }


def _roast_context(args: dict, ba) -> dict[str, Any]:
    day = _day(ba)
    green = args.get("green_item_id", "")
    qty = int(args.get("qty_kg", 0) or 0)
    recipe = _ROAST.get(green, {})
    output = recipe.get("output")
    cap = ba._inventory_cap_kg()
    committed = _committed_kg(ba)
    used_today = int((getattr(ba._env, "_roast_used_today", {}) or {}).get(ba.agent_id, 0))
    return {
        "day": day,
        "days_to_festival": _days_to_festival(day),
        "in_festival": _in_festival(day),
        "input_item": green,
        "output_item": output,
        "qty_kg": qty,
        "labor_spend": round(qty * float(recipe.get("labor", 0.0)), 2),
        "green_on_hand": _held(ba, green),
        "roasted_on_hand": _held(ba, output) if output else 0,
        "committed_kg": committed,
        "cap_headroom_kg": max(0, cap - committed),
        "roast_cap_used_today": used_today,
        "roast_cap_remaining_today": max(0, 50 - used_today),
        "cash": round(float(ba.cash), 2),
        "cash_runway_days": _cash_runway_days(ba),
    }


def _list_roasted_context(args: dict, ba) -> dict[str, Any]:
    """`post_listing(item_id, qty, asking_price, payment_terms_days)` — the roaster
    setting its SELL price for roasted output. THE margin lever: ask vs the market
    median and the margin the ask implies over WAVG cost basis."""
    day = _day(ba)
    item_id = args.get("item_id")
    ask = float(args.get("asking_price", 0.0) or 0.0)
    qty = int(args.get("qty", 0) or 0)
    cost = _cost_basis(ba, item_id) if item_id else 0.0
    median = _rolling_median_price(ba, item_id, day) if item_id else None
    on_hand = _held(ba, item_id) if item_id else 0
    return {
        "day": day,
        "days_to_festival": _days_to_festival(day),
        "in_festival": _in_festival(day),
        "item": item_id,
        "asking_price": ask,
        "qty": qty,
        "list_value": round(ask * qty, 2),
        "payment_terms_days": int(args.get("payment_terms_days", 30) or 30),
        "cost_basis": cost,
        # Margin the ask implies, and how much cushion above break-even.
        "ask_margin_per_kg": round(ask - cost, 4),
        "ask_markup": round(ask / cost, 3) if cost > 0 else None,
        # The KEY signal: ask relative to what the market is actually paying.
        "market_median_price": median,
        "ask_to_median": round(ask / median, 3) if median else None,
        "roasted_on_hand": on_hand,
        "cash": round(float(ba.cash), 2),
        "cash_runway_days": _cash_runway_days(ba),
    }


def _sell_roasted_context(args: dict, ba) -> dict[str, Any]:
    """`accept_offer(offer_id)` — accepting a retailer's bid = executing a SALE of
    roasted output. Realizes the margin; the decision is whether the bid price
    clears an acceptable margin vs cost and vs the market."""
    day = _day(ba)
    off = _offer(ba, args.get("offer_id", ""))
    lst = _listing(ba, off.listing_id) if off is not None else None
    item_id = lst.item_id if lst is not None else None
    bid = float(off.offered_price) if off is not None else 0.0
    qty = int(off.qty) if off is not None else 0
    buyer = off.buyer_id if off is not None else None
    cost = _cost_basis(ba, item_id) if item_id else 0.0
    median = _rolling_median_price(ba, item_id, day) if item_id else None
    return {
        "day": day,
        "days_to_festival": _days_to_festival(day),
        "in_festival": _in_festival(day),
        "item": item_id,
        "buyer": buyer,
        "bid_price": bid,
        "qty": qty,
        "sale_value": round(bid * qty, 2),
        "payment_terms_days": int(off.payment_terms_days) if off is not None else 30,
        "cost_basis": cost,
        "margin_per_kg": round(bid - cost, 4),
        "total_margin": round((bid - cost) * qty, 2),
        "below_cost": bool(cost > 0 and bid < cost),
        "market_median_price": median,
        "bid_to_median": round(bid / median, 3) if median else None,
        "roasted_on_hand": _held(ba, item_id) if item_id else 0,
        "cash": round(float(ba.cash), 2),
        "counterparty_prior_deals": _counterparty_sell_count(ba, buyer) if buyer else 0,
    }


# ---------------------------------------------------------------------------
# Execution-time outcome metrics (structured, from the tool RESULT)
# ---------------------------------------------------------------------------
def _outcome_evaluator(action: str, amount: float | None, result: Any) -> dict | None:
    """Called right after an approved tool runs. Records what the action
    produced at execution time (facts, not yet realized economics — realized
    outcome attribution is a separate deferred labeling pass). Never raises."""
    if not isinstance(result, dict):
        return None
    ok = result.get("status") == "success"
    metrics: dict[str, Any] = {"executed_ok": bool(ok), "action": action}
    if action == "buy_green" and ok:
        metrics.update(
            {
                "offer_id": result.get("offer_id"),
                "seller": result.get("seller_id"),
            }
        )
    elif action == "roast" and ok:
        metrics.update(
            {
                "roasted_output_kg": result.get("roasted_output_kg"),
                "labor_cost": result.get("labor_cost"),
                "ready_day": result.get("ready_day"),
                "roasted_unit_cost_estimate": result.get("roasted_unit_cost_estimate"),
            }
        )
    elif action == "list_roasted" and ok:
        metrics.update(
            {
                "listing_id": result.get("listing_id"),
                "listed_ask": result.get("asking_price"),
                "listed_qty": result.get("qty"),
            }
        )
    elif action == "sell_roasted" and ok:
        metrics.update(
            {
                "deal_id": result.get("deal_id"),
                "buyer": result.get("buyer_id"),
                "sold_qty": result.get("qty"),
                "sold_unit_price": result.get("unit_price"),
                "sold_total": result.get("total_price"),
            }
        )
    elif not ok:
        # e.g. env rejected (cap/cash/capacity) — a useful negative signal.
        metrics["reject_reason"] = str(result.get("message", ""))[:200]
    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# Tool specs + wrapping
# ---------------------------------------------------------------------------
def _intent(ba, fallback: str) -> str:
    """The agent's real stated rationale, or a template fallback (scripted)."""
    real = _agent_intent(ba)
    return real or fallback


def _make_specs() -> dict[str, ProtectedToolSpec]:
    return {
        "make_offer": ProtectedToolSpec(
            action="buy_green",
            amount=lambda args, ba: round(
                float(args.get("offered_price", 0.0) or 0.0) * int(args.get("qty", 0) or 0),
                2,
            ),
            payee=lambda args, ba: (
                (_listing(ba, args.get("listing_id", "")) or _Empty()).seller_id
            ),
            intent=lambda args, ba: _intent(
                ba,
                f"Buy {args.get('qty')} kg green @ ${args.get('offered_price')}/kg "
                f"on {args.get('listing_id')}",
            ),
            context=lambda args, ba: _buy_green_context(args, ba),
        ),
        "roast": ProtectedToolSpec(
            action="roast",
            amount=lambda args, ba: round(
                int(args.get("qty_kg", 0) or 0)
                * float(_ROAST.get(args.get("green_item_id", ""), {}).get("labor", 0.0)),
                2,
            ),
            payee=lambda args, ba: "self",
            intent=lambda args, ba: _intent(
                ba, f"Roast {args.get('qty_kg')} kg {args.get('green_item_id')}"
            ),
            context=lambda args, ba: _roast_context(args, ba),
        ),
        "post_listing": ProtectedToolSpec(
            action="list_roasted",
            amount=lambda args, ba: round(
                float(args.get("asking_price", 0.0) or 0.0) * int(args.get("qty", 0) or 0),
                2,
            ),
            payee=lambda args, ba: "market",
            intent=lambda args, ba: _intent(
                ba,
                f"List {args.get('qty')} kg {args.get('item_id')} @ "
                f"${args.get('asking_price')}/kg",
            ),
            context=lambda args, ba: _list_roasted_context(args, ba),
        ),
        "accept_offer": ProtectedToolSpec(
            action="sell_roasted",
            amount=lambda args, ba: (
                lambda o: round(float(o.offered_price) * int(o.qty), 2) if o else 0.0
            )(_offer(ba, args.get("offer_id", ""))),
            payee=lambda args, ba: (
                (_offer(ba, args.get("offer_id", "")) or _Empty()).buyer_id
            ),
            intent=lambda args, ba: _intent(
                ba, f"Accept sale offer {args.get('offer_id')}"
            ),
            context=lambda args, ba: _sell_roasted_context(args, ba),
        ),
    }


class _Empty:
    seller_id = None
    buyer_id = None


def _reshape_block(result: Any) -> Any:
    """Turn Valto's block dict into CoffeeBench's own tool-rejection shape so
    the agent reads it as an ordinary failed action and re-plans."""
    if isinstance(result, dict) and result.get("blocked_by") == "valto":
        out = {
            "status": "rejected",
            "message": f"[valto:{result.get('verdict')}] {result.get('reason', '')}",
            "verdict": result.get("verdict"),
        }
        if result.get("suggestion"):
            out["suggestion"] = result["suggestion"]
        return out
    return result


def make_valto(actor: str, mode: str, policy, db_path: str | None) -> Valto:
    """Build the embedded Valto client for the focal agent.

    `mode`: "observe" (approve-all, log-only) or "governed" (enforce `policy`).
    `policy`: path/dict; for observe it's the approve-all policy.
    `db_path`: SQLite ledger path (persisted so we can mine it later). None ⇒
    in-memory (nothing kept).
    """
    if db_path is not None:
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Fresh ledger per run (matches the event logger's truncate-on-run
        # semantics) so mining sees exactly one run's decisions, not an
        # accumulation across re-runs. Resumed/governed branches pass a
        # branch-specific db_path so they never collide.
        for suffix in ("", "-journal", "-wal", "-shm"):
            fp = Path(str(db_path) + suffix)
            if fp.exists():
                fp.unlink()
    return Valto.local(
        actor=actor,
        policy=policy,
        db_path=db_path or ":memory:",
        outcome_evaluator=_outcome_evaluator,
        # Governed tools already fail closed if a candidate action can't be
        # resolved; keep the run alive on any transient Valto hiccup.
        protected_fail_closed=True,
    )


def govern_business_app(valto: Valto, ba, tools: list[str]) -> list[str]:
    """Wrap the named bound methods of `ba` with Valto in place. MUST run before
    `_collect_tools(ba)` so the agent registers the wrapped versions. Returns the
    list of tools actually wrapped."""
    specs = _make_specs()
    wrapped: list[str] = []
    for name in tools:
        spec = specs.get(name)
        orig = getattr(ba, name, None)
        if spec is None or not callable(orig):
            continue

        def make_wrapper(orig=orig, spec=spec):
            @functools.wraps(orig)
            def wrapper(**kwargs):
                result = valto.protect_call(
                    spec=spec, args=kwargs, env=ba, run=lambda: orig(**kwargs)
                )
                return _reshape_block(result)

            return wrapper

        setattr(ba, name, make_wrapper())
        wrapped.append(name)
    return wrapped
