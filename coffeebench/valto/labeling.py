"""Tier-1 outcome-attribution ("labeling") pass.

Runs ONCE at run-end. For every governed Valto decision it joins the decision to
its realized consequences drawn from the ENVIRONMENT's truth ledger (the
authoritative, agent-independent ground truth — never the agent's self-report),
computes structured realized-outcome metrics + a waste-based utility, and writes
them back onto the same ledger row via `valto.evaluate(..., metrics=...)`.

Correctness is enforced three ways (see VALTO_EXPERIMENT_PLAN.md §10 / the design
discussion):

  1. Ground truth only    — we read env.truth_ledger / marketplace.deals, not the
                            agent's claims.
  2. Explicit join keys   — a buy_green joins to its deal via the logged offer_id;
                            spoilage joins to the buys/roasts that originated the
                            item, windowed to [d-W, d].
  3. Reconciliation       — Σ attributed waste is checked against the agent's total
                            realized waste (spoilage) over the run; the residual
                            (spoilage we could NOT attribute to an in-window
                            decision) is reported, so we always know how much of
                            the outcome signal is trustworthy.

Tier 1 is deliberately a *proxy*: CoffeeBench uses WAVG cost basis (no lot
tracking), so item spoilage is apportioned across the buys/roasts that could have
caused it rather than tracked per unit. Exact per-decision provenance is Tier 2
(decision-id lot tags threaded through the sim) — deferred until a published ROI
number is needed.
"""

from __future__ import annotations

from collections import defaultdict

DEFAULT_WINDOW_DAYS = 14


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def label_run(env, window_days: int = DEFAULT_WINDOW_DAYS, verbose: bool = True) -> dict:
    """Attribute realized outcomes to every governed decision and write them
    back to the Valto ledger. Returns a reconciliation/coverage report."""
    valto = getattr(env, "_valto", None)
    actor = getattr(env, "_valto_actor", None)
    if valto is None or actor is None:
        return {"labeled": 0, "note": "no Valto client on this run"}

    rows = valto.decisions(limit=1_000_000)
    truth = env.truth_ledger.get(actor, [])
    deals_by_offer = {d.offer_id: d for d in env.marketplace.deals}
    deals_by_id = {d.id: d for d in env.marketplace.deals}
    deals_by_listing: dict[str, list] = defaultdict(list)
    for d in env.marketplace.deals:
        deals_by_listing[d.listing_id].append(d)
    # Invoices the actor ISSUED (its receivables) → realized-payment outcomes on
    # its sales, keyed by the originating deal id.
    ba = env.business_apps.get(actor)
    ar_by_deal = {
        inv.reference: inv
        for inv in (getattr(ba, "accounts_receivable", []) if ba else [])
        if inv.reference
    }

    # Deliveries the actor actually received, keyed by the originating deal id.
    delivered_refs = {
        e.reference
        for e in truth
        if e.entry_type == "inventory_in" and e.reference
    }

    # --- index the governed decisions we can attribute to ---
    buys: list[dict] = []    # buy_green
    roasts: list[dict] = []  # roast
    lists: list[dict] = []   # list_roasted (post_listing)
    sells: list[dict] = []   # sell_roasted (accept_offer)
    for r in rows:
        ctx = r.get("context") or {}
        met = r.get("metrics") or {}
        if r["action"] == "buy_green":
            buys.append(
                {
                    "id": r["id"], "day": ctx.get("day"), "item": ctx.get("item"),
                    "qty": int(ctx.get("qty", 0) or 0),
                    "offered_price": float(ctx.get("offered_price", 0.0) or 0.0),
                    "median": ctx.get("market_median_price"),
                    "offer_id": met.get("offer_id"), "amount": r.get("amount") or 0.0,
                }
            )
        elif r["action"] == "roast":
            roasts.append(
                {
                    "id": r["id"], "day": ctx.get("day"),
                    "output": ctx.get("output_item"),
                    "qty": int(ctx.get("qty_kg", 0) or 0),
                    "amount": r.get("amount") or 0.0,
                }
            )
        elif r["action"] == "list_roasted":
            lists.append(
                {
                    "id": r["id"], "day": ctx.get("day"),
                    "listing_id": met.get("listing_id"),
                    "ask": float(ctx.get("asking_price", 0.0) or 0.0),
                    "listed_qty": int(ctx.get("qty", 0) or 0),
                    "median": ctx.get("market_median_price"),
                }
            )
        elif r["action"] == "sell_roasted":
            sells.append(
                {
                    "id": r["id"], "day": ctx.get("day"),
                    "deal_id": met.get("deal_id"),
                    "total_margin": float(ctx.get("total_margin", 0.0) or 0.0),
                    "median": ctx.get("market_median_price"),
                    "bid": float(ctx.get("bid_price", 0.0) or 0.0),
                }
            )

    # --- windowed spoilage apportionment (the proxy join) ---
    # spoilage of item X on day d is attributed to the buys (for green items) or
    # roasts (for roasted items) of X within (d-W, d], weighted by qty.
    attributed_spoil: dict[str, float] = defaultdict(float)
    total_green_spoil = 0.0
    total_roasted_spoil = 0.0
    unattributed_spoil = 0.0
    for e in truth:
        if e.entry_type != "spoilage_expense":
            continue
        item = e.item or ""
        val = float(e.amount or 0.0)
        d = int(e.day)
        if item.startswith("green"):
            total_green_spoil += val
            origin = buys
            key = "item"
        elif item.startswith("roasted"):
            total_roasted_spoil += val
            origin = roasts
            key = "output"
        else:
            continue
        cands = [
            o for o in origin
            if o[key] == item and o["day"] is not None
            and (d - window_days) < int(o["day"]) <= d
        ]
        tot_qty = sum(o["qty"] for o in cands)
        if tot_qty > 0:
            for o in cands:
                attributed_spoil[o["id"]] += val * (o["qty"] / tot_qty)
        else:
            unattributed_spoil += val

    # --- per-decision metrics + utility, written back via evaluate() ---
    total_overpay = 0.0
    n_wasteful = 0
    labeled = 0

    for b in buys:
        deal = deals_by_offer.get(b["offer_id"]) if b["offer_id"] else None
        dealt = deal is not None
        delivered = bool(deal and deal.id in delivered_refs)
        median = b["median"]
        overpay = (
            round(max(0.0, b["offered_price"] - float(median)) * b["qty"], 2)
            if median else 0.0
        )
        total_overpay += overpay
        spoil = round(attributed_spoil.get(b["id"], 0.0), 2)
        waste = round(overpay + spoil, 2)
        utility = round(-waste, 2)
        spend = float(b["amount"]) or 1.0
        metrics = {
            "dealt": dealt,
            "delivered": delivered,
            "actual_ap": round(float(deal.total_price), 2) if deal else 0.0,
            "overpay_premium": overpay,
            "attributed_spoilage_w": spoil,
            "waste": waste,
            "utility": utility,
            "window_days": window_days,
        }
        useful = utility >= 0.0
        if not useful:
            n_wasteful += 1
        valto.evaluate(b["id"], useful=useful,
                       score=_clamp01(1.0 - waste / spend),
                       note=f"waste=${waste:.2f} (overpay ${overpay:.2f} + spoilage ${spoil:.2f})",
                       metrics=metrics)
        labeled += 1

    for r in roasts:
        spoil = round(attributed_spoil.get(r["id"], 0.0), 2)
        waste = spoil
        utility = round(-waste, 2)
        spend = float(r["amount"]) or 1.0
        metrics = {
            "output_item": r["output"],
            "attributed_spoilage_w": spoil,
            "waste": waste,
            "utility": utility,
            "window_days": window_days,
        }
        useful = utility >= 0.0
        if not useful:
            n_wasteful += 1
        valto.evaluate(r["id"], useful=useful,
                       score=_clamp01(1.0 - waste / spend),
                       note=f"attributed roasted spoilage ${spoil:.2f}",
                       metrics=metrics)
        labeled += 1

    # --- SELL-SIDE: list_roasted (post_listing) realized fill + price capture ---
    total_left_on_table = 0.0
    for lt in lists:
        lid = lt["listing_id"]
        ds = deals_by_listing.get(lid, []) if lid else []
        sold_qty = sum(int(d.qty) for d in ds)
        listed = lt["listed_qty"] or 0
        realized_rev = sum(float(d.total_price) for d in ds)
        realized_avg = round(realized_rev / sold_qty, 4) if sold_qty else None
        fill_rate = round(sold_qty / listed, 3) if listed else None
        first_sale_day = min((d.deal_at // 1440 for d in ds), default=None)
        days_to_first_sale = (first_sale_day - lt["day"]) if first_sale_day is not None and lt["day"] is not None else None
        median = lt["median"]
        # Opportunity cost: if the ask (and thus realized price) sat below the
        # market median, that's margin left on the table on the sold qty.
        left = (
            round(max(0.0, float(median) - realized_avg) * sold_qty, 2)
            if median and realized_avg else 0.0
        )
        total_left_on_table += left
        metrics = {
            "sold_qty": sold_qty,
            "listed_qty": listed,
            "fill_rate": fill_rate,
            "realized_avg_price": realized_avg,
            "days_to_first_sale": days_to_first_sale,
            "price_vs_market": round(realized_avg / median, 3) if (median and realized_avg) else None,
            "margin_left_on_table": left,
            "window_days": window_days,
        }
        valto.evaluate(lt["id"], useful=(left <= 0.0),
                       score=_clamp01(1.0 - left / (lt["ask"] * listed)) if (lt["ask"] and listed) else None,
                       note=f"fill {fill_rate}, realized ${realized_avg}/kg, left ${left:.2f} vs market",
                       metrics=metrics)
        labeled += 1

    # --- SELL-SIDE: sell_roasted (accept_offer) realized margin + payment ---
    total_sell_margin = 0.0
    for s in sells:
        deal = deals_by_id.get(s["deal_id"]) if s["deal_id"] else None
        inv = ar_by_deal.get(s["deal_id"]) if s["deal_id"] else None
        paid = bool(inv and inv.paid)
        bad_debt = bool(inv and getattr(inv, "bad_debt", False))
        margin = s["total_margin"]
        total_sell_margin += margin
        median = s["median"]
        below_market = bool(median and s["bid"] < float(median))
        metrics = {
            "realized_total_margin": round(margin, 2),
            "buyer_paid": paid,
            "became_bad_debt": bad_debt,
            "sold_below_market": below_market,
            "bid_vs_market": round(s["bid"] / median, 3) if median else None,
            "window_days": window_days,
        }
        # A sale is "useful" if it cleared a positive margin and didn't go bad.
        valto.evaluate(s["id"], useful=(margin > 0 and not bad_debt),
                       score=None,
                       note=f"realized margin ${margin:.2f}, paid={paid}, bad_debt={bad_debt}",
                       metrics=metrics)
        labeled += 1

    total_spoil = total_green_spoil + total_roasted_spoil
    attributed_total = sum(attributed_spoil.values())
    report = {
        "actor": actor,
        "window_days": window_days,
        "labeled_decisions": labeled,
        "buys": len(buys),
        "roasts": len(roasts),
        "lists": len(lists),
        "sells": len(sells),
        "wasteful_decisions": n_wasteful,
        "total_sell_margin": round(total_sell_margin, 2),
        "total_margin_left_on_table": round(total_left_on_table, 2),
        "total_overpay": round(total_overpay, 2),
        "total_spoilage": round(total_spoil, 2),
        "attributed_spoilage": round(attributed_total, 2),
        "unattributed_spoilage": round(unattributed_spoil, 2),
        # Reconciliation: what fraction of realized spoilage we could tie to an
        # in-window decision. Low coverage ⇒ the outcome signal is weak/suspect.
        "spoilage_attribution_coverage": (
            round(attributed_total / total_spoil, 3) if total_spoil > 0 else None
        ),
    }
    if verbose:
        _print_report(report)
    return report


def _print_report(r: dict) -> None:
    print("[valto-label] " + "=" * 58)
    print(f"[valto-label] actor={r['actor']} window={r['window_days']}d")
    print(
        f"[valto-label] labeled {r['labeled_decisions']} decisions "
        f"({r['buys']} buys, {r['roasts']} roasts, {r.get('lists', 0)} lists, "
        f"{r.get('sells', 0)} sells)"
    )
    print(
        f"[valto-label] sell side: realized margin ${r.get('total_sell_margin', 0)}, "
        f"margin left on table (under-priced) ${r.get('total_margin_left_on_table', 0)}"
    )
    print(f"[valto-label] buy side: total overpay attributed: ${r['total_overpay']}")
    cov = r["spoilage_attribution_coverage"]
    print(
        f"[valto-label] spoilage reconciliation: attributed "
        f"${r['attributed_spoilage']} / ${r['total_spoilage']} total "
        f"(coverage {cov if cov is not None else 'n/a'}), "
        f"unattributed ${r['unattributed_spoilage']}"
    )
    print("[valto-label] " + "=" * 58)
