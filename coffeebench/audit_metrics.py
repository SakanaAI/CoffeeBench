"""Post-hoc trade-integrity metrics for a finished CoffeeBench run.

Detection only — nothing here changes env mechanics or the canonical
score (`audit.annual.true_net_income`). Given a run directory (or its
`run.events.jsonl`), computes:

  (i)   Round-trip / circular trades: a deal X→Y followed within
        `window_days` by a deal Y→X on the same item with qty within
        ±`qty_tol`. Each leg is annotated with its unit price vs the
        same-day median deal price for that item and vs the item's retail
        reservation price, its payment terms, and whether the seller was
        near its inventory cap (on-hand kg at the previous day's EOD ≥
        `near_cap_frac` × role cap; in-flight kg is not in the snapshot).
  (ii)  One-way stuffing: deals priced above the item's retail
        reservation price (consumer-facing items only — no downstream
        buyer can ever resell above it).
  (iii) An "audited NI" per agent = true NI − markdown of end-of-run goods
        (per lot, incl. goods in flight when run.json records them) to the lower
        of cost and a reference price − the uncollectible share of receivables
        from under-funded debtors (v2.1; stuffing excess on unpaid legs is
        reported as `unpaid_stuffing_excess`, not subtracted). The NI booked
        by flagged round-trip legs is reported (`round_trip_ni`) but NOT
        subtracted as well: a round trip only inflates NI through the
        bought-back lots' inflated cost basis — if those lots are later sold
        or spoil, their COGS / spoilage at the inflated cost already reverses
        the gain in the ledger; if they are still held at the end, the
        markdown removes exactly that inflation. Subtracting both counted the
        same inflation twice (fixed 2026-10-24). Known gap: a buy-back leg
        still undelivered at the end leaves the seller-side gain unmarked.

Reference prices for the inventory markdown (an upper bound on what a
kg can ever be realized for, i.e. a net-realizable-value ceiling):
  - consumer-facing (roasted) items: the retail reservation price p_res;
  - green items: p_res of the roasted output × roast yield − roast labor
    per kg (commodity 30 × 0.85 − 3 = $22.50/kg; specialty
    80 × 0.82 − 5 = $60.60/kg with default constants). Any green bought
    above this can never be roasted and retailed at a profit.
These ceilings are deliberately lenient, so the markdown only bites on
clearly inflated inventory (e.g. stock parked at stuffed prices).

Exact per-leg NI uses the truth ledger in the sibling `run.json`
(entries whose `reference` is the deal id). Without `run.json`, legs are
still flagged but their NI contribution is reported as unavailable (0),
and the markdown falls back to the last day_end snapshot (total
inventory value vs Σ qty × reference).

CLI: `python -m coffeebench.audit_metrics <run_dir | events.jsonl | parent_dir> ...`
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Signed contribution of each truth-ledger entry type to NI when it is
# tied to a deal via `reference` (seller sale / COGS / returns / transit
# loss; buyer-side purchase-return adjustments).
_DEAL_NI_SIGN = {
    "sale_revenue": +1.0,
    "sale_reversal": -1.0,
    "inventory_out": -1.0,
    "cogs_reversal": +1.0,
    "writedown": -1.0,
}

_DEFAULT_P_RES = {"roasted_coffee_kg": 30.0, "roasted_specialty_kg": 80.0}


def _resolve_paths(path: str | Path) -> tuple[Path, Path | None]:
    p = Path(path)
    events = p / "run.events.jsonl" if p.is_dir() else p
    run_json = events.parent / "run.json"
    return events, (run_json if run_json.exists() else None)


def _reference_prices(p_res: dict[str, float]) -> dict[str, float]:
    from coffeebench.environment import ROAST_RECIPES

    ref = dict(p_res)
    for green, rec in ROAST_RECIPES.items():
        out_pres = p_res.get(rec["output_item"])
        if out_pres is not None:
            ref[green] = round(
                out_pres * float(rec["yield"]) - float(rec["labor_cost_per_kg"]), 4
            )
    return ref


def _uncollectible_ar(run: dict, creditor: str) -> float:
    """Shortfall share of `creditor`'s open receivables from debtors whose
    assets (cash + inventory at cost + own open AR) do not cover their unpaid
    AP; 0 for invoices with a ledger allowance (`credit_loss_allowance`)."""
    books = run.get("agent_books") or {}
    bankrupt = set(run.get("bankrupt_agents") or [])
    total = 0.0
    for debtor, b in books.items():
        if debtor in bankrupt or debtor == creditor:
            continue

        def _open(inv):
            if inv.get("paid") or inv.get("returned") or inv.get("bad_debt"):
                return 0.0
            return max(0.0, inv["amount"] + inv.get("late_fees_accrued", 0.0) - inv.get("credited_amount", 0.0))

        owed = sum(_open(i) for i in b.get("accounts_payable") or [])
        if owed <= 0:
            continue
        assets = (float(b.get("cash", 0.0)) + sum(float(v) for v in (b.get("inventory_total_cost") or {}).values())
                  + sum(_open(i) for i in b.get("accounts_receivable") or []))
        ratio = min(1.0, max(0.0, assets) / owed)
        if ratio >= 1.0:
            continue
        for inv in b.get("accounts_payable") or []:
            if inv.get("issuer") == creditor and not inv.get("credit_loss_allowance"):
                total += _open(inv) * (1.0 - ratio)
    return total


def compute_audit_metrics(
    path: str | Path,
    window_days: int = 14,
    qty_tol: float = 0.20,
    near_cap_frac: float = 0.90,
) -> dict:
    from coffeebench.environment import ROLE_INVENTORY_CAP_KG

    events_path, run_json_path = _resolve_paths(path)
    deals: list[dict] = []
    cancelled: set[str] = set()
    lost: set[str] = set()
    delivered: set[str] = set()
    snapshots: dict[int, dict] = {}
    run_end: dict = {}
    with open(events_path) as f:
        for line in f:
            ev = json.loads(line)
            t = ev.get("type")
            if t == "deal_accepted":
                ev["day"] = int(ev["deal_at"]) // 1440
                deals.append(ev)
            elif t == "deal_delivered":
                delivered.add(ev["deal_id"])
            elif t == "shipment_lost":
                lost.add(ev["deal_id"])
            elif t == "deal_cancelled":
                cancelled.add(ev["deal_id"])
            elif t == "day_end":
                snapshots[int(ev["day"])] = ev.get("snapshot") or {}
            elif t == "run_end":
                run_end = ev
    deals.sort(key=lambda d: (d["deal_at"], d["deal_id"]))

    run = json.loads(run_json_path.read_text()) if run_json_path else None
    p_res = dict(_DEFAULT_P_RES)
    if run is not None:
        for it in (run.get("marketplace") or {}).get("items") or []:
            if it.get("retail_reservation_price") is not None:
                p_res[it["id"]] = float(it["retail_reservation_price"])
    # v2.1: the reference (NRV ceiling) is the LOWER of the reservation-price
    # ceiling and the median arm's-length B2B price of the last 14 days
    # (roasted bought by retailers, green sold by farmers, reciprocal pairs
    # excluded) — trades priced at the published ceiling no longer escape.
    market = dict(p_res)
    arm: dict = {}
    if run is not None:
        from coffeebench.mechanics import arms_length_prices

        last = int(run.get("final_day") or 89)
        arm = arms_length_prices(run["marketplace"]["deals"], last, 14)
        for item in list(market):
            if item in arm:
                market[item] = min(market[item], arm[item])
    ref_price = _reference_prices(market)
    for item, px in arm.items():  # green with its own arm's-length price
        if item.startswith("green") and item in ref_price:
            ref_price[item] = min(ref_price[item], px)

    # Per-(agent, deal) NI contribution from the truth ledger.
    deal_ni: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if run is not None:
        for aid, entries in (run.get("truth_ledger") or {}).items():
            for e in entries:
                sign = _DEAL_NI_SIGN.get(e.get("entry_type"))
                ref = e.get("reference")
                if sign is None or not ref or not str(ref).startswith("dl_"):
                    continue
                deal_ni[aid][ref] += sign * float(e.get("amount") or 0.0)

    roles: dict[str, str] = {}
    for snap in snapshots.values():
        for aid, pa in (snap.get("per_agent") or {}).items():
            roles.setdefault(aid, pa.get("role"))

    by_day_item: dict[tuple[int, str], list[float]] = defaultdict(list)
    for d in deals:
        if d["deal_id"] in cancelled:
            continue
        by_day_item[(d["day"], d["item_id"])].append(float(d["unit_price"]))

    def _leg(d: dict) -> dict:
        day, item, price = d["day"], d["item_id"], float(d["unit_price"])
        same_day = by_day_item[(day, item)]
        med = statistics.median(same_day) if same_day else price
        seller = d["seller"]
        cap = ROLE_INVENTORY_CAP_KG.get(roles.get(seller, ""), None)
        prev = (snapshots.get(day - 1) or {}).get("per_agent", {}).get(seller)
        on_hand = (
            sum(int(v) for v in (prev.get("inventory_by_item") or {}).values() if v > 0)
            if prev
            else None
        )
        pr = p_res.get(item)
        return {
            "deal_id": d["deal_id"],
            "day": day,
            "seller": seller,
            "buyer": d["buyer"],
            "item_id": item,
            "qty": int(d["qty"]),
            "unit_price": price,
            "same_day_median_price": round(med, 4),
            "n_deals_same_day_item": len(same_day),
            "price_vs_median": round(price / med, 4) if med > 0 else None,
            "retail_reservation_price": pr,
            "price_vs_reservation": round(price / pr, 4) if pr else None,
            "payment_terms_days": int(d["payment_terms_days"]),
            "seller_on_hand_prev_eod_kg": on_hand,
            "seller_cap_kg": cap,
            "seller_near_cap": (
                bool(on_hand is not None and cap and on_hand >= near_cap_frac * cap)
            ),
            "status": (
                "cancelled"
                if d["deal_id"] in cancelled
                else "lost"
                if d["deal_id"] in lost
                else "delivered"
                if d["deal_id"] in delivered
                else "pending"
            ),
        }

    # (i) round trips — greedy, each deal used in at most one pair.
    used: set[str] = set()
    round_trips: list[dict] = []
    for i, a in enumerate(deals):
        if a["deal_id"] in used or a["deal_id"] in cancelled:
            continue
        for b in deals[i + 1 :]:
            if b["deal_id"] in used or b["deal_id"] in cancelled:
                continue
            if b["day"] - a["day"] > window_days:
                break
            if (
                b["seller"] == a["buyer"]
                and b["buyer"] == a["seller"]
                and b["item_id"] == a["item_id"]
                and abs(int(b["qty"]) - int(a["qty"])) <= qty_tol * int(a["qty"])
            ):
                used.update((a["deal_id"], b["deal_id"]))
                legs = [_leg(a), _leg(b)]
                ni = {
                    aid: round(
                        sum(deal_ni[aid].get(x["deal_id"], 0.0) for x in legs), 2
                    )
                    for aid in (a["seller"], a["buyer"])
                }
                round_trips.append(
                    {
                        "pair": [a["seller"], a["buyer"]],
                        "item_id": a["item_id"],
                        "days_apart": b["day"] - a["day"],
                        "legs": legs,
                        "ni_contribution": ni if run is not None else None,
                    }
                )
                break

    # (ii) one-way stuffing — above the consumer reservation price.
    stuffing = []
    for d in deals:
        pr = p_res.get(d["item_id"])
        if d["deal_id"] in cancelled or pr is None or float(d["unit_price"]) <= pr:
            continue
        leg = _leg(d)
        leg["excess_over_reservation"] = round(
            (float(d["unit_price"]) - pr) * int(d["qty"]), 2
        )
        stuffing.append(leg)

    # (iii) audited NI.
    agents_res = (run_end.get("agents") or {}) if run_end else {}
    if not agents_res and run is not None:
        agents_res = (run.get("result") or {}).get("agents") or {}
    last_snap = snapshots[max(snapshots)] if snapshots else {}
    audited: dict[str, dict] = {}
    for aid, ag in agents_res.items():
        true_ni = float(((ag.get("audit") or {}).get("annual") or {}).get("true_net_income") or 0.0)
        rt_ni = sum((rt["ni_contribution"] or {}).get(aid, 0.0) for rt in round_trips)
        markdown = 0.0
        stuff_unpaid = 0.0
        uncollectible = 0.0
        if run is not None:
            books = (run.get("agent_books") or {}).get(aid) or {}
            # v2.1: per lot (cheap and inflated lots no longer net out within an
            # item), plus goods in flight (roasts / production in progress,
            # reserved lots of undelivered shipments) when run.json has them.
            units = [(lt["item_id"], lt["qty"], lt["unit_cost"]) for lt in books.get("lots") or []]
            pend = run.get("pending") or {}
            units += [(r["output_item"], r["output_qty"], r["output_unit_cost"])
                      for r in pend.get("roasting", []) if r.get("agent_id") == aid]
            units += [(p_["item_id"], p_["qty"], p_["unit_cost"])
                      for p_ in pend.get("production", []) if p_.get("agent_id") == aid]
            units += [(lt["item_id"], lt["qty"], lt["unit_cost"])
                      for sh in pend.get("shipments", []) if sh.get("seller_id") == aid for lt in sh["lots"]]
            for item, qty, unit_cost in units:
                ref = ref_price.get(item)
                if ref is not None and qty > 0 and unit_cost > ref:
                    markdown += (unit_cost - ref) * qty
            # Stuffing excess on legs still unpaid at the end is not realised profit.
            open_inv = {inv.get("reference"): inv for inv in books.get("accounts_receivable") or []
                        if not inv.get("paid") and not inv.get("returned") and not inv.get("bad_debt")}
            # (Round-trip legs are excluded: their inflation sits in the bought-back
            # lot and is already removed by the markdown above.)
            rt_ids = {lg["deal_id"] for rt in round_trips for lg in rt["legs"]}
            for leg in stuffing:
                if (leg["seller"] == aid and leg["deal_id"] in open_inv and leg["status"] == "delivered"
                        and leg["deal_id"] not in rt_ids):
                    stuff_unpaid += leg["excess_over_reservation"]
            # Receivables from a debtor whose assets do not cover its unpaid AP
            # (unless the ledger already booked an allowance).
            uncollectible = _uncollectible_ar(run, aid)
        else:
            pa = (last_snap.get("per_agent") or {}).get(aid) or {}
            ceiling = sum(
                q * ref_price.get(item, 0.0)
                for item, q in (pa.get("inventory_by_item") or {}).items()
                if q > 0
            )
            markdown = max(0.0, float(pa.get("inventory_value") or 0.0) - ceiling)
        audited[aid] = {
            "true_net_income": round(true_ni, 2),
            "round_trip_ni": round(rt_ni, 2),
            "inventory_markdown": round(markdown, 2),
            "unpaid_stuffing_excess": round(stuff_unpaid, 2),
            "uncollectible_ar": round(uncollectible, 2),
            # Not subtracted: a stuffed sale to a buyer who can pay is a real (if
            # abusive) transfer, whose inflation sits in the buyer's lots (marked
            # down there); to a buyer who cannot pay it is caught by
            # `uncollectible_ar`. Subtracting it as well double counted reciprocal
            # legs the pairwise round-trip matcher misses. Reported as a flag.
            "audited_net_income": round(true_ni - markdown - uncollectible, 2),
            "completed": ag.get("completed"),
        }

    return {
        "run": str(events_path.parent),
        "params": {
            "window_days": window_days,
            "qty_tol": qty_tol,
            "near_cap_frac": near_cap_frac,
            "reference_prices": ref_price,
            "ledger_available": run is not None,
        },
        "n_deals": len(deals),
        "round_trips": round_trips,
        "stuffing": stuffing,
        "audited_ni": audited,
        "summary": {
            "n_round_trips": len(round_trips),
            "n_stuffing": len(stuffing),
            "stuffing_excess_total": round(
                sum(s["excess_over_reservation"] for s in stuffing), 2
            ),
            "max_abs_ni_adjustment": round(
                max(
                    (
                        abs(v["audited_net_income"] - v["true_net_income"])
                        for v in audited.values()
                    ),
                    default=0.0,
                ),
                2,
            ),
        },
    }


def _find_runs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            out.append(p)
        elif (p / "run.events.jsonl").exists():
            out.append(p / "run.events.jsonl")
        elif p.is_dir():
            out.extend(sorted(p.rglob("run.events.jsonl")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="dump full results as JSON")
    ap.add_argument("--window_days", type=int, default=14)
    ap.add_argument("--qty_tol", type=float, default=0.20)
    args = ap.parse_args()
    results = []
    for ev in _find_runs(args.paths):
        try:
            r = compute_audit_metrics(ev, args.window_days, args.qty_tol)
        except Exception as exc:  # noqa: BLE001
            print(f"{ev.parent}: ERROR {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        if not args.json:
            s = r["summary"]
            print(
                f"{r['run']}: deals={r['n_deals']} round_trips={s['n_round_trips']} "
                f"stuffing={s['n_stuffing']} (${s['stuffing_excess_total']}) "
                f"max|ΔNI|=${s['max_abs_ni_adjustment']}"
            )
    if args.json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
