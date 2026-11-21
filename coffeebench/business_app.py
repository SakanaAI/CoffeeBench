"""BusinessApp — per-agent toolset for the deal marketplace.

Wraps the shared `Marketplace` (one instance per run) and the agent's
own private books (cash, inventory, AR/AP). Public methods on this
class are auto-registered as tools the LLM agent can call.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Callable, Optional

from coffeebench.time_manager import TimeManager
from coffeebench.typings import Invoice, Lot

from coffeebench.marketplace import Marketplace
from coffeebench.typings import AgentEndowment


# Run-day stamped as `produced_day` on the starter-inventory lots of every
# endowment. Day 0 = the stock is fresh at the run start, so it carries
# its item's full shelf life (green 60 d -> expires day 60; roasted 21 d ->
# expires day 21). The starter stock is sized for 1-2 days of trade, so
# nothing expires before it can be sold; unsold starter stock does
# expire inside the 90-day horizon like any other lot.
INITIAL_LOT_PRODUCED_DAY = 0


def _fifo_key(lot: Lot) -> tuple:
    """FIFO-by-expiry ordering: earliest expiry first (never-expiring
    lots last), then oldest production. Python's sort is stable, so
    ties keep insertion order (deterministic)."""
    return (
        lot.expiry_day is None,
        lot.expiry_day if lot.expiry_day is not None else 0,
        lot.produced_day,
    )


class _LotAggregateView(Mapping):
    """Read-only {item_id: aggregate} view derived live from an agent's
    lots. Lots are the source of truth for inventory; `inventory`
    (sum of qty) and `inventory_total_cost` (sum of qty x unit_cost) are
    exposed through this view so the many existing readers keep working.

    Item assignment is deliberately unsupported (raises TypeError):
    inventory must change through the lot helpers on BusinessApp so the
    qty / cost aggregates can never drift apart. Indexing an item the
    agent does not hold returns 0 (like the old zero-valued dict keys);
    iteration / `dict(view)` only yields items with qty > 0.
    """

    __slots__ = ("_ba", "_field")

    def __init__(self, ba: "BusinessApp", field_name: str):
        self._ba = ba
        self._field = field_name  # "qty" | "cost"

    def _agg(self) -> dict:
        out: dict = {}
        for lot in self._ba.lots:
            if lot.qty <= 0:
                continue
            v = lot.qty if self._field == "qty" else lot.qty * lot.unit_cost
            out[lot.item_id] = out.get(lot.item_id, 0) + v
        return out

    def __getitem__(self, item_id: str):
        return self._agg().get(item_id, 0 if self._field == "qty" else 0.0)

    def get(self, item_id: str, default=None):
        agg = self._agg()
        return agg[item_id] if item_id in agg else default

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._agg()

    def __iter__(self):
        return iter(self._agg())

    def __len__(self) -> int:
        return len(self._agg())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._agg()!r})"


class _CostBasisView:
    """Read-only per-unit average cost view (lot value / lot qty per
    item). Informational only: costing is per-lot FIFO and this average
    is never used to book COGS. Keeps the `.cost_basis.get(item,
    default)` API used across the codebase."""

    __slots__ = ("_inventory", "_total_cost")

    def __init__(self, inventory: Mapping, total_cost: Mapping):
        self._inventory = inventory
        self._total_cost = total_cost

    def get(self, item_id: str, default: float = 0.0) -> float:
        qty = self._inventory.get(item_id, 0)
        if qty <= 0:
            return default
        return self._total_cost.get(item_id, 0.0) / qty

    def __getitem__(self, item_id: str) -> float:
        return self.get(item_id, 0.0)

    def __contains__(self, item_id: str) -> bool:
        return self._inventory.get(item_id, 0) > 0

    def __iter__(self):
        return iter(k for k, v in self._inventory.items() if v > 0)

    def items(self):
        for k, qty in self._inventory.items():
            if qty > 0:
                yield k, self._total_cost.get(k, 0.0) / qty

    def __repr__(self) -> str:
        return f"_CostBasisView({dict(self.items())})"


# Upper bound on payment terms (net-N days) for listings and offers.
# Matches the canonical 90-day horizon: longer terms could never fall
# due inside the run.
MAX_PAYMENT_TERMS_DAYS = 90
# v2.1: hard sanity bound on a unit price (USD/kg; ~1,000× any real price) so
# totals stay finite (1.7e308 used to give total_price = inf).
MAX_UNIT_PRICE = 100_000.0
MAX_MESSAGE_BODY_CHARS = 4000  # v2.1: DM body cap


def _validate_price_and_terms(
    price, payment_terms_days, price_name: str
) -> tuple[float, int, str | None]:
    """Shared validation for post_listing / make_offer. Returns
    (price, terms, error_message_or_None)."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 0.0, 0, f"{price_name} must be a number."
    if not math.isfinite(price) or price <= 0:
        return 0.0, 0, f"{price_name} must be a finite number > 0."
    if price > MAX_UNIT_PRICE:
        return 0.0, 0, f"{price_name} must be at most {MAX_UNIT_PRICE:,.0f} per kg."
    if isinstance(payment_terms_days, bool):
        return 0.0, 0, "payment_terms_days must be an integer."
    try:
        terms_f = float(payment_terms_days)
    except (TypeError, ValueError):
        return 0.0, 0, "payment_terms_days must be an integer."
    if not math.isfinite(terms_f) or not terms_f.is_integer():
        return 0.0, 0, "payment_terms_days must be an integer."
    terms = int(terms_f)
    if not 0 <= terms <= MAX_PAYMENT_TERMS_DAYS:
        return (
            0.0,
            0,
            f"payment_terms_days must be between 0 and {MAX_PAYMENT_TERMS_DAYS}.",
        )
    return price, terms, None


class BusinessApp:
    def __init__(
        self,
        agent_id: str,
        endowment: AgentEndowment,
        marketplace: Marketplace,
        time_manager: TimeManager,
    ):
        self.agent_id = agent_id
        self.display_name = endowment.display_name
        self.role = endowment.role
        self.persona = endowment.persona
        self.marketplace = marketplace
        self.time_manager = time_manager

        self.cash: float = float(endowment.initial_cash)
        # Lot-based inventory (source of truth). Every unit of tangible
        # stock belongs to one `Lot` with its own unit cost and expiry.
        self.lots: list[Lot] = []
        for item_id, qty in endowment.initial_inventory.items():
            if int(qty) <= 0:
                continue
            self._add_lot(
                item_id,
                int(qty),
                float(endowment.initial_cost_basis.get(item_id, 0.0)),
                produced_day=INITIAL_LOT_PRODUCED_DAY,
            )
        # Derived read-only aggregates over `self.lots`:
        #   inventory[item]            = sum of lot.qty
        #   inventory_total_cost[item] = sum of lot.qty * lot.unit_cost
        # `cost_basis` is the per-unit average of the two (display only).
        self.inventory: _LotAggregateView = _LotAggregateView(self, "qty")
        self.inventory_total_cost: _LotAggregateView = _LotAggregateView(self, "cost")
        self.cost_basis: "_CostBasisView" = _CostBasisView(
            self.inventory,
            self.inventory_total_cost,
        )

        # Shop-only: per-item retail prices for consumer-facing storefront.
        # Other roles can still set them but consumers won't visit (the env
        # only generates consumer demand at agents whose role == "retailer").
        # Seeded from `endowment.initial_retail_prices` so the supply chain
        # moves on day 0 even before the retailer calls `set_retail_price`.
        self.retail_prices: dict[str, float] = dict(endowment.initial_retail_prices)

        self.accounts_receivable: list[Invoice] = []
        self.accounts_payable: list[Invoice] = []

        # Set by Environment.__init__
        self._record_truth: Optional[Callable] = None

        self.initial_equity = self._compute_true_equity()

    # ----- internal helpers -----
    def _today(self) -> int:
        return self.time_manager.get_current_day()

    # ----- lot helpers (the only way inventory changes) -----
    def _shelf_life_days(self, item_id: str) -> int | None:
        item = self.marketplace.get_item(item_id)
        if item is None or item.shelf_life_days is None:
            return None
        return int(item.shelf_life_days)

    def _add_lot(
        self,
        item_id: str,
        qty: int,
        unit_cost: float,
        produced_day: int,
        expiry_day: int | None = None,
        source_deal_id: str | None = None,
        compute_expiry: bool = True,
    ) -> Lot:
        """Create a new lot on this agent's books. `expiry_day` defaults
        to produced_day + the item's shelf life (pass compute_expiry=False
        to keep an explicit None)."""
        if expiry_day is None and compute_expiry:
            life = self._shelf_life_days(item_id)
            expiry_day = None if life is None else int(produced_day) + life
        lot = Lot(
            lot_id=self.marketplace.next_lot_id(),
            item_id=item_id,
            qty=int(qty),
            unit_cost=float(unit_cost),
            produced_day=int(produced_day),
            expiry_day=expiry_day,
            source_deal_id=source_deal_id,
        )
        self.lots.append(lot)
        return lot

    def _receive_lots(
        self,
        pieces: list[Lot],
        unit_cost: float | None = None,
        source_deal_id: str | None = None,
    ) -> list[Lot]:
        """Add lot pieces coming from another agent (delivery / return),
        keeping their produced / expiry days. Each piece becomes a new lot
        here (fresh lot_id) at `unit_cost` (None = keep the piece's)."""
        out = []
        for p in pieces:
            if p.qty <= 0:
                continue
            out.append(
                self._add_lot(
                    p.item_id,
                    p.qty,
                    p.unit_cost if unit_cost is None else unit_cost,
                    produced_day=p.produced_day,
                    expiry_day=p.expiry_day,
                    source_deal_id=source_deal_id,
                    compute_expiry=False,
                )
            )
        return out

    def _restore_lots(self, pieces: list[Lot]) -> None:
        """Put this agent's own previously-taken lot pieces back
        unchanged (e.g. a cancelled deal's reservation): merged into the
        surviving lot with the same id, else re-added as that lot."""
        for p in pieces:
            if p.qty <= 0:
                continue
            same = next((lt for lt in self.lots if lt.lot_id == p.lot_id), None)
            if same is not None and same.unit_cost == p.unit_cost:
                same.qty += p.qty
            else:
                self.lots.append(replace(p))

    def _fifo_lots(self, item_id: str, min_expiry_day: int | None = None) -> list[Lot]:
        """This agent's lots of `item_id` in FIFO-by-expiry order,
        optionally restricted to lots still within shelf life on
        `min_expiry_day` (expiry_day >= min_expiry_day)."""
        lots = [
            lt
            for lt in self.lots
            if lt.item_id == item_id
            and lt.qty > 0
            and (
                min_expiry_day is None
                or lt.expiry_day is None
                or lt.expiry_day >= min_expiry_day
            )
        ]
        return sorted(lots, key=_fifo_key)

    def _sellable_qty(self, item_id: str, min_expiry_day: int | None) -> int:
        return sum(lt.qty for lt in self._fifo_lots(item_id, min_expiry_day))

    def _take_lots(
        self,
        item_id: str,
        qty: int,
        min_expiry_day: int | None = None,
        prefer_source_deal: str | None = None,
        only_source_deal: str | None = None,
    ) -> list[Lot]:
        """Remove `qty` units of `item_id` FIFO by expiry and return the
        removed pieces (each keeps its lot_id, unit_cost and dates; the
        sum of piece values is the FIFO cost of the units taken). Lots
        expiring before `min_expiry_day` are skipped. With
        `prefer_source_deal`, lots received through that deal are taken
        first (still FIFO among them). Raises ValueError (no mutation) if
        not enough qualifying units are held."""
        qty = int(qty)
        order = self._fifo_lots(item_id, min_expiry_day)
        if only_source_deal is not None:
            order = [lt for lt in order if lt.source_deal_id == only_source_deal]
        if prefer_source_deal is not None:
            order = [lt for lt in order if lt.source_deal_id == prefer_source_deal] + [
                lt for lt in order if lt.source_deal_id != prefer_source_deal
            ]
        if sum(lt.qty for lt in order) < qty:
            raise ValueError(
                f"{self.agent_id}: insufficient {item_id} lots for {qty} units"
            )
        pieces: list[Lot] = []
        remaining = qty
        for lt in order:
            if remaining <= 0:
                break
            take = min(lt.qty, remaining)
            pieces.append(replace(lt, qty=take))
            lt.qty -= take
            remaining -= take
        self.lots = [lt for lt in self.lots if lt.qty > 0]
        return pieces

    def _freshness_snapshot(
        self, item_id: str, qty: int, delivery_day: int
    ) -> dict | None:
        """Public, static freshness snapshot stored on a listing at posting
        time: only the earliest / latest expiry day of the FIFO lots that
        would cover `qty` units arriving on `delivery_day`, plus the
        as-of day. No quantities, holdings or production days (other
        agents must not learn the seller's stock from it)."""
        remaining = int(qty)
        expiries: list[int] = []
        for lt in self._fifo_lots(item_id, delivery_day):
            if remaining <= 0:
                break
            remaining -= min(lt.qty, remaining)
            if lt.expiry_day is not None:
                expiries.append(lt.expiry_day)
        if not expiries:
            return None
        return {
            "earliest_expiry_day": min(expiries),
            "latest_expiry_day": max(expiries),
            "as_of_day": self._today(),
        }

    def _ship_freshness(self, item_id: str, qty: int, delivery_day: int) -> dict | None:
        """OWNER-ONLY live view: freshness of the FIFO lots that would ship
        if you accepted now, for `qty` units arriving on `delivery_day`
        (no mutation). Only ever shown to the seller itself (it reveals
        holdings); buyers see the listing's posting-time snapshot."""
        remaining = int(qty)
        expiries: list[int] = []
        produced: list[int] = []
        covered = 0
        for lt in self._fifo_lots(item_id, delivery_day):
            if remaining <= 0:
                break
            take = min(lt.qty, remaining)
            remaining -= take
            covered += take
            produced.append(lt.produced_day)
            if lt.expiry_day is not None:
                expiries.append(lt.expiry_day)
        if covered <= 0:
            return None
        out: dict = {"oldest_produced_day": min(produced)}
        if expiries:
            out["earliest_expiry_day"] = min(expiries)
            out["latest_expiry_day"] = max(expiries)
            out["min_days_left_on_arrival"] = min(expiries) - int(delivery_day)
        if covered < int(qty):
            out["note"] = (
                f"seller currently holds only {covered} unit(s) that would "
                f"still be within shelf life on arrival"
            )
        return out

    def _expiring_lots(self, day: int, within_days: int) -> list[Lot]:
        """Lots whose expiry_day is at most `within_days` after `day`."""
        return sorted(
            (
                lt
                for lt in self.lots
                if lt.qty > 0
                and lt.expiry_day is not None
                and lt.expiry_day - int(day) <= int(within_days)
            ),
            key=_fifo_key,
        )

    def _ar_outstanding(self) -> float:
        return sum(max(0.0, inv.net_outstanding - getattr(inv, "credit_loss_allowance", 0.0))
                   for inv in self.accounts_receivable)

    def _ap_outstanding(self) -> float:
        return sum(inv.net_outstanding for inv in self.accounts_payable)

    def _inventory_value(self) -> float:
        # Source of truth = `inventory_total_cost`. Sum directly; this is
        # exactly the $ value reflected in equity.
        return sum(self.inventory_total_cost.values())

    def _total_inventory_kg(self) -> int:
        return sum(int(qty) for qty in self.inventory.values() if qty > 0)

    def _pending_inbound_kg(self) -> int:
        """Kg the agent has committed to receive but hasn't landed yet:
        own queued production + own queued roast output + accepted-but-
        not-delivered purchases (deals where this agent is the buyer).
        The cap-check sites use `held + pending_inbound` so action-time
        gates catch in-flight pile-up and the cap stays a hard cap
        even when batches stack across multiple lag-days."""
        env = getattr(self, "_env", None)
        if env is None:
            return 0
        aid = self.agent_id
        total = 0
        for p in env._pending_production:
            if p.get("agent_id") == aid:
                total += int(p.get("qty", 0))
        for r in env._pending_roasting:
            if r.get("agent_id") == aid:
                total += int(r.get("output_qty", 0))
        for deal in env._pending_shipments:
            if deal.buyer_id == aid:
                total += int(deal.qty)
        return total

    def _effective_held_kg(self) -> int:
        """Current on-hand + in-flight inbound. The quantity that
        matters for hard-cap enforcement."""
        return self._total_inventory_kg() + self._pending_inbound_kg()

    def _inventory_cap_kg(self) -> int:
        from coffeebench.environment import ROLE_INVENTORY_CAP_KG

        cap = int(ROLE_INVENTORY_CAP_KG.get(self.role, 10**9))
        env = getattr(self, "_env", None)
        if env is not None and getattr(env, "_upgrades", None):
            cap += env.inventory_cap_bonus(self.agent_id, self._today())
        return cap

    def _inventory_capacity_remaining_kg(self) -> int:
        return max(0, self._inventory_cap_kg() - self._effective_held_kg())

    def _in_flight_value(self) -> float:
        """In-flight asset value (pending production / roasting +
        reserved-for-sale). Queried from env via `_env` back-ref set
        at Environment.__init__ time. Pre-env (e.g. construction),
        returns 0 — which is correct because no pending state exists
        until the env starts running."""
        env = getattr(self, "_env", None)
        if env is None:
            return 0.0
        return env.pending_value_for(self.agent_id)

    def _compute_true_equity(self) -> float:
        return (
            self.cash
            + self._inventory_value()
            + self._in_flight_value()
            + self._ar_outstanding()
            - self._ap_outstanding()
        )

    # ===== TOOLS — marketplace =====
    def post_listing(
        self,
        item_id: str,
        qty: int,
        asking_price: float,
        payment_terms_days: int = 30,
    ) -> dict:
        """Post a public listing (see §4).

        Args:
            item_id: item to sell.
            qty: kg (at most your current stock).
            asking_price: USD per kg (at most 100,000).
            payment_terms_days: requested terms, 0-90 days (default 30).
        """
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            return {"status": "error", "message": "qty must be an integer."}
        if qty <= 0:
            return {"status": "error", "message": "qty must be > 0."}
        asking_price, payment_terms_days, err = _validate_price_and_terms(
            asking_price, payment_terms_days, "asking_price"
        )
        if err is not None:
            return {"status": "error", "message": err}
        # Catalog is tangible-only; sellers must hold inventory backing.
        if self.inventory.get(item_id, 0) < qty:
            return {
                "status": "error",
                "message": f"You only hold {self.inventory.get(item_id, 0)} units of '{item_id}'.",
            }
        listing = self.marketplace.post_listing(
            seller_id=self.agent_id,
            item_id=item_id,
            qty=qty,
            asking_price=asking_price,
            payment_terms_days=payment_terms_days,
        )
        listing.freshness_at_posting = self._freshness_snapshot(
            item_id,
            qty,
            self._today() + self.marketplace._delivery_lag(self, item_id),
        )
        return {
            "status": "success",
            "listing_id": listing.id,
            "asking_price": asking_price,
            "qty": qty,
        }

    def view_listings(
        self, item_id: str | None = None, only_others: bool = True
    ) -> dict:
        """List open listings, with the expiry range of the seller's lots at
        posting (`freshness_at_posting`; your own listings also `live_freshness`).

        Args:
            item_id: only this item.
            only_others: hide your own listings (default true).
        """
        rows = self.marketplace.listings_dump()
        out = []
        for r in rows:
            if item_id is not None and r["item_id"] != item_id:
                continue
            if only_others and r["seller_id"] == self.agent_id:
                continue
            if r["seller_id"] == self.agent_id:
                r["live_freshness"] = self._ship_freshness(
                    r["item_id"], r["qty"], self._today() + r["delivery_eta_days"]
                )
            out.append(r)
        return {
            "status": "success",
            "count": len(out),
            "listings": out,
            "note": (
                "A listing's payment_terms_days is the seller's requested "
                "terms; a deal uses the terms in the offer the seller accepts."
            ),
        }

    def make_offer(
        self,
        listing_id: str,
        offered_price: float,
        qty: int,
        payment_terms_days: int = 30,
        message: str = "",
    ) -> dict:
        """Make an offer on another business's open listing (see §4).

        Args:
            listing_id: listing id.
            offered_price: USD per kg (at most 100,000).
            qty: kg (at most the listing's remaining qty).
            payment_terms_days: terms, 0-90 days (default 30).
            message: note attached to the offer.
        """
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            return {"status": "error", "message": "qty must be an integer."}
        offered_price, payment_terms_days, err = _validate_price_and_terms(
            offered_price, payment_terms_days, "offered_price"
        )
        if err is not None:
            return {"status": "error", "message": err}
        offer, msg = self.marketplace.make_offer(
            listing_id=listing_id,
            buyer_id=self.agent_id,
            offered_price=offered_price,
            qty=qty,
            payment_terms_days=payment_terms_days,
            message=message or None,
        )
        if offer is None:
            return {"status": "error", "message": msg}
        # Reactive wake on the listing's seller — they need to know
        # an offer is pending so they can accept / counter / ignore
        # without missing it for days at a stretch.
        env = getattr(self.marketplace, "_env", None)
        if env is not None:
            from coffeebench.event_loop import WAKE_OFFER_RECEIVED

            env._wake_agent_externally(
                offer.seller_id,
                f"{WAKE_OFFER_RECEIVED}: {self.agent_id} offered "
                f"{qty} kg @ ${offered_price}/kg on {listing_id}",
            )
        return {"status": "success", "offer_id": offer.id, "seller_id": offer.seller_id}

    def withdraw_offer(self, offer_id: str) -> dict:
        """Withdraw one of your pending offers."""
        ok, msg = self.marketplace.withdraw_offer(offer_id, self.agent_id)
        return {"status": "success" if ok else "error", "message": msg}

    def reject_offer(self, offer_id: str) -> dict:
        """Reject a pending offer on your listing (the listing stays open)."""
        ok, msg = self.marketplace.reject_offer(offer_id, self.agent_id)
        return {"status": "success" if ok else "error", "message": msg}

    def cancel_listing(self, listing_id: str) -> dict:
        """Cancel one of your open listings (its pending offers are rejected)."""
        n, msg = self.marketplace.cancel_listing(listing_id, self.agent_id)
        if n < 0:
            return {"status": "error", "message": msg}
        return {
            "status": "success",
            "message": msg,
            "listing_id": listing_id,
            "pending_offers_rejected": n,
        }

    def accept_offer(self, offer_id: str) -> dict:
        """Accept a pending offer on your listing (binding deal; see §4-5)."""
        deal, msg = self.marketplace.accept_offer(offer_id, self.agent_id)
        if deal is None:
            return {"status": "error", "message": msg}
        return {
            "status": "success",
            "deal_id": deal.id,
            "invoice_id": deal.invoice_id,
            "buyer_id": deal.buyer_id,
            "item_id": deal.item_id,
            "qty": deal.qty,
            "unit_price": deal.unit_price,
            "total_price": deal.total_price,
            "payment_terms_days": deal.payment_terms_days,
            "delivery_day": deal.delivery_at // 1440,
            "delivery_at": deal.delivery_at,
        }

    def view_offers(self, direction: str = "all") -> dict:
        """List offers involving you, with the expiry range of the lots that
        would ship (`live_freshness` on offers you received).

        Args:
            direction: "incoming", "outgoing" or "all" (default).
        """
        return {
            "status": "success",
            "offers": self.marketplace.offers_for(self.agent_id, direction=direction),
        }

    def view_deals(self) -> dict:
        """List all deals you are party to."""
        from coffeebench.event_loop import format_min

        out = []
        for d in self.marketplace.deals:
            if d.seller_id != self.agent_id and d.buyer_id != self.agent_id:
                continue
            row = asdict(d)
            # Internal seller-side bookkeeping (reserved COGS / lot unit
            # costs) is not shown; both parties see the shipped lots'
            # quantities and dates instead.
            row.pop("_reserved_seller_cogs", None)
            row.pop("_reserved_lots", None)
            row["lots"] = [
                {
                    "qty": p.qty,
                    "produced_day": p.produced_day,
                    "expiry_day": p.expiry_day,
                }
                for p in d._reserved_lots
            ]
            row["deal_day"] = d.deal_at // 1440
            row["delivery_day"] = d.delivery_at // 1440
            row["accepted"] = format_min(d.deal_at)
            row["delivers"] = format_min(d.delivery_at)
            out.append(row)
        return {"status": "success", "count": len(out), "deals": out}

    def view_dashboard(self) -> dict:
        """Show a compact snapshot: cash, AR, AP, stock and lots expiring within
        3 days, your listings and offers (both directions), deals in flight,
        overdue and soon-due invoices, new messages, pending production/roasts
        (retailers: prices and yesterday's sales). Long sections are capped.
        """
        from coffeebench.environment import EXPIRY_WARNING_DAYS
        from coffeebench.event_loop import MINUTES_PER_DAY, format_min

        env = getattr(self, "_env", None)
        mp = self.marketplace
        today = self._today()
        now = self.time_manager.get_virtual_min()
        # "Since your last call" is tracked by position in the message log
        # (not by timestamp: agents in one dispatch bucket share a minute).
        last_call = getattr(self, "_last_dashboard_at", None)
        seen_upto = getattr(self, "_last_dashboard_msg_idx", 0)
        self._last_dashboard_at = now
        self._last_dashboard_msg_idx = len(mp.messages)

        def capped(rows: list, cap: int, tool: str) -> dict:
            out = {"rows": rows[:cap]}
            if len(rows) > cap:
                out["omitted"] = len(rows) - cap
                out["more_via"] = tool
            return out

        # Inventory.
        inv_rows = []
        for item_id, qty in sorted(self.inventory.items()):
            lots = self._fifo_lots(item_id)
            exp = [lt.expiry_day for lt in lots if lt.expiry_day is not None]
            inv_rows.append(
                {
                    "item_id": item_id,
                    "qty": int(qty),
                    "value": round(self.inventory_total_cost.get(item_id, 0.0), 2),
                    "earliest_expiry_day": min(exp) if exp else None,
                }
            )
        expiring = [
            {
                "lot_id": lt.lot_id,
                "item_id": lt.item_id,
                "qty": lt.qty,
                "expiry_day": lt.expiry_day,
                "days_left": lt.expiry_day - today,
            }
            for lt in self._expiring_lots(today, EXPIRY_WARNING_DAYS)
        ]

        # Listings / offers.
        offers = mp.offers
        listings = [
            lt
            for lt in mp.listings
            if lt.seller_id == self.agent_id and lt.status == "open"
        ]
        listing_rows = [
            {
                "id": lt.id,
                "item_id": lt.item_id,
                "qty": lt.qty,
                "asking_price": lt.asking_price,
                "pending_offers": sum(
                    1 for o in offers if o.listing_id == lt.id and o.status == "pending"
                ),
            }
            for lt in listings
        ]
        item_of = {lt.id: lt.item_id for lt in mp.listings}
        incoming = [
            {
                "id": o.id,
                "listing_id": o.listing_id,
                "item_id": item_of.get(o.listing_id),
                "buyer": o.buyer_id,
                "price": o.offered_price,
                "qty": o.qty,
                "terms_days": o.payment_terms_days,
                "expires_eod_day": o.posted_at // MINUTES_PER_DAY + 1,
            }
            for o in offers
            if o.seller_id == self.agent_id and o.status == "pending"
        ]
        outgoing = [
            {
                "id": o.id,
                "listing_id": o.listing_id,
                "item_id": item_of.get(o.listing_id),
                "seller": o.seller_id,
                "price": o.offered_price,
                "qty": o.qty,
                "status": o.status,
            }
            for o in reversed(offers)
            if o.buyer_id == self.agent_id
            and (o.status == "pending" or o.posted_at // MINUTES_PER_DAY >= today - 1)
        ]

        # Deals in flight.
        in_flight = []
        for d in env._pending_shipments if env is not None else []:
            if self.agent_id not in (d.buyer_id, d.seller_id):
                continue
            inbound = d.buyer_id == self.agent_id
            in_flight.append(
                {
                    "deal_id": d.id,
                    "direction": "inbound" if inbound else "outbound",
                    "counterparty": d.seller_id if inbound else d.buyer_id,
                    "item_id": d.item_id,
                    "qty": d.qty,
                    "unit_price": d.unit_price,
                    "delivery_day": d.delivery_at // MINUTES_PER_DAY,
                }
            )
        in_flight.sort(key=lambda r: r["delivery_day"])

        # Invoices: overdue or due within 5 days.
        def due_rows(invoices: list) -> list:
            rows = []
            for inv in invoices:
                net = inv.net_outstanding
                if net <= 0.01 or inv.due_date > today + 5:
                    continue
                rows.append(
                    {
                        "id": inv.id,
                        "counterparty": (
                            inv.payer if inv.issuer == self.agent_id else inv.issuer
                        ),
                        "outstanding": round(net, 2),
                        "due_day": inv.due_date,
                        "overdue": inv.due_date < today,
                    }
                )
            return sorted(rows, key=lambda r: (r["due_day"], r["id"]))

        # Messages.
        visible = mp.messages_visible_to(self.agent_id)
        unread = mp.unread_messages_for(self.agent_id)
        new_ids = {m.id for m in mp.messages[seen_upto:]}
        recent = [m for m in visible if m.id in new_ids]
        unread_ids = {m.id for m in unread}
        shown = {m.id: m for m in unread}
        shown.update({m.id: m for m in recent})
        msg_rows = [
            {
                "id": m.id,
                "sent": format_min(m.sent_at),
                "from": m.sender_id,
                "to": m.recipient_id,
                "title": m.title,
                "unread": m.id in unread_ids,
                "thread_id": m.thread_id or m.id,
                "reply_to": self._reply_ref(m),
            }
            for m in sorted(shown.values(), key=lambda m: m.sent_at, reverse=True)
        ]

        out = {
            "status": "success",
            "as_of": format_min(now),
            "headline": {
                "cash": round(self.cash, 2),
                "inventory_value": round(self._inventory_value(), 2),
                "ar": round(self._ar_outstanding(), 2),
                "ap": round(self._ap_outstanding(), 2),
            },
            "inventory": {
                "by_item": inv_rows,
                "expiring_within_3d": capped(expiring, 5, "view_inventory"),
                "on_hand_kg": self._total_inventory_kg(),
                "in_flight_inbound_kg": self._pending_inbound_kg(),
                "cap_kg": self._inventory_cap_kg(),
            },
            "my_open_listings": capped(listing_rows, 5, "view_offers / view_listings"),
            "offers_on_my_listings": capped(incoming, 5, "view_offers('incoming')"),
            "my_offers": capped(outgoing, 5, "view_offers('outgoing')"),
            "deals_in_flight": capped(in_flight, 5, "view_deals"),
            "payables_due": capped(due_rows(self.accounts_payable), 5, "view_payables"),
            "receivables_due": capped(
                due_rows(self.accounts_receivable), 5, "view_receivables"
            ),
            "messages": {
                "unread_total": len(unread),
                "since": (
                    format_min(last_call) if last_call is not None else "run start"
                ),
                **capped(msg_rows, 8, "view_messages"),
            },
        }
        if env is not None and self.role == "farmer":
            out["pending_production"] = [
                {"item_id": p["item_id"], "qty": p["qty"], "ready_day": p["ready_day"]}
                for p in env._pending_production
                if p["agent_id"] == self.agent_id
            ]
        if env is not None and self.role == "roaster":
            out["pending_roasts"] = [
                {
                    "output_item": r["output_item"],
                    "output_qty": r["output_qty"],
                    "ready_day": r["ready_day"],
                }
                for r in env._pending_roasting
                if r["agent_id"] == self.agent_id
            ]
        if self.role == "retailer":
            sales: dict[str, dict] = {}
            for s in getattr(env, "consumer_sales_log", []) or []:
                if s.get("shop_id") != self.agent_id or s.get("day") != today - 1:
                    continue
                row = sales.setdefault(s["item_id"], {"qty": 0, "revenue": 0.0})
                row["qty"] += int(s.get("qty", 0))
                row["revenue"] = round(
                    row["revenue"] + float(s.get("total_price", 0.0)), 2
                )
            out["retail"] = {
                "prices": dict(self.retail_prices),
                "yesterday_consumer_sales": sales,
            }
        return out

    def view_inventory(self, item_id: str | None = None) -> dict:
        """List your stock lot by lot (qty, unit cost, produced and expiry day).

        Args:
            item_id: only this item.
        """
        today = self._today()
        rows = []
        for lt in sorted(self.lots, key=lambda x: (x.item_id, _fifo_key(x))):
            if item_id is not None and lt.item_id != item_id:
                continue
            rows.append(
                {
                    "lot_id": lt.lot_id,
                    "item_id": lt.item_id,
                    "qty": lt.qty,
                    "unit_cost": round(lt.unit_cost, 4),
                    "value": round(lt.value, 2),
                    "produced_day": lt.produced_day,
                    "expiry_day": lt.expiry_day,
                    "days_left": (
                        lt.expiry_day - today if lt.expiry_day is not None else None
                    ),
                }
            )
        totals = {
            k: {
                "qty": int(q),
                "value": round(self.inventory_total_cost.get(k, 0.0), 2),
            }
            for k, q in sorted(self.inventory.items())
            if item_id is None or k == item_id
        }
        return {
            "status": "success",
            "today": today,
            "lots": rows,
            "totals_by_item": totals,
            "note": (
                "Lots listed per item in FIFO order (next to ship / roast / "
                "sell first). days_left = expiry_day - today; 0 = written "
                "off at tonight's EOD."
            ),
        }

    # ===== TOOLS — retail (shop-only effective) =====
    def set_retail_price(self, item_id: str, price_per_unit: float) -> dict:
        """Set your consumer price for a roasted item (retailer only; see §8).

        Args:
            item_id: roasted_coffee_kg or roasted_specialty_kg.
            price_per_unit: USD per kg (0 delists).
        """
        try:
            price = float(price_per_unit)
        except (TypeError, ValueError):
            return {"status": "error", "message": "price_per_unit must be numeric."}
        if not math.isfinite(price) or price < 0 or price > MAX_UNIT_PRICE:
            return {"status": "error", "message": f"price_per_unit must be a finite number in [0, {MAX_UNIT_PRICE:,.0f}]."}
        item = self.marketplace.get_item(item_id)
        if item is None or item.retail_reservation_price is None:
            return {"status": "error", "message": f"'{item_id}' is not a consumer item."}
        if price == 0:
            self.retail_prices.pop(item_id, None)
            return {
                "status": "success",
                "message": f"Removed {item_id} from storefront.",
            }
        self.retail_prices[item_id] = price
        return {
            "status": "success",
            "item_id": item_id,
            "retail_price_per_unit": price,
            "note": (
                "OK — retail price set. Note: consumer demand only flows to retailer-role agents."
                if self.role != "retailer"
                else "OK — retail price set."
            ),
        }

    def view_consumer_sales(self, since_day: int | None = None) -> dict:
        """List your shop's consumer sales (day, item, qty, unit price,
        total, share).

        Args:
            since_day: only from this day on.
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        log = getattr(env, "consumer_sales_log", []) or []
        out = []
        for s in log:
            if s.get("shop_id") != self.agent_id:
                continue
            if since_day is not None and int(s.get("day", 0)) < int(since_day):
                continue
            out.append(dict(s))
        return {"status": "success", "count": len(out), "sales": out}

    def view_market_aggregate(self, days_back: int = 7) -> dict:
        """Show consumer sales across both retailers per (item, day): kg
        sold and average price per kg.

        Args:
            days_back: days up to and including today (default 7).
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        try:
            n = int(days_back)
        except (TypeError, ValueError):
            return {"status": "error", "message": "days_back must be an integer."}
        n = max(1, n)
        today = self._today()
        cutoff = today - n + 1
        log = getattr(env, "consumer_sales_log", []) or []
        # Aggregate by (day, item_id): sum qty, weighted avg unit_price.
        agg: dict[tuple[int, str], dict] = {}
        for s in log:
            d = int(s.get("day", 0))
            if d < cutoff or d > today:
                continue
            key = (d, str(s.get("item_id")))
            row = agg.setdefault(
                key,
                {
                    "day": d,
                    "item_id": key[1],
                    "total_qty": 0,
                    "revenue": 0.0,
                    "n_sales": 0,
                },
            )
            qty = int(s.get("qty", 0) or 0)
            tot = float(s.get("total_price", 0.0) or 0.0)
            row["total_qty"] += qty
            row["revenue"] += tot
            row["n_sales"] += 1
        rows = []
        for (d, item_id), row in sorted(agg.items()):
            qty = int(row["total_qty"])
            rev = float(row["revenue"])
            avg_price = (rev / qty) if qty > 0 else 0.0
            rows.append(
                {
                    "day": d,
                    "item_id": item_id,
                    "market_qty_sold_kg": qty,
                    "market_revenue": round(rev, 2),
                    "avg_unit_price": round(avg_price, 2),
                }
            )
        # Per-item totals across the whole window.
        item_totals: dict[str, dict] = {}
        for r in rows:
            t = item_totals.setdefault(
                r["item_id"],
                {
                    "item_id": r["item_id"],
                    "total_qty_sold_kg": 0,
                    "total_revenue": 0.0,
                    "days_with_sales": 0,
                },
            )
            t["total_qty_sold_kg"] += r["market_qty_sold_kg"]
            t["total_revenue"] += r["market_revenue"]
            t["days_with_sales"] += 1
        item_summary = []
        for it in sorted(item_totals.values(), key=lambda x: x["item_id"]):
            qty = int(it["total_qty_sold_kg"])
            rev = float(it["total_revenue"])
            it["total_revenue"] = round(rev, 2)
            it["avg_qty_per_active_day"] = (
                round(qty / it["days_with_sales"], 2)
                if it["days_with_sales"] > 0
                else 0.0
            )
            it["avg_unit_price"] = round((rev / qty) if qty > 0 else 0.0, 2)
            item_summary.append(it)
        out_extra = {}
        if getattr(env, "mechanics", None) and env.mechanics.get("demand_patterns"):
            from coffeebench.mechanics import weather_forecast

            # E3: the published (accurate) weather forecast.
            out_extra["weather_forecast"] = weather_forecast(env.mechanics, today)
        return {
            "status": "success",
            **out_extra,
            "window_start_day": max(0, cutoff),
            "window_end_day": today,
            "n_days": n,
            "per_day": rows,
            "per_item": item_summary,
        }

    # ===== TOOLS — returns =====
    def return_shipment(
        self,
        invoice_id: str,
        quantity_kg: int,
        reason: str = "",
    ) -> dict:
        """Return goods you bought (see §5).

        Args:
            invoice_id: invoice of the delivery.
            quantity_kg: kg (at most the kg received in that delivery, not yet
                returned, that you still hold with at least 3 days of shelf life).
            reason: note.
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        from coffeebench.environment import RETURN_WINDOW_DAYS

        try:
            qty = int(quantity_kg)
        except (TypeError, ValueError):
            return {"status": "error", "message": "quantity_kg must be an integer."}
        if qty <= 0:
            return {"status": "error", "message": "quantity_kg must be > 0."}

        # Locate the invoice on the buyer (us) side.
        invoice = next(
            (i for i in self.accounts_payable if i.id == invoice_id),
            None,
        )
        if invoice is None:
            return {
                "status": "error",
                "message": f"Invoice '{invoice_id}' not found in your AP.",
            }
        # Locate the deal that issued this invoice.
        deal = next(
            (d for d in self.marketplace.deals if d.id == invoice.reference), None
        )
        if deal is None:
            return {
                "status": "error",
                "message": "Linked deal not found (cannot resolve item / qty / unit_price).",
            }
        if deal.buyer_id != self.agent_id:
            return {
                "status": "error",
                "message": "Only the original buyer can return this shipment.",
            }

        # Item must be tangible.
        item = self.marketplace.get_item(deal.item_id)
        if item is None:
            return {
                "status": "error",
                "message": f"Item '{deal.item_id}' not in catalog.",
            }

        today = self._today()
        if today - invoice.issue_date > RETURN_WINDOW_DAYS:
            return {
                "status": "error",
                "message": (
                    f"Return window ({RETURN_WINDOW_DAYS} days) closed. Invoice issued day "
                    f"{invoice.issue_date}; today is day {today}."
                ),
            }

        # Cumulative-return cap.
        remaining_returnable = deal.qty - deal.returned_qty
        if qty > remaining_returnable:
            return {
                "status": "error",
                "message": (
                    f"Cannot return {qty} — only {remaining_returnable} units remaining "
                    f"on this deal (already returned {deal.returned_qty}/{deal.qty})."
                ),
            }
        # v2.1: only units received through THIS delivery and still saleable
        # (>= RETURN_MIN_SHELF_DAYS of shelf life) can go back — no substituting
        # cheaper lots from another supplier, no dumping near-expiry stock.
        from coffeebench.environment import RETURN_MIN_SHELF_DAYS

        min_expiry = today + RETURN_MIN_SHELF_DAYS
        returnable = sum(
            lt.qty for lt in self._fifo_lots(deal.item_id, min_expiry) if lt.source_deal_id == deal.id
        )
        if qty > returnable:
            return {
                "status": "error",
                "message": (
                    f"Only {returnable} units received in this delivery are still held with at least "
                    f"{RETURN_MIN_SHELF_DAYS} days of shelf life left — cannot return {qty}."
                ),
            }

        # Locate seller. Marketplace.business_apps is the env-wired registry.
        seller = self.marketplace.business_apps.get(deal.seller_id)
        if seller is None:
            return {
                "status": "error",
                "message": "Seller's BusinessApp not registered.",
            }
        if deal.seller_id in env._bankrupt_agents:
            return {
                "status": "error",
                "message": f"Seller {deal.seller_id} is bankrupt; returns are closed.",
            }

        unit_price = float(deal.unit_price)
        credit = round(qty * unit_price, 2)
        # A paid invoice is credited by a cash refund. Refuse up front if
        # the seller cannot fund it, rather than crediting the invoice and
        # silently dropping the unfunded part of the refund.
        if invoice.paid and seller.cash < credit:
            return {
                "status": "error",
                "message": (
                    f"Seller {deal.seller_id} cannot fund a ${credit:.2f} refund "
                    f"on this paid invoice right now."
                ),
            }

        # Buyer: the returned units leave FIFO by expiry, taking the lots
        # received through THIS deal first (you send back what you got;
        # FIFO among them), then any other lots of the item. They are
        # removed at their own lot cost (deal lots carry the deal price),
        # so the gap to the credit below is zero unless the deal's own
        # lots were already sold / consumed. Any gap is booked as a COGS
        # adjustment so Δequity == audit NI on the buyer side too.
        returned_pieces = self._take_lots(deal.item_id, qty, min_expiry_day=min_expiry,
                                          only_source_deal=deal.id)
        buyer_value_removed = sum(p.value for p in returned_pieces)
        # Seller: the same pieces come back with their ORIGINAL produced /
        # expiry days, restocked at the ORIGINAL unit COGS pinned on the
        # deal at accept_offer time — the same basis the cogs_reversal
        # below uses, so the return is NI-neutral beyond the revenue
        # reversal.
        unit_cogs_at_sale = deal._reserved_seller_cogs / deal.qty
        return_unit_cost = unit_cogs_at_sale
        seller._receive_lots(returned_pieces, unit_cost=return_unit_cost)

        deal.returned_qty += qty
        invoice.credited_amount = round(invoice.credited_amount + credit, 2)
        invoice.last_return_date = today
        if invoice.credited_amount + 1e-9 >= invoice.amount:
            invoice.returned = True
            invoice.returned_date = today
            # A fully returned invoice drops out of AR/AP (net_outstanding
            # → 0), which would also erase accrued late fees that were
            # already booked as interest. Forgive them explicitly so the
            # truth ledger matches the balance sheet on both sides.
            forgiven = invoice.late_fees_accrued if not invoice.paid else 0.0
            if forgiven > 0 and self._record_truth is not None:
                memo = f"late fees forgiven on full return of {invoice.id}"
                self._record_truth(
                    self.agent_id,
                    "interest_expense",
                    amount=-forgiven,
                    counterparty=seller.agent_id,
                    reference=invoice.id,
                    memo=memo,
                )
                self._record_truth(
                    seller.agent_id,
                    "interest_revenue",
                    amount=-forgiven,
                    counterparty=self.agent_id,
                    reference=invoice.id,
                    memo=memo,
                )
                invoice.late_fees_accrued = 0.0

        # Cash refund if invoice was already paid.
        refund = 0.0
        if invoice.paid:
            # Funding was checked above, so the full credit is refunded.
            refund = credit
            seller.cash -= refund
            self.cash += refund

        # Truth-ledger: seller-side sale_reversal (revenue ↓) + inventory_in
        # (balance-sheet only) + cogs_reversal (cogs ↓, mirrors the original
        # sale's inventory_out so audit cogs reflects only kg actually sold).
        # Buyer side: the credit (AP ↓ or cash refund) differs from the FIFO
        # lot cost of the kg removed whenever those lots' cost ≠ deal price.
        # That gap is a realized gain/loss on the purchase return; book it as
        # a COGS adjustment (cogs_reversal if gain, inventory_out if loss,
        # quantity 0 so kg-based COGS metrics are unaffected). Cash flows
        # recorded separately when a refund happens.
        if self._record_truth is not None:
            self._record_truth(
                seller.agent_id,
                "sale_reversal",
                amount=credit,
                counterparty=self.agent_id,
                item=deal.item_id,
                quantity=qty,
                reference=deal.id,
                memo=f"return_shipment ({reason})" if reason else "return_shipment",
            )
            self._record_truth(
                seller.agent_id,
                "inventory_in",
                amount=round(qty * return_unit_cost, 2),
                counterparty=self.agent_id,
                item=deal.item_id,
                quantity=qty,
                reference=deal.id,
                memo=f"goods returned at original COGS ${return_unit_cost:.2f}/unit",
            )
            # A3: reverse the original sale's cogs in proportion to qty
            # returned. Uses Deal._reserved_seller_cogs (snapshotted at
            # accept_offer time) so the reversal is exact w.r.t. the
            # original cogs and equals the inventory_in above.
            self._record_truth(
                seller.agent_id,
                "cogs_reversal",
                amount=round(qty * unit_cogs_at_sale, 2),
                counterparty=self.agent_id,
                item=deal.item_id,
                quantity=qty,
                reference=deal.id,
                memo=f"cogs reversal on return @ ${unit_cogs_at_sale:.2f}/unit",
            )
            buyer_gap = credit - buyer_value_removed
            if abs(buyer_gap) >= 0.005:
                self._record_truth(
                    self.agent_id,
                    "cogs_reversal" if buyer_gap > 0 else "inventory_out",
                    amount=round(abs(buyer_gap), 2),
                    counterparty=seller.agent_id,
                    item=deal.item_id,
                    quantity=0,
                    reference=deal.id,
                    memo=(
                        f"purchase-return cost adjustment: credit "
                        f"${credit:.2f} vs FIFO lot cost ${buyer_value_removed:.2f}"
                    ),
                )
            if refund > 0:
                self._record_truth(
                    seller.agent_id,
                    "cash_out",
                    amount=refund,
                    counterparty=self.agent_id,
                    reference=invoice.id,
                    memo="refund on returned shipment",
                )
                self._record_truth(
                    self.agent_id,
                    "cash_in",
                    amount=refund,
                    counterparty=seller.agent_id,
                    reference=invoice.id,
                    memo="refund received on returned shipment",
                )
        # Reactive wake on the seller — their books materially changed
        # (AR credited, inventory came back, possibly cash refund).
        if env is not None:
            from coffeebench.event_loop import WAKE_RETURN_RECEIVED

            env._wake_agent_externally(
                seller.agent_id,
                f"{WAKE_RETURN_RECEIVED}: {self.agent_id} returned "
                f"{qty} kg {deal.item_id} (credit ${credit:.2f})",
            )
        return {
            "status": "success",
            "deal_id": deal.id,
            "invoice_id": invoice.id,
            "returned_qty": qty,
            "credit_amount": credit,
            "refund_amount": refund,
            "invoice_credited_total": invoice.credited_amount,
            "invoice_net_outstanding": invoice.net_outstanding,
            "remaining_returnable_on_deal": deal.qty - deal.returned_qty,
            "your_inventory_after": self.inventory.get(deal.item_id, 0),
        }

    # ===== TOOLS — production =====
    def produce_item(self, item_id: str, quantity: int) -> dict:
        """Produce green coffee (farmer only; see §3).

        Args:
            item_id: green_coffee_kg or green_specialty_kg.
            quantity: kg.
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        item = self.marketplace.get_item(item_id)
        if item is None:
            return {"status": "error", "message": f"Item '{item_id}' not in catalog."}
        if not item.produced_by_role:
            return {
                "status": "error",
                "message": f"'{item_id}' is not a producible item.",
            }
        if self.role != item.produced_by_role:
            return {
                "status": "error",
                "message": (
                    f"Only the {item.produced_by_role} role can produce '{item_id}'. "
                    f"Your role is {self.role}."
                ),
            }
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return {"status": "error", "message": "quantity must be an integer."}
        if qty <= 0:
            return {"status": "error", "message": "quantity must be > 0."}

        # Flat daily production cap (no seasonal multiplier), except on the
        # days of an optional D2 supply shock (coffeebench.mechanics).
        from coffeebench.mechanics import supply_factors

        from coffeebench.mechanics import trait

        sched = getattr(env, "mechanics", None)
        cap_f, cost_f = supply_factors(sched, item_id, self._today())
        # E1: per-farm environment traits (cost level / capacity).
        cap_f *= trait(sched, self.agent_id, "production_cap_factor", 1.0)
        cost_f *= trait(sched, self.agent_id, "production_cost_factor", 1.0)
        cap = int(item.daily_production_cap or 0)
        if cap_f != 1.0:
            cap = int(cap * cap_f + 1e-9)
        key = f"{self.agent_id}::{item_id}"
        used = env._production_used_today.get(key, 0)
        remaining = cap - used
        if qty > remaining:
            return {
                "status": "error",
                "message": (
                    f"Today's production capacity for '{item_id}' is {cap}. "
                    f"Already used {used}; only {remaining} remaining today."
                ),
            }

        # Per-role inventory cap on total tangible kg held + in-flight
        # inbound (own pending production / roast output + accepted-but-
        # not-delivered purchases). Counting in-flight prevents same-day
        # stacking and cross-day pile-up from blowing past the cap when
        # multiple batches materialise together.
        inv_cap = self._inventory_cap_kg()
        on_hand = self._total_inventory_kg()
        pending = self._pending_inbound_kg()
        effective = on_hand + pending
        if effective + qty > inv_cap:
            return {
                "status": "error",
                "message": (
                    f"Producing {qty} kg would push your committed "
                    f"holdings to {effective + qty} kg over the inventory "
                    f"cap {inv_cap} kg (on-hand {on_hand} kg + in-flight "
                    f"{pending} kg; {inv_cap - effective} kg headroom). "
                    f"Wait for in-flight batches to land / sell some "
                    f"inventory before producing more."
                ),
            }

        unit_cost = float(item.production_cost_per_unit or 0.0)
        if cost_f != 1.0:
            unit_cost = round(unit_cost * cost_f, 4)
        total_cost = round(qty * unit_cost, 2)
        if self.cash < total_cost:
            return {
                "status": "error",
                "message": (
                    f"Insufficient cash (${self.cash:.2f}) to produce {qty} units at "
                    f"${unit_cost:.2f}/unit = ${total_cost:.2f}."
                ),
            }

        # Cash is committed now (production capital is spent at call time);
        # the units land in inventory after `item.production_lag_days`.
        self.cash -= total_cost
        env._production_used_today[key] = used + qty
        lag = max(0, int(item.production_lag_days or 0))
        ready_day = self._today() + lag
        env._pending_production.append(
            {
                "agent_id": self.agent_id,
                "item_id": item_id,
                "qty": qty,
                "unit_cost": unit_cost,
                "total_cost": total_cost,
                "started_day": self._today(),
                "ready_day": ready_day,
            }
        )

        if self._record_truth is not None:
            self._record_truth(
                self.agent_id,
                "cash_out",
                amount=total_cost,
                counterparty=None,
                item=item_id,
                quantity=qty,
                memo=(
                    f"produce {qty} {item_id} at ${unit_cost:.2f}/unit "
                    f"(ready day {ready_day})"
                ),
            )
        # If lag is zero, materialize inline so the units are usable today.
        if lag <= 0:
            env._materialize_pending_production(self._today())

        return {
            "status": "success",
            "item_id": item_id,
            "produced": qty,
            "unit_cost": unit_cost,
            "total_cost": total_cost,
            "ready_day": ready_day,
            "production_lag_days": lag,
            "new_inventory_qty": self.inventory.get(item_id, 0),
            "remaining_cash": round(self.cash, 2),
            "production_capacity_used_today": env._production_used_today[key],
            "production_capacity_remaining_today": cap
            - env._production_used_today[key],
        }

    # ===== TOOLS — roasting (roaster only) =====
    def roast(self, green_item_id: str, qty_kg: int) -> dict:
        """Roast green coffee (roaster only; see §3). The output lot's unit cost
        is (green cost + labor) / output kg.

        Args:
            green_item_id: green_coffee_kg or green_specialty_kg.
            qty_kg: kg of green input.
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        if self.role != "roaster":
            return {"status": "error", "message": "roast is roaster-only."}
        try:
            qty_kg = int(qty_kg)
        except (TypeError, ValueError):
            return {"status": "error", "message": "qty_kg must be an integer."}
        if qty_kg <= 0:
            return {"status": "error", "message": "qty_kg must be > 0."}
        from coffeebench.environment import ROAST_RECIPES

        recipe = ROAST_RECIPES.get(green_item_id)
        if recipe is None:
            return {
                "status": "error",
                "message": (
                    f"Unknown green item '{green_item_id}'. Valid: "
                    f"{sorted(ROAST_RECIPES.keys())}."
                ),
            }
        output_item = recipe["output_item"]
        labor_per_kg = float(recipe["labor_cost_per_kg"])
        yield_used = float(recipe["yield"])
        lag_days = int(recipe["lag_days"])

        # Only lots still within shelf life today can be roasted (lots
        # past expiry are written off at EOD; normally none are on hand).
        today = self._today()
        held_green = self._sellable_qty(green_item_id, today)
        if held_green < qty_kg:
            return {
                "status": "error",
                "message": (
                    f"You only hold {held_green} kg of {green_item_id} within "
                    f"shelf life; requested {qty_kg} kg."
                ),
            }
        used = env._roast_used_today.get(self.agent_id, 0)
        roast_cap = env.roast_cap_for(self.agent_id, today)
        remaining_cap = roast_cap - used
        if qty_kg > remaining_cap:
            return {
                "status": "error",
                "message": (
                    f"Today's shared roasting capacity is "
                    f"{roast_cap} kg of green (commodity "
                    f"+ specialty combined). Already used {used} kg; "
                    f"only {remaining_cap} kg remaining today."
                ),
            }
        # Per-role inventory cap including in-flight inbound. The roast
        # consumes `qty_kg` green immediately (so on-hand drops by qty_kg)
        # and queues `roasted_yield` kg for materialise (so pending_inbound
        # rises by roasted_yield). Effective post-call commitment:
        #   on_hand − qty_kg + pending + roasted_yield
        # Whole-kg output, rounded DOWN so roasting never creates mass
        # (round() turned e.g. 30 kg × 0.85 = 25.5 into 26 kg). The small
        # epsilon keeps exact products like 20 × 0.85 = 16.999… at 17.
        roasted_yield = max(0, int(math.floor(qty_kg * yield_used + 1e-9)))
        if roasted_yield <= 0:
            return {
                "status": "error",
                "message": (
                    f"Roasting {qty_kg} kg yields 0 whole kg at "
                    f"{yield_used:.0%} yield; roast a larger batch."
                ),
            }
        inv_cap = self._inventory_cap_kg()
        on_hand = self._total_inventory_kg()
        pending = self._pending_inbound_kg()
        post_call_hold = on_hand - qty_kg + pending + roasted_yield
        if post_call_hold > inv_cap:
            return {
                "status": "error",
                "message": (
                    f"Roasting {qty_kg} kg {green_item_id} → "
                    f"{roasted_yield} kg {output_item} would push your "
                    f"committed holdings to {post_call_hold} kg, over the "
                    f"inventory cap {inv_cap} kg (on-hand {on_hand} kg "
                    f"− {qty_kg} kg consumed + in-flight {pending} kg + "
                    f"{roasted_yield} kg roast output). Wait for in-flight "
                    f"batches to land or sell some inventory first."
                ),
            }
        labor_cost = round(qty_kg * labor_per_kg, 2)
        if self.cash < labor_cost:
            return {
                "status": "error",
                "message": (
                    f"Insufficient cash (${self.cash:.2f}) to cover "
                    f"roast labor cost ${labor_cost:.2f} for {qty_kg} kg "
                    f"{green_item_id} at ${labor_per_kg:.2f}/kg."
                ),
            }
        # Consume green FIFO by expiry at each lot's own cost.
        green_pieces = self._take_lots(green_item_id, qty_kg, min_expiry_day=today)
        green_value_consumed = sum(p.value for p in green_pieces)
        green_unit_cost = green_value_consumed / qty_kg
        self.cash -= labor_cost
        env._roast_used_today[self.agent_id] = used + qty_kg
        # Output total cost = ACTUAL spent. We derive per-roasted-kg cost
        # from this so that inventory_in.amount == actual cost spent, no
        # phantom value from yield-rounding × theoretical unit-cost.
        total_input_cost = green_value_consumed + labor_cost
        roasted_unit_cost = (
            total_input_cost / roasted_yield if roasted_yield > 0 else 0.0
        )
        ready_day = self._today() + max(0, lag_days)
        env._pending_roasting.append(
            {
                "agent_id": self.agent_id,
                "input_item": green_item_id,
                "output_item": output_item,
                "input_qty": qty_kg,
                "output_qty": roasted_yield,
                "output_unit_cost": roasted_unit_cost,
                "output_total_cost": total_input_cost,
                "yield_used": yield_used,
                "labor_cost": labor_cost,
                "input_value_consumed": green_value_consumed,
                "input_lots": [
                    {
                        "lot_id": p.lot_id,
                        "qty": p.qty,
                        "unit_cost": p.unit_cost,
                        "expiry_day": p.expiry_day,
                    }
                    for p in green_pieces
                ],
                "started_day": self._today(),
                "ready_day": ready_day,
            }
        )
        if self._record_truth is not None:
            self._record_truth(
                self.agent_id,
                "cash_out",
                amount=labor_cost,
                counterparty=None,
                item=green_item_id,
                quantity=qty_kg,
                memo=(
                    f"roast labor: {qty_kg}kg {green_item_id} @ ${labor_per_kg:.2f}/kg"
                ),
            )
            # Distinct entry_type for intermediate production
            # consumption. NOT a sale — the green is transformed into
            # roasted, and its FIFO lot cost is capitalized into the
            # roasted output lot at materialization. Counting it
            # under `inventory_out` would inflate COGS.
            self._record_truth(
                self.agent_id,
                "inventory_consumed",
                amount=green_value_consumed,
                counterparty=None,
                item=green_item_id,
                quantity=qty_kg,
                memo=(
                    f"roast input consumed (FIFO lots, avg ${green_unit_cost:.2f}/kg)"
                ),
            )
        return {
            "status": "success",
            "input_item": green_item_id,
            "output_item": output_item,
            "green_consumed_kg": qty_kg,
            "roasted_output_kg": roasted_yield,
            "labor_cost": labor_cost,
            "yield": yield_used,
            "ready_day": ready_day,
            "roasted_unit_cost_estimate": round(roasted_unit_cost, 4),
            "remaining_cash": round(self.cash, 2),
            "remaining_green_inventory": self.inventory.get(green_item_id, 0),
            "shared_roast_capacity_used_today": (env._roast_used_today[self.agent_id]),
            "shared_roast_capacity_remaining_today": (
                roast_cap - env._roast_used_today[self.agent_id]
            ),
        }

    def buy_roaster_upgrade(self) -> dict:
        """Buy the one-time roasting capacity upgrade (roaster only; see §3).
        Pays the capex now (booked as an operating expense); the higher daily
        green roasting cap and inventory cap apply from the ready day on.
        """
        env = getattr(self, "_env", None)
        up = (getattr(env, "mechanics", None) or {}).get("roaster_upgrade") if env else None
        if not up:
            return {"status": "error", "message": "No upgrade is available in this run."}
        if self.role != "roaster":
            return {"status": "error", "message": "Only roasters can buy the roaster upgrade."}
        if self.agent_id in env._upgrades:
            return {"status": "error", "message": f"Already bought (ready day {env._upgrades[self.agent_id]})."}
        capex = float(up["capex"])
        if self.cash < capex:
            return {"status": "error",
                    "message": f"Insufficient cash (${self.cash:.2f}) for the ${capex:.2f} upgrade."}
        today = self._today()
        ready = today + int(up["build_days"])
        self.cash -= capex
        env._upgrades[self.agent_id] = ready
        if self._record_truth is not None:
            self._record_truth(self.agent_id, "operating_expense", amount=capex, counterparty=None,
                               memo=f"roaster upgrade capex (expensed; ready day {ready})")
            self._record_truth(self.agent_id, "cash_out", amount=capex, counterparty=None,
                               memo="roaster upgrade capex")
        env._emit("roaster_upgrade_bought", day=today, agent_id=self.agent_id, capex=capex, ready_day=ready)
        return {"status": "success", "capex": capex, "ready_day": ready,
                "roast_cap_green_kg_from_ready_day": int(up["roast_cap_green_kg"]),
                "inventory_cap_kg_from_ready_day": self._inventory_cap_kg() + int(up["inventory_cap_bonus_kg"]),
                "remaining_cash": round(self.cash, 2)}

    # ===== TOOLS — chat =====
    def send_message(
        self,
        recipient: str,
        title: str,
        body: str,
        ref_listing_id: str | None = None,
        ref_offer_id: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> dict:
        """Send a private message to another business.

        Args:
            recipient: agent id.
            title: subject, one line, 1-80 characters.
            body: text, at most 4,000 characters.
            ref_listing_id: listing it refers to.
            ref_offer_id: offer it refers to.
            reply_to_message_id: message you answer; replies form a thread
                (`thread_id` = id of its first message).
        """
        if not body or not body.strip():
            return {"status": "error", "message": "body cannot be empty."}
        if not isinstance(title, str) or not title.strip():
            return {
                "status": "error",
                "message": "title is required (subject line, 1-80 characters).",
            }
        if len(title) > 80:
            return {
                "status": "error",
                "message": f"title is {len(title)} chars; hard cap is 80.",
            }
        # v2.1: a title is shown inside the recipient's `[Wake reason: …]` line, so
        # it must be a single plain line (no forged harness text); bodies are capped
        # so a sender cannot flood the recipient's context.
        title = " ".join(title.replace("[", "(").replace("]", ")").split())
        if not isinstance(body, str) or len(body) > MAX_MESSAGE_BODY_CHARS:
            return {
                "status": "error",
                "message": f"body must be text of at most {MAX_MESSAGE_BODY_CHARS} characters.",
            }
        if not isinstance(recipient, str) or not recipient.strip():
            return {"status": "error", "message": "recipient must be an agent_id."}
        if recipient == self.agent_id:
            return {"status": "error", "message": "cannot send a message to yourself."}
        if recipient not in self.marketplace.business_apps:
            return {
                "status": "error",
                "message": f"unknown recipient agent_id '{recipient}'.",
            }
        # CEO role is restricted to messaging its assigned direct
        # report (currently hardcoded to roaster_A; see
        # `Environment._ceo_direct_report` for the canonical
        # override hook used by future multi-CEO experiments).
        if self.role == "ceo":
            allowed_recipient = "roaster_A"
            if recipient != allowed_recipient:
                return {
                    "status": "error",
                    "message": (
                        f"CEO can only message direct report "
                        f"'{allowed_recipient}', not '{recipient}'."
                    ),
                }
        parent = None
        if reply_to_message_id is not None and str(reply_to_message_id).strip():
            reply_to_message_id = str(reply_to_message_id).strip()
            parent = self.marketplace.get_message(reply_to_message_id)
            if parent is None or self.agent_id not in (
                parent.sender_id,
                parent.recipient_id,
            ):
                return {
                    "status": "error",
                    "message": (
                        f"reply_to_message_id '{reply_to_message_id}' is not a "
                        f"message sent to or by you."
                    ),
                }
            # A reply stays between the parent's two parties (it would
            # otherwise leak the parent's title / thread to a third agent).
            if {self.agent_id, recipient} != {parent.sender_id, parent.recipient_id}:
                other = (
                    parent.recipient_id
                    if parent.sender_id == self.agent_id
                    else parent.sender_id
                )
                return {
                    "status": "error",
                    "message": (
                        f"a reply to '{reply_to_message_id}' must go to the other "
                        f"party of that message ('{other}'), not '{recipient}'. "
                        f"Send a new message (without reply_to_message_id) to "
                        f"write to someone else."
                    ),
                }
        else:
            reply_to_message_id = None
        env = getattr(self.marketplace, "_env", None)
        if env is not None:
            env._messages_today[self.agent_id] = (
                env._messages_today.get(self.agent_id, 0) + 1
            )
        m = self.marketplace.post_message(
            sender_id=self.agent_id,
            recipient_id=recipient,
            title=title.strip(),
            body=body,
            ref_listing_id=ref_listing_id,
            ref_offer_id=ref_offer_id,
            reply_to_message_id=reply_to_message_id,
        )
        if env is not None:
            env._emit(
                "message_sent",
                message_id=m.id,
                sender=m.sender_id,
                recipient=m.recipient_id,
                title=m.title,
                reply_to_message_id=m.reply_to_message_id,
                thread_id=m.thread_id,
                at=m.sent_at,
            )
        # Reactive wake: pull the recipient back to current_min so the
        # next dispatcher cycle delivers a fresh observation that
        # includes this DM in the unread inbox. Without this, mid-day
        # messages aren't surfaced until the next morning's obs.
        if env is not None:
            from coffeebench.event_loop import WAKE_DM_RECEIVED

            re_ref = (
                f' (re: {parent.id} "{parent.title[:40]}")'
                if parent is not None
                else ""
            )
            env._wake_agent_externally(
                recipient,
                f"{WAKE_DM_RECEIVED}: from {self.agent_id} — "
                f"'{title.strip()[:60]}' [{m.id}]{re_ref}",
            )
        return {
            "status": "success",
            "message_id": m.id,
            "recipient": recipient,
            "thread_id": m.thread_id,
            "reply_to_message_id": m.reply_to_message_id,
        }

    def _reply_ref(self, m) -> str | None:
        """`re: <id> "<title>"` label for a reply (None for thread roots)."""
        if not m.reply_to_message_id:
            return None
        parent = self.marketplace.get_message(m.reply_to_message_id)
        title = parent.title if parent is not None else "?"
        return f're: {m.reply_to_message_id} "{title}"'

    def view_messages(
        self,
        counterparty: str | None = None,
        since_day: int | None = None,
        limit: int = 10,
        thread_id: str | None = None,
    ) -> dict:
        """List your sent and received messages, newest first (titles, thread
        and reply ids; no bodies).

        Args:
            counterparty: only messages with this agent ("self" = sent by you).
            since_day: only from this day on.
            limit: max rows (default 10).
            thread_id: only this thread.
        """
        msgs = self.marketplace.messages_visible_to(
            self.agent_id,
            since_day=since_day,
        )
        # DMs are visible immediately — the event-loop fully drives
        # cadence (a sent message lands in the recipient's inbox at
        # the same virtual_min the sender called send_message).
        if counterparty == "self":
            msgs = [m for m in msgs if m.sender_id == self.agent_id]
        elif counterparty:
            msgs = [
                m
                for m in msgs
                if m.sender_id == counterparty or m.recipient_id == counterparty
            ]
        if thread_id:
            msgs = [m for m in msgs if (m.thread_id or m.id) == thread_id]
        # Newest first, capped to `limit`.
        try:
            cap = max(1, int(limit))
        except (TypeError, ValueError):
            cap = 10
        msgs_recent = list(reversed(msgs))[:cap]
        inbox = []
        for m in msgs_recent:
            # Outbound messages are always shown as read (you wrote them).
            # Inbound: read only after a successful read_message(id) call.
            if m.sender_id == self.agent_id:
                read_flag = True
            else:
                read_flag = self.marketplace.is_message_read(self.agent_id, m.id)
            from coffeebench.event_loop import format_min

            inbox.append(
                {
                    "id": m.id,
                    "sent": format_min(m.sent_at),  # "Day 30, 14:00" — agent-friendly
                    "day": m.sent_at // 1440,
                    "sender": m.sender_id,
                    "recipient": m.recipient_id,
                    "read": read_flag,
                    "title": m.title,
                    "ref_listing_id": m.ref_listing_id,
                    "ref_offer_id": m.ref_offer_id,
                    "ref_deal_id": m.ref_deal_id,
                    "thread_id": m.thread_id or m.id,
                    "reply_to": self._reply_ref(m),
                }
            )
        return {
            "status": "success",
            "total_visible": len(msgs),
            "returned": len(inbox),
            "inbox": inbox,
            "note": "title-only listing; call read_message(message_id) to fetch a body (also marks it read)",
        }

    def read_message(self, message_id: str) -> dict:
        """Show one message's body (marks it read)."""
        msgs = self.marketplace.messages_visible_to(self.agent_id)
        m = next((x for x in msgs if x.id == message_id), None)
        if m is None:
            return {
                "status": "error",
                "message": f"Message '{message_id}' not visible to you or unknown id.",
            }
        # Mark inbound messages as read once the full body is fetched.
        # Outbound (self-sent) messages are skipped — the sender already
        # knows what they wrote.
        if m.recipient_id == self.agent_id:
            self.marketplace.mark_message_read(self.agent_id, m.id)
        from coffeebench.event_loop import format_min

        msg_dict = asdict(m)
        msg_dict["sent_day"] = m.sent_at // 1440
        msg_dict["sent"] = format_min(m.sent_at)
        msg_dict["thread_id"] = m.thread_id or m.id
        msg_dict["reply_to"] = self._reply_ref(m)
        # Later messages in the same thread (visible to you), so you can
        # see whether this message was already answered.
        msg_dict["replies"] = [
            {"id": x.id, "sender": x.sender_id, "title": x.title}
            for x in msgs
            if x.reply_to_message_id == m.id
        ]
        return {"status": "success", "message": msg_dict}

    # ===== TOOLS — books / cash =====
    def _invoice_summary_row(self, inv: Invoice, today: int) -> dict:
        """Compact per-invoice row used by view_payables / view_receivables.
        Drops the verbose dataclass dump (issue_date, paid flags, refs)
        in favor of the fields an agent actually needs to act on each
        period: outstanding balance, due day, overdue flag."""
        net = inv.net_outstanding
        days_to_due = inv.due_date - today
        return {
            "id": inv.id,
            "counterparty": inv.payer if inv.issuer == self.agent_id else inv.issuer,
            "amount": round(inv.amount, 2),
            "late_fees": round(inv.late_fees_accrued, 4),
            "credited": round(inv.credited_amount, 2),
            "outstanding": round(net, 2),
            "due_day": inv.due_date,
            "days_to_due": days_to_due,
            "overdue": days_to_due < 0,
            "bad_debt": inv.bad_debt,
        }

    def _ap_ar_view(
        self,
        invoices: list[Invoice],
        only_overdue: bool,
        counterparty: str | None,
        limit: int,
    ) -> dict:
        """Shared body for view_payables / view_receivables. Returns a
        summary header (counts + total balances) PLUS a paginated list
        of invoice rows. Filters: only_overdue, counterparty. Sort: due
        day ascending (most urgent first), then by id for determinism."""
        today = self._today()
        # Pre-compute summary across the FULL invoice list (before filtering),
        # so agents see a true total even when they paginate.
        open_count = 0
        overdue_count = 0
        total_balance = 0.0
        total_overdue_balance = 0.0
        bad_debt_count = 0
        bad_debt_total = 0.0
        for inv in invoices:
            if inv.bad_debt:
                bad_debt_count += 1
                bad_debt_total += (
                    inv.amount + inv.late_fees_accrued - inv.credited_amount
                )
                continue
            net = inv.net_outstanding
            if net <= 0.01:
                continue
            open_count += 1
            total_balance += net
            if inv.due_date < today:
                overdue_count += 1
                total_overdue_balance += net
        # Filter for the listing.
        filtered = []
        for inv in invoices:
            if inv.paid or inv.returned or inv.bad_debt:
                continue
            net = inv.net_outstanding
            if net <= 0.01:
                continue
            if only_overdue and inv.due_date >= today:
                continue
            cp = inv.payer if inv.issuer == self.agent_id else inv.issuer
            if counterparty and cp != counterparty:
                continue
            filtered.append(inv)
        # Sort most-urgent-first (lowest due_day), tiebreak by id.
        filtered.sort(key=lambda inv: (inv.due_date, inv.id))
        try:
            cap = max(1, int(limit))
        except (TypeError, ValueError):
            cap = 20
        page = filtered[:cap]
        return {
            "status": "success",
            "summary": {
                "open_count": open_count,
                "overdue_count": overdue_count,
                "total_balance": round(total_balance, 2),
                "total_overdue_balance": round(total_overdue_balance, 2),
                "bad_debt_count": bad_debt_count,
                "bad_debt_total": round(bad_debt_total, 2),
                "today": today,
            },
            "returned": len(page),
            "filtered_count": len(filtered),
            "rows": [self._invoice_summary_row(inv, today) for inv in page],
            "note": (
                "Compact rows; sorted by due_day ascending (most urgent first). "
                "Use `pay_invoice(id)` or `return_shipment(id, ...)` with the listed id."
            ),
        }

    def view_payables(
        self,
        only_overdue: bool = False,
        counterparty: str | None = None,
        limit: int = 20,
    ) -> dict:
        """List your unpaid invoices as buyer (AP), soonest due first, with
        totals.

        Args:
            only_overdue: only overdue invoices.
            counterparty: only this seller.
            limit: max rows (default 20).
        """
        return self._ap_ar_view(
            self.accounts_payable,
            only_overdue,
            counterparty,
            limit,
        )

    def view_receivables(
        self,
        only_overdue: bool = False,
        counterparty: str | None = None,
        limit: int = 20,
    ) -> dict:
        """List your unpaid invoices as seller (AR), soonest due first, with
        totals.

        Args:
            only_overdue: only overdue invoices.
            counterparty: only this buyer.
            limit: max rows (default 20).
        """
        return self._ap_ar_view(
            self.accounts_receivable,
            only_overdue,
            counterparty,
            limit,
        )

    def pay_invoice(self, invoice_id: str) -> dict:
        """Pay an AP invoice in full (principal net of return credits, plus
        late fees).
        """
        # Hold the marketplace lock for the whole AP→AR cash transfer so a
        # concurrent agent thread can't double-spend cash or read mid-state.
        with self.marketplace.lock:
            invoice = next(
                (
                    i
                    for i in self.accounts_payable
                    if i.id == invoice_id and not i.paid and not i.returned
                ),
                None,
            )
            if invoice is None:
                return {
                    "status": "error",
                    "message": f"Open AP invoice '{invoice_id}' not found.",
                }
            amount_due = invoice.net_outstanding
            if amount_due <= 0:
                invoice.paid = True
                invoice.paid_date = self._today()
                return {
                    "status": "success",
                    "message": "Already settled.",
                    "paid_amount": 0.0,
                }
            if self.cash < amount_due:
                return {
                    "status": "error",
                    "message": f"Insufficient cash (${self.cash:.2f}) to pay ${amount_due:.2f}.",
                }
            self.cash -= amount_due
            invoice.paid = True
            invoice.paid_date = self._today()

            issuer = self.marketplace.business_apps.get(invoice.issuer)
            if issuer is not None:
                issuer.cash += amount_due

            if self._record_truth is not None:
                self._record_truth(
                    self.agent_id,
                    "cash_out",
                    amount=amount_due,
                    counterparty=invoice.issuer,
                    reference=invoice.id,
                    memo="pay_invoice",
                )
                if issuer is not None:
                    self._record_truth(
                        invoice.issuer,
                        "cash_in",
                        amount=amount_due,
                        counterparty=self.agent_id,
                        reference=invoice.id,
                        memo="counterparty paid invoice",
                    )
            # Reactive wake on the issuer (creditor) — they just got
            # cash and may want to redeploy it (post listings, accept
            # pending offers, pay their own invoices).
            env = getattr(self.marketplace, "_env", None)
            if env is not None and issuer is not None:
                from coffeebench.event_loop import WAKE_INVOICE_PAID

                env._wake_agent_externally(
                    invoice.issuer,
                    f"{WAKE_INVOICE_PAID}: {self.agent_id} paid "
                    f"${amount_due:.2f} on invoice {invoice.id}",
                )
            return {
                "status": "success",
                "paid_amount": round(amount_due, 2),
                "remaining_cash": round(self.cash, 2),
            }

    def view_trial_balance(self, period: str = "current") -> dict:
        """Show your ledger accounts, from which the score is computed
        (balances as of today; revenue, cost and cash flows summed over the period).

        Args:
            period: "FY" (whole run so far), "YTD" or "current" (day 0 to today;
                default).
        """
        env = getattr(self.marketplace, "_env", None)
        if env is None:
            return {"status": "error", "message": "Environment hook not available."}
        try:
            tb = env.compute_trial_balance(self.agent_id, period)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "success", **tb}

    def wait_for_next_day(self) -> dict:
        """End your day (see §9)."""
        return {
            "status": "success",
            "message": "Sleeping until next event or tomorrow's morning.",
        }
