"""BusinessApp — per-agent toolset for the deal marketplace.

Wraps the shared `Marketplace` (one instance per run) and the agent's
own private books (cash, inventory, AR/AP). Public methods on this
class are auto-registered as tools the LLM agent can call.
"""

from dataclasses import asdict
from typing import Callable, Optional

from coffeebench.time_manager import TimeManager
from coffeebench.typings import Invoice

from coffeebench.marketplace import Marketplace
from coffeebench.typings import AgentEndowment


class _CostBasisView:
    """Read-only WAVG view derived from inventory + inventory_total_cost.

    The bench tracks `inventory_total_cost` (total $ held) as the source
    of truth and exposes per-unit cost as `inventory_total_cost / qty`
    on demand. Maintains the existing `.cost_basis.get(item, default)`
    API used across the codebase (env, business_app, tests). Writes are
    intentionally not supported — callers must update
    `inventory_total_cost` directly so that the qty × total invariant
    stays exact (no phantom-from-WAVG-rounding artefacts).
    """

    __slots__ = ("_inventory", "_total_cost")

    def __init__(self, inventory: dict[str, int], total_cost: dict[str, float]):
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
        # {item_id: qty}
        self.inventory: dict[str, int] = dict(endowment.initial_inventory)
        # {item_id: total $ value of currently-held units}. The source of
        # truth for inventory valuation under the WAVG accounting scheme.
        # Always co-mutated with `self.inventory` so that
        #   per-unit avg = inventory_total_cost[item] / inventory[item]
        # `cost_basis` below is a read-only view; do not write to it.
        self.inventory_total_cost: dict[str, float] = {
            item_id: float(qty) * float(endowment.initial_cost_basis.get(item_id, 0.0))
            for item_id, qty in self.inventory.items()
        }
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

    def _ar_outstanding(self) -> float:
        return sum(inv.net_outstanding for inv in self.accounts_receivable)

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

        return int(ROLE_INVENTORY_CAP_KG.get(self.role, 10**9))

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
        """Post a public listing on the marketplace. All other agents will see it.

        Args:
            item_id: id of the item from your inventory you want to sell.
            qty: units to offer (must not exceed your current inventory of that item).
            asking_price: PER-UNIT asking price in USD.
            payment_terms_days: net-N from delivery (default 30). Pass a larger
                number to extend (slower cash collection); pass a smaller one
                to demand faster payment.
        """
        try:
            qty = int(qty)
            asking_price = float(asking_price)
            payment_terms_days = int(payment_terms_days)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "qty must be int, asking_price float, payment_terms_days int.",
            }
        if qty <= 0:
            return {"status": "error", "message": "qty must be > 0."}
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
        return {
            "status": "success",
            "listing_id": listing.id,
            "asking_price": asking_price,
            "qty": qty,
        }

    def view_listings(
        self, item_id: str | None = None, only_others: bool = True
    ) -> dict:
        """List currently OPEN listings on the marketplace.

        Args:
            item_id: filter to a single item id (optional).
            only_others: if True (default), hide listings YOU posted.
        """
        rows = self.marketplace.listings_dump()
        out = []
        for r in rows:
            if item_id is not None and r["item_id"] != item_id:
                continue
            if only_others and r["seller_id"] == self.agent_id:
                continue
            out.append(r)
        return {"status": "success", "count": len(out), "listings": out}

    def make_offer(
        self,
        listing_id: str,
        offered_price: float,
        qty: int,
        payment_terms_days: int = 30,
        message: str = "",
    ) -> dict:
        """Post an offer (bid) on someone else's listing. The seller can accept, reject, or counter via message.

        Args:
            listing_id: target listing id.
            offered_price: PER-UNIT price you're willing to pay.
            qty: how many units you want (≤ listing qty).
            payment_terms_days: net-N you're proposing for payment (default 30).
            message: optional message attached to the offer.
        """
        try:
            offered_price = float(offered_price)
            qty = int(qty)
            payment_terms_days = int(payment_terms_days)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Type error in offer params."}
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
        """Withdraw an offer YOU previously made (only while it's still pending)."""
        ok, msg = self.marketplace.withdraw_offer(offer_id, self.agent_id)
        return {"status": "success" if ok else "error", "message": msg}

    def accept_offer(self, offer_id: str) -> dict:
        """Accept an offer made on YOUR listing. Creates a binding deal: inventory transfers next day, an invoice is issued (your AR / buyer's AP) with the agreed payment terms.

        Args:
            offer_id: pending offer on a listing you own.
        """
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
        """View offers involving you.

        Args:
            direction: "incoming" (offers on your listings, you decide), "outgoing" (offers you made), or "all".
        """
        return {
            "status": "success",
            "offers": self.marketplace.offers_for(self.agent_id, direction=direction),
        }

    def view_deals(self) -> dict:
        """List every closed deal you participated in (as buyer or seller)."""
        from coffeebench.event_loop import format_min

        out = []
        for d in self.marketplace.deals:
            if d.seller_id != self.agent_id and d.buyer_id != self.agent_id:
                continue
            row = asdict(d)
            row["deal_day"] = d.deal_at // 1440
            row["delivery_day"] = d.delivery_at // 1440
            row["accepted"] = format_min(d.deal_at)
            row["delivers"] = format_min(d.delivery_at)
            out.append(row)
        return {"status": "success", "count": len(out), "deals": out}

    # ===== TOOLS — retail (shop-only effective) =====
    def set_retail_price(self, item_id: str, price_per_unit: float) -> dict:
        """Set your storefront price for a consumer-facing item. The env
        runs a scripted consumer-demand model each day across all
        retailers; consumers buy at the price you set, capped by your
        on-hand inventory of that item, only if your price is at or
        below their reservation price. Setting `price_per_unit=0`
        delists the item.

        Two consumer-facing items: `roasted_coffee_kg` (commodity,
        reservation $30/kg) and `roasted_specialty_kg` (premium,
        reservation $80/kg). Price each independently. Pricing above
        the reservation forfeits BOTH the elastic share AND the
        inelastic-floor demand for that item that day. COGS is
        deducted at WAVG cost on every sale.

        Args:
            item_id: id of a consumer-facing item (`roasted_coffee_kg`
                or `roasted_specialty_kg`).
            price_per_unit: USD per kg. Set to 0 to delist.
        """
        try:
            price = float(price_per_unit)
        except (TypeError, ValueError):
            return {"status": "error", "message": "price_per_unit must be numeric."}
        if price < 0:
            return {"status": "error", "message": "price_per_unit must be ≥ 0."}
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
        """Return YOUR shop's consumer-sale history (env auto-sells from
        your inventory at the retail prices you set; this tool lets you
        read those sales back).

        Each row carries: day, item_id, qty, unit_price, total_price,
        share. Retailer-only — other roles see no sales here.

        Args:
            since_day: only sales on or after this day.
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
        """Snapshot total CONSUMER-side market activity over the last
        `days_back` days, aggregated across BOTH retailers (i.e. the
        whole market — not just your own sales). One row per
        (item, day) reporting total qty sold and average per-kg price
        achieved across the market that day. Useful for forecasting
        demand and pricing pressure regardless of role: farmers /
        roasters can plan B2B supply against observed downstream
        consumer pull, and retailers can benchmark their own share
        against the aggregate.
        Args:
            days_back: integer days back from today to include
                (e.g. 7 = last 7 days incl. today). Clamped to ≥ 1.
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
        return {
            "status": "success",
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
        """Return some or all of a delivered tangible-goods order against an
        invoice. Only available to the original BUYER, and only within
        the env's `RETURN_WINDOW_DAYS` from the invoice's issue_date.

        On a successful return:
          - The returned qty leaves your inventory and goes back to the seller's.
          - The invoice's `credited_amount` grows by qty × original unit_price.
            If the credited amount equals the principal, the invoice is flagged
            `returned`. If you had already paid the invoice, the seller refunds
            you the credited amount in cash.
          - Truth ledger: seller logs `sale_reversal` and `inventory_in`; buyer
            logs `inventory_out` (and `cash_in` for any refund). The seller's
            recognized revenue effectively drops by qty × unit_price.

        Args:
            invoice_id: AP invoice id you want to return against (must be a
                tangible-goods deal you bought).
            quantity_kg: integer units to return. Bounded by `coffeebench.qty - already_returned`
                AND by the qty you currently hold in inventory.
            reason: optional free-form note (e.g. "quality issue", "over-stock").
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
        # Inventory cap.
        on_hand = self.inventory.get(deal.item_id, 0)
        if qty > on_hand:
            return {
                "status": "error",
                "message": f"You hold only {on_hand} units of '{deal.item_id}' — cannot return {qty}.",
            }

        # Locate seller. Marketplace.business_apps is the env-wired registry.
        seller = self.marketplace.business_apps.get(deal.seller_id)
        if seller is None:
            return {
                "status": "error",
                "message": "Seller's BusinessApp not registered.",
            }

        unit_price = float(deal.unit_price)
        credit = round(qty * unit_price, 2)

        # Apply: buyer inventory --> seller inventory (at seller's pre-sale WAVG).
        # We approximate seller's original COGS-per-unit by storing it on the
        # truth ledger via the original sale's inventory_out entry; for the
        # current value we use the seller's CURRENT WAVG cost, which is the
        # closest available proxy. (Returned goods rejoining inventory blend
        # into seller's stock at this proxy cost, so a back-and-forth doesn't
        # gradually shift the basis far from reality.)
        # Buyer: drop qty + proportional total_cost.
        buyer_avg = (
            (self.inventory_total_cost.get(deal.item_id, 0.0) / on_hand)
            if on_hand > 0
            else 0.0
        )
        buyer_value_removed = qty * buyer_avg
        self.inventory[deal.item_id] = on_hand - qty
        self.inventory_total_cost[deal.item_id] = (
            self.inventory_total_cost.get(deal.item_id, 0.0) - buyer_value_removed
        )
        # Seller: blend returned units back at seller's current WAVG.
        old_qty = seller.inventory.get(deal.item_id, 0)
        old_total = seller.inventory_total_cost.get(deal.item_id, 0.0)
        old_cost = (old_total / old_qty) if old_qty > 0 else 0.0
        return_unit_cost = old_cost
        return_value = qty * return_unit_cost
        seller.inventory[deal.item_id] = old_qty + qty
        seller.inventory_total_cost[deal.item_id] = old_total + return_value

        deal.returned_qty += qty
        invoice.credited_amount = round(invoice.credited_amount + credit, 2)
        invoice.last_return_date = today
        if invoice.credited_amount + 1e-9 >= invoice.amount:
            invoice.returned = True
            invoice.returned_date = today

        # Cash refund if invoice was already paid.
        refund = 0.0
        if invoice.paid:
            refund = round(min(credit, seller.cash), 2)
            # If seller is broke, they refund what they have; the rest stays
            # owed informally — no AP/AR mechanic for refund balance in v1.
            seller.cash -= refund
            self.cash += refund

        # Truth-ledger: seller-side sale_reversal (revenue ↓) + inventory_in
        # (balance-sheet only) + cogs_reversal (cogs ↓, mirrors the original
        # sale's inventory_out so audit cogs reflects only kg actually sold).
        # Buyer side has NO income-statement entry — a purchase return is not
        # a sale, so it carries no cogs; the AP credit + inventory drop already
        # close the books on the equity side. Cash flows recorded separately
        # when a refund happens.
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
                memo=f"goods returned at WAVG ${return_unit_cost:.2f}/unit",
            )
            # A3: reverse the original sale's cogs in proportion to qty
            # returned. Uses Deal._reserved_seller_cogs (snapshotted at
            # accept_offer time) so the reversal is exact w.r.t. the
            # original cogs, not the seller's current WAVG.
            unit_cogs_at_sale = deal._reserved_seller_cogs / deal.qty
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
        """Produce more units of a producible item from your cash. Each item
        carries `produced_by_role` (only that role can mint it),
        `production_cost_per_unit` (paid from your cash at call time),
        `daily_production_cap` (per-agent per-item kg/day cap, resets at
        the start of each day), and `production_lag_days` (delay between
        the call and the units landing in inventory). Output blends into
        your WAVG cost basis at the production cost when it materialises.

        Args:
            item_id: id of an item whose `produced_by_role` matches your role.
            quantity: integer units to produce, > 0, ≤ today's remaining
                capacity for this item, and ≤ what your cash covers.
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

        # Flat daily production cap (no seasonal multiplier).
        cap = int(item.daily_production_cap or 0)
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
        """Convert green beans from your inventory into roasted beans.
        Roaster-role only. One unified action covers BOTH product
        lines via `green_item_id`:

          - `green_coffee_kg`    → `roasted_coffee_kg`    (commodity:
                                   $3/kg labor, yield 0.85, 1-day lag).
          - `green_specialty_kg` → `roasted_specialty_kg` (premium:
                                   $5/kg labor, yield 0.82, 1-day lag).

        Both recipes SHARE one 50 kg/day total green-input cap (same
        physical equipment). Cash + green leave immediately; the
        roasted output lands in inventory after the recipe's lag.
        Effective COGS per roasted kg = (green WAVG + labor) / yield,
        blended into the output's WAVG cost basis at materialize time.

        Args:
            green_item_id: input item id (`green_coffee_kg` or
                `green_specialty_kg`).
            qty_kg: integer kg of green beans to roast. Must be > 0,
                ≤ today's remaining shared cap, ≤ green inventory you
                hold of that item, and your cash must cover labor.
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
        from coffeebench.environment import (
            ROAST_DAILY_CAP_GREEN_KG,
            ROAST_RECIPES,
        )

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

        held_green = self.inventory.get(green_item_id, 0)
        if held_green < qty_kg:
            return {
                "status": "error",
                "message": (
                    f"You only hold {held_green} kg of {green_item_id}; "
                    f"requested {qty_kg} kg."
                ),
            }
        used = env._roast_used_today.get(self.agent_id, 0)
        remaining_cap = ROAST_DAILY_CAP_GREEN_KG - used
        if qty_kg > remaining_cap:
            return {
                "status": "error",
                "message": (
                    f"Today's shared roasting capacity is "
                    f"{ROAST_DAILY_CAP_GREEN_KG} kg of green (commodity "
                    f"+ specialty combined). Already used {used} kg; "
                    f"only {remaining_cap} kg remaining today."
                ),
            }
        # Per-role inventory cap including in-flight inbound. The roast
        # consumes `qty_kg` green immediately (so on-hand drops by qty_kg)
        # and queues `roasted_yield` kg for materialise (so pending_inbound
        # rises by roasted_yield). Effective post-call commitment:
        #   on_hand − qty_kg + pending + roasted_yield
        roasted_yield = max(0, int(round(qty_kg * yield_used)))
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
        # Consume green at current WAVG; deduct both qty and total cost.
        green_total = self.inventory_total_cost.get(green_item_id, 0.0)
        green_unit_cost = (green_total / held_green) if held_green > 0 else 0.0
        green_value_consumed = qty_kg * green_unit_cost
        self.inventory[green_item_id] = held_green - qty_kg
        self.inventory_total_cost[green_item_id] = green_total - green_value_consumed
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
            # roasted, and its WAVG cost is capitalized into the
            # roasted output's basis at materialization. Counting it
            # under `inventory_out` would inflate COGS.
            self._record_truth(
                self.agent_id,
                "inventory_consumed",
                amount=green_value_consumed,
                counterparty=None,
                item=green_item_id,
                quantity=qty_kg,
                memo=(f"roast input consumed (WAVG ${green_unit_cost:.2f}/kg)"),
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
            "remaining_green_inventory": self.inventory[green_item_id],
            "shared_roast_capacity_used_today": (env._roast_used_today[self.agent_id]),
            "shared_roast_capacity_remaining_today": (
                ROAST_DAILY_CAP_GREEN_KG - env._roast_used_today[self.agent_id]
            ),
        }

    # ===== TOOLS — chat =====
    def send_message(
        self,
        recipient: str,
        title: str,
        body: str,
        ref_listing_id: str | None = None,
        ref_offer_id: str | None = None,
    ) -> dict:
        """Send a private message to another agent.

        Args:
            recipient: the recipient agent's id.
            title: REQUIRED short subject line (≤80 chars hard cap, ≤60
                recommended). Must summarize the message intent — empty
                or whitespace-only titles are rejected.
            body: free-form message text.
            ref_listing_id: optionally tag this message to a specific listing.
            ref_offer_id: optionally tag this message to a specific offer.
        """
        if not body or not body.strip():
            return {"status": "error", "message": "body cannot be empty."}
        if not isinstance(title, str) or not title.strip():
            return {
                "status": "error",
                "message": "title is required (short subject line, ≤60 chars recommended).",
            }
        if len(title) > 80:
            return {
                "status": "error",
                "message": f"title is {len(title)} chars; hard cap is 80.",
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
        )
        # Reactive wake: pull the recipient back to current_min so the
        # next dispatcher cycle delivers a fresh observation that
        # includes this DM in the unread inbox. Without this, mid-day
        # messages aren't surfaced until the next morning's obs.
        if env is not None:
            from coffeebench.event_loop import WAKE_DM_RECEIVED

            env._wake_agent_externally(
                recipient,
                f"{WAKE_DM_RECEIVED}: from {self.agent_id} — '{title.strip()[:60]}'",
            )
        return {"status": "success", "message_id": m.id, "recipient": recipient}

    def view_messages(
        self,
        counterparty: str | None = None,
        since_day: int | None = None,
        limit: int = 10,
    ) -> dict:
        """Return a compact INBOX of messages visible to you (sent by you
        or sent to you), most-recent first. Each row carries the message
        id, sender, recipient, day, and the **title** (subject line). The
        body is NOT included in the listing view — call
        `read_message(message_id)` to fetch the body of a specific message.

        This title-only format keeps your context window small even on long
        runs — bodies are fetched on demand only for messages you care
        about.

        Args:
            counterparty: filter to messages exchanged with one specific
                agent (sender or recipient = this agent). Use "self" for
                only messages YOU sent. None = all your messages in/out.
            since_day: only messages sent on or after this day.
            limit: max rows to return (default 10, most-recent first).
                Increase explicitly when you need to scan further back.
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
        """Fetch the full body of a single message previously seen via
        `view_messages` (the inbox carries the id). The message must be
        one you sent or one sent to you.

        Args:
            message_id: id from a `view_messages` inbox row.
        """
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
        """List your unpaid supplier/peer invoices (AP) with a summary
        header. Compact one-row-per-invoice — drops the verbose dataclass
        dump. Sorted by due day ascending (most urgent first).

        Args:
            only_overdue: if True, return only invoices already past
                their due_date.
            counterparty: filter to invoices from one specific seller
                (issuer agent_id).
            limit: max rows to return (default 20). The summary block
                still reports the FULL counts/totals regardless of limit.
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
        """List your unpaid customer/peer invoices (AR) with a summary
        header. Compact one-row-per-invoice — drops the verbose dataclass
        dump. Sorted by due day ascending (most urgent first).

        Args:
            only_overdue: if True, return only invoices already past
                their due_date.
            counterparty: filter to invoices from one specific buyer
                (payer agent_id).
            limit: max rows to return (default 20). The summary block
                still reports the FULL counts/totals regardless of limit.
        """
        return self._ap_ar_view(
            self.accounts_receivable,
            only_overdue,
            counterparty,
            limit,
        )

    def pay_invoice(self, invoice_id: str) -> dict:
        """Pay one of your AP invoices in full from cash. Cash is wired to the issuer.

        Args:
            invoice_id: AP invoice id.
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
        """Period trial balance — chart-of-accounts balances over the
        period as a flat dict (NO IS/BS/CF subtotals computed for you).
        Useful for self-monitoring your own NI components day-to-day.

        Each account record carries:
          - balance : float, rounded to 2 decimals
          - kind    : "snapshot" → point-in-time figure at the period's
                      `as_of` day (Balance Sheet style)
                      "period"   → sum over [day_lo .. day_hi]
                                   (Income Statement / cash-flow style)

        Account list (chart of accounts):
          Snapshot (Balance Sheet at `as_of`):
            cash, accounts_receivable, inventory, accounts_payable
          Period (Income Statement):
            sales_revenue, sales_returns, cogs,
            rent_utilities_expense, spoilage_expense,
            other_operating_expense,
            interest_expense, interest_revenue
          Period (Cash flow inputs):
            cash_received, cash_paid, cash_paid_for_production

        Use it to inspect the env's truth ledger as it has been
        compiled so far for the period — the same data the env uses
        to compute your end-of-run score.

        `period` values:
            "Q1" / "Q2" / "Q3" / "Q4" — full quarter window.
            "FY"                      — full year (days 0..364).
            "YTD"                     — year-to-date (days 0..today).
            "current"                 — today's quarter (default).

        Mid-period queries (e.g. "Q1" on day 30) cap at today and return
        partial accumulation; `complete: false` flags this. Asking for a
        future period (e.g. "Q3" on day 50) returns an error.

        For per-record drill-down (e.g. "which sale_revenue events
        composed Q1's revenue?"), use `view_econ_events` with explicit
        filter args.
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
        """Signal you have nothing more to do right now and want to
        sleep until something happens for you.

        You will be woken automatically when an external event lands
        for you within today's business hours: a new direct message
        arrives, a counterparty accepts one of your offers, a shipment
        you are expecting is delivered, or your in-flight production
        completes. If no event fires for you, you stay asleep until
        tomorrow's morning observation. This is the cost-efficient way
        to end your turn — you spend zero LLM calls while sleeping.
        """
        return {
            "status": "success",
            "message": "Sleeping until next event or tomorrow's morning.",
        }
