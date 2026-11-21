"""Marketplace — shared, environment-owned state for the deal benchmark."""

import random
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING, Callable, Optional

from coffeebench.time_manager import TimeManager
from coffeebench.typings import Deal, Invoice, Item, Listing, Message, Offer

if TYPE_CHECKING:
    from coffeebench.business_app import BusinessApp


class Marketplace:
    def __init__(self, time_manager: TimeManager) -> None:
        self.time_manager = time_manager
        self.items: dict[str, Item] = {}  # item_id -> Item
        self.listings: list[Listing] = []
        self.offers: list[Offer] = []
        self.deals: list[Deal] = []
        self.messages: list[Message] = []
        # Per-agent set of message ids that have been read via
        # `read_message(id)`. Messages surfaced only as previews in
        # `view_messages` are NOT auto-marked — the agent must commit to
        # reading the full body. Used to compute the morning observation's
        # unread-DM line and the post-run `audit.message_volume.unread_at_end`.
        self._read_message_ids: dict[str, set[str]] = {}
        # Re-entrant lock guarding ALL mutations of the shared marketplace
        # state PLUS the on_deal_accepted callback (which mutates BusinessApp
        # state too — so the deal is atomic). Reads with .lock-protected
        # snapshotting so concurrent agent threads see consistent views.
        self.lock = threading.RLock()
        # Set by Environment so accept_offer can spawn AR/AP invoices on both
        # sides' books and emit truth-ledger entries.
        self.on_deal_accepted: Optional[Callable[[Deal], Invoice]] = None
        # Maps agent_id -> BusinessApp; populated by Environment after init
        # so the marketplace can hand offers / messages to recipients and run
        # inventory transfers atomically.
        self.business_apps: dict[str, "BusinessApp"] = {}
        # Per-run sequential inventory-lot ids ("lot_00001", ...). Kept on
        # the shared marketplace (not a module global) so two runs in the
        # same process assign identical ids — lot ids are deterministic.
        self._lot_seq: int = 0
        # Object ids (listings / offers / deals / messages / journal
        # entries) are 8 hex chars drawn from this RNG, seeded per run by
        # `seed_ids` (build_run) — ids are reproducible given the seed and
        # survive checkpoint/resume (the RNG state is pickled). Separate
        # from `env.rng`, so the demand/delivery draws are unaffected.
        self._id_rng = random.Random()

    def seed_ids(self, seed: int | None) -> None:
        self._id_rng.seed(f"coffeebench-ids-{seed}")

    def new_hex_id(self) -> str:
        with self.lock:
            return f"{self._id_rng.getrandbits(32):08x}"

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("lock", None)  # RLock is not picklable; recreated below
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.lock = threading.RLock()

    def next_lot_id(self) -> str:
        with self.lock:
            self._lot_seq += 1
            return f"lot_{self._lot_seq:05d}"

    # --- catalog ---
    def register_item(self, item: Item) -> None:
        self.items[item.id] = item

    def get_item(self, item_id: str) -> Item | None:
        return self.items.get(item_id)

    # --- listings ---
    def post_listing(
        self,
        seller_id: str,
        item_id: str,
        qty: int,
        asking_price: float,
        payment_terms_days: int,
    ) -> Listing:
        with self.lock:
            listing = Listing(
                id="lst_" + self.new_hex_id(),
                seller_id=seller_id,
                item_id=item_id,
                qty=int(qty),
                asking_price=float(asking_price),
                payment_terms_days=int(payment_terms_days),
                posted_at=self.time_manager.get_virtual_min(),
            )
            self.listings.append(listing)
            return listing

    # --- offers ---
    def make_offer(
        self,
        listing_id: str,
        buyer_id: str,
        offered_price: float,
        qty: int,
        payment_terms_days: int,
        message: str | None = None,
    ) -> tuple[Offer | None, str]:
        with self.lock:
            listing = next((lt for lt in self.listings if lt.id == listing_id), None)
            if listing is None:
                return None, f"Listing {listing_id} not found."
            if listing.status != "open":
                return None, f"Listing {listing_id} is {listing.status}."
            if buyer_id == listing.seller_id:
                return None, "Cannot offer on your own listing."
            if self._is_bankrupt(listing.seller_id):
                return None, f"Seller {listing.seller_id} is bankrupt."
            if int(qty) <= 0 or int(qty) > listing.qty:
                return None, f"Offer qty must be 1..{listing.qty}."
            # Mirror the farmer-bulk constraint on the offer side. Without
            # this, a buyer could offer for a small fraction of a 50 kg lot,
            # farmer accepts, and the deal lands at e.g. 30 kg — bypassing
            # the bulk-breaker moat entirely.
            offer = Offer(
                id="off_" + self.new_hex_id(),
                listing_id=listing_id,
                buyer_id=buyer_id,
                seller_id=listing.seller_id,
                offered_price=float(offered_price),
                qty=int(qty),
                payment_terms_days=int(payment_terms_days),
                posted_at=self.time_manager.get_virtual_min(),
                message=message,
            )
            self.offers.append(offer)
            return offer, "ok"

    def withdraw_offer(self, offer_id: str, by_buyer_id: str) -> tuple[bool, str]:
        with self.lock:
            offer = next((o for o in self.offers if o.id == offer_id), None)
            if offer is None:
                return False, f"Offer {offer_id} not found."
            if offer.buyer_id != by_buyer_id:
                return False, "Only the buyer can withdraw this offer."
            if offer.status != "pending":
                return False, f"Offer is already {offer.status}."
            offer.status = "withdrawn"
            return True, "Offer withdrawn."

    def reject_offer(self, offer_id: str, by_seller_id: str) -> tuple[bool, str]:
        with self.lock:
            offer = next((o for o in self.offers if o.id == offer_id), None)
            if offer is None:
                return False, f"Offer {offer_id} not found."
            if offer.seller_id != by_seller_id:
                return False, "Only the seller can reject this offer."
            if offer.status != "pending":
                return False, f"Offer is already {offer.status}."
            offer.status = "rejected"
            return True, "Offer rejected."

    def cancel_listing(self, listing_id: str, by_seller_id: str) -> tuple[int, str]:
        """Cancel an open listing. Listings do not reserve inventory
        (stock is only reserved when an offer is accepted), so the only
        thing to release is the listing's still-pending offers, which
        are marked `rejected`. Returns (n_offers_rejected, msg); n < 0
        signals an error."""
        with self.lock:
            listing = next((lt for lt in self.listings if lt.id == listing_id), None)
            if listing is None:
                return -1, f"Listing {listing_id} not found."
            if listing.seller_id != by_seller_id:
                return -1, "Only the seller can cancel this listing."
            if listing.status != "open":
                return -1, f"Listing is already {listing.status}."
            listing.status = "cancelled"
            n = 0
            for o in self.offers:
                if o.listing_id == listing.id and o.status == "pending":
                    o.status = "rejected"
                    n += 1
            return n, "Listing cancelled."

    def expire_offers(self, posted_before_day: int) -> list[Offer]:
        """Mark every still-pending offer posted before `posted_before_day`
        as `expired`. Called from the env's EOD handler so an offer lapses
        at the end of the day AFTER it was posted."""
        from coffeebench.event_loop import MINUTES_PER_DAY

        with self.lock:
            out: list[Offer] = []
            for o in self.offers:
                if o.status != "pending":
                    continue
                if o.posted_at // MINUTES_PER_DAY < posted_before_day:
                    o.status = "expired"
                    out.append(o)
            return out

    def cancel_open_orders_for(self, agent_id: str) -> dict:
        """Called on bankruptcy: cancel the agent's open listings and every
        pending offer it is a party to (as buyer or seller)."""
        with self.lock:
            n_listings = 0
            for lt in self.listings:
                if lt.seller_id == agent_id and lt.status == "open":
                    lt.status = "cancelled"
                    n_listings += 1
            n_offers = 0
            for o in self.offers:
                if o.status != "pending":
                    continue
                if o.buyer_id == agent_id or o.seller_id == agent_id:
                    o.status = "cancelled"
                    n_offers += 1
            return {"listings_cancelled": n_listings, "offers_cancelled": n_offers}

    def _is_bankrupt(self, agent_id: str) -> bool:
        env = getattr(self, "_env", None)
        return env is not None and agent_id in env._bankrupt_agents

    def accept_offer(self, offer_id: str, by_seller_id: str) -> tuple[Deal | None, str]:
        with self.lock:
            offer = next((o for o in self.offers if o.id == offer_id), None)
            if offer is None:
                return None, f"Offer {offer_id} not found."
            if offer.seller_id != by_seller_id:
                return None, "Only the seller can accept this offer."
            if offer.status != "pending":
                return None, f"Offer is already {offer.status}."
            listing = next(
                (lt for lt in self.listings if lt.id == offer.listing_id), None
            )
            if listing is None or listing.status != "open":
                return None, "Listing is no longer open."
            if offer.qty > listing.qty:
                return (
                    None,
                    f"Offer qty {offer.qty} exceeds remaining listing qty {listing.qty}.",
                )

            seller = self.business_apps.get(offer.seller_id)
            buyer = self.business_apps.get(offer.buyer_id)
            if seller is None or buyer is None:
                return None, "Internal: counterparty BusinessApp not registered."
            # A bankrupt party can neither ship nor pay: refuse the deal
            # so no uncollectable receivable is ever booked.
            for party in (offer.buyer_id, offer.seller_id):
                if self._is_bankrupt(party):
                    offer.status = "cancelled"
                    return None, f"{party} is bankrupt; the offer was cancelled."

            today = self.time_manager.get_current_day()
            now_at = self.time_manager.get_virtual_min()
            lag = self._delivery_lag(seller, listing.item_id)
            delivery_day = today + lag
            if seller.inventory.get(listing.item_id, 0) < offer.qty:
                return (
                    None,
                    f"Seller no longer holds enough of {listing.item_id} to fulfill.",
                )
            # Lots that would be past expiry on arrival are not shipped
            # (they stay with the seller and are written off at its EOD).
            shippable = seller._sellable_qty(listing.item_id, delivery_day)
            if shippable < offer.qty:
                return (
                    None,
                    f"Only {shippable} kg of your {listing.item_id} is still "
                    f"within shelf life on the delivery day (day "
                    f"{delivery_day}); the rest expires before it would "
                    f"arrive. Accept a smaller offer or let the buyer re-offer.",
                )
            # Buyer-side inventory-cap gate. Counts on-hand + in-flight
            # inbound (other accepted-but-not-delivered purchases, own
            # pending production / roast output) so the cap is a hard
            # cap against pile-up across multiple lag-days, not just an
            # action-time check on currently-held kg.
            buyer_cap = buyer._inventory_cap_kg()
            buyer_on_hand = buyer._total_inventory_kg()
            buyer_pending = buyer._pending_inbound_kg()
            buyer_effective = buyer_on_hand + buyer_pending
            if buyer_effective + offer.qty > buyer_cap:
                # The caller is the SELLER — don't leak the buyer's
                # private on-hand / in-flight kg; the role cap is public.
                return None, (
                    f"Buyer {buyer.agent_id} lacks inventory headroom for "
                    f"{offer.qty} kg (its role cap is {buyer_cap} kg incl. "
                    f"in-flight inbound). Ask the buyer to re-offer a "
                    f"smaller quantity."
                )
            # (lag / delivery_day computed above via `_delivery_lag`: item
            # base lag + farmer-from-origin surcharge.)
            # Delivery fires at the start of business hours on
            # `delivery_day` (09:00 in vt terms) — `_process_deliveries`
            # is invoked from `_handle_morning_open`, which runs at
            # BUSINESS_HOURS_START. We bake that 09:00 offset into
            # `delivery_at` so the timestamp the agent sees matches
            # when the delivery actually fires.
            from coffeebench.environment import BUSINESS_HOURS_START
            from coffeebench.event_loop import MINUTES_PER_DAY

            delivery_at = delivery_day * MINUTES_PER_DAY + BUSINESS_HOURS_START
            total = round(offer.offered_price * offer.qty, 2)

            deal = Deal(
                id="dl_" + self.new_hex_id(),
                listing_id=listing.id,
                offer_id=offer.id,
                seller_id=offer.seller_id,
                buyer_id=offer.buyer_id,
                item_id=listing.item_id,
                qty=offer.qty,
                unit_price=offer.offered_price,
                total_price=total,
                payment_terms_days=offer.payment_terms_days,
                deal_at=now_at,
                delivery_at=delivery_at,
                invoice_id="",
                notes=None,
            )

            offer.status = "accepted"
            listing.qty -= offer.qty
            if listing.qty <= 0:
                listing.status = "closed"
                for o in self.offers:
                    if (
                        o.listing_id == listing.id
                        and o.status == "pending"
                        and o.id != offer.id
                    ):
                        o.status = "rejected"

            if self.on_deal_accepted is None:
                return None, "Internal: marketplace.on_deal_accepted not wired."
            invoice = self.on_deal_accepted(deal)
            deal.invoice_id = invoice.id
            self.deals.append(deal)
            return deal, "ok"

    # --- messages ---
    def post_message(
        self,
        sender_id: str,
        recipient_id: str,
        title: str,
        body: str,
        ref_listing_id: str | None = None,
        ref_offer_id: str | None = None,
        ref_deal_id: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> Message:
        """Append a message. `reply_to_message_id` must already be
        validated by the caller (visible to the sender); the new message
        inherits the parent's `thread_id` (a root message is its own
        thread)."""
        with self.lock:
            msg_id = "msg_" + self.new_hex_id()
            thread_id = msg_id
            if reply_to_message_id is not None:
                parent = self.get_message(reply_to_message_id)
                if parent is not None:
                    thread_id = parent.thread_id or parent.id
            msg = Message(
                id=msg_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                title=title,
                body=body,
                sent_at=self.time_manager.get_virtual_min(),
                ref_listing_id=ref_listing_id,
                ref_offer_id=ref_offer_id,
                ref_deal_id=ref_deal_id,
                reply_to_message_id=reply_to_message_id,
                thread_id=thread_id,
            )
            self.messages.append(msg)
            # The reactive wake for the recipient is pushed by
            # BusinessApp.send_message (WAKE_DM_RECEIVED), not here.
            return msg

    def get_message(self, message_id: str) -> Message | None:
        with self.lock:
            return next((m for m in self.messages if m.id == message_id), None)

    def messages_visible_to(
        self,
        agent_id: str,
        since_day: int | None = None,
    ) -> list[Message]:
        with self.lock:
            from coffeebench.event_loop import MINUTES_PER_DAY

            out: list[Message] = []
            for m in self.messages:
                if since_day is not None and (m.sent_at // MINUTES_PER_DAY) < since_day:
                    continue
                # A message is visible iff you are its sender or recipient.
                if m.sender_id != agent_id and m.recipient_id != agent_id:
                    continue
                out.append(m)
            return out

    def mark_message_read(self, agent_id: str, message_id: str) -> None:
        """Record that `agent_id` has fetched the full body of `message_id`.
        Idempotent. Caller is responsible for verifying visibility."""
        with self.lock:
            self._read_message_ids.setdefault(agent_id, set()).add(message_id)

    def is_message_read(self, agent_id: str, message_id: str) -> bool:
        with self.lock:
            return message_id in self._read_message_ids.get(agent_id, set())

    def unread_messages_for(self, agent_id: str) -> list[Message]:
        """Return messages where agent is the recipient and the full body
        has not yet been fetched via read_message. Sender-side messages are
        not 'unread' (the agent wrote them). Ordered oldest-first."""
        with self.lock:
            read = self._read_message_ids.get(agent_id, set())
            return [
                m
                for m in self.messages
                if m.recipient_id == agent_id and m.id not in read
            ]

    # --- views (used by agent tools) ---
    def listings_dump(self) -> list[dict]:
        """Return all OPEN listings, augmented with the buyer-relevant
        delivery ETA. The ETA bakes in `Item.delivery_lag_days` plus the
        seller-role surcharge (`FARMER_DELIVERY_EXTRA_DAYS` for farmer
        on a tangible), so an agent can decide which listing to bid on
        without having to compute the lead time themselves."""
        from coffeebench.environment import FARMER_DELIVERY_EXTRA_DAYS
        from coffeebench.event_loop import format_min

        with self.lock:
            out: list[dict] = []
            for lt in self.listings:
                if lt.status != "open":
                    continue
                d = asdict(lt)
                # Derived day-int + Day-N-HH:MM string. asdict only
                # serializes fields, so `posted_day` (a property) isn't
                # there — re-add it for agent ergonomics.
                d["posted_day"] = lt.posted_at // 1440
                d["posted"] = format_min(lt.posted_at)
                item = self.items.get(lt.item_id)
                base_lag = (
                    max(0, int(item.delivery_lag_days)) if item is not None else 1
                )
                seller = self.business_apps.get(lt.seller_id)
                extra = (
                    FARMER_DELIVERY_EXTRA_DAYS
                    if (seller is not None and seller.role == "farmer")
                    else 0
                )
                eta = base_lag + extra
                d["delivery_eta_days"] = eta
                if extra:
                    d["delivery_eta_note"] = (
                        f"{base_lag} (base) + {extra} (farmer ships from origin)"
                    )
                else:
                    d["delivery_eta_note"] = f"{base_lag} (base)"
                # `freshness_at_posting` (from asdict) is the static
                # posting-time snapshot; never recomputed live here.
                out.append(d)
            return out

    def offers_for(self, agent_id: str, direction: str = "all") -> list[dict]:
        from coffeebench.event_loop import format_min

        with self.lock:
            out: list[Offer] = []
            for o in self.offers:
                if direction in ("incoming", "all") and o.seller_id == agent_id:
                    out.append(o)
                elif direction in ("outgoing", "all") and o.buyer_id == agent_id:
                    out.append(o)
            today = self.time_manager.get_current_day()
            rows: list[dict] = []
            for o in out:
                d = asdict(o)
                d["posted_day"] = o.posted_at // 1440
                d["posted"] = format_min(o.posted_at)
                listing = next(
                    (lt for lt in self.listings if lt.id == o.listing_id), None
                )
                if listing is not None:
                    d["item_id"] = listing.item_id
                seller = self.business_apps.get(o.seller_id)
                if listing is not None and o.seller_id == agent_id:
                    # Owner-only live view of the lots that would ship.
                    if o.status == "pending" and seller is not None:
                        d["live_freshness"] = seller._ship_freshness(
                            listing.item_id,
                            o.qty,
                            today + self._delivery_lag(seller, listing.item_id),
                        )
                elif listing is not None:
                    d["freshness_at_posting"] = listing.freshness_at_posting
                rows.append(d)
            return rows

    def _delivery_lag(self, seller: "BusinessApp", item_id: str) -> int:
        """Accept→delivery lag in days (item base lag + farmer surcharge);
        mirrors `accept_offer`."""
        from coffeebench.environment import FARMER_DELIVERY_EXTRA_DAYS

        item = self.items.get(item_id)
        lag = max(0, int(item.delivery_lag_days)) if item is not None else 1
        if seller.role == "farmer":
            lag += FARMER_DELIVERY_EXTRA_DAYS
        return lag
