"""HeuristicRetailerModel — fixed rule-based retailer for the rule background market.

`get_model("heuristic_retailer")` → `--models retailer_A:heuristic_retailer`
(Harbor: `launch_sim.py --bg Rule`, which runs every non-focal retailer with it).
$0, deterministic given the run seed, checkpointable (plain-data state).

Inputs: the rules prompt (own id, reservation and starting retail prices,
opex, inventory cap, festival windows, run length) and its own tool results /
observations. Never reads env internals or messages.

Morning routine (≤ 19 actions/day, then wait_for_next_day):
  views   view_dashboard (cash, AP due, stock, in-flight purchases, retail
          prices, yesterday's consumer sales), view_inventory (lot cost /
          expiry), view_offers(outgoing), view_listings (roasted items)
  pay     every invoice due today / tomorrow / overdue while cash covers it
  price   per roasted item, once a day, on a base price (opening = the current
          retail price, i.e. the starting price from the rules prompt):
            +5% if yesterday's stock sold out (sales ≥ stock on hand),
            −5% if stock was on hand and it sold < 20% of it (leftover),
            else hold; clamped to [avg lot cost × 1.2, reservation × 0.95]
          (never above the reservation price). Near-expiry clearance: when
          lots with ≤ 3 days left exceed the expected sales until they expire,
          today's price is cut 10% per day inside the window (floor ½ lot
          cost); the base price is unchanged.
  buy     per roasted item: target stock = expected daily sales × 3 days
          cover (× 2 in the 3 days before a public festival, × 1.2 inside
          it), limited to the days left in the run. Expected sales = mean of
          the last 5 days of the same kind (festival vs normal), stock-out
          days counted as sales × 1.3 (censored demand); prior per shop
          (commodity 18, specialty 3.5 kg/day; × 2 for a festival with no
          observation yet) blended in while < 5 days are observed.
          need = target − (on hand + inbound deals + pending bids), within the
          inventory-cap headroom and the cash budget (cash − 7 days opex −
          open AP − pending bids). Buys from the CHEAPEST eligible listings:
          ask ≤ 70% of its own base retail price (keeps ≥ 30% gross margin),
          ≥ 5 days of shelf life left at delivery; bids exactly the ask with
          the listing's requested terms (capped at 30 d); ≤ 4 bids/day.
          Pending bids from earlier days are withdrawn when the ask is above
          today's limit or a ≥ 5% cheaper eligible listing exists.
  recheck sellers re-post listings during the day, so a bid can hit a cancelled
          listing: after a failed bid, or on a mid-day wake (message / deal —
          used only as a content-free trigger), it re-reads view_offers +
          view_listings and runs the buy rule again (≤ 3 rechecks/day; a
          listing that rejected its bid is skipped for the day).
          Empty-shelf days count as stock-outs for pricing but the raise stops
          at the starting price × 1.25.
  sell    retailers never post B2B listings; any incoming offer is rejected.
Parameters: DEFAULT_PARAMS; a per-seat profile overrides some keys
(`HeuristicRetailerModel(params=...)`).
Messages: none sent; DMs (and any free-text negotiation) are ignored — known
limitation: a focal agent can only trade with it through listings/offers.
"""

from __future__ import annotations

from coffeebench.models._rule_base import RuleAgentBase

ROASTED = ("roasted_coffee_kg", "roasted_specialty_kg")

# Rule parameters (untuned defaults; see module docstring). A per-agent
# profile = a dict overriding some keys (`HeuristicRetailerModel(params=...)`).
DEFAULT_PARAMS: dict = {
    "price_step": 0.05,
    "empty_shelf_raise_cap": 1.25,  # raises on empty-shelf days stop at start price × this
    "low_sell_through": 0.20,
    "floor_markup": 1.2,
    "cap_of_reservation": 0.95,
    "max_buy_ratio": 0.70,        # max wholesale price = base retail price × this
    "cover_days": 3,
    "ramp_factor": 2.0,
    "festival_factor": 1.2,
    "festival_prior_factor": 2.0,
    "sales_window_days": 5,
    "censored_bump": 1.3,
    "prior_sales": {"roasted_coffee_kg": 18.0, "roasted_specialty_kg": 3.5},
    "cash_buffer_days": 7,
    "min_shelf_days": 5,
    "max_terms_days": 30,
    "max_bids_per_day": 4,
    "min_bid_kg": 2,
    "cheaper_for_withdraw": 0.95,
    "clearance_window_days": 3,
    "clearance_step": 0.10,
    "max_rechecks_per_day": 3,    # extra listing checks per day (wake / stale listing)
    # E2 negotiation (structured actions only): open at ask × open_bid_frac
    # (never above the max-buy ceiling); each rejected / expired bid to a
    # seller for an item raises the next one by bid_raise_step × ask, up to
    # the ask and never above the ceiling. Deterministic.
    "negotiate": False,
    "open_bid_frac": 0.85,
    "bid_raise_step": 0.05,
}

# Per-seat policy profiles (`heuristic_retailer@<name>`).
PROFILES: dict[str, dict] = {
    "default": {},
    "negotiator": {"negotiate": True},
}
P = DEFAULT_PARAMS


def next_retail_price(price: float, sold_out: bool, leftover_cut: bool, floor: float, cap: float,
                      step: float = P["price_step"]) -> float:
    if sold_out:
        price *= 1 + step
    elif leftover_cut:
        price *= 1 - step
    return round(min(max(price, floor), cap), 2)


def clearance_retail(price: float, unit_cost: float, days_left: int, p: dict = P) -> float:
    disc = p["clearance_step"] * (p["clearance_window_days"] + 1 - max(0, days_left))
    return round(max(price * (1 - disc), unit_cost * 0.5), 2)


class HeuristicRetailerModel(RuleAgentBase):
    ROLE = "retailer"
    TAG = "hr"

    def __init__(self, model: str = "heuristic_retailer", params: dict | None = None):
        super().__init__(model)
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.rechecks = 0
        self.rejected_listings: list[str] = []  # listings that rejected our bid today
        self.bid_frac: dict[str, float] = {}     # "seller::item" → current bid / ask
        self.seen_outcomes: list[str] = []       # offer ids whose outcome was processed
        self.on_hand_kg = 0
        self.in_flight_kg = 0
        self.stock_by_item: dict[str, int] = {}
        self.inbound_by_item: dict[str, int] = {}
        self.lots: list[dict] = []
        self.retail_prices: dict[str, float] = {}
        self.base_price: dict[str, float] = {}
        self.eval_day: dict[str, int] = {}
        self.stock_seen: dict[str, list] = {}  # item → [day, sellable kg at morning]
        self.sales: list[dict] = []  # {day, item, qty, censored, fest}
        self.listings_market: list[dict] = []
        self.outgoing: list[dict] = []
        self.incoming: list[dict] = []
        self.last_buy_price: dict[str, float] = {}
        self.bids_today = 0

    # ---------- day ----------------------------------------------------------

    def _start_day(self) -> None:
        self.bids_today = 0
        self.rechecks = 0
        self.rejected_listings = []
        self.queue = [["call", "view_dashboard", {}], ["call", "view_inventory", {}],
                      ["call", "view_offers", {"direction": "all"}], ["call", "view_listings", {}]] + [
            ["phase", p, {}] for p in ("reject", "pay", "price", "buy")]

    def _recheck(self) -> None:
        """Re-read the public listings and run the buy rule again (bounded).
        Sellers re-post listings during the day, so a listing seen in the
        morning may be gone; a mid-day wake (message, deal) is used only as a
        content-free trigger for this check."""
        if self.rechecks >= self.p["max_rechecks_per_day"] or any(q[1] == "buy" for q in self.queue):
            return
        self.rechecks += 1
        self.queue = [q for q in self.queue if q[1] != "make_offer"]
        self.queue[0:0] = [["call", "view_offers", {"direction": "outgoing"}],
                           ["call", "view_listings", {}], ["phase", "buy", {}]]

    def _on_wake(self, reason: str) -> None:
        if reason.startswith(("dm_received", "deal_accepted")):
            self._recheck()

    # ---------- results ------------------------------------------------------

    def _on_result(self, name: str, data: dict) -> None:
        if name == "make_offer" and data.get("status") != "success":
            self.bids_today = max(0, self.bids_today - 1)
            self._recheck()
            return
        if data.get("status") != "success":
            return
        args = self._last_args(name)
        if name == "view_dashboard":
            h = data.get("headline") or {}
            self.cash = float(h.get("cash", self.cash))
            self.ap_total = float(h.get("ap", 0.0))
            inv = data.get("inventory") or {}
            self.on_hand_kg = int(inv.get("on_hand_kg", 0))
            self.in_flight_kg = int(inv.get("in_flight_inbound_kg", 0))
            self.inv_cap = int(inv.get("cap_kg", self.inv_cap))
            self.stock_by_item = {r["item_id"]: int(r["qty"]) for r in inv.get("by_item") or []}
            dif = data.get("deals_in_flight") or {}
            self.inbound_by_item = {}
            for r in dif.get("rows") or []:
                if r.get("direction") == "inbound":
                    self.inbound_by_item[r["item_id"]] = self.inbound_by_item.get(r["item_id"], 0) + int(r["qty"])
            unknown = self.in_flight_kg - sum(self.inbound_by_item.values())
            if unknown > 0:  # rows capped: attribute the rest to commodity
                self.inbound_by_item["roasted_coffee_kg"] = self.inbound_by_item.get("roasted_coffee_kg", 0) + unknown
            self._ingest_payables(data)
            if (data.get("payables_due") or {}).get("omitted"):
                self.queue.insert(self._after_views(), ["call", "view_payables", {}])
            retail = data.get("retail") or {}
            self.retail_prices = {k: float(v) for k, v in (retail.get("prices") or {}).items()}
            ysales = retail.get("yesterday_consumer_sales") or {}
            fest = self._phase_name_for(self.day - 1) == "festival"
            for item in ROASTED:
                q = int((ysales.get(item) or {}).get("qty", 0))
                seen = self.stock_seen.get(item)
                prev = int(seen[1]) if seen and seen[0] == self.day - 1 else 0
                if self.day > 0 and seen and seen[0] == self.day - 1:
                    # prev == 0: out of stock all day (counted as a stock-out
                    # for pricing, not as a demand observation).
                    self.sales.append({"day": self.day - 1, "item": item, "qty": q,
                                       "censored": q >= prev, "fest": fest, "stock": prev})
            self.sales = self.sales[-200:]
        elif name == "view_payables":
            for r in data.get("rows") or []:
                self.payables[str(r["id"])] = {"id": r["id"], "outstanding": r.get("outstanding", 0),
                                               "due_day": r.get("due_day", self.day)}
        elif name == "view_inventory":
            self.lots = [lt for lt in data.get("lots") or [] if int(lt.get("qty", 0)) > 0]
            for item in ROASTED:
                sellable = sum(int(lt["qty"]) for lt in self.lots if lt.get("item_id") == item
                               and (lt.get("days_left") is None or int(lt["days_left"]) >= 0))
                self.stock_seen[item] = [self.day, sellable]
        elif name == "view_offers":
            offers = data.get("offers") or []
            self.outgoing = [o for o in offers if o.get("status") == "pending" and o.get("buyer_id") == self.me]
            for o in offers:
                oid = str(o.get("id"))
                if (o.get("buyer_id") == self.me and o.get("status") in ("rejected", "expired")
                        and oid not in self.seen_outcomes and self.p["negotiate"]):
                    key = f"{o.get('seller_id')}::{o.get('item_id')}"
                    f = self.bid_frac.get(key, self.p["open_bid_frac"])
                    self.bid_frac[key] = round(min(1.0, f + self.p["bid_raise_step"]), 4)
                if o.get("buyer_id") == self.me and o.get("status") != "pending" and oid not in self.seen_outcomes:
                    self.seen_outcomes.append(oid)
            self.seen_outcomes = self.seen_outcomes[-600:]
            self.rejected_listings = sorted({str(o.get("listing_id")) for o in offers
                                             if o.get("buyer_id") == self.me and o.get("status") == "rejected"
                                             and int(o.get("posted_day", -1)) >= self.day - 1})
            if args.get("direction", "all") == "all":
                self.incoming = [o for o in offers if o.get("status") == "pending" and o.get("seller_id") == self.me]
        elif name == "view_listings":
            self.listings_market = [r for r in data.get("listings") or [] if r.get("item_id") in ROASTED]
        elif name == "set_retail_price":
            self.retail_prices[args.get("item_id")] = float(args.get("price_per_unit", 0))
        elif name == "make_offer":
            meta = self._last_meta()
            self.outgoing.append({"id": data.get("offer_id"), "listing_id": args.get("listing_id"),
                                  "offered_price": args.get("offered_price"), "qty": args.get("qty"),
                                  "item_id": meta.get("item"), "posted_day": self.day, "status": "pending"})
        elif name == "withdraw_offer":
            self.outgoing = [o for o in self.outgoing if str(o.get("id")) != str(args.get("offer_id"))]
        elif name == "pay_invoice":
            self._on_pay_result(data)

    def _after_views(self) -> int:
        return next((i for i, q in enumerate(self.queue) if q[0] == "phase"), len(self.queue))

    # ---------- derived ------------------------------------------------------

    def _phase_name_for(self, day: int) -> str | None:
        from coffeebench.models._rule_base import festival_phase

        return festival_phase(day, self.festivals)

    def _lots(self, item: str) -> list[dict]:
        return [lt for lt in self.lots if lt.get("item_id") == item]

    @staticmethod
    def _avg_cost(lots: list[dict]) -> float:
        q = sum(int(lt["qty"]) for lt in lots)
        return sum(int(lt["qty"]) * float(lt["unit_cost"]) for lt in lots) / q if q else 0.0

    def _expected_sales(self, item: str) -> float:
        fest = self._phase_name() == "festival"
        rows = [s for s in self.sales if s["item"] == item and s["fest"] == fest
                and s["stock"] > 0][-self.p["sales_window_days"]:]
        prior = self.p["prior_sales"].get(item, 3.0) * (self.p["festival_prior_factor"] if fest else 1.0)
        tot = sum(s["qty"] * (self.p["censored_bump"] if s["censored"] else 1.0) for s in rows)
        return (tot + prior * (self.p["sales_window_days"] - len(rows))) / self.p["sales_window_days"]

    def _base(self, item: str) -> float:
        if item not in self.base_price:
            self.base_price[item] = float(self.retail_prices.get(item) or self.start_retail.get(item)
                                          or self.reservation.get(item, 30.0) * 0.6)
        return self.base_price[item]

    # ---------- phases -------------------------------------------------------

    def _bid_price(self, r: dict, limit: dict) -> float | None:
        """Price to bid on listing `r`, or None if not worth bidding."""
        ask = float(r["asking_price"])
        lim = limit[r["item_id"]]
        if not self.p["negotiate"]:
            return ask if ask <= lim else None
        f = self.bid_frac.get(f"{r.get('seller_id')}::{r['item_id']}", self.p["open_bid_frac"])
        bid = round(min(ask * f, lim), 2)
        return bid if bid >= ask * self.p["open_bid_frac"] - 1e-9 else None

    def _forecast_factor(self, days: int) -> float:
        fs = [self.forecast.get(str(d), 1.0) for d in range(self.day, self.day + max(1, days))]
        return sum(fs) / len(fs)

    def _phase_reject(self) -> list:
        out = [["call", "reject_offer", {"offer_id": str(o["id"])}] for o in self.incoming]
        self.incoming = []
        return out

    def _phase_price(self) -> list:
        out = []
        for item in ROASTED:
            base = self._base(item)
            lots = self._lots(item)
            cost = self._avg_cost(lots) or self.last_buy_price.get(item, 0.0)
            floor = cost * self.p["floor_markup"]
            cap = self.reservation.get(item, 1e9) * self.p["cap_of_reservation"]
            if self.eval_day.get(item, -1) < self.day:
                last = next((s for s in reversed(self.sales) if s["item"] == item and s["day"] == self.day - 1), None)
                sold_out = bool(last and last["censored"])
                if last and last["stock"] == 0:
                    # Empty shelf all day: scarcity signal, but bounded (no
                    # sales to learn from) at the starting price × 1.25.
                    start = self.start_retail.get(item) or base
                    sold_out = base < start * self.p["empty_shelf_raise_cap"]
                cut = bool(last and not last["censored"] and last["qty"] < self.p["low_sell_through"] * last["stock"])
                base = next_retail_price(base, sold_out, cut, min(floor, cap), cap, self.p["price_step"])
                self.base_price[item] = base
                self.eval_day[item] = self.day
            price = base
            near = [lt for lt in lots if lt.get("days_left") is not None
                    and 0 <= int(lt["days_left"]) <= self.p["clearance_window_days"]]
            if near:
                dl = min(int(lt["days_left"]) for lt in near)
                near_qty = sum(int(lt["qty"]) for lt in near)
                if near_qty > self._expected_sales(item) * (dl + 1):
                    price = clearance_retail(base, self._avg_cost(near), dl, self.p)
            if abs(float(self.retail_prices.get(item, 0.0)) - price) >= 0.005:
                out.append(["call", "set_retail_price", {"item_id": item, "price_per_unit": price}])
        return out

    def _phase_buy(self) -> list:
        out = []
        phase = self._phase_name()
        factor = self.p["ramp_factor"] if phase == "ramp" else self.p["festival_factor"] if phase == "festival" else 1.0
        days_left = max(0, self.max_days - self.day - 1)
        cover = min(self.p["cover_days"], days_left)
        limit = {it: round(self._base(it) * self.p["max_buy_ratio"], 2) for it in ROASTED}
        cands = sorted(
            (r for r in self.listings_market
             if int(r.get("qty", 0)) > 0 and self._bid_price(r, limit) is not None
             and ((r.get("freshness_at_posting") or {}).get("earliest_expiry_day") is None
                  or int(r["freshness_at_posting"]["earliest_expiry_day"])
                  >= self.day + int(r.get("delivery_eta_days", 1)) + self.p["min_shelf_days"])),
            key=lambda r: (float(r["asking_price"]), str(r["id"])))
        keep = []
        for o in self.outgoing:
            item = o.get("item_id")
            price = float(o.get("offered_price") or 0)
            stale = int(o.get("posted_day", self.day)) < self.day
            cheaper = any(c["item_id"] == item and float(c["asking_price"]) < price * self.p["cheaper_for_withdraw"]
                          for c in cands)
            if stale and item in limit and (price > limit[item] or cheaper):
                out.append(["call", "withdraw_offer", {"offer_id": str(o["id"])}])
            else:
                keep.append(o)
        self.outgoing = keep
        pend_qty = {it: sum(int(o.get("qty") or 0) for o in keep if o.get("item_id") == it) for it in ROASTED}
        pend_val = sum(int(o.get("qty") or 0) * float(o.get("offered_price") or 0) for o in keep)
        need = {}
        for it in ROASTED:
            target = self._expected_sales(it) * cover * factor * self._forecast_factor(cover)
            pos = self.stock_by_item.get(it, 0) + self.inbound_by_item.get(it, 0) + pend_qty[it]
            need[it] = int(target - pos)
        headroom = self.inv_cap - self.on_hand_kg - self.in_flight_kg - sum(pend_qty.values())
        budget = self.cash - self.p["cash_buffer_days"] * self.opex - self.ap_total - pend_val
        bid_on = {str(o.get("listing_id")) for o in keep}
        for r in cands:
            if self.bids_today >= self.p["max_bids_per_day"] or headroom < self.p["min_bid_kg"] or budget <= 0:
                break
            it = r["item_id"]
            if str(r["id"]) in bid_on or str(r["id"]) in self.rejected_listings or need[it] < self.p["min_bid_kg"]:
                continue
            ask = self._bid_price(r, limit)
            qty = min(int(r["qty"]), need[it], headroom, int(budget // ask) if ask > 0 else 0)
            if qty < self.p["min_bid_kg"]:
                continue
            need[it] -= qty
            headroom -= qty
            budget -= qty * ask
            self.bids_today += 1
            self.last_buy_price[it] = ask
            out.append(self._call("make_offer", {
                "listing_id": str(r["id"]), "offered_price": ask, "qty": qty,
                "payment_terms_days": min(self.p["max_terms_days"], int(r.get("payment_terms_days", 30)))}, item=it))
        return out
