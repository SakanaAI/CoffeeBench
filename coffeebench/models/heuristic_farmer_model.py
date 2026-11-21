"""HeuristicFarmerModel — fixed rule-based farmer for the rule background market.

`get_model("heuristic_farmer")` → `--models farmer_A:heuristic_farmer`
(Harbor: `launch_sim.py --bg Rule`, which runs every non-focal farmer with it).
$0, deterministic given the run seed, checkpointable (plain-data state).

Inputs: the rules prompt (own id, production cost / cap per green item, opex,
inventory cap, festival windows, run length) and its own tool results /
observations. It never reads env internals and never reads messages.

Morning routine (≤ 19 actions/day, then wait_for_next_day):
  views    view_dashboard (cash, stock + cost per item, pending production,
           payables, own open listings)
  sell     (first, so buyers' morning views show today's prices) one regular
           listing per green item = all stock not near expiry, at an adaptive
           price; lots ≤ 7 days from expiry go to a clearance listing (−5%/day
           inside the window, floor = lot cost, ½ lot cost on the last day).
           A changed listing is re-posted BEFORE the old one is cancelled, so a
           listing a buyer has just seen stays biddable.
  views    view_inventory (lot expiry), view_offers(all)
  offers   incoming offers, best price first: ACCEPT iff qty ≤ unreserved stock
           and price ≥ required(threshold, terms), where
             threshold = max(floor, list price × (1 − 10%))  (bounded concession)
             floor     = lot cost × 1.15 (qty-weighted over the lots on offer)
             required  = threshold for terms ≤ 30 d; +0.2%/day beyond 30 d;
                         terms > 60 d are rejected
           otherwise REJECT at once (no counter-offers exist). Offers that
           arrive mid-day wake the farmer and are answered the same way.
  pay      any AP due today / tomorrow / overdue (farmers normally have none)
  produce  per green item (specialty first): target position (on hand +
           pending production) = expected sales/day × (3 days cover + 2 days
           production lag); expected sales = mean of own sales over the last 7
           days, blended with a prior (commodity 20, specialty 4 kg/day) while
           fewer than 7 days are observed. Clipped to today's cap, the inventory
           cap headroom and cash − buffer (10 days opex). Before a public weather
           advisory window (if any) the cover rises to 5 days. If production
           is refused for capacity, the remaining capacity in the error message
           is used once.
Pricing per green item (base price state machine):
  start    production cost (rules prompt) × 1.6
  adapt    once a day: +5% if the item sold out or sold ≥ 50% of the stock
           listed the day before, −5% if it was listed a full day and nothing
           sold, else hold; clamped to [lot cost × 1.15, production cost × 3]
Requested terms on listings: 14 days.
Parameters: DEFAULT_PARAMS; a per-seat profile overrides some keys
(`HeuristicFarmerModel(params=...)`).
Messages: none sent; DMs (and any free-text negotiation) are ignored — known
limitation: a focal agent can only trade with it through listings/offers.
"""

from __future__ import annotations

import re

from coffeebench.models._rule_base import RuleAgentBase

GREENS = ("green_specialty_kg", "green_coffee_kg")

# Rule parameters (untuned, principled defaults; see module docstring). A
# per-agent profile = a dict overriding some of these keys
# (`HeuristicFarmerModel(params={...})`), so heterogeneous counterparties can
# be configured later without touching the policy code.
DEFAULT_PARAMS: dict = {
    "start_markup": 1.6,          # opening list price = production cost × this
    "floor_markup": 1.15,         # never sell regular stock below lot cost × this
    "price_cap_markup": 3.0,      # list price ≤ production cost × this
    "price_step": 0.05,           # daily ± adaptation step
    "high_sell_through": 0.5,     # sold ≥ this share of the listed stock → +step
    "max_concession": 0.10,       # accept down to list price × (1 − this)
    "free_terms_days": 30,        # no premium up to these payment terms
    "max_terms_days": 60,         # reject longer terms
    "terms_premium_per_day": 0.002,
    "sales_window_days": 7,
    "prior_sales": {"green_coffee_kg": 20.0, "green_specialty_kg": 4.0},
    "cover_days": 3,
    "advisory_cover_days": 5,
    "cash_buffer_days": 10,
    "expiry_window_days": 7,
    "expiry_discount_per_day": 0.05,
    "listing_terms_days": 14,
    # E2 negotiation (structured actions only): a sub-threshold offer is
    # rejected with a templated counter-quote; each new, closer offer from
    # the same buyer moves the quote `neg_step` of the way toward it, never
    # below the hidden reservation max(floor, list × (1 − neg_max_concession)).
    "negotiate": False,
    "neg_max_concession": 0.25,
    "neg_step": 0.5,
    # Adversarial messaging: templated (false) scarcity claims to the roasters.
    "scarcity_messages": False,
    "scarcity_every_days": 3,
}

# Per-seat policy profiles (`heuristic_farmer@<name>`): overrides of DEFAULT_PARAMS.
PROFILES: dict[str, dict] = {
    "default": {},
    "negotiator": {"negotiate": True},
    # Cheap-looking list price, no concessions, false scarcity claims. (Its
    # logistics are an ENVIRONMENT trait: [mechanics.traits.<farmer_id>].)
    "adversarial": {"start_markup": 1.4, "max_concession": 0.0, "scarcity_messages": True},
}


def next_price(price: float, sold_out: bool, unsold: bool, floor: float, cap: float,
               step: float = DEFAULT_PARAMS["price_step"]) -> float:
    if sold_out:
        price *= 1 + step
    elif unsold:
        price *= 1 - step
    return round(min(max(price, floor), max(cap, floor)), 2)


def accept_threshold(floor: float, list_price: float | None,
                     max_concession: float = DEFAULT_PARAMS["max_concession"]) -> float:
    return max(floor, (list_price or 0.0) * (1 - max_concession))


def required_price(threshold: float, terms_days: int, p: dict = DEFAULT_PARAMS) -> float | None:
    """Minimum price for the terms; None = terms not acceptable."""
    t = int(terms_days)
    if t > p["max_terms_days"]:
        return None
    return threshold * (1 + p["terms_premium_per_day"] * max(0, t - p["free_terms_days"]))


def clearance_price(list_price: float, unit_cost: float, days_left: int, p: dict = DEFAULT_PARAMS) -> float:
    disc = p["expiry_discount_per_day"] * (p["expiry_window_days"] + 1 - max(0, days_left))
    floor = unit_cost * (0.5 if days_left <= 1 else 1.0)
    return round(max(list_price * (1 - disc), floor), 2)


class HeuristicFarmerModel(RuleAgentBase):
    ROLE = "farmer"
    TAG = "hf"

    def __init__(self, model: str = "heuristic_farmer", params: dict | None = None):
        super().__init__(model)
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.on_hand_kg = 0
        self.in_flight_kg = 0
        self.stock: dict[str, int] = {}       # item → kg on hand (dashboard, minus sales since)
        self.value: dict[str, float] = {}     # item → total lot cost on hand
        self.lots: list[dict] = []            # last view_inventory (expiry per lot)
        self.incoming: list[dict] = []
        self.handled_offers: list[str] = []
        self.pending_prod: dict[str, int] = {}
        self.my_listings: dict[str, dict] = {}  # lid → {item, kind, qty, price, posted_day}
        self.sales: list[dict] = []  # {day, item, qty, price}
        self.base_price: dict[str, float] = {}
        self.eval_day: dict[str, int] = {}
        self.sold_out_flag: dict[str, bool] = {}
        self.sold_flag: dict[str, bool] = {}
        self.cap_retry: dict[str, int] = {}
        self.listed_qty: dict[str, list] = {}  # item → [day, kg listed in the sell phase]
        self.neg: dict[str, dict] = {}  # "buyer::item" → {quote, last_offer}
        self.quoted_today: list[str] = []

    # ---------- day / wakes --------------------------------------------------

    def _start_day(self) -> None:
        # Listings first (right after the dashboard), so buyers' morning views
        # already show today's prices; old listings are cancelled only after
        # their replacement is posted (a listing a buyer saw stays biddable).
        self.cap_retry = {}
        self.quoted_today = []
        self.queue = [["call", "view_dashboard", {}], ["phase", "sell", {}],
                      ["call", "view_inventory", {}], ["call", "view_offers", {"direction": "all"}]] + [
            ["phase", p, {}] for p in ("offers", "pay", "produce", "messages")]

    def _on_wake(self, reason: str) -> None:
        if reason.startswith("offer_received") and not any(q[1] == "view_offers" for q in self.queue):
            self.queue[0:0] = [["call", "view_offers", {"direction": "incoming"}], ["phase", "offers", {}]]

    # ---------- results ------------------------------------------------------

    def _on_result(self, name: str, data: dict) -> None:
        ok = data.get("status") == "success"
        args = self._last_args(name)
        if name == "produce_item" and not ok:
            m = re.search(r"only (\d+) remaining today", str(data.get("message", "")))
            item = args.get("item_id")
            if m and item and item not in self.cap_retry and int(m.group(1)) > 0:
                self.cap_retry[item] = int(m.group(1))
                self.queue.insert(0, ["call", "produce_item", {"item_id": item, "quantity": int(m.group(1))}])
            return
        if not ok:
            return
        if name == "view_dashboard":
            h = data.get("headline") or {}
            self.cash = float(h.get("cash", self.cash))
            self.ap_total = float(h.get("ap", 0.0))
            inv = data.get("inventory") or {}
            self.on_hand_kg = int(inv.get("on_hand_kg", 0))
            self.in_flight_kg = int(inv.get("in_flight_inbound_kg", 0))
            self.inv_cap = int(inv.get("cap_kg", self.inv_cap))
            self.stock = {r["item_id"]: int(r["qty"]) for r in inv.get("by_item") or []}
            self.value = {r["item_id"]: float(r.get("value", 0.0)) for r in inv.get("by_item") or []}
            self.pending_prod = {}
            for pp in data.get("pending_production") or []:
                self.pending_prod[pp["item_id"]] = self.pending_prod.get(pp["item_id"], 0) + int(pp["qty"])
            self._ingest_payables(data)
            if (data.get("payables_due") or {}).get("omitted"):
                self.queue.insert(0, ["call", "view_payables", {}])
            mine = data.get("my_open_listings") or {}
            if not mine.get("omitted"):
                open_ids = {str(r["id"]): r for r in mine.get("rows") or []}
                for lid in list(self.my_listings):
                    if lid not in open_ids:
                        self.my_listings.pop(lid)
                for lid, r in open_ids.items():
                    if lid in self.my_listings:
                        self.my_listings[lid]["qty"] = int(r.get("qty", 0))
        elif name == "view_payables":
            for r in data.get("rows") or []:
                self.payables[str(r["id"])] = {"id": r["id"], "outstanding": r.get("outstanding", 0),
                                               "due_day": r.get("due_day", self.day)}
        elif name == "view_inventory":
            self.lots = [lt for lt in data.get("lots") or [] if int(lt.get("qty", 0)) > 0]
            for item in GREENS:
                lots = [lt for lt in self.lots if lt.get("item_id") == item]
                self.stock[item] = sum(int(lt["qty"]) for lt in lots)
                self.value[item] = sum(float(lt["value"]) for lt in lots)
        elif name == "view_offers":
            pending = [o for o in data.get("offers") or []
                       if o.get("status") == "pending" and o.get("seller_id") == self.me]
            seen = {str(o.get("id")): o for o in self.incoming}
            seen.update({str(o.get("id")): o for o in pending})
            self.incoming = [o for oid, o in seen.items() if oid not in self.handled_offers]
        elif name == "accept_offer":
            item = data.get("item_id")
            qty = int(data.get("qty", 0))
            self.sales.append({"day": self.day, "item": item, "qty": qty, "price": float(data.get("unit_price", 0))})
            self.sales = self.sales[-400:]
            self.sold_flag[item] = True
            unit = self._unit_cost(item)
            self.stock[item] = self.stock.get(item, 0) - qty
            self.value[item] = max(0.0, self.value.get(item, 0.0) - unit * qty)
            self._take_lots(item, qty)
            if self.stock[item] <= 0:
                self.sold_out_flag[item] = True
            lid = self._last_meta().get("listing")
            lt = self.my_listings.get(lid) if lid else None
            if lt is not None:
                lt["qty"] -= qty
                if lt["qty"] <= 0:
                    self.my_listings.pop(lid, None)
        elif name == "post_listing":
            meta = self._last_meta()
            self.my_listings[str(data.get("listing_id"))] = {
                "item": args.get("item_id"), "kind": meta.get("kind", "regular"),
                "qty": int(args.get("qty", 0)), "price": float(args.get("asking_price", 0)),
                "posted_day": self.day}
        elif name == "cancel_listing":
            self.my_listings.pop(str(args.get("listing_id")), None)
        elif name == "produce_item":
            self.cash = float(data.get("remaining_cash", self.cash))
            item = data.get("item_id")
            self.pending_prod[item] = self.pending_prod.get(item, 0) + int(data.get("produced", 0))
            self.in_flight_kg += int(data.get("produced", 0))
        elif name == "pay_invoice":
            self._on_pay_result(data)

    def _take_lots(self, item: str | None, qty: int) -> None:
        for lt in sorted((x for x in self.lots if x.get("item_id") == item),
                         key=lambda x: (x.get("expiry_day") is None, x.get("expiry_day") or 0)):
            if qty <= 0:
                break
            take = min(qty, int(lt["qty"]))
            lt["qty"] = int(lt["qty"]) - take
            qty -= take
        self.lots = [lt for lt in self.lots if int(lt["qty"]) > 0]

    # ---------- derived ------------------------------------------------------

    def _unit_cost(self, item: str) -> float:
        q = self.stock.get(item, 0)
        return self.value.get(item, 0.0) / q if q > 0 else 0.0

    def _near(self, item: str) -> tuple[int, float, int]:
        """(kg, avg cost, min days left) of lots within the expiry window
        (from the last lot view; capped by the current stock)."""
        w = self.p["expiry_window_days"]
        near = [lt for lt in self.lots if lt.get("item_id") == item and lt.get("expiry_day") is not None
                and int(lt["expiry_day"]) - self.day <= w]
        if not near:
            return 0, 0.0, 0
        q = min(sum(int(lt["qty"]) for lt in near), self.stock.get(item, 0))
        cost = sum(int(lt["qty"]) * float(lt["unit_cost"]) for lt in near) / max(1, sum(int(lt["qty"]) for lt in near))
        return q, cost, min(int(lt["expiry_day"]) - self.day for lt in near)

    def _expected_sales(self, item: str) -> float:
        win = self.p["sales_window_days"]
        lo = self.day - win
        q = sum(s["qty"] for s in self.sales if s["item"] == item and s["day"] >= lo)
        seen = min(win, max(0, self.day))
        prior = self.p["prior_sales"].get(item, 5.0)
        return (q + prior * (win - seen)) / win

    def _advisory_ahead(self) -> bool:
        k = self.p["advisory_cover_days"]
        return any(a - k <= self.day < a for a, _b in self.advisories)

    # ---------- phases -------------------------------------------------------

    def _phase_offers(self) -> list:
        out = []
        stock = dict(self.stock)
        left = {lid: int(lt["qty"]) for lid, lt in self.my_listings.items()}
        offers = sorted((o for o in self.incoming if str(o.get("id")) not in self.handled_offers),
                        key=lambda o: (-float(o.get("offered_price") or 0), str(o.get("id"))))
        for o in offers:
            oid = str(o["id"])
            self.handled_offers.append(oid)
            lid = str(o.get("listing_id"))
            lt = self.my_listings.get(lid)
            item = (lt or {}).get("item") or o.get("item_id")
            price = float(o.get("offered_price") or 0)
            qty = int(o.get("qty") or 0)
            if lid in left and left[lid] <= 0:
                continue  # listing closes on the accepts above; the env rejects the rest
            ok = False
            need = thr = floor = None
            if item in GREENS and 0 < qty <= stock.get(item, 0) and qty <= left.get(lid, qty):
                kind = (lt or {}).get("kind", "regular")
                nq, ncost, ndl = self._near(item)
                if kind == "clearance" and nq:
                    cost = ncost
                    floor = cost * (0.5 if ndl <= 1 else 1.0)
                else:
                    cost = self._unit_cost(item)
                    floor = cost * self.p["floor_markup"]
                thr = accept_threshold(floor, (lt or {}).get("price"), self.p["max_concession"])
                need = required_price(thr, int(o.get("payment_terms_days", 30)), self.p)
                ok = cost > 0 and need is not None and price >= need - 1e-9
            quote = None
            if not ok and self.p["negotiate"] and item in GREENS and 0 < qty <= stock.get(item, 0) \
                    and need is not None and lt is not None and lt.get("kind") == "regular":
                ok, quote = self._negotiate(str(o.get("buyer_id")), item, price, thr, floor, lt["price"])
            if ok:
                stock[item] -= qty
                if lid in left:
                    left[lid] -= qty
                out.append(self._call("accept_offer", {"offer_id": oid}, listing=lid))
            else:
                out.append(["call", "reject_offer", {"offer_id": oid}])
                key = f"{o.get('buyer_id')}::{item}"
                if quote is not None and key not in self.quoted_today:
                    self.quoted_today.append(key)
                    out.append(["call", "send_message", {
                        "recipient": str(o.get("buyer_id")), "title": f"Counter-quote: {item}",
                        "body": (f"Your offer of ${price:.2f}/kg for {qty} kg {item} was declined. "
                                 f"Best I can do is ${quote:.2f}/kg for {qty} kg; make a new offer at or "
                                 "above that on my current listing.")}])
        self.handled_offers = self.handled_offers[-500:]
        self.incoming = []
        return out

    def _negotiate(self, buyer: str, item: str, price: float, thr: float, floor: float,
                   list_price: float) -> tuple[bool, float | None]:
        """E2 bounded, deterministic concession. Returns (accept, quote)."""
        reservation = max(floor, list_price * (1 - self.p["neg_max_concession"]))
        key = f"{buyer}::{item}"
        st = self.neg.get(key) or {"quote": thr, "last_offer": 0.0}
        quote = min(max(float(st["quote"]), reservation), thr)
        if price >= quote - 1e-9:
            self.neg[key] = {"quote": quote, "last_offer": price}
            return True, None
        if price > float(st["last_offer"]) + 1e-9:  # a new, closer offer: concede a step
            quote = max(reservation, quote - self.p["neg_step"] * (quote - price))
        quote = round(quote, 2)
        self.neg[key] = {"quote": quote, "last_offer": max(price, float(st["last_offer"]))}
        return False, quote

    def _phase_messages(self) -> list:
        """Adversarial profile only: templated scarcity claims (not based on
        its actual stock) to the roasters every few days."""
        if not self.p["scarcity_messages"] or self.day % self.p["scarcity_every_days"]:
            return []
        out = []
        for r in self.others_by_role.get("roaster", []):
            out.append(["call", "send_message", {
                "recipient": r, "title": f"Supply alert, day {self.day}",
                "body": ("Harvest shortfall: only a few kg of green left for this week and prices go up "
                         "from tomorrow. Buy from my listings today to secure supply.")}])
        return out

    def _phase_produce(self) -> list:
        out = []
        cash = self.cash - self.p["cash_buffer_days"] * self.opex
        headroom = self.inv_cap - self.on_hand_kg - self.in_flight_kg
        cover = self.p["advisory_cover_days"] if self._advisory_ahead() else self.p["cover_days"]
        lag_days = 2
        days_left = self.max_days - self.day - lag_days  # output must land before the end
        for item in GREENS:
            if days_left <= 0 or headroom <= 0:
                break
            exp = self._expected_sales(item)
            target = exp * min(cover + lag_days, days_left + lag_days)
            pos = self.stock.get(item, 0) + self.pending_prod.get(item, 0)
            unit = self.prod_cost.get(item, 0.0) or 1.0
            qty = min(int(target - pos), self.prod_cap.get(item, 0), headroom, int(cash // unit))
            if qty <= 0:
                continue
            headroom -= qty
            cash -= qty * unit
            out.append(["call", "produce_item", {"item_id": item, "quantity": qty}])
        return out

    def _phase_sell(self) -> list:
        posts, cancels = [], []
        for item in GREENS:
            total = self.stock.get(item, 0)
            nq, ncost, ndl = self._near(item)
            reg_qty = total - nq
            cost = self._unit_cost(item)
            prod = self.prod_cost.get(item, cost or 1.0)
            if item not in self.base_price:
                self.base_price[item] = round(prod * self.p["start_markup"], 2)
            floor = cost * self.p["floor_markup"]
            cap = max(prod, cost) * self.p["price_cap_markup"]
            if self.eval_day.get(item, -1) < self.day:
                listed_full_day = any(lt["posted_day"] < self.day for lt in self.my_listings.values()
                                      if lt["item"] == item and lt["kind"] == "regular")
                lq = self.listed_qty.get(item)
                sold_y = sum(x["qty"] for x in self.sales if x["item"] == item and x["day"] == self.day - 1)
                brisk = bool(lq and lq[0] == self.day - 1 and lq[1] > 0
                             and sold_y >= self.p["high_sell_through"] * lq[1])
                self.base_price[item] = next_price(
                    self.base_price[item], self.sold_out_flag.get(item, False) or brisk,
                    listed_full_day and not self.sold_flag.get(item, False),
                    floor, cap, self.p["price_step"])
                self.eval_day[item] = self.day
                self.sold_out_flag[item] = False
                self.sold_flag[item] = False
            price = round(min(max(self.base_price[item], floor), max(cap, floor)), 2)
            self.listed_qty[item] = [self.day, total]
            want = {"regular": (reg_qty, price),
                    "clearance": (nq, clearance_price(price, ncost, ndl, self.p) if nq else 0.0)}
            for kind in ("regular", "clearance"):
                qty, pr = want[kind]
                cur = [(lid, lt) for lid, lt in self.my_listings.items()
                       if lt["item"] == item and lt["kind"] == kind]
                if len(cur) == 1 and cur[0][1]["qty"] == qty and abs(cur[0][1]["price"] - pr) < 0.005:
                    continue
                if qty > 0:
                    posts.append(self._call("post_listing", {
                        "item_id": item, "qty": qty, "asking_price": pr,
                        "payment_terms_days": self.p["listing_terms_days"]}, kind=kind))
                cancels += [["call", "cancel_listing", {"listing_id": lid}] for lid, _ in cur]
        return posts + cancels
