"""HeuristicRoasterV2Model — stronger rule-based baseline for the roaster role.

`get_model("heuristic_roaster_v2")` → `--models roaster_A:heuristic_roaster_v2`
(in-process; Harbor: `launch_sim.py --local roaster_A:heuristic_roaster_v2`).
The original 20-line `heuristic_roaster` stays registered unchanged for
continuity with earlier runs.

Same contract as every other model: one tool call per turn (30 virtual min
each, `wait_for_next_day` free), decisions made only from what the agent can
observe — the rules prompt (system prompt: roast recipes, caps, opex, festival
windows, reservation prices, starting retail prices) and tool results / wake
observations in its own message history. It never reads env internals. The
static numbers it uses (recipes, roast cap, opex, shelf life, festival
WINDOWS) are the module constants the rules prompt is rendered from; the
festival demand multiplier (not disclosed to agents) is never used.

All rules are untuned, principled defaults (no parameter search):

Day structure (state kept across the whole day; a mid-day wake never restarts
the routine). Morning (new "Day N" observation):
  views    view_dashboard (cash, AP, cap, roasts, payables due), view_inventory
           (lot costs/expiry), view_offers (in+out), view_listings, and
           view_market_aggregate on day 0, weekly, or when a price is unanchored
  offers   incoming offers: accept if price ≥ floor(listing) and terms ≤ 30 d,
           or price ≥ floor × (1 + 0.2%/day beyond 30 d) (≈ 2× the late-fee
           rate as time value + default risk); otherwise reject at once (no
           counter-offers exist, so a sub-floor offer is never left to sit)
  pay      AP due today or overdue (late fee starts after the due day); AP due
           tomorrow only if the cash buffer survives the payment
  roast    to the 50 kg/day green cap, item with the higher margin per green kg
           first (specialty usually), FIFO by expiry is automatic; never over
           the inventory cap or cash
  sell     one regular listing per roasted item (+ one clearance listing for
           lots ≤ 5 days from expiry), re-posted only when price/qty changed
  bids     withdraw pending bids above today's max bid or beaten by a ≥5%
           cheaper listing, then bid the asking price/terms on the cheapest
           eligible green listings (≤ 3 bids/day)
  messages ≤ 1 templated availability note per retailer per day, only when
           the listed qty/prices changed since the last note and ≤ 15 actions
           were used; no negotiation
  then     wait_for_next_day
Wakes: offer_received → view_offers + evaluate (inserted ahead of the rest of
the routine if it is still running); deal_accepted / invoice_paid /
return_received / dm_received → noted, no action (next morning's views
refresh state). At most ACTION_BUDGET (19) actions/day.

Costs: floors use the actual lot unit costs from view_inventory (qty-weighted
over the lots a listing would ship), never hard-coded cost bases.

Pricing per roasted item (state machine on a base price, festival markup on
top):
  anchor   first own deal price for the item if any, else consumer average
           price (view_market_aggregate) × RETAILER_MARGIN 0.6 (the retailer
           keeps 40% gross margin), else the retailers' starting retail price
           from the rules prompt × 0.6, else cost × 1.5; capped at the
           cheapest competing listing for the same item (match, don't
           undercut); floor = lot cost × 1.2; cap = consumer reservation price
  adapt    once a day: +5% if a regular listing sold out since the last
           evaluation, −5% if it was listed a full day and nothing sold
           (cancel + repost), else hold; clamped to [floor, cap]
  expiry   lots with days_left ≤ 5 go to a clearance listing at 5% off per day
           inside the window (5 d → −5%, 0 d → −30%), floor = lot cost (≤ 1 day
           left: half the lot cost — anything beats a 100% write-off)
  festival from the public calendar: ramp = the 5 days before the window
           (no price cuts, larger green and roasted-stock targets), window =
           listed at base price × 1.10 (the daily ±5% rule keeps adapting the
           base) and the same larger targets

Procurement (both commodity and specialty green):
  max bid  = expected roasted price × yield / 1.2 − labor per green kg, i.e. a
           bid at which the roasted lot still clears the 1.2× floor
  target   green position (on hand after today's roast + accepted in-flight
           purchases + pending bids) up to one day of roast capacity (1.5 days
           in ramp/festival), within inventory-cap headroom (committed holdings
           incl. in-flight and pending bids)
  ranking  margin per green kg (expected price × yield − labor − ask), best
           first; an item is skipped while its roasted stock + pending output
           exceeds 4 days of its recent sales (min one day of roast output;
           ×2 in ramp/festival)
  cash     spend only cash − buffer, buffer = 10 days opex + all open AP +
           tomorrow's roast labor + value of pending bids
  freshness skip green whose earliest lot expires within 5 days of delivery

Limitations (by design): no DM reading or negotiation, no counter-party
selection beyond price, no strategic use of payment terms when buying (uses
the seller's requested terms), no green resale, no reaction to roaster_B's
behaviour beyond matching its listing price.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from coffeebench.models.types import ModelResponse, ToolCall

ROASTED = ("roasted_specialty_kg", "roasted_coffee_kg")
GREEN_FOR = {"roasted_coffee_kg": "green_coffee_kg", "roasted_specialty_kg": "green_specialty_kg"}

# --- rule constants (untuned defaults; see module docstring) -----------------
FLOOR_MARKUP = 1.2
FALLBACK_MARKUP = 1.5
RETAILER_MARGIN = 0.6
PRICE_STEP = 0.05
EXPIRY_WINDOW_DAYS = 5
EXPIRY_DISCOUNT_PER_DAY = 0.05
FESTIVAL_RAMP_DAYS = 5
FESTIVAL_MARKUP = 1.10
FESTIVAL_STOCK_FACTOR = 2.0
FESTIVAL_GREEN_DAYS = 1.5
CASH_BUFFER_DAYS = 10
FREE_TERMS_DAYS = 30
TERMS_PREMIUM_PER_DAY = 0.002
SALES_COVER_DAYS = 4
SALES_WINDOW_DAYS = 7
GREEN_MIN_SHELF_DAYS = 5
COMPETITOR_UNDERCUT_FOR_WITHDRAW = 0.95
MAX_BIDS_PER_DAY = 3
ACTION_BUDGET = 19
MESSAGE_ACTION_LIMIT = 15
LISTING_TERMS_DAYS = 14
MARKET_REFRESH_DAYS = 7


# --- pure decision functions (unit-tested in scripts/test_heuristic_roaster_v2.py)

def festival_phase(day: int, windows: list[tuple[int, int]], ramp_days: int = FESTIVAL_RAMP_DAYS) -> str | None:
    """'festival' inside a window, 'ramp' in the `ramp_days` before it, else None."""
    for start, end in windows:
        if start <= day <= end:
            return "festival"
        if start - ramp_days <= day < start:
            return "ramp"
    return None


def max_bid(expected_price: float, yield_: float, labor: float, markup: float = FLOOR_MARKUP) -> float:
    """Highest green price per kg at which the roasted output still clears
    the `markup` × cost floor at `expected_price`."""
    return max(0.0, expected_price * yield_ / markup - labor)


def next_price(price: float, sold_out: bool, unsold: bool, floor: float, cap: float,
               allow_cut: bool = True, step: float = PRICE_STEP) -> float:
    """One daily adaptive step: +step on a sell-out, −step when a full day
    passed without a sale (unless cuts are paused), clamped to [floor, cap]."""
    if sold_out:
        price *= 1 + step
    elif unsold and allow_cut:
        price *= 1 - step
    return round(min(max(price, floor), max(cap, floor)), 2)


def clearance_price(list_price: float, unit_cost: float, days_left: int) -> float:
    """Discounted price for a lot `days_left` days from expiry (≤ window)."""
    disc = EXPIRY_DISCOUNT_PER_DAY * (EXPIRY_WINDOW_DAYS + 1 - max(0, days_left))
    floor = unit_cost * (0.5 if days_left <= 1 else 1.0)
    return round(max(list_price * (1 - disc), floor), 2)


def required_price(floor: float, terms_days: int) -> float:
    """Minimum acceptable offer price for the given payment terms."""
    extra = max(0, int(terms_days) - FREE_TERMS_DAYS)
    return floor * (1 + TERMS_PREMIUM_PER_DAY * extra)


def invoice_action(due_day: int, today: int, outstanding: float, cash: float, buffer: float) -> str:
    """'pay' AP due today/overdue whenever cash covers it, AP due tomorrow
    only if the buffer survives; otherwise 'defer'."""
    if outstanding <= 0 or cash < outstanding:
        return "defer"
    if due_day <= today:
        return "pay"
    if due_day == today + 1 and cash - outstanding >= buffer:
        return "pay"
    return "defer"


def anchor_price(cost: float, own_last: float | None = None, consumer_avg: float | None = None,
                 start_retail: float | None = None, competitor_ask: float | None = None,
                 cap: float | None = None) -> float:
    """Initial wholesale price from observed signals (see module docstring)."""
    if own_last:
        p = own_last
    elif consumer_avg:
        p = consumer_avg * RETAILER_MARGIN
    elif start_retail:
        p = start_retail * RETAILER_MARGIN
    else:
        p = cost * FALLBACK_MARKUP
    if competitor_ask:
        p = min(p, competitor_ask)
    floor = cost * FLOOR_MARKUP
    p = max(p, floor)
    if cap:
        p = min(p, max(cap, floor))
    return round(p, 2)


def _extract_day(text: str) -> int | None:
    m = re.search(r"\bDay\s+(\d+)\b", text or "")
    return int(m.group(1)) if m else None


class HeuristicRoasterV2Model:
    """Scripted roaster policy; state is plain data (checkpointable via vars())."""

    DEFAULT_MAX_INPUT_TOKENS = 1_000_000

    def __init__(self, model: str = "heuristic_roaster_v2"):
        self.model = model
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.system_prompt = ""  # set by Agent.init (rules prompt); parsed lazily
        self._prompt_parsed = False

        # Public rules (parsed from the rules prompt; fallbacks = the module
        # constants that prompt is rendered from).
        self.festivals: list[tuple[int, int]] = []
        self.reservation: dict[str, float] = {}
        self.start_retail: dict[str, float] = {}
        self.retailers: list[str] = []
        self.me: str | None = None

        self.day = -1
        self.queue: list[list] = []  # [kind, name, args]; kind ∈ {"call", "phase"}
        self.actions_today = 0
        self._seq = 0
        self._last_call: dict | None = None

        # Observed state (refreshed by views).
        self.cash = 0.0
        self.ap_total = 0.0
        self.on_hand_kg = 0
        self.in_flight_kg = 0
        self.cap_kg = 120
        self.pending_roast_out: dict[str, int] = {}
        self.payables: dict[str, dict] = {}
        self.lots: list[dict] = []
        self.listings_market: list[dict] = []
        self.incoming: list[dict] = []
        self.outgoing: list[dict] = []
        self.consumer_avg: dict[str, float] = {}
        self.market_day = -99
        self.roast_used = 0

        # Own history.
        self.my_listings: dict[str, dict] = {}  # lid → {item, kind, qty, price, posted_day, sold}
        self.sales: list[dict] = []  # {day, item, qty, price}
        self.last_deal_price: dict[str, float] = {}
        self.base_price: dict[str, float] = {}
        self.eval_day: dict[str, int] = {}
        self.sold_out_flag: dict[str, bool] = {}
        self.sold_flag: dict[str, bool] = {}
        self.handled_offers: list[str] = []
        self.paid: list[str] = []
        self.bids_today = 0
        self.messaged_today: list[str] = []
        self.last_message: dict[str, str] = {}
        self.notes: list[str] = []

    # ---------- Model protocol ----------------------------------------------

    def query(self, messages, tools=None) -> ModelResponse:  # noqa: ARG002
        self.n_calls += 1
        if not self._prompt_parsed:
            self._parse_rules(self.system_prompt or "")
        tail = []
        for m in reversed(messages or []):
            if m.get("role") == "assistant":
                break
            tail.append(m)
        for m in reversed(tail):
            role = m.get("role")
            if role == "tool":
                try:
                    data = json.loads(m.get("content") or "{}")
                except (TypeError, ValueError):
                    data = {}
                self._on_result(m.get("name") or "", data if isinstance(data, dict) else {})
            elif role == "user":
                self._on_observation(m.get("content") or "")
        name, args, meta = self._next_call()
        self._seq += 1
        self._last_call = {"name": name, "args": dict(args), "meta": meta}
        if name != "wait_for_next_day":
            self.actions_today += 1
        return ModelResponse(
            content="(heuristic_roaster_v2)", thinking="",
            tool_calls=[ToolCall(id=f"hv2_{self._seq:05d}", name=name, input=args)],
            stop_reason="tool_use", cost=0.0, raw=None,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        return {"model": self.model, "n_calls": self.n_calls, "cost": 0.0,
                "total_input_tokens": 0, "total_output_tokens": 0, "last_input_tokens": 0}

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        return content

    # ---------- rules prompt -------------------------------------------------

    def _parse_rules(self, text: str) -> None:
        from coffeebench import environment as env_mod

        self._prompt_parsed = True
        fest = re.findall(r"`[\w-]+` days (\d+)[–-](\d+)", text)
        self.festivals = [(int(a), int(b)) for a, b in fest] or [
            (s, e) for s, e, _m, _n in env_mod.FESTIVALS
        ]
        for item in ROASTED:
            m = re.search(r"consumers never pay more than the reservation price \(([^)]*)\)", text)
            res = re.search(rf"{item} \$([\d,.]+)/kg", m.group(1)) if m else None
            self.reservation[item] = float(res.group(1).replace(",", "")) if res else (
                {"roasted_coffee_kg": 30.0, "roasted_specialty_kg": 80.0}[item])
            sr = re.search(rf"starting retail prices[^\n]*?{item} \$([\d,.]+)/kg", text)
            if sr:
                self.start_retail[item] = float(sr.group(1).replace(",", ""))
        self.retailers = re.findall(r"(\w+) \(retailer\)", text) or ["retailer_A", "retailer_B"]
        me = re.search(r"^# CoffeeBench — .*\(`(\w+)`, roaster\)", text, re.M)
        self.me = me.group(1) if me else None

    # ---------- statics ------------------------------------------------------

    @staticmethod
    def _recipe(green: str) -> dict:
        from coffeebench.environment import ROAST_RECIPES

        return ROAST_RECIPES[green]

    @staticmethod
    def _roast_cap() -> int:
        from coffeebench.environment import ROAST_DAILY_CAP_GREEN_KG

        return int(ROAST_DAILY_CAP_GREEN_KG)

    @staticmethod
    def _opex() -> float:
        from coffeebench.environment import DAILY_OPEX_BY_ROLE

        return float(DAILY_OPEX_BY_ROLE.get("roaster", 30.0))

    # ---------- observation / result ingestion -------------------------------

    def _on_observation(self, text: str) -> None:
        day = _extract_day(text)
        wake = re.search(r"\[Wake reason: ([^\]]*)\]", text)
        m = re.search(r"Cash \$(-?[\d.]+)", text)
        if m:
            self.cash = float(m.group(1))
        if day is not None and day != self.day:
            self._start_day(day)
            return
        if wake:
            reason = wake.group(1)
            self.notes.append(f"d{self.day}: {reason[:80]}")
            self.notes = self.notes[-50:]
            if reason.startswith("offer_received") and not any(
                q[1] == "view_offers" for q in self.queue
            ):
                self.queue[0:0] = [["call", "view_offers", {"direction": "incoming"}],
                                   ["phase", "offers", {}]]

    def _start_day(self, day: int) -> None:
        self.day = day
        self.actions_today = 0
        self.roast_used = 0
        self.bids_today = 0
        self.messaged_today = []
        views = [["call", "view_dashboard", {}], ["call", "view_inventory", {}],
                 ["call", "view_offers", {"direction": "all"}], ["call", "view_listings", {}]]
        self.queue = views + [["phase", "market", {}]] + [
            ["phase", p, {}] for p in ("offers", "pay", "roast", "sell", "bids", "messages")
        ]

    def _on_result(self, name: str, data: dict) -> None:
        ok = data.get("status") == "success"
        args = (self._last_call or {}).get("args", {}) if (self._last_call or {}).get("name") == name else {}
        if not ok:
            return
        if name == "view_dashboard":
            h = data.get("headline") or {}
            self.cash = float(h.get("cash", self.cash))
            self.ap_total = float(h.get("ap", 0.0))
            inv = data.get("inventory") or {}
            self.on_hand_kg = int(inv.get("on_hand_kg", 0))
            self.in_flight_kg = int(inv.get("in_flight_inbound_kg", 0))
            self.cap_kg = int(inv.get("cap_kg", self.cap_kg))
            self.pending_roast_out = {}
            for r in data.get("pending_roasts") or []:
                it = r.get("output_item")
                self.pending_roast_out[it] = self.pending_roast_out.get(it, 0) + int(r.get("output_qty", 0))
            pay = data.get("payables_due") or {}
            self.payables = {str(r["id"]): r for r in pay.get("rows") or []}
            if pay.get("omitted"):
                self._insert_after_views(["call", "view_payables", {}])
            mine = data.get("my_open_listings") or {}
            if not mine.get("omitted"):
                open_ids = {str(r["id"]) for r in mine.get("rows") or []}
                for lid in list(self.my_listings):
                    if lid not in open_ids:
                        self._close_listing(lid)
                for r in mine.get("rows") or []:
                    lt = self.my_listings.get(str(r["id"]))
                    if lt is not None:
                        lt["qty"] = int(r.get("qty", lt["qty"]))
        elif name == "view_payables":
            for r in data.get("rows") or []:
                self.payables[str(r["id"])] = {"id": r["id"], "outstanding": r.get("outstanding", 0),
                                               "due_day": r.get("due_day", self.day)}
        elif name == "view_inventory":
            self.lots = [lt for lt in data.get("lots") or [] if int(lt.get("qty", 0)) > 0]
        elif name == "view_listings":
            self.listings_market = list(data.get("listings") or [])
        elif name == "view_offers":
            pending = [o for o in data.get("offers") or [] if o.get("status") == "pending"]
            # Merge (a mid-day incoming-only refresh must not drop offers
            # seen earlier but not yet answered).
            seen = {str(o.get("id")): o for o in self.incoming}
            seen.update({str(o.get("id")): o for o in pending if self._is_incoming(o)})
            self.incoming = [o for oid, o in seen.items() if oid not in self.handled_offers]
            if args.get("direction", "all") == "all":
                self.outgoing = [o for o in pending if not self._is_incoming(o)]
        elif name == "view_market_aggregate":
            self.market_day = self.day
            for r in data.get("per_item") or []:
                if float(r.get("avg_unit_price") or 0) > 0:
                    self.consumer_avg[r["item_id"]] = float(r["avg_unit_price"])
        elif name == "accept_offer":
            item = data.get("item_id")
            qty = int(data.get("qty", 0))
            price = float(data.get("unit_price", 0.0))
            self.sales.append({"day": self.day, "item": item, "qty": qty, "price": price})
            self.sales = self.sales[-400:]
            self.last_deal_price[item] = price
            self.sold_flag[item] = True
            self._take_stock(item, qty)
            lid = ((self._last_call or {}).get("meta") or {}).get("listing")
            lt = self.my_listings.get(lid) if lid else None
            if lt is not None:
                lt["qty"] -= qty
                lt["sold"] += qty
                if lt["qty"] <= 0:
                    if lt["kind"] == "regular":
                        self.sold_out_flag[item] = True
                    self._close_listing(lid)
        elif name == "post_listing":
            lid = str(data.get("listing_id"))
            meta = (self._last_call or {}).get("meta") or {}
            self.my_listings[lid] = {"item": args.get("item_id"), "kind": meta.get("kind", "regular"),
                                     "qty": int(args.get("qty", 0)), "price": float(args.get("asking_price", 0)),
                                     "posted_day": self.day, "sold": 0}
        elif name == "cancel_listing":
            self._close_listing(str(args.get("listing_id")))
        elif name == "roast":
            self.cash = float(data.get("remaining_cash", self.cash))
            self.roast_used = int(data.get("shared_roast_capacity_used_today", self.roast_used))
            green = data.get("input_item")
            out = data.get("output_item")
            q = int(data.get("green_consumed_kg", 0))
            self._take_stock(green, q)
            self.pending_roast_out[out] = self.pending_roast_out.get(out, 0) + int(data.get("roasted_output_kg", 0))
            self.on_hand_kg -= q
            self.in_flight_kg += int(data.get("roasted_output_kg", 0))
        elif name == "pay_invoice":
            self.cash = float(data.get("remaining_cash", self.cash - float(data.get("paid_amount", 0))))
            self.paid.append(str(args.get("invoice_id")))
            self.paid = self.paid[-200:]
        elif name == "make_offer":
            self.outgoing.append({"id": data.get("offer_id"), "listing_id": args.get("listing_id"),
                                  "offered_price": args.get("offered_price"), "qty": args.get("qty"),
                                  "item_id": (self._last_call or {}).get("meta", {}).get("item"),
                                  "posted_day": self.day, "status": "pending"})
        elif name == "withdraw_offer":
            self.outgoing = [o for o in self.outgoing if str(o.get("id")) != str(args.get("offer_id"))]

    def _is_incoming(self, offer: dict) -> bool:
        if self.me:
            return offer.get("seller_id") == self.me
        return str(offer.get("listing_id")) in self.my_listings

    def _close_listing(self, lid: str) -> None:
        self.my_listings.pop(lid, None)

    def _insert_after_views(self, call: list) -> None:
        idx = next((i for i, q in enumerate(self.queue) if q[0] == "phase"), len(self.queue))
        self.queue.insert(idx, call)

    def _take_stock(self, item: str | None, qty: int) -> None:
        """Mirror FIFO consumption on the local lot copy (earliest expiry first)."""
        for lt in sorted((x for x in self.lots if x.get("item_id") == item),
                         key=lambda x: (x.get("expiry_day") is None, x.get("expiry_day") or 0)):
            if qty <= 0:
                break
            take = min(qty, int(lt["qty"]))
            lt["qty"] = int(lt["qty"]) - take
            qty -= take
        self.lots = [lt for lt in self.lots if int(lt["qty"]) > 0]

    # ---------- derived quantities -----------------------------------------

    def _stock(self, item: str) -> int:
        return sum(int(lt["qty"]) for lt in self.lots if lt.get("item_id") == item)

    def _split_lots(self, item: str) -> tuple[list[dict], list[dict]]:
        lots = [lt for lt in self.lots if lt.get("item_id") == item]
        near = [lt for lt in lots if lt.get("days_left") is not None and int(lt["days_left"]) <= EXPIRY_WINDOW_DAYS]
        reg = [lt for lt in lots if lt not in near]
        return reg, near

    @staticmethod
    def _avg_cost(lots: list[dict]) -> float:
        q = sum(int(lt["qty"]) for lt in lots)
        return sum(int(lt["qty"]) * float(lt["unit_cost"]) for lt in lots) / q if q else 0.0

    def _unit_cost(self, item: str) -> float:
        reg, near = self._split_lots(item)
        return self._avg_cost(reg) or self._avg_cost(near)

    def _floor(self, item: str, kind: str = "regular") -> float:
        reg, near = self._split_lots(item)
        if kind == "clearance" and near:
            dl = min(int(lt["days_left"]) for lt in near)
            return self._avg_cost(near) * (0.5 if dl <= 1 else 1.0)
        cost = self._avg_cost(reg) or self._avg_cost(near)
        return cost * FLOOR_MARKUP

    def _phase(self) -> str | None:
        return festival_phase(self.day, self.festivals)

    def _competitor_ask(self, item: str) -> float | None:
        asks = [float(r["asking_price"]) for r in self.listings_market
                if r.get("item_id") == item and int(r.get("qty", 0)) > 0]
        return min(asks) if asks else None

    def _expected_price(self, item: str) -> float:
        """Base (non-festival) price we expect to realise for a roasted item."""
        if item in self.base_price:
            return self.base_price[item]
        green = GREEN_FOR[item]
        rec = self._recipe(green)
        cost = self._unit_cost(item)
        if not cost:
            asks = [float(r["asking_price"]) for r in self.listings_market if r.get("item_id") == green]
            g = min(asks) if asks else 0.0
            cost = (g + rec["labor_cost_per_kg"]) / rec["yield"] if g else 0.0
        return anchor_price(cost, self.last_deal_price.get(item), self.consumer_avg.get(item),
                            self.start_retail.get(item), self._competitor_ask(item),
                            self.reservation.get(item))

    def _buffer(self) -> float:
        return CASH_BUFFER_DAYS * self._opex() + self.ap_total

    def _recent_sales_per_day(self, item: str) -> float:
        lo = self.day - SALES_WINDOW_DAYS
        q = sum(s["qty"] for s in self.sales if s["item"] == item and s["day"] >= lo)
        return q / SALES_WINDOW_DAYS

    # ---------- action selection -------------------------------------------

    def _next_call(self) -> tuple[str, dict, dict]:
        while self.queue:
            if self.actions_today >= ACTION_BUDGET:
                self.queue = []
                break
            entry = self.queue.pop(0)
            kind, name, args = entry[0], entry[1], entry[2]
            if kind == "call":
                return name, dict(args), dict(entry[3]) if len(entry) > 3 else {}
            self.queue[0:0] = getattr(self, f"_phase_{name}")()
        return "wait_for_next_day", {}, {}

    @staticmethod
    def _call(name: str, args: dict, **meta) -> list:
        """Queued call; `meta` travels to the result handler via `_last_call`."""
        return ["call", name, args, meta]

    def _phase_market(self) -> list:
        need = any(it not in self.base_price and self._stock(it) > 0 for it in ROASTED)
        if self.day - self.market_day >= MARKET_REFRESH_DAYS or (need and self.market_day < self.day):
            return [["call", "view_market_aggregate", {"days_back": 3}]]
        return []

    def _phase_offers(self) -> list:
        out = []
        stock = {it: self._stock(it) for it in ROASTED}
        left = {lid: int(lt["qty"]) for lid, lt in self.my_listings.items()}
        offers = sorted((o for o in self.incoming if str(o.get("id")) not in self.handled_offers),
                        key=lambda o: -float(o.get("offered_price") or 0))
        for o in offers:
            oid = str(o["id"])
            self.handled_offers.append(oid)
            lid = str(o.get("listing_id"))
            lt = self.my_listings.get(lid)
            item = (lt or {}).get("item") or o.get("item_id")
            kind = (lt or {}).get("kind", "regular")
            price = float(o.get("offered_price") or 0)
            qty = int(o.get("qty") or 0)
            if lid in left and left[lid] <= 0:
                continue  # listing closes on the accepts above; the env rejects the rest
            fl = self._floor(item, kind) if item in stock else 0.0
            ok = (item in stock and fl > 0 and qty <= stock[item] and qty <= left.get(lid, qty)
                  and price >= required_price(fl, int(o.get("payment_terms_days", 30))))
            if ok:
                stock[item] -= qty
                if lid in left:
                    left[lid] -= qty
                out.append(self._call("accept_offer", {"offer_id": oid}, listing=lid))
            else:
                out.append(["call", "reject_offer", {"offer_id": oid}])
        self.handled_offers = self.handled_offers[-500:]
        self.incoming = []
        return out

    def _phase_pay(self) -> list:
        out = []
        cash = self.cash
        buf = self._buffer()
        for r in sorted(self.payables.values(), key=lambda r: (r.get("due_day", 0), str(r["id"]))):
            iid = str(r["id"])
            if iid in self.paid:
                continue
            amt = float(r.get("outstanding") or 0)
            if invoice_action(int(r.get("due_day", self.day)), self.day, amt, cash, buf) == "pay":
                cash -= amt
                out.append(["call", "pay_invoice", {"invoice_id": iid}])
        return out

    def _phase_roast(self) -> list:
        out = []
        remaining = self._roast_cap() - self.roast_used
        committed = self.on_hand_kg + self.in_flight_kg
        cash = self.cash
        ranked = []
        for item in ROASTED:
            green = GREEN_FOR[item]
            rec = self._recipe(green)
            avail = sum(int(lt["qty"]) for lt in self.lots if lt.get("item_id") == green
                        and (lt.get("days_left") is None or int(lt["days_left"]) >= 0))
            m = self._expected_price(item) * rec["yield"] - rec["labor_cost_per_kg"] - self._unit_cost(green)
            ranked.append((m, green, rec, avail))
        for _m, green, rec, avail in sorted(ranked, key=lambda x: -x[0]):
            qty = min(avail, remaining, int(cash // rec["labor_cost_per_kg"]))
            if qty <= 0:
                continue
            outq = int(math.floor(qty * rec["yield"] + 1e-9))
            if outq <= 0 or committed - qty + outq > self.cap_kg:
                continue
            remaining -= qty
            committed += outq - qty
            cash -= qty * rec["labor_cost_per_kg"]
            out.append(["call", "roast", {"green_item_id": green, "qty_kg": qty}])
        return out

    def _phase_sell(self) -> list:
        out = []
        phase = self._phase()
        for item in ROASTED:
            reg, near = self._split_lots(item)
            reg_qty = sum(int(lt["qty"]) for lt in reg)
            near_qty = sum(int(lt["qty"]) for lt in near)
            cost = self._avg_cost(reg) or self._avg_cost(near)
            if cost <= 0:  # no stock: a listing would advertise goods we lack
                out += [["call", "cancel_listing", {"listing_id": lid}]
                        for lid, lt in self.my_listings.items() if lt["item"] == item]
                continue
            floor = cost * FLOOR_MARKUP
            cap = self.reservation.get(item, 1e9)
            if item not in self.base_price:
                self.base_price[item] = self._expected_price(item)
            elif self.eval_day.get(item, -1) < self.day:
                mine = [lt for lt in self.my_listings.values()
                        if lt["item"] == item and lt["kind"] == "regular"]
                listed_full_day = any(lt["posted_day"] < self.day for lt in mine)
                self.base_price[item] = next_price(
                    self.base_price[item], self.sold_out_flag.get(item, False),
                    listed_full_day and not self.sold_flag.get(item, False),
                    floor, cap, allow_cut=phase != "ramp")
            self.eval_day[item] = self.day
            self.sold_out_flag[item] = False
            self.sold_flag[item] = False
            base = max(self.base_price[item], floor)
            price = round(min(base * (FESTIVAL_MARKUP if phase == "festival" else 1.0), max(cap, floor)), 2)
            want = {"regular": (reg_qty, price)}
            if near_qty:
                dl = min(int(lt["days_left"]) for lt in near)
                want["clearance"] = (near_qty, clearance_price(price, self._avg_cost(near), dl))
            for kind in ("regular", "clearance"):
                qty, p = want.get(kind, (0, 0.0))
                cur = [(lid, lt) for lid, lt in self.my_listings.items()
                       if lt["item"] == item and lt["kind"] == kind]
                if len(cur) == 1 and cur[0][1]["qty"] == qty and abs(cur[0][1]["price"] - p) < 0.005:
                    continue
                out += [["call", "cancel_listing", {"listing_id": lid}] for lid, _ in cur]
                if qty > 0:
                    out.append(self._call("post_listing", {
                        "item_id": item, "qty": qty, "asking_price": p,
                        "payment_terms_days": LISTING_TERMS_DAYS}, kind=kind))
        return out

    def _phase_bids(self) -> list:
        out = []
        phase = self._phase()
        open_ids = {str(r["id"]): r for r in self.listings_market}
        green_items = list(GREEN_FOR.values())
        info = {}
        for item, green in GREEN_FOR.items():
            rec = self._recipe(green)
            exp = self._expected_price(item)
            info[green] = {"item": item, "rec": rec, "exp": exp,
                           "max": max_bid(exp, rec["yield"], rec["labor_cost_per_kg"])}
        cands = sorted(
            (r for r in self.listings_market
             if r.get("item_id") in green_items and int(r.get("qty", 0)) > 0
             and float(r["asking_price"]) <= info[r["item_id"]]["max"]
             and ((r.get("freshness_at_posting") or {}).get("earliest_expiry_day") is None
                  or int(r["freshness_at_posting"]["earliest_expiry_day"])
                  >= self.day + int(r.get("delivery_eta_days", 1)) + GREEN_MIN_SHELF_DAYS)),
            key=lambda r: -(info[r["item_id"]]["exp"] * info[r["item_id"]]["rec"]["yield"]
                            - info[r["item_id"]]["rec"]["labor_cost_per_kg"] - float(r["asking_price"])))
        # Withdraw stale bids (posted before today) that are above today's max
        # bid or beaten by a clearly cheaper eligible listing.
        keep = []
        for o in self.outgoing:
            g = o.get("item_id") or (open_ids.get(str(o.get("listing_id"))) or {}).get("item_id")
            price = float(o.get("offered_price") or 0)
            stale = int(o.get("posted_day", self.day)) < self.day
            cheaper = any(c["item_id"] == g and float(c["asking_price"]) < price * COMPETITOR_UNDERCUT_FOR_WITHDRAW
                          for c in cands)
            if g in info and stale and (price > info[g]["max"] or cheaper):
                out.append(["call", "withdraw_offer", {"offer_id": str(o["id"])}])
            else:
                o["item_id"] = g
                keep.append(o)
        pend_qty = sum(int(o.get("qty") or 0) for o in keep)
        pend_val = sum(int(o.get("qty") or 0) * float(o.get("offered_price") or 0) for o in keep)
        # Position: green on hand + purchases in flight + pending bids.
        green_on_hand = sum(self._stock(g) for g in green_items)
        roast_out = sum(self.pending_roast_out.values())
        inbound_purchases = max(0, self.in_flight_kg - roast_out)
        target = self._roast_cap() * (FESTIVAL_GREEN_DAYS if phase else 1.0)
        need = int(target - green_on_hand - inbound_purchases - pend_qty)
        headroom = self.cap_kg - self.on_hand_kg - self.in_flight_kg - pend_qty
        need = min(need, headroom)
        budget = self.cash - self._buffer() - pend_val - self._roast_cap() * max(
            r["labor_cost_per_kg"] for r in (self._recipe(g) for g in green_items))
        bid_listings = {str(o.get("listing_id")) for o in keep}
        stock_factor = FESTIVAL_STOCK_FACTOR if phase else 1.0
        for r in cands:
            if need <= 0 or budget <= 0 or self.bids_today >= MAX_BIDS_PER_DAY:
                break
            if str(r["id"]) in bid_listings:
                continue
            g = r["item_id"]
            item, rec = info[g]["item"], info[g]["rec"]
            roasted_pos = self._stock(item) + self.pending_roast_out.get(item, 0)
            limit = stock_factor * max(SALES_COVER_DAYS * self._recent_sales_per_day(item),
                                       self._roast_cap() * rec["yield"])
            if roasted_pos >= limit:
                continue
            ask = float(r["asking_price"])
            qty = min(int(r["qty"]), need, int(budget // ask) if ask > 0 else 0)
            if qty <= 0:
                continue
            need -= qty
            budget -= qty * ask
            self.bids_today += 1
            out.append(self._call("make_offer", {
                "listing_id": str(r["id"]), "offered_price": ask, "qty": qty,
                "payment_terms_days": int(r.get("payment_terms_days", 30))}, item=g))
        self.outgoing = keep
        return out

    def _phase_messages(self) -> list:
        if self.actions_today >= MESSAGE_ACTION_LIMIT:
            return []
        rows = [lt for lt in self.my_listings.values() if lt["qty"] > 0]
        if not rows:
            return []
        body = "; ".join(f"{lt['qty']} kg {lt['item']} at ${lt['price']:.2f}/kg"
                         + (" (near expiry)" if lt["kind"] == "clearance" else "")
                         for lt in sorted(rows, key=lambda x: (x["item"], x["kind"])))
        out = []
        for r in self.retailers:
            # ≤ 1/day per retailer, and only when the offer changed (no spam).
            if r in self.messaged_today or self.last_message.get(r) == body:
                continue
            self.messaged_today.append(r)
            self.last_message[r] = body
            out.append(["call", "send_message", {
                "recipient": r, "title": f"Roaster availability, day {self.day}",
                "body": f"Listed now: {body}. Offers on the listings are answered the same day."}])
        return out
