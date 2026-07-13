"""Deterministic scripted-policy models for an active, zero-LLM economy.

These models conform to the same `Model` protocol as the provider wrappers and
plug into the standard `Agent` harness — one tool call per turn, read from the
agent's own message history, no network, no API keys, no provider tokens, no
cost, and no unseeded randomness. They exist so the whole six-agent supply chain
can be exercised for free and, critically, *deterministically* — which is the
substrate the checkpoint/branch verification relies on (see
`docs/deterministic_checkpointing.md`).

There is one policy per role:

    HeuristicFarmerModel    — produce green, list it, accept roaster bids, pay AP
    HeuristicRoasterModel   — buy green, roast it, list roasted, accept retailer bids
    HeuristicRetailerModel  — buy roasted, price the storefront, pay AP

Both agents of a role share the same policy class. The goal is NOT optimal play;
it is a simple, readable, deterministic routine that keeps every economic
channel active (production, listings, offers, deals, shipments, roasting,
consumer sales, AR/AP, payments, spoilage, and both good and bad outcomes)
across the whole horizon.

Design notes shared by all three:

* **State from the transcript.** Each turn we read the *latest* message. A fresh
  morning observation ("Day N") resets the per-day routine; a mid-day reactive
  wake (an offer landed, a delivery arrived) re-queues the views needed to react
  (so incoming offers get accepted same-day) but keeps the one-shot flags so we
  never re-produce / re-roast / re-buy within a day.
* **Authoritative own-listing view.** Sellers call `view_listings(only_others=
  False)` so they see BOTH the market (to buy from) and their OWN open listings
  (to (a) accept incoming offers by resolving the offer's item, and (b) top up
  only the *unlisted* portion of inventory rather than re-posting every day —
  which is what keeps listings from accumulating without bound).
* **Deals always close.** Buyers bid at the listing's asking price (capped by a
  fixed max); sellers accept any bid at or above a fixed cost-plus floor. Ask >
  floor by construction, so the handshake is deterministic.
* **Checkpointable.** `checkpoint_state()` / `restore_checkpoint_state()` capture
  the full internal routine so a model can resume mid-run without repeating
  completed actions and without any pickling.
"""

from __future__ import annotations

import json
import re
from typing import Any

from coffeebench.models.types import ModelResponse, ToolCall


def _extract_day(text: str) -> int | None:
    """Pull the day index out of a morning / wake observation ('Day N')."""
    if not text:
        return None
    m = re.search(r"\bDay\s+(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_agent_id(text: str) -> str | None:
    """Pull the agent id from the observation line 'You are X (agent_id).'."""
    if not text:
        return None
    m = re.search(r"You are .+?\(([a-zA-Z_]+_[A-Z])\)", text)
    return m.group(1) if m else None


class ScriptedModel:
    """Base class for the deterministic role policies.

    Subclasses set `ROLE`, `DEFAULT_COST_BASIS`, `SELLABLE`, and implement
    `_next_role_action()` (and optionally `_role_result`).
    """

    DEFAULT_MAX_INPUT_TOKENS = 1_000_000
    ROLE = "generic"
    # Fallback per-item WAVG cost basis. `view_trial_balance` surfaces total
    # inventory value but not per-item WAVG, so — like the original heuristic
    # roaster — we use canonical steady-state costs. They drift slightly after
    # the agent's own deals but stay close enough for a cost-plus policy.
    DEFAULT_COST_BASIS: dict[str, float] = {}
    # Items this role sells B2B (posts listings for). Empty for retailers.
    SELLABLE: tuple[str, ...] = ()
    MARKUP_FLOOR = 1.2  # accept offers at/above cost * this
    MIN_LIST_CHUNK = 1  # only (re)list when >= this much inventory is unlisted

    def __init__(self, model: str):
        self.model = model
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.system_prompt = ""

        # Long-lived state (persists across days).
        self.agent_id: str = ""
        self.cash: float = 0.0
        self.inventory: dict[str, int] = {}
        self.cost_basis: dict[str, float] = dict(self.DEFAULT_COST_BASIS)

        self._call_seq = 0
        self._last_call: dict | None = None

        self.day = -1
        self._init_day_queues()
        # Transient caches rebuilt from view_listings each observation (NOT
        # checkpointed — the resume's first morning view repopulates them).
        self.own_open: dict[str, str] = {}  # my open listing_id -> item_id
        self.own_listed_qty: dict[str, int] = {}  # item_id -> kg currently listed
        self.market_listings: list[dict] = []  # OTHER agents' open listings

    # ---------- Model protocol ---------------------------------------------

    def query(self, messages, tools=None) -> ModelResponse:  # noqa: ARG002
        self.n_calls += 1
        self._ingest(messages)
        call = self._next_action()
        return ModelResponse(
            content=f"(scripted {self.ROLE} policy)",
            thinking="",
            tool_calls=[call],
            stop_reason="tool_use",
            cost=0.0,
            raw=None,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_calls": self.n_calls,
            "cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "last_input_tokens": 0,
        }

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        return content

    # ---------- checkpoint interface ---------------------------------------

    _STATE_KEYS = (
        "n_calls",
        "agent_id",
        "cash",
        "inventory",
        "cost_basis",
        "_call_seq",
        "_last_call",
        "day",
        "views_pending",
        "accepts_pending",
        "pays_pending",
        "one_shots",
    )

    def checkpoint_state(self) -> dict:
        """Serializable snapshot of the routine (no pickling). Transient
        listing caches are omitted — the resume's first daily view rebuilds
        them before any action is taken."""
        state: dict[str, Any] = {}
        for key in self._STATE_KEYS:
            val = getattr(self, key, None)
            if isinstance(val, dict):
                state[key] = dict(val)
            elif isinstance(val, list):
                state[key] = list(val)
            else:
                state[key] = val
        return state

    def restore_checkpoint_state(self, state: dict) -> None:
        for key in self._STATE_KEYS:
            if key not in state:
                continue
            val = state[key]
            if isinstance(val, dict):
                setattr(self, key, dict(val))
            elif isinstance(val, list):
                setattr(self, key, list(val))
            else:
                setattr(self, key, val)

    # ---------- per-day routine --------------------------------------------

    def _init_day_queues(self) -> None:
        self.views_pending: list[str] = []
        self.accepts_pending: list[str] = []
        self.pays_pending: list[str] = []
        self.one_shots: dict[str, bool] = {}

    def _reset_for_new_day(self, day: int) -> None:
        self.day = day
        self.views_pending = list(self._daily_views())
        self.accepts_pending = []
        self.pays_pending = []
        self.one_shots = {}

    def _reobserve_same_day(self) -> None:
        """Reactive same-day wake: re-check offers / listings / cash so newly
        arrived offers get accepted, but keep one-shot flags intact."""
        self.views_pending = list(self._wake_views())

    def _daily_views(self) -> list[str]:
        return ["view_trial_balance", "view_listings", "view_offers", "view_payables"]

    def _wake_views(self) -> list[str]:
        return ["view_trial_balance", "view_listings", "view_offers"]

    # ---------- state ingestion --------------------------------------------

    def _ingest(self, messages: list[dict]) -> None:
        if not messages:
            return
        last = messages[-1]
        role = last.get("role")
        if role == "user":
            content = last.get("content") or ""
            if not self.agent_id:
                aid = _extract_agent_id(content)
                if aid:
                    self.agent_id = aid
            new_day = _extract_day(content)
            if new_day is not None and new_day != self.day:
                self._reset_for_new_day(new_day)
            else:
                self._reobserve_same_day()
            return
        if role == "tool":
            name = last.get("name") or ""
            try:
                data = json.loads(last.get("content") or "{}")
            except (TypeError, ValueError):
                return
            self._handle_result(name, data)

    def _handle_result(self, name: str, data: dict) -> None:
        if name == "view_trial_balance":
            accounts = data.get("accounts") or {}
            cash_acc = accounts.get("cash") or {}
            self.cash = float(cash_acc.get("balance") or 0.0)
            self.inventory = {
                k: int(v) for k, v in (data.get("inventory_by_item") or {}).items()
            }
        elif name == "view_listings":
            self._parse_listings(data.get("listings") or [])
        elif name == "view_offers":
            offers = data.get("offers") or []
            self.accepts_pending = [
                str(o["id"])
                for o in offers
                if o.get("status") in (None, "open", "pending")
                and self._offer_above_floor(o)
            ]
        elif name == "view_payables":
            rows = data.get("rows") or []
            self.pays_pending = [
                str(r["id"]) for r in rows if float(r.get("outstanding") or 0.0) > 0.0
            ]
        self._role_result(name, data)

    def _parse_listings(self, listings: list[dict]) -> None:
        """Split the (only_others=False) listing dump into MY open listings
        (for acceptance + top-up) and the market (for procurement)."""
        own_open: dict[str, str] = {}
        own_qty: dict[str, int] = {}
        market: list[dict] = []
        for li in listings:
            if int(li.get("qty", 0) or 0) <= 0:
                continue
            seller = li.get("seller_id")
            if self.agent_id and seller == self.agent_id:
                own_open[str(li.get("id"))] = str(li.get("item_id"))
                own_qty[str(li.get("item_id"))] = own_qty.get(
                    str(li.get("item_id")), 0
                ) + int(li.get("qty", 0) or 0)
            else:
                market.append(li)
        self.own_open = own_open
        self.own_listed_qty = own_qty
        self.market_listings = market

    def _offer_above_floor(self, offer: dict) -> bool:
        # Resolve the item via my authoritative open-listing map, so acceptance
        # is robust for ANY of my listings (not a separately-tracked subset).
        item_id = self.own_open.get(str(offer.get("listing_id") or ""))
        if item_id is None:
            return False
        price = float(offer.get("offered_price") or 0.0)
        cost = float(self.cost_basis.get(item_id, 0.0))
        if cost <= 0.0:
            return price > 0.0
        return price >= cost * self.MARKUP_FLOOR

    # ---------- action selection -------------------------------------------

    def _next_action(self) -> ToolCall:
        # 1) Views first (view_listings carries only_others=False).
        if self.views_pending:
            tool = self.views_pending.pop(0)
            args = {"only_others": False} if tool == "view_listings" else {}
            return self._mk(tool, args)
        # 2) Accept profitable incoming offers, one per turn.
        if self.accepts_pending:
            return self._mk("accept_offer", {"offer_id": self.accepts_pending.pop(0)})
        # 3) Settle outstanding AP, one per turn.
        if self.pays_pending:
            return self._mk("pay_invoice", {"invoice_id": self.pays_pending.pop(0)})
        # 4) Role-specific production / procurement / pricing.
        call = self._next_role_action()
        if call is not None:
            return call
        # 5) Post any unlisted sellable inventory (top-up, not duplicate).
        call = self._post_unlisted()
        if call is not None:
            return call
        # 6) Nothing left — sleep until next event.
        return self._mk("wait_for_next_day", {})

    def _post_unlisted(self) -> ToolCall | None:
        """List the portion of a sellable item that isn't already on an open
        listing. Because we subtract what's already listed, a fully-listed
        inventory produces no new listing — which stops unbounded accumulation
        while still surfacing freshly produced / roasted stock."""
        for item_id in self.SELLABLE:
            if not self._one_shot(f"post_{item_id}"):
                continue
            held = self._held(item_id)
            listed = int(self.own_listed_qty.get(item_id, 0))
            delta = held - listed
            cost = float(self.cost_basis.get(item_id, 0.0))
            if delta >= self.MIN_LIST_CHUNK and cost > 0:
                ask = round(cost * self._list_markup(item_id), 2)
                return self._mk(
                    "post_listing",
                    {"item_id": item_id, "qty": delta, "asking_price": ask},
                )
        return None

    # ---------- helpers ----------------------------------------------------

    def _mk(self, tool_name: str, args: dict) -> ToolCall:
        self._call_seq += 1
        self._last_call = {"name": tool_name, "args": dict(args)}
        return ToolCall(id=f"scr_{self._call_seq:06d}", name=tool_name, input=args)

    def _held(self, item_id: str) -> int:
        return int(self.inventory.get(item_id, 0) or 0)

    def _one_shot(self, key: str) -> bool:
        if self.one_shots.get(key):
            return False
        self.one_shots[key] = True
        return True

    def _cheapest_market(self, item_id: str) -> dict | None:
        rows = [
            li
            for li in self.market_listings
            if li.get("item_id") == item_id and int(li.get("qty", 0) or 0) > 0
        ]
        if not rows:
            return None
        rows.sort(
            key=lambda li: (float(li.get("asking_price") or 1e9), str(li.get("id")))
        )
        return rows[0]

    def _bid(self, item_id, reorder, max_price, chunk) -> ToolCall | None:
        """Bid on the cheapest market listing of `item_id` if we're below the
        reorder level and it clears our max price. Deals close because the
        seller accepts any bid at/above its floor."""
        if self._held(item_id) >= reorder:
            return None
        listing = self._cheapest_market(item_id)
        if not listing:
            return None
        ask = float(listing.get("asking_price") or 0.0)
        if ask <= 0 or ask > max_price:
            return None
        qty = min(int(listing.get("qty", 0) or 0), chunk)
        if qty <= 0 or self.cash < qty * ask:
            return None
        return self._mk(
            "make_offer",
            {"listing_id": str(listing.get("id") or ""), "offered_price": ask, "qty": qty},
        )

    # ---------- subclass hooks ---------------------------------------------

    def _role_result(self, name: str, data: dict) -> None:
        """Parse role-specific tool results (default: nothing extra)."""

    def _next_role_action(self) -> ToolCall | None:
        """Next role action, or None when the day's role routine is done."""
        return None

    def _list_markup(self, item_id: str) -> float:
        return 1.5


# ---------------------------------------------------------------------------
# Farmer
# ---------------------------------------------------------------------------
class HeuristicFarmerModel(ScriptedModel):
    """Produce green coffee to target levels, list it cost-plus, accept
    roaster bids, and settle any payables."""

    ROLE = "farmer"
    SELLABLE = ("green_coffee_kg", "green_specialty_kg")
    DEFAULT_COST_BASIS = {"green_coffee_kg": 2.00, "green_specialty_kg": 10.00}
    TARGET_COMMODITY_KG = 45
    TARGET_SPECIALTY_KG = 18
    PRODUCE_COMMODITY_CAP = 30
    PRODUCE_SPECIALTY_CAP = 10
    LIST_MARKUP = {"green_coffee_kg": 1.5, "green_specialty_kg": 1.4}

    def __init__(self, model: str = "heuristic_farmer"):
        super().__init__(model)

    def _list_markup(self, item_id: str) -> float:
        return self.LIST_MARKUP.get(item_id, 1.5)

    def _next_role_action(self) -> ToolCall | None:
        if self._one_shot("produce_commodity"):
            qty = max(
                0,
                min(
                    self.TARGET_COMMODITY_KG - self._held("green_coffee_kg"),
                    self.PRODUCE_COMMODITY_CAP,
                ),
            )
            if qty > 0:
                return self._mk(
                    "produce_item", {"item_id": "green_coffee_kg", "quantity": qty}
                )
        if self._one_shot("produce_specialty"):
            qty = max(
                0,
                min(
                    self.TARGET_SPECIALTY_KG - self._held("green_specialty_kg"),
                    self.PRODUCE_SPECIALTY_CAP,
                ),
            )
            if qty > 0:
                return self._mk(
                    "produce_item", {"item_id": "green_specialty_kg", "quantity": qty}
                )
        return None


# ---------------------------------------------------------------------------
# Roaster
# ---------------------------------------------------------------------------
class HeuristicRoasterModel(ScriptedModel):
    """Buy green when low, roast it, list roasted cost-plus, accept retailer
    bids, and settle payables. A generalized, actively-procuring successor to
    the original single-file heuristic roaster."""

    ROLE = "roaster"
    SELLABLE = ("roasted_specialty_kg", "roasted_coffee_kg")
    DEFAULT_COST_BASIS = {
        "green_coffee_kg": 3.00,
        "green_specialty_kg": 14.00,
        "roasted_coffee_kg": 7.10,  # ($3 green + $3 labor) / 0.85
        "roasted_specialty_kg": 23.20,  # ($14 green + $5 labor) / 0.82
    }
    REORDER_COMMODITY_KG = 30
    REORDER_SPECIALTY_KG = 12
    MAX_BUY_COMMODITY = 6.0
    MAX_BUY_SPECIALTY = 16.0
    BUY_CHUNK_COMMODITY = 28
    BUY_CHUNK_SPECIALTY = 12
    ROAST_CAP_GREEN = 50
    LIST_MARKUP_VAL = 1.5

    def __init__(self, model: str = "heuristic_roaster"):
        super().__init__(model)

    def _list_markup(self, item_id: str) -> float:
        return self.LIST_MARKUP_VAL

    def _next_role_action(self) -> ToolCall | None:
        # 1) Procure green if low.
        if self._one_shot("buy_commodity"):
            call = self._bid(
                "green_coffee_kg", self.REORDER_COMMODITY_KG,
                self.MAX_BUY_COMMODITY, self.BUY_CHUNK_COMMODITY,
            )
            if call is not None:
                return call
        if self._one_shot("buy_specialty"):
            call = self._bid(
                "green_specialty_kg", self.REORDER_SPECIALTY_KG,
                self.MAX_BUY_SPECIALTY, self.BUY_CHUNK_SPECIALTY,
            )
            if call is not None:
                return call
        # 2) Roast specialty first (higher margin, smaller pool).
        if self._one_shot("roast_specialty"):
            qty = min(self._held("green_specialty_kg"), self.ROAST_CAP_GREEN)
            if qty > 0:
                return self._mk(
                    "roast", {"green_item_id": "green_specialty_kg", "qty_kg": qty}
                )
        # 3) Roast commodity within remaining shared cap.
        if self._one_shot("roast_commodity"):
            spec = min(self._held("green_specialty_kg"), self.ROAST_CAP_GREEN)
            qty = min(self._held("green_coffee_kg"), max(0, self.ROAST_CAP_GREEN - spec))
            if qty > 0:
                return self._mk(
                    "roast", {"green_item_id": "green_coffee_kg", "qty_kg": qty}
                )
        return None


# ---------------------------------------------------------------------------
# Retailer
# ---------------------------------------------------------------------------
class HeuristicRetailerModel(ScriptedModel):
    """Replenish roasted stock from roasters, price the storefront below
    reservation (with a small inventory-based nudge), and settle payables."""

    ROLE = "retailer"
    SELLABLE = ()  # retailers sell to consumers via set_retail_price, not B2B
    DEFAULT_COST_BASIS = {"roasted_coffee_kg": 11.00, "roasted_specialty_kg": 34.00}
    REORDER_COMMODITY_KG = 28
    REORDER_SPECIALTY_KG = 8
    MAX_WHOLESALE_COMMODITY = 16.0
    MAX_WHOLESALE_SPECIALTY = 42.0
    BUY_CHUNK_COMMODITY = 24
    BUY_CHUNK_SPECIALTY = 8
    BASE_PRICE_COMMODITY = 22.0  # reservation $30
    BASE_PRICE_SPECIALTY = 60.0  # reservation $80
    PRICE_STEP = 2.0

    def __init__(self, model: str = "heuristic_retailer"):
        super().__init__(model)

    def _daily_views(self) -> list[str]:
        return [
            "view_trial_balance",
            "view_listings",
            "view_offers",
            "view_payables",
            "view_consumer_sales",
        ]

    def _next_role_action(self) -> ToolCall | None:
        if self._one_shot("buy_commodity"):
            call = self._bid(
                "roasted_coffee_kg", self.REORDER_COMMODITY_KG,
                self.MAX_WHOLESALE_COMMODITY, self.BUY_CHUNK_COMMODITY,
            )
            if call is not None:
                return call
        if self._one_shot("buy_specialty"):
            call = self._bid(
                "roasted_specialty_kg", self.REORDER_SPECIALTY_KG,
                self.MAX_WHOLESALE_SPECIALTY, self.BUY_CHUNK_SPECIALTY,
            )
            if call is not None:
                return call
        if self._one_shot("price_commodity"):
            return self._price(
                "roasted_coffee_kg", self.BASE_PRICE_COMMODITY,
                self.REORDER_COMMODITY_KG,
            )
        if self._one_shot("price_specialty"):
            return self._price(
                "roasted_specialty_kg", self.BASE_PRICE_SPECIALTY,
                self.REORDER_SPECIALTY_KG,
            )
        return None

    def _price(self, item_id: str, base: float, reorder: int) -> ToolCall:
        held = self._held(item_id)
        price = base
        if held > int(1.5 * reorder):
            price = base - self.PRICE_STEP
        elif held < int(0.5 * reorder):
            price = base + self.PRICE_STEP
        return self._mk(
            "set_retail_price", {"item_id": item_id, "price_per_unit": round(price, 2)}
        )
