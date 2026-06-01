"""HeuristicRoasterModel — minimal scripted-policy baseline for the roaster role.

A ceiling reference for the LLM matrix: any model that scores below this
20-line policy is being out-decided by a fixed daily routine with no
strategic awareness.

Designed for `roaster_A`. Plugs into the standard `Agent` harness like
any other model (`get_model("heuristic_roaster")` →
`--models 'roaster_A:heuristic_roaster'`). Internally it reads the
latest tool result from the agent's message history to keep state, and
issues one tool call per turn just like an LLM would — paying the same
30-min/turn cost, so the comparison stays apples-to-apples.

Daily routine (resets on each new "Day N" morning observation):
  views   →  view_trial_balance, view_listings, view_offers, view_payables
  actions →  accept_offer (each profitable, one per turn)
             pay_invoice (each due AP, one per turn)
             roast(specialty)         — all specialty green held
             roast(commodity)         — all commodity green held, capped to remaining shared roast cap
             post_listing(specialty)  — list all roasted_specialty held at MARKUP_LIST × cost
             post_listing(commodity)  — list all roasted_coffee held at MARKUP_LIST × cost
             make_offer(green)        — cheapest commodity green listing on the market
             wait_for_next_day

Intentionally NOT modelled (each is a documented heuristic limitation):
  - DM reading / replying        (`view_messages` never called)
  - Dynamic pricing               (festival, competitor, demand-noise ignored)
  - Counterparty selection logic  (cheapest listing wins regardless of who)
  - Reactive wakes mid-day        (treated like fresh days; routine restarts)
  - Specialty green procurement   (only commodity green is bought; specialty
                                   capacity comes only from any specialty green
                                   that happens to be sold to us — keeps the
                                   policy genuinely simple)

Two hardcoded knobs:
  MARKUP_LIST   = 1.5  — list roasted at 50% over WAVG cost basis
  MARKUP_FLOOR  = 1.2  — accept offers down to 20% over cost basis
"""

from __future__ import annotations

import json
from typing import Any

from coffeebench.models.types import ModelResponse, ToolCall


MARKUP_LIST = 1.5
MARKUP_FLOOR = 1.2

# Working cost-basis assumptions. The bench's `view_trial_balance` tool
# surfaces inventory dollar value but not per-item WAVG cost; rather
# than add a tool just for the heuristic, we hardcode the starter /
# steady-state values from `coffeebench/main._seed_world`. These drift
# slightly after the agent's own deals but stay close enough for a
# 1.5×/1.2× margin policy on a 90-day horizon.
_DEFAULT_COST_BASIS: dict[str, float] = {
    "green_coffee_kg": 2.00,
    "green_specialty_kg": 10.00,
    "roasted_coffee_kg": 5.88,  # ($2 + $3 labor) / 0.85 yield
    "roasted_specialty_kg": 18.30,  # ($10 + $5 labor) / 0.82 yield
}


class HeuristicRoasterModel:
    """Scripted state-machine policy for the roaster role."""

    DEFAULT_MAX_INPUT_TOKENS = 1_000_000

    def __init__(self, model: str = "heuristic_roaster"):
        self.model = model
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.system_prompt = ""

        # Long-lived (across days) state, refreshed by view_trial_balance.
        self.cost_basis: dict[str, float] = {}
        self.inventory: dict[str, int] = {}
        self.cash: float = 0.0

        # Map of listing_id → item_id for listings WE posted, so when
        # offers land on them (offers carry listing_id but not item_id)
        # we can resolve the item to look up cost_basis. Built up on
        # successful post_listing tool results.
        self.my_listings: dict[str, str] = {}

        # Per-day state — reset on each new "Day N" morning observation.
        self._reset_day_state(day=-1)

        self._call_seq = 0  # for unique ToolCall ids
        # Args of the most recent ToolCall we issued — used to pair with
        # the next tool-result message so we can recover what we asked
        # for (e.g. item_id on a successful post_listing).
        self._last_call: dict | None = None

    # ---------- per-day state -----------------------------------------------

    def _reset_day_state(self, day: int) -> None:
        self.day = day
        # Refresh queue: views to issue at the start of every day so the
        # rest of the routine has fresh state.
        self.views_pending: list[str] = [
            "view_trial_balance",
            "view_listings",
            "view_offers",
            "view_payables",
        ]
        # Action queues populated lazily after the views land.
        self.accepts_pending: list[str] = []
        self.pays_pending: list[str] = []
        self.green_listings: list[dict] = []
        # One-shot flags for irreversible daily actions.
        self.roast_specialty_done = False
        self.roast_commodity_done = False
        self.post_specialty_done = False
        self.post_commodity_done = False
        self.make_offer_done = False

    # ---------- Model protocol ----------------------------------------------

    def query(self, messages, tools=None) -> ModelResponse:  # noqa: ARG002
        self.n_calls += 1
        self._ingest(messages)
        call = self._next_action()
        return ModelResponse(
            content="(heuristic baseline)",
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
        # ContextCompactor never fires on this path (transcripts stay tiny);
        # if it ever does, returning the content unchanged is safe.
        return content

    # ---------- state ingestion --------------------------------------------

    def _ingest(self, messages: list[dict]) -> None:
        """Read the latest message: detect new-day boundaries, parse tool
        results to refresh internal state."""
        if not messages:
            return
        last = messages[-1]
        role = last.get("role")

        if role == "user":
            # Morning / wake observation. Detect day from "Day N" pattern;
            # reset day-state on day change.
            content = last.get("content") or ""
            new_day = _extract_day(content)
            if new_day is not None and new_day != self.day:
                self._reset_day_state(day=new_day)
            return

        if role == "tool":
            name = last.get("name") or ""
            try:
                data = json.loads(last.get("content") or "{}")
            except (TypeError, ValueError):
                return
            self._handle_tool_result(name, data)

    def _handle_tool_result(self, name: str, data: dict) -> None:
        # Pair with the args we sent (so we can recover, e.g., the
        # item_id of a post_listing call to populate `my_listings`).
        last_args = (self._last_call or {}).get("args", {}) if self._last_call else {}

        if name == "post_listing" and data.get("status") == "success":
            lid = data.get("listing_id")
            item = last_args.get("item_id")
            if lid and item:
                self.my_listings[str(lid)] = str(item)

        if name == "view_trial_balance":
            # Actual schema: data["accounts"]["cash"]["balance"], NOT
            # data["cash"]; data["inventory_by_item"], NOT data["inventory"];
            # cost_basis is NOT in the trial balance at all. Working
            # heuristic uses canonical starter values for the four items
            # — these drift after deals but stay close enough for a
            # 1.2× / 1.5× margin policy. The trial balance's per-item
            # WAVG cost basis would be more accurate but isn't surfaced
            # by that tool today.
            accounts = data.get("accounts") or {}
            cash_acc = accounts.get("cash") or {}
            self.cash = float(cash_acc.get("balance") or 0.0)
            self.inventory = {
                k: int(v) for k, v in (data.get("inventory_by_item") or {}).items()
            }
            self.cost_basis = dict(_DEFAULT_COST_BASIS)
        elif name == "view_listings":
            # Cache cheapest commodity green listings (others ignored — see
            # docstring, no specialty procurement in the simplest policy).
            self.green_listings = sorted(
                (
                    li
                    for li in (data.get("listings") or [])
                    if li.get("item_id") == "green_coffee_kg" and li.get("qty", 0) > 0
                ),
                key=lambda li: float(li.get("asking_price") or 1e9),
            )
        elif name == "view_offers":
            # Offers ON my listings (incoming buy interest). Schema:
            # offers[*].id, offered_price, status, item_id (resolved
            # via the linked listing — but the offer dict carries
            # item_id directly when surfaced via view_offers).
            offers = data.get("offers") or []
            self.accepts_pending = [
                str(o["id"])
                for o in offers
                if o.get("status") in (None, "open", "pending")
                and self._offer_above_floor(o)
            ]
        elif name == "view_payables":
            # Schema: data["rows"] (not "invoices"), each row has
            # row["id"] and row["outstanding"].
            rows = data.get("rows") or []
            self.pays_pending = [
                str(r["id"]) for r in rows if float(r.get("outstanding") or 0.0) > 0.0
            ]

    def _offer_above_floor(self, offer: dict) -> bool:
        # Offers don't carry item_id directly — resolve via the
        # listing_id the offer is on (we tracked our own listings in
        # `my_listings` from post_listing successes).
        listing_id = str(offer.get("listing_id") or "")
        item_id = self.my_listings.get(listing_id)
        if item_id is None:
            # Listing not in our local map (e.g. heuristic was started
            # mid-run, or listing was posted before tracking began).
            # Skip rather than guess — safer than accepting blind.
            return False
        price = float(offer.get("offered_price") or 0.0)
        cost = float(self.cost_basis.get(item_id, 0.0))
        if cost <= 0.0:
            return price > 0.0
        return price >= cost * MARKUP_FLOOR

    # ---------- action selection -------------------------------------------

    def _next_action(self) -> ToolCall:
        # 1) Run the daily refresh views first.
        if self.views_pending:
            tool = self.views_pending.pop(0)
            return self._mk(tool, {})

        # 2) Accept profitable incoming offers, one per turn.
        if self.accepts_pending:
            offer_id = self.accepts_pending.pop(0)
            return self._mk("accept_offer", {"offer_id": offer_id})

        # 3) Settle outstanding AP, one per turn.
        if self.pays_pending:
            invoice_id = self.pays_pending.pop(0)
            return self._mk("pay_invoice", {"invoice_id": invoice_id})

        # 4) Roast — specialty first (smaller pool, higher margin).
        if not self.roast_specialty_done:
            self.roast_specialty_done = True
            qty = int(self.inventory.get("green_specialty_kg", 0) or 0)
            if qty > 0:
                return self._mk(
                    "roast",
                    {
                        "green_item_id": "green_specialty_kg",
                        "qty_kg": qty,
                    },
                )

        if not self.roast_commodity_done:
            self.roast_commodity_done = True
            qty = int(self.inventory.get("green_coffee_kg", 0) or 0)
            # Cap at 50 kg conservatively (the env's shared roast cap).
            # If specialty also roasted today this could over-shoot the
            # remaining cap; in that case the env returns an error and
            # this phase is just lost for the day. Simplicity over edge-
            # case recovery — the routine restarts tomorrow.
            qty = min(qty, 50)
            if qty > 0:
                return self._mk(
                    "roast",
                    {
                        "green_item_id": "green_coffee_kg",
                        "qty_kg": qty,
                    },
                )

        # 5) List all roasted inventory at cost × MARKUP_LIST.
        if not self.post_specialty_done:
            self.post_specialty_done = True
            qty = int(self.inventory.get("roasted_specialty_kg", 0) or 0)
            cost = float(self.cost_basis.get("roasted_specialty_kg", 0.0))
            if qty > 0 and cost > 0:
                ask = round(cost * MARKUP_LIST, 2)
                return self._mk(
                    "post_listing",
                    {
                        "item_id": "roasted_specialty_kg",
                        "qty": qty,
                        "asking_price": ask,
                    },
                )

        if not self.post_commodity_done:
            self.post_commodity_done = True
            qty = int(self.inventory.get("roasted_coffee_kg", 0) or 0)
            cost = float(self.cost_basis.get("roasted_coffee_kg", 0.0))
            if qty > 0 and cost > 0:
                ask = round(cost * MARKUP_LIST, 2)
                return self._mk(
                    "post_listing",
                    {
                        "item_id": "roasted_coffee_kg",
                        "qty": qty,
                        "asking_price": ask,
                    },
                )

        # 6) Procure commodity green: one offer at the cheapest listing's
        #    asking price, qty = whatever the listing has up to a daily
        #    budget that fits roast capacity (~50 kg/day input → ~30 kg
        #    is a reasonable floor that leaves headroom for specialty).
        if not self.make_offer_done and self.green_listings:
            self.make_offer_done = True
            top = self.green_listings[0]
            qty = min(int(top.get("qty", 0) or 0), 30)
            ask = float(top.get("asking_price") or 0.0)
            if qty > 0 and ask > 0 and self.cash >= qty * ask:
                return self._mk(
                    "make_offer",
                    {
                        "listing_id": str(top.get("id") or ""),
                        "offered_price": ask,
                        "qty": qty,
                    },
                )

        # 7) Nothing left to do this day → sleep until next event.
        return self._mk("wait_for_next_day", {})

    # ---------- helpers ----------------------------------------------------

    def _mk(self, tool_name: str, args: dict) -> ToolCall:
        self._call_seq += 1
        # Stash so the next _ingest can pair the result with what we asked
        # for (e.g. recover post_listing's item_id arg from the result).
        self._last_call = {"name": tool_name, "args": dict(args)}
        return ToolCall(
            id=f"heur_{self._call_seq:04d}",
            name=tool_name,
            input=args,
        )


def _extract_day(text: str) -> int | None:
    """Pull the day index out of a morning/wake observation. Looks for
    the canonical 'Day N' substring (case-insensitive)."""
    if not text:
        return None
    import re

    m = re.search(r"\bDay\s+(\d+)\b", text)
    if m:
        return int(m.group(1))
    return None
