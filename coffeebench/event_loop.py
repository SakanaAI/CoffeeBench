"""Event-driven scheduler for CoffeeBench."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

# 1 day = 1440 virtual minutes. Tool costs and event scheduling are all
# expressed in integer minutes; `virtual_min` is a single int counter.
MINUTES_PER_DAY: int = 24 * 60


# ---------------------------------------------------------------------------
# Tool time costs (virtual minutes per call)
# ---------------------------------------------------------------------------
# Default cost for any tool not in this table is `DEFAULT_TOOL_COST_MIN`.
# Costs are coarse — they exist to give events a temporal order an LLM
# can reason about (morning vs afternoon, before vs after a DM), not to
# model real-world labor. Edit centrally so prompt + scheduling stay in
# sync.
TOOL_TIME_COST_MIN: dict[str, int] = {
    # Uniform 30-min cost for every tool. Keeping all costs equal
    # means every post-bucket reschedule lands on the same vt as
    # every other agent's, so the dispatcher's bucket-parallel pop
    # continues to collect the full set of live agents instead of
    # fragmenting into 1-2 agent buckets when costs were mixed.
    "view_listings": 30,
    "view_offers": 30,
    "view_deals": 30,
    "view_messages": 30,
    "view_payables": 30,
    "view_receivables": 30,
    "view_trial_balance": 30,
    "view_consumer_sales": 30,
    "read_message": 30,
    "post_listing": 30,
    "make_offer": 30,
    "accept_offer": 30,
    "withdraw_offer": 30,
    "send_message": 30,
    "produce_item": 30,
    "roast": 30,
    "set_retail_price": 30,
    "pay_invoice": 30,
    "return_shipment": 30,
    "view_market_aggregate": 30,
    # Skip — handled by env (jumps available_at to next day's open),
    # not by per-tool cost. Listed so the lookup is exhaustive.
    "wait_for_next_day": 0,
}

DEFAULT_TOOL_COST_MIN: int = 30


def tool_cost_minutes(tool_name: str) -> int:
    return TOOL_TIME_COST_MIN.get(tool_name, DEFAULT_TOOL_COST_MIN)


# ---------------------------------------------------------------------------
# (virtual_min: int) ↔ (day, hour, minute) helpers
# ---------------------------------------------------------------------------
def min_to_day_hour_minute(m: int) -> tuple[int, int, int]:
    """Convert a minute counter to (day, hour, minute).

    `day` is the 0-indexed sim day. `hour` ∈ [0, 23], `minute` ∈ [0, 59].
    """
    day, rem = divmod(int(m), MINUTES_PER_DAY)
    hour, minute = divmod(rem, 60)
    return day, hour, minute


def format_min(m: int) -> str:
    """Render `m` (minutes since sim start) as `Day N, HH:MM` for
    agent-visible observations."""
    day, hour, minute = min_to_day_hour_minute(m)
    return f"Day {day}, {hour:02d}:{minute:02d}"


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------
# Concrete event kinds the env's dispatcher knows how to handle.
# Centralizing them as named constants (rather than ad-hoc strings)
# lets type-checkers and grep find every producer / consumer.
#
# Payload contract for each kind (read by handlers in `Environment`):
#
#   AGENT_WAKE
#     target_agent: str            — who is being woken
#     payload.reason: str          — "morning_heartbeat" | "deal_accepted" |
#                                    "delivery_arrived" | "production_ready" |
#                                    "partner_dm" | etc.
#     Stale-check: if `_available_at[target_agent] != self.virtual_min`,
#                  this event was superseded — skip it.
#
#   MORNING_OPEN
#     payload.day: int             — start-of-day mechanics for `day`
#                                    (charge opex/wages, materialize pending
#                                    production, process deliveries, inject
#                                    partner_dm, schedule per-agent wakes).
#
#   EOD_MECHANICS
#     payload.day: int             — end-of-day mechanics for `day`
#                                    (consumer_sales, spoilage, late-fee,
#                                    settle_due, day_end emit).
#                                    After firing, schedules next day's
#                                    MORNING_OPEN + EOD_MECHANICS unless
#                                    `day+1 >= max_days`.
#
EVENT_AGENT_WAKE: str = "agent_wake"
EVENT_MORNING_OPEN: str = "morning_open"
EVENT_EOD_MECHANICS: str = "eod_mechanics"

# Reasons attached to AGENT_WAKE events (str — used as `payload.reason`).
WAKE_MORNING: str = "morning_heartbeat"
WAKE_DEAL_ACCEPTED: str = "deal_accepted"
WAKE_DELIVERY_ARRIVED: str = "delivery_arrived"
WAKE_PRODUCTION_READY: str = "production_ready"
WAKE_DM_RECEIVED: str = "dm_received"
WAKE_OFFER_RECEIVED: str = "offer_received"
WAKE_INVOICE_PAID: str = "invoice_paid"
WAKE_RETURN_RECEIVED: str = "return_received"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """A scheduled simulation event.

    Compared by (virtual_min, _counter) only — `_counter` is set by
    EventLoop on push to break ties deterministically (insertion order).
    All other fields are payload and never participate in ordering.

    Use the `EVENT_*` constants for `kind` and the `WAKE_*` constants
    for `payload['reason']` when scheduling AGENT_WAKE events. See the
    "Event kinds" section above for the per-kind payload schema.
    """

    virtual_min: int
    kind: str
    target_agent: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EventLoop
# ---------------------------------------------------------------------------
class EventLoop:
    """Min-heap-backed scheduler.

    The heap stores ``(virtual_min, counter, event)`` tuples; the
    counter is incremented on every push so events scheduled at the
    same virtual_min pop in insertion order. `pop()` advances
    `current_min` to the popped event's virtual_min before returning it
    — handlers can inspect `current_min` without computing it from the
    event they were given.
    """

    def __init__(self, start_min: int = 0):
        self._heap: list[tuple[int, int, Event]] = []
        self._counter: int = 0
        self.current_min: int = int(start_min)

    # -- scheduling --------------------------------------------------------

    def schedule(self, event: Event) -> None:
        """Push an event onto the queue.

        Events scheduled in the past (`virtual_min < current_min`) are
        clamped UP to `current_min` so they fire on the next tick.
        Without this, a handler that schedules a successor with `delay=0`
        could push it before `current_min` if the handler itself was
        run after the heap pop advanced the clock — leading to events
        firing in reverse causal order.
        """
        if event.virtual_min < self.current_min:
            event.virtual_min = self.current_min
        heapq.heappush(self._heap, (event.virtual_min, self._counter, event))
        self._counter += 1

    def schedule_at(
        self,
        virtual_min: int,
        kind: str,
        *,
        target_agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        ev = Event(
            virtual_min=int(virtual_min),
            kind=kind,
            target_agent=target_agent,
            payload=dict(payload or {}),
        )
        self.schedule(ev)
        return ev

    # -- consumption -------------------------------------------------------

    def pop(self) -> Event | None:
        """Pop the next event, advancing `current_min`. None when empty."""
        if not self._heap:
            return None
        m, _, ev = heapq.heappop(self._heap)
        if m > self.current_min:
            self.current_min = m
        return ev

    def pop_bucket(self) -> list[Event]:
        """Pop ALL events at the smallest `virtual_min` together. Used
        by the async dispatcher to gather concurrently-eligible
        AGENT_WAKE events into one parallel batch. Within the bucket
        events keep their insertion-counter order (so handlers can
        still rely on a deterministic intra-bucket sequence). Empty
        list when the heap is exhausted."""
        if not self._heap:
            return []
        first_min = self._heap[0][0]
        if first_min > self.current_min:
            self.current_min = first_min
        bucket: list[tuple[int, Event]] = []
        while self._heap and self._heap[0][0] == first_min:
            _, counter, ev = heapq.heappop(self._heap)
            bucket.append((counter, ev))
        bucket.sort(key=lambda x: x[0])
        return [ev for _, ev in bucket]

    def peek_time(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    # -- time advancement --------------------------------------------------

    def advance_minutes(self, minutes: int) -> int:
        """Move `current_min` forward by `minutes` (no event firing).

        Used by the env to charge per-tool time costs WITHOUT firing
        events scheduled in the interim. (Event firing is the explicit
        responsibility of the caller via `pop()`.)

        Returns the new `current_min`.
        """
        if minutes < 0:
            raise ValueError(f"minutes must be non-negative, got {minutes}")
        self.current_min += int(minutes)
        return self.current_min

    def get_current_day(self) -> int:
        """Integer day-of-sim. Compatible drop-in for `TimeManager.get_current_day`."""
        return self.current_min // MINUTES_PER_DAY
