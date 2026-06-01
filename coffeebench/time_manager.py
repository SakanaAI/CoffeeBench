"""Sim-time authority. Tracks a single `virtual_min` integer (minutes
since sim start) which is the source of truth for both the integer day
and the intra-day timestamp used by the event-driven scheduler.
"""

from coffeebench.event_loop import MINUTES_PER_DAY


class TimeManager:
    def __init__(self) -> None:
        self.virtual_min: int = 0

    # -- legacy day-level API ---------------------------------------------
    def get_current_day(self) -> int:
        return self.virtual_min // MINUTES_PER_DAY

    def advance_day(self, days: int = 1) -> None:
        target_day = self.get_current_day() + int(days)
        self.virtual_min = target_day * MINUTES_PER_DAY

    def reset(self) -> None:
        self.virtual_min = 0

    # -- minute-level API -------------------------------------------------
    def get_virtual_min(self) -> int:
        return self.virtual_min

    def advance_minutes(self, minutes: int) -> None:
        if minutes < 0:
            raise ValueError(f"minutes must be non-negative, got {minutes}")
        self.virtual_min += int(minutes)
