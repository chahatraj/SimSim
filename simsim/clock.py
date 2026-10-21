"""Simulated time: map steps to clock times.

    clock = Clock(start="2024-03-04 08:00", step="30min")
    Simulation(env, agents, clock=clock)    # agents see "# Current time", memories get timestamps
    clock.time(t)                           # datetime for step t
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_UNITS = {"s": "seconds", "sec": "seconds", "min": "minutes", "m": "minutes", "h": "hours", "hour": "hours",
          "d": "days", "day": "days", "w": "weeks", "week": "weeks"}


def parse_step(step: str | timedelta) -> timedelta:
    if isinstance(step, timedelta):
        return step
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+?)s?\s*", step.lower())
    if not m or m.group(2) not in _UNITS:
        raise ValueError(f"Cannot parse step {step!r}; use e.g. '15min', '1h', '1 day'")
    return timedelta(**{_UNITS[m.group(2)]: float(m.group(1))})


class Clock:
    def __init__(self, start: str | datetime = "2024-01-01 08:00", step: str | timedelta = "1h",
                 fmt: str = "%A %d %B %Y, %H:%M"):
        self.start = datetime.fromisoformat(start) if isinstance(start, str) else start
        self.step = parse_step(step)
        self.fmt = fmt

    def time(self, t: int) -> datetime:
        return self.start + t * self.step

    def __call__(self, t: int) -> str:
        return self.time(t).strftime(self.fmt)
