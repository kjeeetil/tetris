"""Light-weight performance measurement helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, List, Optional


@dataclass
class PerfStat:
    """Aggregated timing information for a single label."""

    count: int = 0
    total: float = 0.0
    self_time: float = 0.0
    min_time: Optional[float] = None
    max_time: float = 0.0

    def add(self, total: float, exclusive: float) -> None:
        """Update the aggregates with a new timing sample."""

        self.count += 1
        self.total += total
        self.self_time += exclusive
        if self.min_time is None or total < self.min_time:
            self.min_time = total
        if total > self.max_time:
            self.max_time = total

    @property
    def average(self) -> float:
        """Return the average inclusive time in seconds."""

        return self.total / self.count if self.count else 0.0

    @property
    def self_average(self) -> float:
        """Return the average exclusive time in seconds."""

        return self.self_time / self.count if self.count else 0.0


@dataclass
class _ActiveTimer:
    name: str
    start: float
    children: float = 0.0


class _PerfTimer:
    """Context manager that records a section's runtime."""

    __slots__ = ("_tracker", "_token", "_name")

    def __init__(self, tracker: "PerformanceTracker", name: str) -> None:
        self._tracker = tracker
        self._token: Optional[_ActiveTimer] = None
        self._name = name

    def __enter__(self) -> "_PerfTimer":
        self._token = self._tracker._start(self._name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tracker._stop(self._token)
        self._token = None
        return False


class PerformanceTracker:
    """Collect execution time statistics for labelled code sections."""

    def __init__(
        self,
        *,
        clock: Optional[Callable[[], float]] = None,
        enabled: bool = True,
    ) -> None:
        self._clock = clock or time.perf_counter
        self.enabled = enabled
        self._stats: Dict[str, PerfStat] = {}
        self._stack: List[_ActiveTimer] = []

    def enable(self) -> None:
        """Enable timing collection."""

        self.enabled = True

    def disable(self) -> None:
        """Disable timing collection."""

        self.enabled = False

    def reset(self) -> None:
        """Clear accumulated statistics and active timers."""

        self._stats.clear()
        self._stack.clear()

    # Internal helpers -------------------------------------------------
    def _start(self, name: str) -> Optional[_ActiveTimer]:
        if not self.enabled:
            return None
        token = _ActiveTimer(name=name, start=self._clock())
        self._stack.append(token)
        return token

    def _stop(self, token: Optional[_ActiveTimer]) -> float:
        if not self.enabled or token is None:
            return 0.0
        end = self._clock()
        if not self._stack or self._stack[-1] is not token:
            raise RuntimeError("Timer stack out of sync")
        self._stack.pop()
        elapsed = end - token.start
        child = token.children
        exclusive = elapsed - child
        if exclusive < 0:
            exclusive = 0.0
        stat = self._stats.get(token.name)
        if stat is None:
            stat = PerfStat()
            self._stats[token.name] = stat
        stat.add(elapsed, exclusive)
        if self._stack:
            self._stack[-1].children += elapsed
        return elapsed

    # Public API -------------------------------------------------------
    def section(self, name: str) -> _PerfTimer:
        """Return a context manager tracking ``name``'s runtime."""

        return _PerfTimer(self, name)

    def snapshot(self) -> Dict[str, PerfStat]:
        """Return a copy of the accumulated statistics."""

        return {name: replace(stat) for name, stat in self._stats.items()}

    def iter_stats(self) -> Iterator[tuple[str, PerfStat]]:
        """Yield ``(name, PerfStat)`` pairs for all recorded sections."""

        for name, stat in self._stats.items():
            yield name, stat

    def summary(
        self, *, sort_by: str = "total", descending: bool = True
    ) -> List[Dict[str, float | int]]:
        """Return a sorted summary of the collected statistics."""

        key_map = {
            "total": lambda item: item[1].total,
            "self": lambda item: item[1].self_time,
            "count": lambda item: item[1].count,
            "average": lambda item: item[1].average,
            "self_average": lambda item: item[1].self_average,
            "max": lambda item: item[1].max_time,
            "min": lambda item: item[1].min_time if item[1].min_time is not None else 0.0,
        }
        if sort_by not in key_map:
            raise ValueError(f"Unknown sort key: {sort_by}")
        items = sorted(self._stats.items(), key=key_map[sort_by], reverse=descending)
        summary: List[Dict[str, float | int]] = []
        for name, stat in items:
            summary.append(
                {
                    "name": name,
                    "count": stat.count,
                    "total": stat.total,
                    "self": stat.self_time,
                    "average": stat.average,
                    "self_average": stat.self_average,
                    "min": stat.min_time if stat.min_time is not None else 0.0,
                    "max": stat.max_time,
                }
            )
        return summary

    def time_function(self, name: str, func: Callable[..., object], *args, **kwargs):
        """Execute ``func`` inside a named section and return its result."""

        with self.section(name):
            return func(*args, **kwargs)


__all__ = ["PerfStat", "PerformanceTracker"]
