"""Profile placement enumeration using :mod:`tetris.perf`.

Run with::

    PYTHONPATH=src python examples/profile_placements.py
"""

from __future__ import annotations

from tetris.perf import PerformanceTracker
from tetris.placement_env import PlacementEnv


def run_env(steps: int, tracker: PerformanceTracker) -> None:
    env = PlacementEnv(profiler=tracker)
    obs, info = env.reset(seed=42)
    for _ in range(steps):
        actions = info["action_list"]
        if not actions:
            break
        obs, _, done, info = env.step(0)
        if done:
            obs, info = env.reset(seed=43)


def print_summary(tracker: PerformanceTracker, limit: int = 10) -> None:
    summary = tracker.summary(sort_by="total")
    if not summary:
        print("No timings recorded.")
        return
    width = max(len(row["name"]) for row in summary[:limit])
    header = f"{'Section':<{width}}  Total (ms)  Self (ms)  Count  Avg (ms)"
    print(header)
    print("-" * len(header))
    for row in summary[:limit]:
        total_ms = row["total"] * 1000.0
        self_ms = row["self"] * 1000.0
        avg_ms = row["average"] * 1000.0
        print(
            f"{row['name']:<{width}}  {total_ms:10.3f}  {self_ms:8.3f}"
            f"  {int(row['count']):5d}  {avg_ms:8.3f}"
        )


def main() -> None:
    tracker = PerformanceTracker()
    run_env(steps=500, tracker=tracker)
    print_summary(tracker)


if __name__ == "__main__":
    main()
