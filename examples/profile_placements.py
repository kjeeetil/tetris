"""Profile placement enumeration using :mod:`tetris.perf`.

Run with::

    PYTHONPATH=src python examples/profile_placements.py

Pass ``--help`` to see options for running multiple simulations and enabling
periodic performance logging summaries.
"""

from __future__ import annotations

import argparse
import logging

from tetris.perf import PerformanceTracker
from tetris.placement_env import PlacementEnv


LOGGER = logging.getLogger(__name__)


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


def _format_summary(summary: list[dict[str, float | int]], limit: int = 10) -> str:
    if not summary:
        return "No timings recorded."
    parts: list[str] = []
    for row in summary[:limit]:
        total_ms = row["total"] * 1000.0
        self_ms = row["self"] * 1000.0
        avg_ms = row["average"] * 1000.0
        parts.append(
            (
                f"{row['name']}: total={total_ms:.3f}ms, self={self_ms:.3f}ms, "
                f"count={int(row['count'])}, avg={avg_ms:.3f}ms"
            )
        )
    return "; ".join(parts)


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


def log_summary(tracker: PerformanceTracker, *, limit: int, index: int) -> list[dict[str, float | int]]:
    summary = tracker.summary(sort_by="total")
    limit = max(0, limit)
    limited_summary = summary[:limit] if limit else []
    message = _format_summary(limited_summary, limit=limit)
    LOGGER.info("Simulation %d performance: %s", index, message)
    return limited_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per simulation.")
    parser.add_argument("--simulations", type=int, default=1, help="How many simulations to run.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Emit a performance summary every N simulations (0 disables periodic logging).",
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=10,
        help="Maximum number of sections to include in summaries.",
    )
    parser.add_argument(
        "--no-table",
        dest="print_table",
        action="store_false",
        help="Skip printing the final tabular summary (logging only).",
    )
    parser.set_defaults(print_table=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")

    tracker = PerformanceTracker()
    last_summary: list[dict[str, float | int]] = []
    for sim_idx in range(1, args.simulations + 1):
        run_env(steps=args.steps, tracker=tracker)
        should_log = False
        if args.log_interval > 0 and sim_idx % args.log_interval == 0:
            should_log = True
        elif sim_idx == args.simulations:
            should_log = True
        if should_log:
            last_summary = log_summary(tracker, limit=args.summary_limit, index=sim_idx)
            if sim_idx != args.simulations:
                tracker.reset()

    if args.print_table and last_summary:
        print_summary(tracker, limit=args.summary_limit)


if __name__ == "__main__":
    main()
