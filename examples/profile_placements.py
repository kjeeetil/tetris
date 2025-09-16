"""Profile placement enumeration in :mod:`tetris.placement_env`.

Run with::

    PYTHONPATH=src python examples/profile_placements.py

The script executes a small number of environment steps and prints the top
entries from the cumulative-time profile so it is easy to confirm which helper
functions dominate the runtime.
"""

from __future__ import annotations

import cProfile
import pstats

from tetris.placement_env import PlacementEnv


def run_env(steps: int = 500) -> None:
    env = PlacementEnv()
    obs, info = env.reset(seed=42)
    for _ in range(steps):
        actions = info["action_list"]
        if not actions:
            break
        obs, _, done, info = env.step(0)
        if done:
            obs, info = env.reset(seed=43)


def main() -> None:
    prof = cProfile.Profile()
    prof.enable()
    run_env()
    prof.disable()
    stats = pstats.Stats(prof)
    stats.sort_stats("cumtime")
    stats.print_stats(10)


if __name__ == "__main__":
    main()
