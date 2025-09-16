import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from examples.profile_placements import log_summary
from tetris.perf import PerformanceTracker


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def advance(self, delta: float) -> None:
        self.current += delta

    def __call__(self) -> float:
        return self.current


def test_log_summary_limits_rows_and_output(caplog):
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)
    with tracker.section("slow"):
        clock.advance(0.5)
    with tracker.section("fast"):
        clock.advance(0.1)

    with caplog.at_level(logging.INFO, logger="examples.profile_placements"):
        summary = log_summary(tracker, limit=1, index=7)

    assert len(summary) == 1
    assert summary[0]["name"] == "slow"
    message = "".join(caplog.messages)
    assert "slow" in message
    assert "fast" not in message
