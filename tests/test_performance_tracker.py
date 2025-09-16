import pytest

from tetris.perf import PerformanceTracker


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def advance(self, delta: float) -> None:
        self.current += delta

    def __call__(self) -> float:
        return self.current


def test_tracker_records_basic_stats():
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)
    with tracker.section("outer"):
        clock.advance(0.5)
    summary = tracker.summary()
    assert len(summary) == 1
    row = summary[0]
    assert row["name"] == "outer"
    assert row["count"] == 1
    assert row["total"] == pytest.approx(0.5)
    assert row["self"] == pytest.approx(0.5)
    assert row["average"] == pytest.approx(0.5)
    assert row["min"] == pytest.approx(0.5)
    assert row["max"] == pytest.approx(0.5)


def test_tracker_nested_sections_compute_exclusive_time():
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)
    with tracker.section("outer"):
        clock.advance(0.5)
        with tracker.section("inner"):
            clock.advance(0.2)
        clock.advance(0.3)
    summary = tracker.summary(sort_by="total")
    stats = {row["name"]: row for row in summary}
    assert stats["outer"]["total"] == pytest.approx(1.0)
    assert stats["outer"]["self"] == pytest.approx(0.8)
    assert stats["inner"]["total"] == pytest.approx(0.2)
    assert stats["inner"]["self"] == pytest.approx(0.2)


def test_summary_sorting_and_errors():
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)
    with tracker.section("a"):
        clock.advance(0.1)
    with tracker.section("b"):
        clock.advance(0.3)
    summary = tracker.summary(sort_by="total")
    assert [row["name"] for row in summary] == ["b", "a"]
    summary = tracker.summary(sort_by="self")
    assert summary[0]["name"] == "b"
    with pytest.raises(ValueError):
        tracker.summary(sort_by="unknown")


def test_disable_enable_and_reset():
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)
    tracker.disable()
    with tracker.section("ignored"):
        clock.advance(0.4)
    assert tracker.summary() == []
    tracker.enable()
    with tracker.section("active"):
        clock.advance(0.2)
    assert tracker.summary()[0]["name"] == "active"
    tracker.reset()
    assert tracker.summary() == []


def test_time_function_helper_records_duration():
    clock = FakeClock()
    tracker = PerformanceTracker(clock=clock)

    def slow_call() -> str:
        clock.advance(0.25)
        return "ok"

    result = tracker.time_function("call", slow_call)
    assert result == "ok"
    summary = tracker.summary()
    assert summary[0]["total"] == pytest.approx(0.25)
    assert summary[0]["name"] == "call"
