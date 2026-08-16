"""ForecastCache TTL behaviour, time-weighted MOER, and the process-wide singleton."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from carbonsight_core.carbon import build_synthetic_forecast, synthetic_moer_for_region
from carbonsight_core.config import DEFAULT_FORECAST_CACHE_TTL_SECONDS, Config
from carbonsight_core.watttime.cache import (
    ForecastCache,
    get_global_forecast_cache,
    mixture_weighted_moer,
    reset_global_forecast_cache,
    time_weighted_moer,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _points(start: datetime, values: list[float], step_minutes: int = 60) -> list[dict]:
    return [
        {
            "point_time": (start + timedelta(minutes=step_minutes * index))
            .isoformat()
            .replace("+00:00", "Z"),
            "value": value,
        }
        for index, value in enumerate(values)
    ]


class _FakeClock:
    """Stand-in for the ``time`` module inside cache.py, with a movable monotonic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr("carbonsight_core.watttime.cache.time", clock)
    return clock


# --- TTL ---------------------------------------------------------------------


def test_default_ttl_is_15_minutes() -> None:
    assert DEFAULT_FORECAST_CACHE_TTL_SECONDS == 900
    assert ForecastCache().ttl_seconds == 900


def test_set_get_roundtrip() -> None:
    cache = ForecastCache(ttl_seconds=60)
    forecast = build_synthetic_forecast("CAISO_NORTH", horizon_hours=2)
    cache.set("CAISO_NORTH", forecast)

    cached = cache.get("CAISO_NORTH")
    assert cached is not None
    assert len(cached) == len(forecast)
    assert cached[0]["value"] == forecast[0]["value"]


def test_get_returns_a_copy_so_callers_cannot_mutate_the_cache() -> None:
    cache = ForecastCache()
    cache.set("R", _points(START, [100.0]))
    cache.get("R").append({"point_time": "2026-01-01T05:00:00Z", "value": 999.0})
    assert len(cache.get("R")) == 1


def test_ttl_expiry_drops_the_entry(fake_clock: _FakeClock) -> None:
    cache = ForecastCache(ttl_seconds=10)
    cache.set("R", _points(START, [100.0]))
    assert cache.get("R") is not None

    fake_clock.advance(11)
    assert cache.get("R") is None
    assert cache.regions() == []


def test_get_or_expired_survives_the_ttl(fake_clock: _FakeClock) -> None:
    cache = ForecastCache(ttl_seconds=10)
    cache.set("R", _points(START, [100.0]))
    fake_clock.advance(11)

    assert cache.get_or_expired("R") is not None
    assert cache.get_or_expired("MISSING") is None


def test_regions_lists_only_live_entries(fake_clock: _FakeClock) -> None:
    cache = ForecastCache(ttl_seconds=10)
    cache.set("OLD", _points(START, [100.0]))
    fake_clock.advance(6)
    cache.set("NEW", _points(START, [100.0]))
    fake_clock.advance(6)

    assert cache.regions() == ["NEW"]


def test_clear_empties_the_cache() -> None:
    cache = ForecastCache()
    cache.set("R", _points(START, [100.0]))
    cache.clear()
    assert cache.get("R") is None


def test_set_accepts_a_watttime_envelope() -> None:
    cache = ForecastCache()
    cache.set(
        "R",
        {"data": [{"point_time": "2026-01-01T00:00:00Z", "value": 42.0}], "units": "lbs_co2_per_mwh"},
    )
    cached = cache.get("R")
    assert cached is not None
    assert cached[0]["value"] == 42.0


def test_set_accepts_an_empty_envelope() -> None:
    cache = ForecastCache()
    cache.set("R", {"units": "lbs_co2_per_mwh"})
    assert cache.get("R") == []


def test_concurrent_set_and_get_are_safe() -> None:
    cache = ForecastCache()
    barrier = threading.Barrier(8)

    def hammer(worker: int) -> int:
        barrier.wait(timeout=5)
        for _ in range(200):
            cache.set(f"R{worker % 3}", _points(START, [float(worker)]))
            cache.get(f"R{worker % 3}")
            cache.regions()
        return worker

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sorted(pool.map(hammer, range(8))) == list(range(8))
    assert sorted(cache.regions()) == ["R0", "R1", "R2"]


# --- time-weighted MOER ------------------------------------------------------


def test_time_weighted_moer_simple_average() -> None:
    points = _points(START, [100.0, 200.0, 300.0])
    assert time_weighted_moer(points, START, START + timedelta(hours=2)) == pytest.approx(150.0)


def test_time_weighted_moer_fill_forward_past_last_point() -> None:
    points = _points(START, [100.0, 200.0])
    # One hour at 100, then four hours holding the last value of 200.
    average = time_weighted_moer(points, START, START + timedelta(hours=5))
    assert average == pytest.approx(180.0)


def test_time_weighted_moer_divides_by_covered_seconds_not_window_length() -> None:
    """Deliberate: an uncovered window head must not dilute the average toward zero.

    Points start one hour into a two-hour window. Dividing by the full window would
    report 50 lb/MWh — half the real grid intensity — and make an unforecast region
    look greener than a forecast one. Covered-seconds reports the honest 100.
    """
    points = _points(START + timedelta(hours=1), [100.0])
    average = time_weighted_moer(points, START, START + timedelta(hours=2))
    assert average == pytest.approx(100.0)


def test_time_weighted_moer_partial_head_coverage() -> None:
    """Half the window covered by two equal-length points averages those two, not four."""
    points = _points(START + timedelta(hours=2), [100.0, 300.0])
    average = time_weighted_moer(points, START, START + timedelta(hours=4))
    assert average == pytest.approx(200.0)


def test_time_weighted_moer_window_entirely_before_the_series() -> None:
    points = _points(START + timedelta(hours=10), [700.0, 800.0])
    assert time_weighted_moer(points, START, START + timedelta(hours=1)) == pytest.approx(700.0)


def test_time_weighted_moer_empty_series_is_zero() -> None:
    assert time_weighted_moer([], START, START + timedelta(hours=1)) == 0.0


def test_time_weighted_moer_degenerate_window_uses_the_first_value() -> None:
    points = _points(START, [123.0, 456.0])
    assert time_weighted_moer(points, START, START) == pytest.approx(123.0)


def test_time_weighted_moer_sorts_unordered_points() -> None:
    ordered = _points(START, [100.0, 200.0, 300.0])
    shuffled = [ordered[2], ordered[0], ordered[1]]
    window_end = START + timedelta(hours=2)
    assert time_weighted_moer(shuffled, START, window_end) == pytest.approx(
        time_weighted_moer(ordered, START, window_end)
    )


def test_time_weighted_moer_accepts_naive_point_times() -> None:
    points = [
        {"point_time": "2026-01-01T00:00:00", "value": 100.0},
        {"point_time": "2026-01-01T01:00:00", "value": 200.0},
    ]
    assert time_weighted_moer(points, START, START + timedelta(hours=2)) == pytest.approx(150.0)


# --- mixtures ----------------------------------------------------------------


def test_mixture_weighted_moer_blends_by_weight() -> None:
    window_end = START + timedelta(hours=1)
    clean = _points(START, [100.0, 100.0])
    dirty = _points(START, [400.0, 400.0])
    blended = mixture_weighted_moer([(clean, 0.25), (dirty, 0.75)], START, window_end)
    assert blended == pytest.approx(325.0)


def test_mixture_weighted_moer_normalizes_weights_that_do_not_sum_to_one() -> None:
    window_end = START + timedelta(hours=1)
    clean = _points(START, [100.0, 100.0])
    dirty = _points(START, [300.0, 300.0])
    assert mixture_weighted_moer([(clean, 2.0), (dirty, 2.0)], START, window_end) == pytest.approx(
        200.0
    )


def test_mixture_weighted_moer_ignores_negative_weights() -> None:
    window_end = START + timedelta(hours=1)
    clean = _points(START, [100.0, 100.0])
    dirty = _points(START, [300.0, 300.0])
    assert mixture_weighted_moer([(clean, 1.0), (dirty, -5.0)], START, window_end) == pytest.approx(
        100.0
    )


def test_mixture_weighted_moer_empty_is_zero() -> None:
    assert mixture_weighted_moer([], START, START + timedelta(hours=1)) == 0.0


# --- get_moer ----------------------------------------------------------------


def test_get_moer_uses_the_lbar_window() -> None:
    cache = ForecastCache()
    cache.set("X", _points(START, [100.0, 300.0, 500.0]))
    assert cache.get_moer("X", window_start=START, lbar_hours=2.0) == pytest.approx(200.0)


def test_get_moer_defaults_to_a_one_hour_window() -> None:
    cache = ForecastCache()
    cache.set("X", _points(START, [100.0, 300.0]))
    assert cache.get_moer("X", window_start=START) == pytest.approx(100.0)


def test_get_moer_accepts_an_explicit_window() -> None:
    cache = ForecastCache()
    cache.set("X", _points(START, [100.0, 300.0]))
    moer = cache.get_moer("X", window_start=START, window_end=START + timedelta(hours=2))
    assert moer == pytest.approx(200.0)


def test_get_moer_coerces_naive_window_bounds() -> None:
    cache = ForecastCache()
    cache.set("X", _points(START, [100.0, 300.0]))
    moer = cache.get_moer(
        "X", window_start=datetime(2026, 1, 1), window_end=datetime(2026, 1, 1, 2)
    )
    assert moer == pytest.approx(200.0)


def test_get_moer_mixture_weighting() -> None:
    cache = ForecastCache()
    cache.set("A", _points(START, [100.0, 100.0]))
    cache.set("B", _points(START, [300.0, 300.0]))
    moer = cache.get_moer(
        "A", window_start=START, lbar_hours=1.0, mixture=[("A", 0.5), ("B", 0.5)]
    )
    assert moer == pytest.approx(200.0)


def test_get_moer_returns_none_for_an_uncached_region() -> None:
    assert ForecastCache().get_moer("NOPE", window_start=START, lbar_hours=1.0) is None


def test_get_moer_returns_none_when_any_mixture_leg_is_missing() -> None:
    cache = ForecastCache()
    cache.set("A", _points(START, [100.0, 100.0]))
    moer = cache.get_moer(
        "A", window_start=START, lbar_hours=1.0, mixture=[("A", 0.5), ("B", 0.5)]
    )
    assert moer is None


def test_get_moer_reflects_the_per_region_synthetic_spread() -> None:
    """The cache must carry the carbon lever through, not flatten it."""
    cache = ForecastCache()
    for region in ("SE", "IND"):
        cache.set(region, build_synthetic_forecast(region, horizon_hours=3, now=START))
    clean = cache.get_moer("SE", window_start=START, lbar_hours=3.0)
    dirty = cache.get_moer("IND", window_start=START, lbar_hours=3.0)
    assert clean is not None and dirty is not None
    assert abs(clean - dirty) == pytest.approx(
        abs(synthetic_moer_for_region("SE") - synthetic_moer_for_region("IND"))
    )


# --- process-wide singleton --------------------------------------------------


def test_global_cache_is_a_singleton() -> None:
    first = get_global_forecast_cache()
    second = get_global_forecast_cache()
    assert first is second

    first.set("Z", build_synthetic_forecast("Z", horizon_hours=1))
    assert second.get("Z") is not None


def test_global_cache_ttl_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARBONSIGHT_FORECAST_CACHE_TTL", "42")
    assert Config.from_env().forecast_cache_ttl_seconds == 42
    assert get_global_forecast_cache().ttl_seconds == 42


def test_global_cache_ttl_argument_wins_on_first_construction() -> None:
    assert get_global_forecast_cache(ttl_seconds=7).ttl_seconds == 7
    # Later calls reuse the existing instance rather than rebuilding it.
    assert get_global_forecast_cache(ttl_seconds=99).ttl_seconds == 7


def test_reset_global_cache_drops_state() -> None:
    get_global_forecast_cache().set("Z", _points(START, [100.0]))
    reset_global_forecast_cache()
    assert get_global_forecast_cache().get("Z") is None
