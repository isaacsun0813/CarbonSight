"""ForecastCache TTL + time-weighted MOER tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.watttime.cache import (
    DEFAULT_TTL,
    ForecastCache,
    get_global_forecast_cache,
    mixture_weighted_moer,
    reset_global_forecast_cache,
    time_weighted_moer,
)
from carbonsight_core.watttime.client import build_synthetic_forecast, synthetic_moer_for_region


def _pts(start: datetime, values: list[float], step_min: int = 60) -> list[dict]:
    out = []
    for i, v in enumerate(values):
        t = start + timedelta(minutes=step_min * i)
        out.append({"point_time": t.isoformat().replace("+00:00", "Z"), "value": v})
    return out


def test_default_ttl_is_15_minutes():
    assert DEFAULT_TTL == 900
    c = ForecastCache()
    assert c.ttl_seconds == 900


def test_set_get_roundtrip():
    c = ForecastCache(ttl_seconds=60)
    data = build_synthetic_forecast("CAISO_NORTH", horizon_hours=2)
    c.set("CAISO_NORTH", data)
    got = c.get("CAISO_NORTH")
    assert got is not None
    assert len(got) == len(data)
    assert got[0]["value"] == data[0]["value"]


def test_ttl_expiry(monkeypatch):
    c = ForecastCache(ttl_seconds=1)
    c.set("R", [{"point_time": "2026-01-01T00:00:00Z", "value": 100.0}])
    assert c.get("R") is not None
    # Force expiry by rewriting store expiry
    key = "R"
    exp, data = c._store[key]
    c._store[key] = (exp - 10, data)
    assert c.get("R") is None


def test_time_weighted_moer_simple_average():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pts = _pts(start, [100.0, 200.0, 300.0], step_min=60)
    # Window covers first two full hours equally -> avg 150
    end = start + timedelta(hours=2)
    avg = time_weighted_moer(pts, start, end)
    assert avg == pytest.approx(150.0, rel=1e-6)


def test_time_weighted_moer_fill_forward_past_last_point():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pts = _pts(start, [100.0, 200.0], step_min=60)
    # Window extends 3h past last point — last value held
    end = start + timedelta(hours=5)
    avg = time_weighted_moer(pts, start, end)
    assert avg > 100.0
    # Heavily weighted toward 200 after hour 1
    assert avg == pytest.approx(180.0, rel=0.05)


def test_get_moer_uses_lbar_window():
    c = ForecastCache()
    start = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    pts = _pts(start, [100.0, 300.0, 500.0], step_min=60)
    c.set("X", pts)
    moer = c.get_moer("X", window_start=start, lbar_hours=2.0)
    assert moer is not None
    assert moer == pytest.approx(200.0, rel=1e-6)


def test_get_moer_mixture_weighting():
    c = ForecastCache()
    start = datetime(2026, 6, 1, tzinfo=UTC)
    c.set("A", _pts(start, [100.0, 100.0], step_min=60))
    c.set("B", _pts(start, [300.0, 300.0], step_min=60))
    moer = c.get_moer(
        "A",
        window_start=start,
        lbar_hours=1.0,
        mixture=[("A", 0.5), ("B", 0.5)],
    )
    assert moer == pytest.approx(200.0, rel=1e-6)


def test_mixture_weighted_moer_helper():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    a = _pts(start, [100.0, 100.0])
    b = _pts(start, [400.0, 400.0])
    m = mixture_weighted_moer([(a, 0.25), (b, 0.75)], start, end)
    assert m == pytest.approx(325.0, rel=1e-6)


def test_global_cache_singleton():
    reset_global_forecast_cache()
    a = get_global_forecast_cache()
    b = get_global_forecast_cache()
    assert a is b
    a.set("Z", build_synthetic_forecast("Z", horizon_hours=1))
    assert b.get("Z") is not None
    reset_global_forecast_cache()


def test_synthetic_moer_deterministic():
    assert synthetic_moer_for_region("CAISO_NORTH") == synthetic_moer_for_region("CAISO_NORTH")
    assert synthetic_moer_for_region("CAISO_NORTH") != synthetic_moer_for_region("DIRTY_GRID_ZZ")


def test_set_accepts_watttime_dict_payload():
    c = ForecastCache()
    c.set("R", {"data": [{"point_time": "2026-01-01T00:00:00Z", "value": 42.0}], "units": "lbs_co2_per_mwh"})
    got = c.get("R")
    assert got is not None
    assert got[0]["value"] == 42.0
