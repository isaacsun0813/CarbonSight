"""Unit tests for ForecastCache TTL + time-weighted MOER + mixture — P0 central cache."""

from datetime import datetime
from unittest.mock import MagicMock

from carbonsight_core.watttime import WattTimeClient
from carbonsight_core.watttime.cache import ForecastCache, _time_weighted_moer


def test_time_weighted_non_uniform_intervals() -> None:
    pts = [
        {"point_time": "2026-01-01T00:00:00+00:00", "value": 100},
        {"point_time": "2026-01-01T01:00:00+00:00", "value": 200},
        {"point_time": "2026-01-01T02:00:00+00:00", "value": 300},
    ]
    start = datetime.fromisoformat("2026-01-01T00:30:00+00:00")
    end = datetime.fromisoformat("2026-01-01T01:30:00+00:00")
    avg = _time_weighted_moer(pts, start, end)
    assert abs(avg - 150) < 1e-6


def test_cache_hit_avoids_second_http_call() -> None:
    pts = [
        {"point_time": "2026-01-01T00:00:00+00:00", "value": 100},
        {"point_time": "2026-01-01T01:00:00+00:00", "value": 100},
    ]
    mock_client = MagicMock(spec=WattTimeClient)
    mock_client.get_forecast.return_value = {"data": pts}
    cache = ForecastCache(mock_client, ttl_seconds=60)
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    end = datetime.fromisoformat("2026-01-01T01:00:00+00:00")
    moer1 = cache.get_time_weighted_moer([("CAISO_NORTH", 1.0)], start, end)
    assert cache.fetch_count == 1
    moer2 = cache.get_time_weighted_moer([("CAISO_NORTH", 1.0)], start, end)
    assert cache.fetch_count == 1  # cached
    assert moer1 == moer2


def test_mixture_weighting() -> None:
    def side_effect(region, horizon_hours=24, client=None):
        if region == "A":
            return {"data": [{"point_time": "2026-01-01T00:00:00+00:00", "value": 100}]}
        return {"data": [{"point_time": "2026-01-01T00:00:00+00:00", "value": 200}]}

    mock_client = MagicMock(spec=WattTimeClient)
    mock_client.get_forecast.side_effect = side_effect
    cache = ForecastCache(mock_client, ttl_seconds=60)
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    end = datetime.fromisoformat("2026-01-01T01:00:00+00:00")
    moer = cache.get_time_weighted_moer([("A", 0.6), ("B", 0.4)], start, end)
    assert abs(moer - 140) < 1e-6


def test_single_point_fallback() -> None:
    pts = [{"point_time": "2026-01-01T00:00:00+00:00", "value": 123}]
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    end = datetime.fromisoformat("2026-01-01T05:00:00+00:00")
    avg = _time_weighted_moer(pts, start, end)
    assert avg == 123

    mock_client = MagicMock(spec=WattTimeClient)
    mock_client.get_forecast.return_value = {"data": pts}
    cache = ForecastCache(mock_client)
    moer = cache.get_time_weighted_moer([("R", 1.0)], start, end)
    assert moer == 123
