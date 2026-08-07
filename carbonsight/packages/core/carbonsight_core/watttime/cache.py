"""
In-memory TTL forecast cache with time-weighted MOER over [t, t+Lbar].

Central design: the API server holds one ForecastCache (and optionally one WattTime
credential). CLI clients hit the API; they never need personal WattTime credentials.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import DEFAULT_FORECAST_CACHE_TTL_SECONDS, Config


def _parse_utc(timestamp: str | datetime) -> datetime:
    """Parse a WattTime point_time (ISO string or datetime) as an aware UTC datetime."""
    if isinstance(timestamp, datetime):
        return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def time_weighted_moer(
    points: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """Time-weighted average MOER (lb/MWh) over [window_start, window_end].

    Each point covers [t_i, t_{i+1}); the final point is held constant out to
    ``window_end``, so a window extending past the end of the series fills forward.

    The weighted sum is divided by the seconds actually **covered** by points, not by
    the full window length. Dividing by the window length silently understates MOER
    whenever the head of the window has no forecast coverage, which would make an
    uncovered region look greener than a covered one. Callers integrate from wall-clock
    starts against a 5-minute point grid, so that head gap is the normal case.
    """
    if not points:
        return 0.0
    if window_end <= window_start:
        return float(points[0].get("value", 0.0))

    sorted_points = sorted(points, key=lambda point: _parse_utc(point["point_time"]))

    weighted_sum = 0.0
    covered_seconds = 0.0
    for index, point in enumerate(sorted_points):
        point_start = _parse_utc(point["point_time"])
        if index + 1 < len(sorted_points):
            point_end = _parse_utc(sorted_points[index + 1]["point_time"])
        else:
            point_end = window_end
        interval_start = max(point_start, window_start)
        interval_end = min(point_end, window_end)
        overlap_seconds = max(0.0, (interval_end - interval_start).total_seconds())
        if overlap_seconds <= 0:
            continue
        weighted_sum += float(point["value"]) * overlap_seconds
        covered_seconds += overlap_seconds

    if covered_seconds <= 0:
        # Window lies entirely before the first point: use the first value.
        return float(sorted_points[0]["value"])
    return weighted_sum / covered_seconds


def mixture_weighted_moer(
    region_series: list[tuple[list[dict[str, Any]], float]],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """Blend multiple forecast series with mixture weights (sum preferably 1.0)."""
    if not region_series:
        return 0.0
    total_weight = sum(max(0.0, weight) for _, weight in region_series) or 1.0
    blended = 0.0
    for points, weight in region_series:
        share = max(0.0, weight) / total_weight
        blended += share * time_weighted_moer(points, window_start, window_end)
    return blended


class ForecastCache:
    """Thread-safe in-memory TTL cache of MOER forecast series keyed by region."""

    def __init__(self, ttl_seconds: float = DEFAULT_FORECAST_CACHE_TTL_SECONDS) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        # region -> (expires_at_monotonic, points)
        self._store: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    @property
    def ttl_seconds(self) -> float:
        """Time-to-live applied to every entry, in seconds."""
        return self._ttl_seconds

    def clear(self) -> None:
        """Drop every cached region."""
        with self._lock:
            self._store.clear()

    def set(self, region: str, forecast: list[dict[str, Any]] | dict[str, Any]) -> None:
        """Store forecast points for ``region``. Accepts a point list or a WattTime envelope."""
        if isinstance(forecast, dict):
            points = list(forecast.get("data") or [])
        else:
            points = list(forecast)
        with self._lock:
            self._store[region] = (time.monotonic() + self._ttl_seconds, points)

    def get(self, region: str) -> list[dict[str, Any]] | None:
        """Return cached points if present and not expired; else None."""
        with self._lock:
            entry = self._store.get(region)
            if entry is None:
                return None
            expires_at, points = entry
            if time.monotonic() > expires_at:
                del self._store[region]
                return None
            return list(points)

    def get_or_expired(self, region: str) -> list[dict[str, Any]] | None:
        """Return points even if expired (graceful degradation); None if never cached."""
        with self._lock:
            entry = self._store.get(region)
            if entry is None:
                return None
            return list(entry[1])

    def regions(self) -> list[str]:
        """Region codes with a live (unexpired) entry."""
        with self._lock:
            now_monotonic = time.monotonic()
            return [
                region
                for region, (expires_at, _) in self._store.items()
                if expires_at >= now_monotonic
            ]

    def get_moer(
        self,
        region: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        lbar_hours: float | None = None,
        *,
        mixture: list[tuple[str, float]] | None = None,
    ) -> float | None:
        """Time-weighted average MOER over [t, t+Lbar] (or an explicit window).

        Parameters
        ----------
        region:
            Primary WattTime region code.
        window_start / window_end:
            Explicit integration window. If omitted, start=now and end=start+lbar_hours.
        lbar_hours:
            Expected remaining job hours (Lbar). Used when window_end is None.
        mixture:
            Optional list of (region, weight) to blend multiple grids; ``region`` is
            ignored when this is given.

        Returns None if no live cache entry exists for the required region(s).
        """
        start = window_start or datetime.now(UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if window_end is not None:
            end = window_end if window_end.tzinfo else window_end.replace(tzinfo=UTC)
        else:
            hours = float(lbar_hours) if lbar_hours is not None else 1.0
            end = start + timedelta(hours=max(hours, 1e-6))

        if mixture:
            region_series: list[tuple[list[dict[str, Any]], float]] = []
            for mixture_region, weight in mixture:
                points = self.get(mixture_region)
                if points is None:
                    return None
                region_series.append((points, weight))
            return mixture_weighted_moer(region_series, start, end)

        points = self.get(region)
        if points is None:
            return None
        return time_weighted_moer(points, start, end)


# Process-wide cache used by the API server (one credential, many clients)
_global_forecast_cache: ForecastCache | None = None
_global_cache_lock = threading.Lock()


def get_global_forecast_cache(ttl_seconds: float | None = None) -> ForecastCache:
    """Return the process-wide ForecastCache singleton.

    On first call the TTL comes from ``ttl_seconds`` when given, otherwise from
    ``Config.from_env().forecast_cache_ttl_seconds``. Later calls reuse the instance.
    """
    global _global_forecast_cache
    with _global_cache_lock:
        if _global_forecast_cache is None:
            if ttl_seconds is None:
                ttl_seconds = Config.from_env().forecast_cache_ttl_seconds
            _global_forecast_cache = ForecastCache(ttl_seconds=ttl_seconds)
        return _global_forecast_cache


def reset_global_forecast_cache() -> None:
    """Drop the singleton (test helper and server shutdown hook)."""
    global _global_forecast_cache
    with _global_cache_lock:
        _global_forecast_cache = None
