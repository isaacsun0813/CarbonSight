"""
In-memory TTL forecast cache with time-weighted MOER over [t, t+Lbar].

Central design: the API server holds one ForecastCache (and optionally one WattTime
credential). CLI clients hit the API; they never need personal WattTime creds.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import CACHE_TTL_SECONDS

DEFAULT_TTL = CACHE_TTL_SECONDS  # 15 minutes


def _parse_utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def time_weighted_moer(
    points: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """Time-weighted average MOER (lb/MWh) over [window_start, window_end].

    Each point covers [t_i, t_{i+1}). If the window extends past the last point,
    the last value is held constant (SkyNomad-style fill-forward).
    """
    if not points:
        return 0.0
    if window_end <= window_start:
        return float(points[0].get("value", 0.0))

    sorted_pts = sorted(points, key=lambda p: _parse_utc(p["point_time"]))
    # Ensure fill-forward coverage past the last point
    last = sorted_pts[-1]
    last_t = _parse_utc(last["point_time"])
    if last_t < window_end:
        sorted_pts = list(sorted_pts) + [
            {
                "point_time": (window_end + timedelta(seconds=1)).isoformat(),
                "value": float(last["value"]),
            }
        ]

    total_seconds = (window_end - window_start).total_seconds()
    if total_seconds <= 0:
        return float(sorted_pts[0]["value"])

    weighted = 0.0
    covered = 0.0
    for i, p in enumerate(sorted_pts):
        t_i = _parse_utc(p["point_time"])
        if i + 1 < len(sorted_pts):
            t_next = _parse_utc(sorted_pts[i + 1]["point_time"])
        else:
            t_next = window_end
        interval_start = max(t_i, window_start)
        interval_end = min(t_next, window_end)
        overlap = max(0.0, (interval_end - interval_start).total_seconds())
        if overlap <= 0:
            continue
        weighted += float(p["value"]) * overlap
        covered += overlap

    if covered <= 0:
        # Window entirely before first point: use first value
        return float(sorted_pts[0]["value"])
    return weighted / covered


def mixture_weighted_moer(
    region_series: list[tuple[list[dict[str, Any]], float]],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """Blend multiple forecast series with mixture weights (sum preferably 1.0)."""
    if not region_series:
        return 0.0
    total_w = sum(max(0.0, w) for _, w in region_series) or 1.0
    acc = 0.0
    for pts, w in region_series:
        acc += (max(0.0, w) / total_w) * time_weighted_moer(pts, window_start, window_end)
    return acc


class ForecastCache:
    """Thread-safe in-memory TTL cache of MOER forecast series keyed by region."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL) -> None:
        self._ttl = float(ttl_seconds)
        self._lock = threading.RLock()
        # region -> (expires_at_monotonic, data_list)
        self._store: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def set(self, region: str, data: list[dict[str, Any]] | dict[str, Any]) -> None:
        """Store forecast points for ``region``. Accepts list or WattTime dict payload."""
        if isinstance(data, dict):
            points = list(data.get("data") or [])
        else:
            points = list(data)
        with self._lock:
            self._store[region] = (time.monotonic() + self._ttl, points)

    def get(self, region: str) -> list[dict[str, Any]] | None:
        """Return cached points if present and not expired; else None."""
        with self._lock:
            entry = self._store.get(region)
            if entry is None:
                return None
            expires_at, data = entry
            if time.monotonic() > expires_at:
                del self._store[region]
                return None
            return list(data)

    def get_or_expired(self, region: str) -> list[dict[str, Any]] | None:
        """Return data even if expired (for graceful degradation); None if missing."""
        with self._lock:
            entry = self._store.get(region)
            if entry is None:
                return None
            return list(entry[1])

    def regions(self) -> list[str]:
        with self._lock:
            now = time.monotonic()
            return [r for r, (exp, _) in self._store.items() if exp >= now]

    def get_moer(
        self,
        region: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        lbar_hours: float | None = None,
        *,
        mixture: list[tuple[str, float]] | None = None,
    ) -> float | None:
        """Time-weighted average MOER over [t, t+Lbar] (or explicit window).

        Parameters
        ----------
        region:
            Primary WattTime region code.
        window_start / window_end:
            Explicit integration window. If omitted, start=now and end=now+lbar_hours.
        lbar_hours:
            Expected remaining job hours (SkyNomad Lbar). Used when window_end is None.
        mixture:
            Optional list of (region, weight) to blend multiple grids.

        Returns None if no cache entry exists for the required region(s).
        """
        now = datetime.now(UTC)
        start = window_start or now
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if window_end is not None:
            end = window_end if window_end.tzinfo else window_end.replace(tzinfo=UTC)
        else:
            hours = float(lbar_hours) if lbar_hours is not None else 1.0
            end = start + timedelta(hours=max(hours, 1e-6))

        if mixture:
            series: list[tuple[list[dict[str, Any]], float]] = []
            for reg, weight in mixture:
                pts = self.get(reg)
                if pts is None:
                    return None
                series.append((pts, weight))
            return mixture_weighted_moer(series, start, end)

        pts = self.get(region)
        if pts is None:
            return None
        return time_weighted_moer(pts, start, end)


# Process-wide cache used by the API server (one credential, many clients)
_GLOBAL_CACHE: ForecastCache | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_forecast_cache(ttl_seconds: float | None = None) -> ForecastCache:
    """Return the process-wide ForecastCache singleton."""
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = ForecastCache(ttl_seconds=ttl_seconds or DEFAULT_TTL)
        return _GLOBAL_CACHE


def reset_global_forecast_cache() -> None:
    """Test helper: drop the singleton."""
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        _GLOBAL_CACHE = None
