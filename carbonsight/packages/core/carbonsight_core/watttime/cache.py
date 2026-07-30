"""
ForecastCache: centralizes WattTime forecast fetching + time-weighted window queries.

Design for P0 centralized creds:
- One WattTimeClient holds single credential server-side, token cached 25min
- ForecastCache adds second layer: per-region forecast points cached TTL (default 15min)
- Provides get_time_weighted_moer(wt_regions, start, end) which is needed for joint 2D rank:
  MOER(r, [t, t+L̄]) not just point now.

This file does NOT call DB — DB layer is in api/routes/carbon.py which reuses same
_time_weighted_moer logic. In-memory cache is fallback for CLI without DB and for unit tests.

Fallback: if WattTimeError or no data, returns 400 lb/MWh (existing pattern in carbon_model.py:86)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from carbonsight_core.watttime import WattTimeClient

FALLBACK_MOER_LB_PER_MWH = 400.0
DEFAULT_TTL_SECONDS = 15 * 60  # 15min matches fetcher interval
DEFAULT_HORIZON_HOURS = 72


@dataclass
class _CacheEntry:
    fetched_at: float
    points: list[dict[str, Any]]


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _time_weighted_moer(
    points: list[dict[str, Any]],
    actual_start: datetime,
    actual_end: datetime,
) -> float:
    """
    Copy of carbon_model._time_weighted_moer to avoid circular import.
    Each point covers [point_time, next_point_time), clipped to [start,end).
    """
    if not points:
        return FALLBACK_MOER_LB_PER_MWH
    if len(points) == 1:
        try:
            return float(points[0]["value"])
        except Exception:
            return FALLBACK_MOER_LB_PER_MWH

    total_seconds = (actual_end - actual_start).total_seconds()
    if total_seconds <= 0:
        try:
            return float(points[0]["value"])
        except Exception:
            return FALLBACK_MOER_LB_PER_MWH

    sorted_points = sorted(points, key=lambda p: p["point_time"])
    weighted_sum = 0.0
    for i, p in enumerate(sorted_points):
        try:
            t_i = _parse_utc(p["point_time"])
        except Exception:
            continue
        if i + 1 < len(sorted_points):
            try:
                t_next = _parse_utc(sorted_points[i + 1]["point_time"])
            except Exception:
                t_next = actual_end
        else:
            t_next = actual_end

        interval_start = max(t_i, actual_start)
        interval_end = min(t_next, actual_end)
        overlap = max(0.0, (interval_end - interval_start).total_seconds())
        try:
            weighted_sum += float(p["value"]) * overlap
        except Exception:
            continue

    if weighted_sum == 0:
        # fallback to first point if no overlap
        try:
            return float(sorted_points[0]["value"])
        except Exception:
            return FALLBACK_MOER_LB_PER_MWH
    return weighted_sum / total_seconds


class ForecastCache:
    """
    In-memory TTL cache for WattTime forecasts.

    Usage:
        cache = ForecastCache(watt_client, ttl_seconds=900)
        moer = cache.get_time_weighted_moer([("CAISO_NORTH",1.0)], start, end)
    """

    def __init__(self, client: WattTimeClient, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._fetch_count = 0  # for tests to verify hit/miss

    def clear(self) -> None:
        self._cache.clear()

    def get_forecast_points(
        self,
        wt_region: str,
        horizon_hours: int = DEFAULT_HORIZON_HOURS,
    ) -> list[dict[str, Any]]:
        now = time.time()
        entry = self._cache.get(wt_region)
        if entry and (now - entry.fetched_at) < self._ttl:
            return entry.points

        try:
            data = self._client.get_forecast(wt_region, horizon_hours=horizon_hours)
            points = data.get("data", []) or []
            if not points:
                points = []
        except Exception:
            # On any error, return cached if exists, else empty -> fallback upstream
            if entry:
                return entry.points
            return []

        self._cache[wt_region] = _CacheEntry(fetched_at=now, points=points)
        self._fetch_count += 1
        return points

    def get_time_weighted_moer(
        self,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        horizon_hours: int = DEFAULT_HORIZON_HOURS,
    ) -> float:
        """
        Weighted sum over mixture wt_regions of time-weighted MOER in [start,end].
        If wt_regions empty, returns fallback.
        """
        if not wt_regions:
            return FALLBACK_MOER_LB_PER_MWH

        total = 0.0
        total_weight = 0.0
        for region, weight in wt_regions:
            points = self.get_forecast_points(region, horizon_hours=horizon_hours)
            if not points:
                moer = FALLBACK_MOER_LB_PER_MWH
            else:
                moer = _time_weighted_moer(points, start, end)
            total += weight * moer
            total_weight += weight

        if total_weight == 0:
            return FALLBACK_MOER_LB_PER_MWH
        # Normalize if weights don't sum to 1 (defensive)
        if abs(total_weight - 1.0) > 1e-6:
            total = total / total_weight if total_weight else FALLBACK_MOER_LB_PER_MWH
        return total

    # For tests
    @property
    def fetch_count(self) -> int:
        return self._fetch_count
