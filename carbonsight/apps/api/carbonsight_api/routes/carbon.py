"""
GET /v1/carbon/forecast — the central MOER cache.

The server holds the one WattTime credential; CLI clients read this endpoint via
``ApiCarbonProvider`` and need no credentials of their own.

Resolution order per region:

    1. in-memory ForecastCache (TTL 15 min)
    2. Postgres ``grid_signal_cache`` when DATABASE_URL is set
    3. live WattTime fetch when credentials are present (upserted back into 2)
    4. synthetic curve, so the endpoint never fails closed

Postgres access is sqlalchemy only — one driver, one code path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS
from carbonsight_core.watttime.cache import get_global_forecast_cache, time_weighted_moer
from carbonsight_core.watttime.client import WattTimeClient, build_synthetic_forecast
from fastapi import APIRouter, Query

router = APIRouter()

SIGNAL_TYPE = "co2_moer"

_SELECT_POINTS = """
    SELECT point_time, value, units
    FROM grid_signal_cache
    WHERE wt_region = :region AND signal_type = :signal_type
      AND point_time BETWEEN :start AND :end
    ORDER BY point_time ASC
"""

_UPSERT_POINT = """
    INSERT INTO grid_signal_cache (wt_region, signal_type, point_time, value, units)
    VALUES (:region, :signal_type, :point_time, :value, :units)
    ON CONFLICT (wt_region, signal_type, point_time)
    DO UPDATE SET value = EXCLUDED.value, units = EXCLUDED.units
"""


def _parse_utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _db_points(db_url: str, region: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Read cached MOER points for a region and window. Empty list on any failure."""
    try:
        from sqlalchemy import create_engine, text

        with create_engine(db_url, future=True).connect() as conn:
            rows = conn.execute(
                text(_SELECT_POINTS),
                {"region": region, "signal_type": SIGNAL_TYPE, "start": start, "end": end},
            ).fetchall()
    except Exception:
        return []
    return [
        {
            "point_time": _parse_utc(row[0]).isoformat(),
            "value": float(row[1]),
            "units": row[2] or "lbs_co2_per_mwh",
        }
        for row in rows
    ]


def _upsert_points(db_url: str, region: str, points: list[dict[str, Any]]) -> int:
    """Persist freshly fetched points. Best effort; returns how many were written."""
    if not points:
        return 0
    try:
        from sqlalchemy import create_engine, text

        with create_engine(db_url, future=True).begin() as conn:
            for p in points:
                conn.execute(
                    text(_UPSERT_POINT),
                    {
                        "region": region,
                        "signal_type": SIGNAL_TYPE,
                        "point_time": _parse_utc(p["point_time"]),
                        "value": float(p["value"]),
                        "units": p.get("units") or "lbs_co2_per_mwh",
                    },
                )
    except Exception:
        return 0
    return len(points)


def resolve_forecast(region: str, horizon_hours: int = 24) -> tuple[list[dict[str, Any]], str]:
    """Return ``(points, source)`` following cache -> Postgres -> live -> synthetic."""
    cache = get_global_forecast_cache()
    cached = cache.get(region)
    if cached is not None:
        return cached, "cache"

    cfg = Config.from_env()
    now = datetime.now(UTC)

    if cfg.database_url:
        pts = _db_points(
            cfg.database_url,
            region,
            now - timedelta(hours=1),
            now + timedelta(hours=horizon_hours),
        )
        if pts:
            cache.set(region, pts)
            return pts, "db"

    if cfg.watttime_username and cfg.watttime_password:
        try:
            client = WattTimeClient(cfg, allow_synthetic=False)
            pts = WattTimeClient.normalize_forecast_payload(
                client.get_forecast(region, horizon_hours=horizon_hours)
            )
        except Exception:
            pts = []
        if pts:
            cache.set(region, pts)
            if cfg.database_url:
                _upsert_points(cfg.database_url, region, pts)
            return pts, "live"

    pts = build_synthetic_forecast(region, horizon_hours=horizon_hours)
    cache.set(region, pts)
    return pts, "synthetic"


@router.get("/carbon/forecast")
def get_carbon_forecast(
    region: str = Query(..., description="WattTime region code, e.g. CAISO_NORTH"),
    horizon_hours: int = Query(24, ge=1, le=168),
    lbar_hours: float | None = Query(None, description="Optional window length for average MOER"),
) -> dict[str, Any]:
    """MOER forecast points for one grid region, from the central cache."""
    pts, source = resolve_forecast(region, horizon_hours=horizon_hours)
    body: dict[str, Any] = {
        "region": region,
        "units": "lbs_co2_per_mwh",
        "data": pts,
        "points": pts,
        "source": source,
    }
    if lbar_hours is not None:
        start = datetime.now(UTC)
        body["moer_avg"] = time_weighted_moer(pts, start, start + timedelta(hours=lbar_hours))
        body["lbar_hours"] = lbar_hours
    return body


@router.get("/carbon/regions")
def list_carbon_regions() -> list[dict[str, str]]:
    """Grid regions the server can serve forecasts for."""
    return [{"wt_region": r} for r in DEFAULT_CARBON_REGIONS]


def warm_all_regions() -> int:
    """Warm the global cache for every default region (called on API startup)."""
    for region in DEFAULT_CARBON_REGIONS:
        resolve_forecast(region)
    return len(DEFAULT_CARBON_REGIONS)
