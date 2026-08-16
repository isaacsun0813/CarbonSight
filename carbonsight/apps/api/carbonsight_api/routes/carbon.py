"""GET /v1/carbon/forecast — serve MOER from the shared cache.

The middle of the central-credential design. The worker holds the one WattTime
login and keeps the cache warm; this route hands that data to CLI clients, which
set ``CARBONSIGHT_API_URL`` and need no credentials of their own.

Read-only and cheap by construction: it serves whatever the worker last wrote and
never fetches on the request path. A miss is a 503 telling the caller to wait for
the next refresh — not a synchronous fan-out to WattTime, and never a synthetic
curve. Both would be worse than an honest "not yet": the fan-out turns one HTTP
request into 21 upstream calls, and the synthetic curve is a hash of the region
name that the client cannot distinguish from a real reading.

``source`` on every response says where the data came from, so a client can tell
a live reading from a stale one.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from carbonsight_core.carbon import CarbonProviderError, ForecastBackedProvider
from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import default_grid_regions
from carbonsight_core.watttime import get_global_forecast_cache
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("carbonsight.api.carbon")

router = APIRouter()

_SELECT_POINTS = """
SELECT point_time, value, units
FROM grid_signal_cache
WHERE wt_region = :wt_region AND signal_type = 'co2_moer'
ORDER BY point_time
"""


def _limit_horizon(points: list[dict[str, Any]], horizon_hours: int) -> list[dict[str, Any]]:
    """Return at most ``horizon_hours`` from the first usable forecast point."""
    if not points:
        return []
    try:
        start = datetime.fromisoformat(str(points[0]["point_time"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return points
    end = start + timedelta(hours=horizon_hours)
    limited: list[dict[str, Any]] = []
    for point in points:
        try:
            point_time = datetime.fromisoformat(
                str(point["point_time"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if point_time < end:
            limited.append(point)
    return limited


def read_points_from_db(region: str, *, database_url: str | None = None) -> list[dict[str, Any]]:
    """Read a region's cached points from Postgres. Empty list when unavailable.

    In a real deployment the worker and the API are separate processes, so the
    in-memory cache is *not* shared between them — ``grid_signal_cache`` is the
    only store they both see. The in-process cache stays as a per-process
    accelerator for the single-process case.
    """
    dsn = database_url if database_url is not None else os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        return []
    try:
        import sqlalchemy as sa
        from sqlalchemy.exc import SQLAlchemyError

        engine = sa.create_engine(dsn)
        with engine.connect() as connection:
            rows = connection.execute(sa.text(_SELECT_POINTS), {"wt_region": region}).fetchall()
        return [
            {
                "point_time": row.point_time.isoformat().replace("+00:00", "Z"),
                "value": float(row.value),
                "units": row.units,
            }
            for row in rows
        ]
    except (SQLAlchemyError, ImportError) as err:
        logger.warning("grid_signal_cache read failed for %s: %s", region, err)
        return []


def read_cached_forecast(region: str, horizon_hours: int = 24) -> dict[str, Any]:
    """Read one forecast from process cache or Postgres without upstream I/O."""
    config = Config.from_env()
    cache = get_global_forecast_cache(config.forecast_cache_ttl_seconds)

    any_points = cache.get_or_expired(region)
    fresh_points = cache.get(region)
    if fresh_points is not None:
        return {
            "region": region,
            "source": "cache",
            "data": _limit_horizon(fresh_points, horizon_hours),
        }

    db_points = read_points_from_db(region)
    if db_points:
        cache.set(region, db_points)
        return {
            "region": region,
            "source": "db",
            "data": _limit_horizon(db_points, horizon_hours),
        }

    if any_points is not None:
        return {
            "region": region,
            "source": "stale",
            "data": _limit_horizon(any_points, horizon_hours),
        }

    raise CarbonProviderError(
        f"No forecast cached for {region}. The refresh worker has not stored usable data."
    )


class StoredCarbonProvider(ForecastBackedProvider):
    """Server-local provider backed by the shared forecast store."""

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        del anchor
        return list(read_cached_forecast(region, horizon_hours)["data"])

    def list_regions(self) -> list[str]:
        return default_grid_regions()


@router.get("/carbon/forecast")
def get_carbon_forecast(
    region: str = Query(..., description="WattTime grid region code, e.g. CAISO_NORTH"),
    horizon_hours: int = Query(24, ge=1, le=168),
) -> dict[str, Any]:
    """Return cached MOER points for one grid region."""
    try:
        return read_cached_forecast(region, horizon_hours)
    except CarbonProviderError as err:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{err} Check that the refresh worker is running and that "
                "WATTTIME_USERNAME/PASSWORD are set on the server."
            ),
        ) from err


@router.get("/carbon/regions")
def list_carbon_regions() -> dict[str, Any]:
    """Grid regions the worker refreshes, and which currently have data."""
    config = Config.from_env()
    cache = get_global_forecast_cache(config.forecast_cache_ttl_seconds)
    warm = set(cache.regions())
    return {
        "regions": [
            {"wt_region": region, "cached": region in warm}
            for region in default_grid_regions()
        ],
        "cached_count": len(warm),
    }
