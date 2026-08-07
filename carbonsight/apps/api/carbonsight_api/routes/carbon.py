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
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.watttime import SYNTHETIC_WATTTIME_REGIONS, get_global_forecast_cache
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("carbonsight.api.carbon")

router = APIRouter()

_SELECT_POINTS = """
SELECT point_time, value, units
FROM grid_signal_cache
WHERE wt_region = :wt_region AND signal_type = 'co2_moer'
ORDER BY point_time
"""


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
    except Exception as err:
        logger.warning("grid_signal_cache read failed for %s: %s", region, err)
        return []


@router.get("/carbon/forecast")
def get_carbon_forecast(
    region: str = Query(..., description="WattTime grid region code, e.g. CAISO_NORTH"),
    horizon_hours: int = Query(24, ge=1, le=168),
) -> dict[str, Any]:
    """Return cached MOER points for one grid region."""
    del horizon_hours  # the worker decides the horizon; kept for client compatibility
    config = Config.from_env()
    cache = get_global_forecast_cache(config.forecast_cache_ttl_seconds)

    # Read the expired-tolerant copy FIRST: ForecastCache.get() evicts an expired
    # entry as a side effect, so asking it before get_or_expired() would leave
    # nothing to serve and make the stale branch unreachable.
    any_points = cache.get_or_expired(region)
    fresh_points = cache.get(region)

    if fresh_points is not None:
        return {"region": region, "source": "cache", "data": fresh_points}

    # Nothing fresh in this process. When the worker runs in a *different* process
    # (the compose setup), Postgres is the only store both sides see.
    db_points = read_points_from_db(region)
    if db_points:
        cache.set(region, db_points)
        return {"region": region, "source": "db", "data": db_points}

    # Expired but present: a stale real reading beats nothing, clearly labelled.
    if any_points is not None:
        return {"region": region, "source": "stale", "data": any_points}

    raise HTTPException(
        status_code=503,
        detail=(
            f"No forecast cached for {region}. The refresh worker populates the cache "
            "every 15 minutes; if this persists, check that the worker is running and "
            "that WATTTIME_USERNAME/PASSWORD are set on the server."
        ),
    )


@router.get("/carbon/regions")
def list_carbon_regions() -> dict[str, Any]:
    """Grid regions the worker refreshes, and which currently have data."""
    config = Config.from_env()
    cache = get_global_forecast_cache(config.forecast_cache_ttl_seconds)
    warm = set(cache.regions())
    return {
        "regions": [
            {"wt_region": region, "cached": region in warm}
            for region in SYNTHETIC_WATTTIME_REGIONS
        ],
        "cached_count": len(warm),
    }
