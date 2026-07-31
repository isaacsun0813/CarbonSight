"""GET /v1/carbon/forecast — central MOER cache proxy (ONE WattTime credential server-side)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS, get_carbon_provider
from carbonsight_core.watttime.cache import get_global_forecast_cache, time_weighted_moer
from carbonsight_core.watttime.client import WattTimeClient, build_synthetic_forecast
from fastapi import APIRouter, Query

router = APIRouter()


def _ensure_region_cached(region: str, horizon_hours: int = 24) -> list[dict[str, Any]]:
    """Populate global cache from WattTime or synthetic; always return points."""
    cache = get_global_forecast_cache()
    cached = cache.get(region)
    if cached is not None:
        return cached

    cfg = Config.from_env()
    pts: list[dict[str, Any]] = []
    if cfg.watttime_username and cfg.watttime_password:
        try:
            client = WattTimeClient(cfg, allow_synthetic=True)
            payload = client.get_forecast(region, horizon_hours=horizon_hours)
            pts = WattTimeClient.normalize_forecast_payload(payload)
        except Exception:
            pts = []
    if not pts:
        pts = build_synthetic_forecast(region, horizon_hours=horizon_hours)
    cache.set(region, pts)
    return pts


@router.get("/carbon/forecast")
def get_carbon_forecast(
    region: str = Query(..., description="WattTime region code, e.g. CAISO_NORTH"),
    horizon_hours: int = Query(24, ge=1, le=168),
    lbar_hours: float | None = Query(None, description="Optional window length for avg MOER"),
) -> dict[str, Any]:
    """Return MOER forecast points; uses server cache + optional WattTime creds."""
    pts = _ensure_region_cached(region, horizon_hours=horizon_hours)
    body: dict[str, Any] = {
        "region": region,
        "units": "lbs_co2_per_mwh",
        "data": pts,
        "source": "cache_or_synthetic",
    }
    if lbar_hours is not None:
        start = datetime.now(UTC)
        end = start + timedelta(hours=lbar_hours)
        body["moer_avg"] = time_weighted_moer(pts, start, end)
        body["lbar_hours"] = lbar_hours
    return body


@router.get("/carbon/regions")
def list_carbon_regions() -> list[dict[str, str]]:
    """List known WattTime / synthetic grid regions (17)."""
    return [{"wt_region": r} for r in DEFAULT_CARBON_REGIONS]


def warm_all_regions() -> int:
    """Warm global cache for all default regions (API lifespan)."""
    n = 0
    for r in DEFAULT_CARBON_REGIONS:
        _ensure_region_cached(r)
        n += 1
    # Also touch provider factory path
    try:
        get_carbon_provider()
    except Exception:
        pass
    return n
