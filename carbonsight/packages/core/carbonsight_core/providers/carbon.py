"""
Carbon intensity providers.

Factory preference (P0 central-credential design):
  1. CARBONSIGHT_API_URL  -> ApiCarbonProvider  (CLI without personal WattTime creds)
  2. WATTTIME_USERNAME    -> WattTimeCarbonProvider
  3. else                 -> SyntheticCarbonProvider
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import httpx

from carbonsight_core.config import Config
from carbonsight_core.watttime.cache import ForecastCache, get_global_forecast_cache, time_weighted_moer
from carbonsight_core.watttime.client import (
    SYNTHETIC_WT_REGIONS,
    WattTimeClient,
    build_synthetic_forecast,
    synthetic_moer_for_region,
)

# Canonical list of ~17 grid regions used for synthetic rankings / API fallback
DEFAULT_CARBON_REGIONS: list[str] = list(SYNTHETIC_WT_REGIONS)


@runtime_checkable
class CarbonIntensityProvider(Protocol):
    """Protocol for MOER forecast access."""

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        """Return list of {point_time, value} MOER points (lb/MWh)."""
        ...

    def get_moer(
        self,
        region: str,
        *,
        lbar_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        """Time-weighted average MOER over [t, t+Lbar]."""
        ...

    def list_regions(self) -> list[str]:
        """Known region codes (at least the synthetic 17)."""
        ...


class SyntheticCarbonProvider:
    """Deterministic fake MOER data for tests and no-creds demos."""

    def __init__(self, regions: list[str] | None = None) -> None:
        self._regions = list(regions or DEFAULT_CARBON_REGIONS)
        self._cache = ForecastCache()

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        pts = build_synthetic_forecast(region, horizon_hours=horizon_hours)
        self._cache.set(region, pts)
        return pts

    def get_moer(
        self,
        region: str,
        *,
        lbar_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        pts = self.get_forecast(region)
        start = window_start or datetime.now(UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        end = start + timedelta(hours=max(lbar_hours, 1e-6))
        return time_weighted_moer(pts, start, end)

    def list_regions(self) -> list[str]:
        return list(self._regions)

    def point_moer(self, region: str) -> float:
        """Instantaneous synthetic MOER (no time weighting)."""
        return synthetic_moer_for_region(region)


class WattTimeCarbonProvider:
    """Direct WattTime access + local ForecastCache. Requires WATTTIME_* env."""

    def __init__(
        self,
        client: WattTimeClient | None = None,
        cache: ForecastCache | None = None,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._client = client or WattTimeClient(self._config, allow_synthetic=True)
        self._cache = cache or ForecastCache(ttl_seconds=self._config.cache_ttl_seconds)

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        payload = self._client.get_forecast(region, horizon_hours=horizon_hours)
        pts = WattTimeClient.normalize_forecast_payload(payload)
        self._cache.set(region, pts)
        return pts

    def get_moer(
        self,
        region: str,
        *,
        lbar_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        pts = self.get_forecast(region)
        start = window_start or datetime.now(UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        end = start + timedelta(hours=max(lbar_hours, 1e-6))
        return time_weighted_moer(pts, start, end)

    def list_regions(self) -> list[str]:
        return list(DEFAULT_CARBON_REGIONS)


class ApiCarbonProvider:
    """Proxy carbon forecasts through CarbonSight API (central credential).

    CLI sets CARBONSIGHT_API_URL=http://localhost:8001 — no personal WattTime creds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 15.0,
        cache: ForecastCache | None = None,
    ) -> None:
        cfg = Config.from_env()
        self._base = (base_url or cfg.carbonsight_api_url or "").rstrip("/")
        self._timeout = timeout
        self._cache = cache or ForecastCache(ttl_seconds=cfg.cache_ttl_seconds)
        self._fallback = SyntheticCarbonProvider()

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        if not self._base:
            return self._fallback.get_forecast(region, horizon_hours=horizon_hours)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.get(
                    f"{self._base}/v1/carbon/forecast",
                    params={"region": region, "horizon_hours": horizon_hours},
                )
                r.raise_for_status()
                body = r.json()
            if isinstance(body, dict):
                pts = list(body.get("data") or body.get("points") or [])
            elif isinstance(body, list):
                pts = body
            else:
                pts = []
            if not pts:
                return self._fallback.get_forecast(region, horizon_hours=horizon_hours)
            self._cache.set(region, pts)
            return pts
        except Exception:
            return self._fallback.get_forecast(region, horizon_hours=horizon_hours)

    def get_moer(
        self,
        region: str,
        *,
        lbar_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        pts = self.get_forecast(region)
        start = window_start or datetime.now(UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        end = start + timedelta(hours=max(lbar_hours, 1e-6))
        return time_weighted_moer(pts, start, end)

    def list_regions(self) -> list[str]:
        if not self._base:
            return self._fallback.list_regions()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.get(f"{self._base}/v1/regions")
                r.raise_for_status()
                body = r.json()
            if isinstance(body, list) and body:
                # API may return cloud regions or WT regions
                out: list[str] = []
                for item in body:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        code = item.get("wt_region") or item.get("region_code") or item.get("region")
                        if code:
                            out.append(str(code))
                return out or self._fallback.list_regions()
        except Exception:
            pass
        return self._fallback.list_regions()


def get_carbon_provider(config: Config | None = None) -> CarbonIntensityProvider:
    """Factory: CARBONSIGHT_API_URL > WATTTIME_USERNAME > synthetic."""
    cfg = config or Config.from_env()
    api_url = cfg.carbonsight_api_url or os.environ.get("CARBONSIGHT_API_URL", "")
    if api_url.strip():
        return ApiCarbonProvider(base_url=api_url.strip())
    if cfg.watttime_username and cfg.watttime_password:
        return WattTimeCarbonProvider(config=cfg, cache=get_global_forecast_cache(cfg.cache_ttl_seconds))
    return SyntheticCarbonProvider()
