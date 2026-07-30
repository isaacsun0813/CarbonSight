"""
CarbonIntensityProvider: dual-mode abstraction so everyone can schedule without WattTime creds.

Protocol:
- get_moer_lb_per_mwh(wt_regions, start, end) -> float  # time-weighted
- get_kg_per_hr(job, wt_regions, start, end) -> float
- get_cost_per_hr(..., carbon_price_usd_per_ton) -> float

Impls:
- WattTimeCarbonProvider: direct WattTimeClient + ForecastCache (server side, ONE credential)
- ApiCarbonProvider: calls CarbonSight API GET /v1/carbon/forecast (client side, NO WattTime creds)
- SyntheticCarbonProvider: 400 lb/MWh fallback for CI/replay/no-creds

Factory: prefers CARBONSIGHT_API_URL if set, else WATTTIME creds, else synthetic.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import httpx

from carbonsight_core.estimator.power_model import sample_power_params
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime import LB_TO_KG, WattTimeClient
from carbonsight_core.watttime.cache import ForecastCache, _time_weighted_moer

FALLBACK_MOER = 400.0


@runtime_checkable
class CarbonIntensityProvider(Protocol):
    def get_moer_lb_per_mwh(
        self,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float: ...

    def get_kg_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float: ...

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float: ...


def _facility_mwh_per_hour(job: JobSpec, pue: float = 1.2) -> float:
    """Mean facility MWh per hour using sampled power params (fixed seed for determinism)."""
    import random

    rng = random.Random(0)
    params = sample_power_params(job, rng)
    # sample_power_params returns P_IT and PUE, we use mean facility MWh per hour
    p_it_w = params.p_it_w
    pue_val = params.pue
    # same as carbon_model _facility_mwh_per_hour * 1h
    return (p_it_w / 1000.0) * 1.0 * (pue_val / 1000.0)


class WattTimeCarbonProvider:
    """Direct WattTime mode — holds ONE credential server-side, uses ForecastCache."""

    def __init__(self, client: WattTimeClient | None = None, ttl_seconds: int = 900) -> None:
        self._client = client or WattTimeClient()
        self._cache = ForecastCache(self._client, ttl_seconds=ttl_seconds)

    def get_moer_lb_per_mwh(
        self,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        return self._cache.get_time_weighted_moer(wt_regions, start, end)

    def get_kg_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        moer = self.get_moer_lb_per_mwh(wt_regions, start, end)
        mwh_per_hr = _facility_mwh_per_hour(job)
        return mwh_per_hr * moer * LB_TO_KG

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        kg_per_hr = self.get_kg_per_hr(job, wt_regions, start, end)
        return (kg_per_hr / 1000.0) * carbon_price_usd_per_ton * carbon_weight


class ApiCarbonProvider:
    """
    Proxy mode — CLI has NO WattTime creds, just CARBONSIGHT_API_URL.
    Calls GET /v1/carbon/forecast?wt_region=...&start=&end=&horizon_hours=
    and does time-weighting locally using same logic.
    """

    def __init__(self, api_url: str, timeout: int = 15) -> None:
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout

    def _fetch_points(self, wt_region: str, horizon_hours: int = 72) -> list[dict]:
        url = f"{self._api_url}/v1/carbon/forecast"
        params = {"wt_region": wt_region, "horizon_hours": horizon_hours}
        try:
            with httpx.Client(timeout=self._timeout) as c:
                r = c.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                # API returns {"wt_region":..., "points": [...] } or {"data": [...]}
                if isinstance(data, dict):
                    if "points" in data:
                        return data["points"]
                    if "data" in data:
                        return data["data"]
                if isinstance(data, list):
                    return data
                return []
        except Exception:
            return []

    def get_moer_lb_per_mwh(
        self,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        if not wt_regions:
            return FALLBACK_MOER
        total = 0.0
        total_weight = 0.0
        for region, weight in wt_regions:
            points = self._fetch_points(region)
            if not points:
                moer = FALLBACK_MOER
            else:
                # Local time-weighting, same function as cache
                moer = _time_weighted_moer(points, start, end)
            total += weight * moer
            total_weight += weight
        if total_weight == 0:
            return FALLBACK_MOER
        if abs(total_weight - 1.0) > 1e-6:
            total = total / total_weight if total_weight else FALLBACK_MOER
        return total

    def get_kg_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        moer = self.get_moer_lb_per_mwh(wt_regions, start, end)
        mwh_per_hr = _facility_mwh_per_hour(job)
        return mwh_per_hr * moer * LB_TO_KG

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        kg_per_hr = self.get_kg_per_hr(job, wt_regions, start, end)
        return (kg_per_hr / 1000.0) * carbon_price_usd_per_ton * carbon_weight


class SyntheticCarbonProvider:
    """Fallback for CI/replay/no-creds — deterministic 400 lb/MWh."""

    def __init__(self, fallback_moer: float = FALLBACK_MOER) -> None:
        self._fallback = fallback_moer

    def get_moer_lb_per_mwh(
        self,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        return self._fallback

    def get_kg_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
    ) -> float:
        mwh_per_hr = _facility_mwh_per_hour(job)
        return mwh_per_hr * self._fallback * LB_TO_KG

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        kg_per_hr = self.get_kg_per_hr(job, wt_regions, start, end)
        return (kg_per_hr / 1000.0) * carbon_price_usd_per_ton * carbon_weight


def get_carbon_provider(
    api_url: str | None = None,
    watt_client: WattTimeClient | None = None,
) -> CarbonIntensityProvider:
    """
    Factory: prefers CARBONSIGHT_API_URL, then WattTime creds, then synthetic.

    Env:
      CARBONSIGHT_API_URL=https://api.carbonsight.internal
      WATTTIME_USERNAME/PASSWORD for direct mode
    """
    api_url = api_url or os.environ.get("CARBONSIGHT_API_URL", "").strip()
    if api_url:
        return ApiCarbonProvider(api_url)

    # Check if WattTime creds exist
    wt_user = os.environ.get("WATTTIME_USERNAME", "").strip()
    wt_pass = os.environ.get("WATTTIME_PASSWORD", "").strip()
    if wt_user and wt_pass:
        return WattTimeCarbonProvider(client=watt_client)

    if watt_client is not None:
        # If caller passed explicit client, use it even without env (tests)
        return WattTimeCarbonProvider(client=watt_client)

    return SyntheticCarbonProvider()
