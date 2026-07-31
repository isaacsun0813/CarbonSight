"""
Carbon intensity providers — one protocol, three sources.

Every provider answers the same four questions:

    get_forecast(region)                       raw MOER points for one grid region
    get_moer_lb_per_mwh(wt_regions, s, e)      mixture- and time-weighted MOER
    get_kg_per_hr(job, wt_regions, s, e)       kgCO2/hr for *this* job's power draw
    get_cost_per_hr(job, wt_regions, s, e)     the same, dollarised

``wt_regions`` is the ``[(wt_region, weight)]`` mixture from the mapping registry,
so a cloud region straddling two grids is priced correctly. ``get_kg_per_hr``
runs the real power model (``estimator/power_model.sample_power_params``), so GPU
type, GPU count and ``--gpu-util`` all move the number.

Factory preference (central-credential design):
  1. CARBONSIGHT_API_URL  -> ApiCarbonProvider  (CLI needs no personal creds)
  2. WATTTIME_USERNAME    -> WattTimeCarbonProvider
  3. else                 -> SyntheticCarbonProvider
"""

from __future__ import annotations

import os
import random
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import httpx

from carbonsight_core.config import Config
from carbonsight_core.estimator.power_model import sample_power_params
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime.cache import (
    ForecastCache,
    get_global_forecast_cache,
    mixture_weighted_moer,
    time_weighted_moer,
)
from carbonsight_core.watttime.client import (
    LB_TO_KG,
    SYNTHETIC_WT_REGIONS,
    WattTimeClient,
    build_synthetic_forecast,
    synthetic_moer_for_region,
)

# Canonical list of grid regions used for synthetic rankings / API fallback
DEFAULT_CARBON_REGIONS: list[str] = list(SYNTHETIC_WT_REGIONS)
FALLBACK_MOER = 400.0


@runtime_checkable
class CarbonIntensityProvider(Protocol):
    """Mixture- and job-aware MOER access."""

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        """MOER points ``{point_time, value}`` (lb/MWh) for one grid region."""
        ...

    def get_moer_lb_per_mwh(
        self, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        """Mixture-weighted, time-weighted MOER over ``[start, end]``."""
        ...

    def get_kg_per_hr(
        self, job: JobSpec, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        """kgCO2 per wall-clock hour for this job in this grid mixture."""
        ...

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        """``get_kg_per_hr`` dollarised at the social cost of carbon."""
        ...

    def list_regions(self) -> list[str]:
        """Known grid region codes."""
        ...


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def facility_mwh_per_hr(job: JobSpec, *, samples: int = 64, seed: int = 0) -> float:
    """Mean facility MWh drawn per wall-clock hour: P_IT * PUE / 1e6.

    Averages ``samples`` draws from the same power model the estimator uses, on
    a fixed seed so rankings stay reproducible. A single draw is a sample of the
    utilisation and PUE distributions, not their mean — the previous one-shot
    version happened to sit near the 72nd percentile of PUE.

    ``job.gpu_utilization`` (``--gpu-util`` / ``--nvidia-smi``) pins the GPU term
    instead of sampling it, in which case only PUE and CPU load still vary.
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(samples):
        params = sample_power_params(job, rng)
        total += params.p_it_w * params.pue
    return total / samples / 1_000_000.0


class _ForecastBackedProvider:
    """Shared mixture/job math; subclasses only supply ``get_forecast``."""

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_regions(self) -> list[str]:
        return list(DEFAULT_CARBON_REGIONS)

    def get_moer_lb_per_mwh(
        self, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        series = [
            (points, weight)
            for points, weight in ((self.get_forecast(r), w) for r, w in wt_regions)
            if points
        ]
        if not series:
            return FALLBACK_MOER
        return mixture_weighted_moer(series, _utc(start), _utc(end)) or FALLBACK_MOER

    def get_kg_per_hr(
        self, job: JobSpec, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        moer = self.get_moer_lb_per_mwh(wt_regions, start, end)
        return facility_mwh_per_hr(job) * moer * LB_TO_KG

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        kg = self.get_kg_per_hr(job, wt_regions, start, end)
        return kg / 1000.0 * carbon_price_usd_per_ton * carbon_weight

    def get_moer(
        self,
        region: str,
        *,
        lbar_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        """Single-region convenience: time-weighted MOER over ``[t, t+Lbar]``."""
        start = _utc(window_start or datetime.now(UTC))
        end = start + timedelta(hours=max(lbar_hours, 1e-6))
        return time_weighted_moer(self.get_forecast(region), start, end)


class SyntheticCarbonProvider(_ForecastBackedProvider):
    """Deterministic per-region MOER curves for tests and no-creds demos.

    Curves differ region to region (see ``synthetic_moer_for_region``) so the
    carbon lever is visible in the default demo rather than flat everywhere.
    """

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

    def list_regions(self) -> list[str]:
        return list(self._regions)

    def point_moer(self, region: str) -> float:
        """Instantaneous synthetic MOER (no time weighting)."""
        return synthetic_moer_for_region(region)


class WattTimeCarbonProvider(_ForecastBackedProvider):
    """Direct WattTime access plus a local ForecastCache. Requires WATTTIME_*."""

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


class ApiCarbonProvider(_ForecastBackedProvider):
    """Proxy forecasts through the CarbonSight API (one central credential).

    The CLI sets ``CARBONSIGHT_API_URL=http://localhost:8001`` and needs no
    WattTime credentials of its own. Falls back to synthetic when the server is
    unreachable so the CLI degrades instead of failing.
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
        # Where the last forecast for each region actually came from, as reported
        # by the server ("db"/"live"/"cache"/"synthetic") or "local_synthetic"
        # when we never reached it. Without this a caller cannot tell a real
        # WattTime reading from a made-up curve.
        self.sources: dict[str, str] = {}

    def last_source(self, region: str) -> str | None:
        """Provenance of the most recent forecast for ``region``."""
        return self.sources.get(region)

    def get_forecast(self, region: str, *, horizon_hours: int = 24) -> list[dict[str, Any]]:
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        if not self._base:
            self.sources[region] = "local_synthetic"
            return self._fallback.get_forecast(region, horizon_hours=horizon_hours)
        source = "local_synthetic"
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
                source = str(body.get("source") or "api")
            elif isinstance(body, list):
                pts = body
                source = "api"
            else:
                pts = []
        except Exception:
            pts = []
        if not pts:
            self.sources[region] = "local_synthetic"
            return self._fallback.get_forecast(region, horizon_hours=horizon_hours)
        self.sources[region] = source
        self._cache.set(region, pts)
        return pts

    def list_regions(self) -> list[str]:
        if not self._base:
            return self._fallback.list_regions()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.get(f"{self._base}/v1/regions")
                r.raise_for_status()
                body = r.json()
            out: list[str] = []
            for item in body if isinstance(body, list) else ():
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    code = item.get("wt_region") or item.get("region_code") or item.get("region")
                    if code:
                        out.append(str(code))
            if out:
                return out
        except Exception:
            pass
        return self._fallback.list_regions()


def get_carbon_provider(config: Config | None = None) -> CarbonIntensityProvider:
    """Factory: CARBONSIGHT_API_URL > WATTTIME_USERNAME/PASSWORD > synthetic."""
    cfg = config or Config.from_env()
    api_url = (cfg.carbonsight_api_url or os.environ.get("CARBONSIGHT_API_URL", "")).strip()
    if api_url:
        return ApiCarbonProvider(base_url=api_url)
    if cfg.watttime_username and cfg.watttime_password:
        return WattTimeCarbonProvider(
            config=cfg, cache=get_global_forecast_cache(cfg.cache_ttl_seconds)
        )
    return SyntheticCarbonProvider()
