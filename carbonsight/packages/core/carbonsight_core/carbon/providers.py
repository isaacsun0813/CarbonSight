"""The three carbon-intensity sources, and the factory that picks one.

Preference order (central-credential design):

  1. ``CARBONSIGHT_API_URL``  -> :class:`ApiCarbonProvider`     CLI needs no personal creds
  2. ``WATTTIME_USERNAME``    -> :class:`WattTimeCarbonProvider` direct, one login per user
  3. neither                  -> :class:`SyntheticCarbonProvider` offline demo

**On fabrication.** Synthetic curves are a *demo* affordance, not a *fallback*. They
are deterministic hashes of the region name, so they are arbitrary with respect to
real grids — Sweden can score dirtier than India. Serving them when a configured
source fails would hand the user a confident, physically inverted ranking with
nothing marking it fake. So a configured-but-broken source raises; only the genuine
"nothing is configured" case is answered synthetically, and it is labelled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from carbonsight_core.carbon.base import ForecastBackedProvider, as_utc
from carbonsight_core.config import Config
from carbonsight_core.watttime import (
    SYNTHETIC_WATTTIME_REGIONS,
    ForecastCache,
    WattTimeClient,
    build_synthetic_forecast,
    get_global_forecast_cache,
    synthetic_moer_for_region,
)


class CarbonProviderError(RuntimeError):
    """A configured carbon source could not be reached or returned nothing usable."""


class SyntheticCarbonProvider(ForecastBackedProvider):
    """Deterministic per-region MOER curves for tests and no-credentials demos.

    Curves differ region to region (see ``synthetic_moer_for_region``) so the carbon
    lever is visible in the default demo rather than flat everywhere. The values are
    *not* physically meaningful — see the module docstring.
    """

    def __init__(self, regions: list[str] | None = None) -> None:
        self._regions = list(regions or SYNTHETIC_WATTTIME_REGIONS)
        self._cache = ForecastCache()

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        # Key on the anchor too: the curve is a function of (region, window), so a
        # pinned `now` yields the same series on any calendar date.
        cache_key = region if anchor is None else f"{region}@{as_utc(anchor).isoformat()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        points = build_synthetic_forecast(region, horizon_hours=horizon_hours, now=anchor)
        self._cache.set(cache_key, points)
        return points

    def list_regions(self) -> list[str]:
        return list(self._regions)

    def point_moer(self, region: str) -> float:
        """Instantaneous synthetic MOER, no time weighting."""
        return synthetic_moer_for_region(region)


class WattTimeCarbonProvider(ForecastBackedProvider):
    """Direct WattTime access plus a local :class:`ForecastCache`.

    ``allow_synthetic=False`` on the client: a credentialled provider that cannot
    reach WattTime must fail, not invent numbers.
    """

    def __init__(
        self,
        client: WattTimeClient | None = None,
        cache: ForecastCache | None = None,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._client = client or WattTimeClient(self._config, allow_synthetic=False)
        self._cache = cache or ForecastCache(
            ttl_seconds=self._config.forecast_cache_ttl_seconds
        )

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        del anchor  # live data is whatever the grid reported; the window is applied later
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        payload = self._client.get_forecast(region, horizon_hours=horizon_hours)
        points = WattTimeClient.normalize_forecast_payload(payload)
        self._cache.set(region, points)
        return points

    def list_regions(self) -> list[str]:
        return list(SYNTHETIC_WATTTIME_REGIONS)


class ApiCarbonProvider(ForecastBackedProvider):
    """Proxy forecasts through a CarbonSight API holding one central credential.

    The CLI sets ``CARBONSIGHT_API_URL`` and needs no WattTime login of its own.
    If the server is configured but unreachable, this raises rather than serving
    synthetic data — see the module docstring.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 15.0,
        cache: ForecastCache | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config.from_env()
        self._base_url = (base_url or cfg.carbonsight_api_url or "").rstrip("/")
        if not self._base_url:
            raise ValueError("ApiCarbonProvider needs a base URL (CARBONSIGHT_API_URL)")
        self._timeout = timeout
        self._cache = cache or ForecastCache(ttl_seconds=cfg.forecast_cache_ttl_seconds)
        # What the server said each forecast's origin was ("db"/"live"/"cache"/
        # "synthetic"). Without this a caller cannot tell a real reading from a
        # server-side demo curve.
        self.sources: dict[str, str] = {}

    def last_source(self, region: str) -> str | None:
        """Provenance of the most recent forecast for ``region``."""
        return self.sources.get(region)

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        del anchor  # the server decides the series; the window is applied later
        cached = self._cache.get(region)
        if cached is not None:
            return cached
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    f"{self._base_url}/v1/carbon/forecast",
                    params={"region": region, "horizon_hours": horizon_hours},
                )
                response.raise_for_status()
                body = response.json()
        except Exception as err:
            raise CarbonProviderError(
                f"CarbonSight API at {self._base_url} is unreachable: {err}"
            ) from err

        if isinstance(body, dict):
            points = list(body.get("data") or body.get("points") or [])
            source = str(body.get("source") or "api")
        elif isinstance(body, list):
            points, source = body, "api"
        else:
            points, source = [], "api"

        if not points:
            raise CarbonProviderError(
                f"CarbonSight API returned no forecast points for {region}"
            )
        self.sources[region] = source
        self._cache.set(region, points)
        return points

    def list_regions(self) -> list[str]:
        """Grid regions the server can serve MOER for.

        Hits ``/v1/carbon/regions``, not ``/v1/regions``. The two are different
        namespaces and the codes are not interchangeable: ``/v1/regions`` returns
        *cloud* regions (``us-east-1``) from the mapping registry, while this
        provider deals in *grid* regions (``CAISO_NORTH``). Asking the wrong one
        returns plausible-looking strings that no MOER lookup will ever match.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(f"{self._base_url}/v1/carbon/regions")
                response.raise_for_status()
                body = response.json()
        except Exception as err:
            raise CarbonProviderError(
                f"CarbonSight API at {self._base_url} is unreachable: {err}"
            ) from err

        entries = body.get("regions", []) if isinstance(body, dict) else body
        regions: list[str] = []
        for item in entries if isinstance(entries, list) else ():
            if isinstance(item, str):
                regions.append(item)
            elif isinstance(item, dict) and item.get("wt_region"):
                regions.append(str(item["wt_region"]))
        if not regions:
            # Do NOT fall back to the built-in list. A configured server that
            # answers with nothing we recognise is a misconfiguration — most
            # likely pointed at the wrong endpoint — and quietly substituting
            # the local list makes that invisible and unfixable.
            raise CarbonProviderError(
                f"{self._base_url}/v1/carbon/regions returned no recognisable grid "
                "regions. Expected objects with a 'wt_region' key.",
            )
        return regions


def get_carbon_provider(config: Config | None = None) -> ForecastBackedProvider:
    """Pick a source: API URL > WattTime credentials > synthetic."""
    cfg = config or Config.from_env()
    if cfg.carbonsight_api_url:
        return ApiCarbonProvider(config=cfg)
    if cfg.watttime_username and cfg.watttime_password:
        return WattTimeCarbonProvider(
            config=cfg,
            cache=get_global_forecast_cache(cfg.forecast_cache_ttl_seconds),
        )
    return SyntheticCarbonProvider()
