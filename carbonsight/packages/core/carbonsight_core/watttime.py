"""
WattTime v3 API client.
Design doc: /login (basic auth), token cache, /v3/my-access, /v3/region-from-loc,
/v3/forecast, /v3/historical, /v3/signal-index. Units: lbs_co2_per_mwh. 401 refresh, 429 backoff.
"""

import time
from typing import Any

import httpx

from carbonsight_core.config import Config

WATTTIME_BASE = "https://api.watttime.org"
SUPPORTED_MOER_UNIT = "lbs_co2_per_mwh"
LB_TO_KG = 0.45359237


class WattTimeError(Exception):
    """WattTime API or usage error."""

    pass


class WattTimeClient:
    """WattTime v3 client with login, token cache, and MOER endpoints."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config.from_env()
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._timeout = self._config.request_timeout
        self._max_retries = self._config.max_retries

    def _ensure_token(self, client: httpx.Client) -> None:
        if self._config.cache_tokens and self._token and time.time() < self._token_expires_at:
            return
        # WattTime v2 login returns token; v3 may differ - use v2 login per common docs
        r = client.post(
            f"{WATTTIME_BASE}/v2/login",
            auth=(self._config.watttime_username, self._config.watttime_password),
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        self._token = data.get("token")
        if not self._token:
            raise WattTimeError("No token in login response")
        # Token ~30 min; cache for 25 min to be safe
        self._token_expires_at = time.time() + 25 * 60

    def _request(
        self,
        method: str,
        path: str,
        client: httpx.Client,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        retry_on_401: bool = True,
    ) -> httpx.Response:
        self._ensure_token(client)
        url = f"{WATTTIME_BASE}{path}" if path.startswith("/") else f"{WATTTIME_BASE}/{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        for attempt in range(self._max_retries + 1):
            r = client.request(method, url, headers=headers, params=params, json=json, timeout=self._timeout)
            if r.status_code == 401 and retry_on_401:
                self._token = None
                self._token_expires_at = 0.0
                self._ensure_token(client)
                headers["Authorization"] = f"Bearer {self._token}"
                retry_on_401 = False
                continue
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return r
        return r

    def my_access(self, client: httpx.Client | None = None) -> dict[str, Any]:
        """GET /v3/my-access: list regions, signal types, models."""
        with client or httpx.Client() as c:
            r = self._request("GET", "/v3/my-access", c)
            r.raise_for_status()
            return r.json()

    def region_from_loc(
        self,
        latitude: float,
        longitude: float,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/region-from-loc?signal_type=co2_moer&latitude=...&longitude=..."""
        with client or httpx.Client() as c:
            r = self._request(
                "GET",
                "/v3/region-from-loc",
                c,
                params={"signal_type": signal_type, "latitude": latitude, "longitude": longitude},
            )
            r.raise_for_status()
            return r.json()

    def get_forecast(
        self,
        region: str,
        signal_type: str = "co2_moer",
        horizon_hours: int = 24,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/forecast. Returns data with units (must be lbs_co2_per_mwh for kg)."""
        with client or httpx.Client() as c:
            r = self._request(
                "GET",
                "/v3/forecast",
                c,
                params={"region": region, "signal_type": signal_type, "horizon_hours": horizon_hours},
            )
            r.raise_for_status()
            data = r.json()
            self._assert_moer_units(data)
            return data

    def get_signal_index(
        self,
        region: str,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/signal-index. Fallback for ranking; percentile, not raw MOER."""
        with client or httpx.Client() as c:
            r = self._request(
                "GET",
                "/v3/signal-index",
                c,
                params={"region": region, "signal_type": signal_type},
            )
            r.raise_for_status()
            return r.json()

    def _assert_moer_units(self, data: dict[str, Any]) -> None:
        """Fail closed if MOER data does not have supported units for kg CO2."""
        units = data.get("units") or (data.get("data", [{}]) and data["data"][0].get("units"))
        if units and units != SUPPORTED_MOER_UNIT:
            raise WattTimeError(
                f"Unsupported units for kg CO2: {units}. Required: {SUPPORTED_MOER_UNIT}. Refusing to compute."
            )

    @staticmethod
    def moer_lb_per_mwh_to_kg_co2(lb_per_mwh: float, facility_mwh: float) -> float:
        """CO2_kg = (MOER_lb_per_MWh * E_facility_MWh) * 0.45359237."""
        return lb_per_mwh * facility_mwh * LB_TO_KG
