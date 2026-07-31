"""
WattTime v3 API client.

Auth: username/password -> bearer token (v2 login), then
GET /v3/forecast?region=...&signal_type=co2_moer -> MOER list (point_time, value lb/MWh).

Works without credentials via synthetic fallback so tests and demos never require secrets.
"""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from carbonsight_core.config import Config

WATTTIME_BASE = "https://api.watttime.org"
SUPPORTED_MOER_UNIT = "lbs_co2_per_mwh"
LB_TO_KG = 0.45359237

# 17 common AWS GPU regions' WattTime grid codes used for synthetic forecasts
SYNTHETIC_WT_REGIONS: list[str] = [
    "PJM_DC",
    "PJM_EASTERN_OH",
    "CAISO_NORTH",
    "PACW",
    "HQ",
    "IE",
    "UK",
    "FR",
    "DE",
    "SE",
    "IT",
    "JP_TK",
    "KOR",
    "SGP",
    "NEM_NSW",
    "IND",
    "BRA",
]


class WattTimeError(Exception):
    """WattTime API or usage error."""


def synthetic_moer_for_region(region: str, *, base: float = 200.0) -> float:
    """Deterministic fake MOER (lb/MWh) from region name. Spread is large for lever tests."""
    h = int(hashlib.md5(region.encode("utf-8")).hexdigest()[:8], 16)
    # Spread ~50..900 so green vs dirty ranking flips are visible under high carbon price
    return float(base + (h % 850))


def build_synthetic_forecast(
    region: str,
    *,
    horizon_hours: int = 24,
    step_minutes: int = 5,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a synthetic MOER series for tests / no-creds fallback."""
    t0 = now or datetime.now(UTC)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=UTC)
    base = synthetic_moer_for_region(region)
    points: list[dict[str, Any]] = []
    n = max(1, int(horizon_hours * 60 / step_minutes))
    for i in range(n):
        # Mild diurnal wave so time-weighted averages are non-trivial
        wave = 30.0 * ((i % 12) / 12.0 - 0.5)
        pt = t0 + timedelta(minutes=step_minutes * i)
        points.append(
            {
                "point_time": pt.isoformat().replace("+00:00", "Z"),
                "value": max(10.0, base + wave),
                "units": SUPPORTED_MOER_UNIT,
            }
        )
    return points


class WattTimeClient:
    """WattTime v3 client with login, token cache, MOER endpoints, and synthetic fallback."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        allow_synthetic: bool = True,
    ) -> None:
        self._config = config or Config.from_env()
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._timeout = self._config.request_timeout
        self._max_retries = self._config.max_retries
        self._allow_synthetic = allow_synthetic

    @property
    def has_credentials(self) -> bool:
        return bool(self._config.watttime_username and self._config.watttime_password)

    def _ensure_token(self, client: httpx.Client) -> None:
        if not self.has_credentials:
            raise WattTimeError("WattTime credentials not configured")
        if self._config.cache_tokens and self._token and time.time() < self._token_expires_at:
            return
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
        r: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            r = client.request(
                method, url, headers=headers, params=params, json=json, timeout=self._timeout
            )
            if r.status_code == 401 and retry_on_401:
                self._token = None
                self._token_expires_at = 0.0
                self._ensure_token(client)
                headers["Authorization"] = f"Bearer {self._token}"
                retry_on_401 = False
                continue
            if r.status_code == 429:
                time.sleep(2**attempt)
                continue
            return r
        assert r is not None
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
                params={
                    "signal_type": signal_type,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            )
            r.raise_for_status()
            return r.json()

    def get_forecast(
        self,
        region: str,
        signal_type: str = "co2_moer",
        horizon_hours: int = 24,
        client: httpx.Client | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """GET /v3/forecast.

        Returns a dict with ``data`` list when talking to the real API.
        Without credentials (or on hard failure with allow_synthetic), returns a
        plain list of ``{point_time, value}`` for drop-in use by ForecastCache.
        """
        if not self.has_credentials:
            if not self._allow_synthetic:
                raise WattTimeError("WattTime credentials not configured")
            return build_synthetic_forecast(region, horizon_hours=horizon_hours)

        try:
            with client or httpx.Client() as c:
                r = self._request(
                    "GET",
                    "/v3/forecast",
                    c,
                    params={
                        "region": region,
                        "signal_type": signal_type,
                        "horizon_hours": horizon_hours,
                    },
                )
                r.raise_for_status()
                data = r.json()
                self._assert_moer_units(data)
                return data
        except Exception:
            if self._allow_synthetic:
                return build_synthetic_forecast(region, horizon_hours=horizon_hours)
            raise

    def get_historical(
        self,
        region: str,
        start: datetime,
        end: datetime,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]:
        """GET /v3/historical for a time window."""
        if not self.has_credentials:
            if not self._allow_synthetic:
                raise WattTimeError("WattTime credentials not configured")
            hours = max(1, int((end - start).total_seconds() / 3600) or 1)
            return build_synthetic_forecast(region, horizon_hours=hours, now=start)

        with client or httpx.Client() as c:
            r = self._request(
                "GET",
                "/v3/historical",
                c,
                params={
                    "region": region,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "signal_type": signal_type,
                },
            )
            r.raise_for_status()
            data = r.json()
            self._assert_moer_units(data)
            return data.get("data", [])

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
        units: str | None = data.get("units")
        if units is None and data.get("data"):
            units = data["data"][0].get("units") if data["data"] else None
        if units is None:
            raise WattTimeError(
                "Missing 'units' field in WattTime response; cannot verify CO2 unit safety"
            )
        if units != SUPPORTED_MOER_UNIT:
            raise WattTimeError(
                f"Unsupported unit '{units}'; expected '{SUPPORTED_MOER_UNIT}'. Refusing to compute."
            )

    @staticmethod
    def moer_lb_per_mwh_to_kg_co2(lb_per_mwh: float, facility_mwh: float) -> float:
        """CO2_kg = (MOER_lb_per_MWh * E_facility_MWh) * 0.45359237."""
        return lb_per_mwh * facility_mwh * LB_TO_KG

    @staticmethod
    def normalize_forecast_payload(
        payload: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize API dict or list into a plain list of forecast points."""
        if isinstance(payload, list):
            return payload
        return list(payload.get("data") or [])
