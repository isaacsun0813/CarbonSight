"""
WattTime v3 API client.
Design doc: /login (basic auth), token cache, /v3/my-access, /v3/region-from-loc,
/v3/forecast, /v3/historical, /v3/signal-index. Units: lbs_co2_per_mwh. 401 refresh, 429 backoff.

Forecast and historical calls also work with no credentials at all: they fall back to a
per-region deterministic synthetic MOER curve so demos and tests never require secrets.
The unit gate is never bypassed by that fallback — a WattTimeError always propagates.
"""

import hashlib
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from carbonsight_core.config import Config

WATTTIME_BASE = "https://api.watttime.org"
SUPPORTED_MOER_UNIT = "lbs_co2_per_mwh"
LB_TO_KG = 0.45359237

# WattTime grid codes for the 17 cloud regions that carry GPU capacity; used to seed
# synthetic forecasts when no credentials are configured.
SYNTHETIC_WATTTIME_REGIONS: tuple[str, ...] = (
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
)

SYNTHETIC_BASE_MOER_LB_PER_MWH = 200.0
SYNTHETIC_MOER_SPREAD_LB_PER_MWH = 850
SYNTHETIC_DIURNAL_AMPLITUDE_LB_PER_MWH = 30.0
SYNTHETIC_MIN_MOER_LB_PER_MWH = 10.0


class WattTimeError(Exception):
    """WattTime API or usage error."""

    pass


def synthetic_moer_for_region(
    region: str,
    *,
    base_lb_per_mwh: float = SYNTHETIC_BASE_MOER_LB_PER_MWH,
) -> float:
    """
    Deterministic stand-in MOER (lb/MWh) derived from the region name.

    The spread is deliberately wide (~200..1050 lb/MWh) rather than one flat constant:
    a flat value would make the carbon lever invisible when ranking regions without
    credentials. The same region always yields the same value.
    """
    digest = hashlib.md5(region.encode("utf-8"), usedforsecurity=False).hexdigest()
    region_offset = int(digest[:8], 16) % SYNTHETIC_MOER_SPREAD_LB_PER_MWH
    return float(base_lb_per_mwh + region_offset)


def build_synthetic_forecast(
    region: str,
    *,
    horizon_hours: int = 24,
    step_minutes: int = 5,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a synthetic MOER point series for demos and the no-credentials fallback."""
    series_start = now or datetime.now(UTC)
    if series_start.tzinfo is None:
        series_start = series_start.replace(tzinfo=UTC)
    base_lb_per_mwh = synthetic_moer_for_region(region)
    point_count = max(1, int(horizon_hours * 60 / step_minutes))
    points: list[dict[str, Any]] = []
    for index in range(point_count):
        # Mild diurnal wave so time-weighted averages are not trivially constant.
        wave_lb_per_mwh = SYNTHETIC_DIURNAL_AMPLITUDE_LB_PER_MWH * ((index % 12) / 12.0 - 0.5)
        point_time = series_start + timedelta(minutes=step_minutes * index)
        points.append(
            {
                "point_time": point_time.isoformat().replace("+00:00", "Z"),
                "value": max(SYNTHETIC_MIN_MOER_LB_PER_MWH, base_lb_per_mwh + wave_lb_per_mwh),
                "units": SUPPORTED_MOER_UNIT,
            }
        )
    return points


def build_synthetic_forecast_payload(
    region: str,
    *,
    horizon_hours: int = 24,
    step_minutes: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Synthetic forecast wrapped in the same envelope the live /v3/forecast returns."""
    return {
        "data": build_synthetic_forecast(
            region, horizon_hours=horizon_hours, step_minutes=step_minutes, now=now
        ),
        "units": SUPPORTED_MOER_UNIT,
        "meta": {"region": region, "source": "synthetic"},
    }


class WattTimeClient:
    """WattTime v3 client with login, token cache, MOER endpoints, and synthetic fallback."""

    def __init__(self, config: Config | None = None, *, allow_synthetic: bool = True) -> None:
        self._config = config or Config.from_env()
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._timeout = self._config.request_timeout
        self._max_retries = self._config.max_retries
        self._allow_synthetic = allow_synthetic

    @property
    def has_credentials(self) -> bool:
        """True when both WattTime username and password are configured."""
        return bool(self._config.watttime_username and self._config.watttime_password)

    @property
    def allow_synthetic(self) -> bool:
        """True when this client may substitute a synthetic MOER curve for live data."""
        return self._allow_synthetic

    def _ensure_token(self, http_client: httpx.Client) -> None:
        if not self.has_credentials:
            raise WattTimeError(
                "WattTime credentials not configured "
                "(set WATTTIME_USERNAME and WATTTIME_PASSWORD)"
            )
        if self._config.cache_tokens and self._token and time.time() < self._token_expires_at:
            return
        # WattTime v2 login returns token; v3 may differ - use v2 login per common docs
        login_response = http_client.post(
            f"{WATTTIME_BASE}/v2/login",
            auth=(self._config.watttime_username, self._config.watttime_password),
            timeout=self._timeout,
        )
        login_response.raise_for_status()
        self._token = login_response.json().get("token")
        if not self._token:
            raise WattTimeError("No token in login response")
        # Token ~30 min; cache for 25 min to be safe
        self._token_expires_at = time.time() + 25 * 60

    def _request(
        self,
        method: str,
        path: str,
        http_client: httpx.Client,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        retry_on_401: bool = True,
    ) -> httpx.Response:
        self._ensure_token(http_client)
        url = f"{WATTTIME_BASE}{path}" if path.startswith("/") else f"{WATTTIME_BASE}/{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            response = http_client.request(
                method, url, headers=headers, params=params, json=json, timeout=self._timeout
            )
            if response.status_code == 401 and retry_on_401:
                self._token = None
                self._token_expires_at = 0.0
                self._ensure_token(http_client)
                headers["Authorization"] = f"Bearer {self._token}"
                retry_on_401 = False
                continue
            if response.status_code == 429:
                time.sleep(2**attempt)
                continue
            return response
        if response is None:
            raise WattTimeError(f"No response from WattTime for {method} {path}")
        return response

    def my_access(self, client: httpx.Client | None = None) -> dict[str, Any]:
        """GET /v3/my-access: list regions, signal types, models."""
        with client or httpx.Client() as http_client:
            response = self._request("GET", "/v3/my-access", http_client)
            response.raise_for_status()
            return response.json()

    def region_from_loc(
        self,
        latitude: float,
        longitude: float,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/region-from-loc?signal_type=co2_moer&latitude=...&longitude=...

        Deliberately has no synthetic fallback: mapping drift checks must never compare
        stored WattTime regions against invented ones.
        """
        with client or httpx.Client() as http_client:
            response = self._request(
                "GET",
                "/v3/region-from-loc",
                http_client,
                params={"signal_type": signal_type, "latitude": latitude, "longitude": longitude},
            )
            response.raise_for_status()
            return response.json()

    def get_forecast(
        self,
        region: str,
        signal_type: str = "co2_moer",
        horizon_hours: int = 24,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/forecast. Returns data with units (must be lbs_co2_per_mwh for kg).

        Without credentials — or on a transport failure when ``allow_synthetic`` is set —
        returns a synthetic payload in the same ``{"data": [...], "units": ...}`` envelope,
        so callers never branch on where the forecast came from.
        """
        if not self.has_credentials:
            if not self._allow_synthetic:
                raise WattTimeError("WattTime credentials not configured")
            return build_synthetic_forecast_payload(region, horizon_hours=horizon_hours)

        try:
            with client or httpx.Client() as http_client:
                response = self._request(
                    "GET",
                    "/v3/forecast",
                    http_client,
                    params={
                        "region": region,
                        "signal_type": signal_type,
                        "horizon_hours": horizon_hours,
                    },
                )
                response.raise_for_status()
                forecast_payload = response.json()
                self._assert_moer_units(forecast_payload)
                return forecast_payload
        except WattTimeError:
            # The unit gate fails closed; synthetic data must never paper over it.
            raise
        except Exception:
            if self._allow_synthetic:
                return build_synthetic_forecast_payload(region, horizon_hours=horizon_hours)
            raise

    def get_historical(
        self,
        region: str,
        start: datetime,
        end: datetime,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]:
        """GET /v3/historical for a time window.

        Returns a list of ``{"point_time": str, "value": float}`` dicts with MOER
        values in lbs CO₂ per MWh, ordered by ``point_time`` ascending. Without
        credentials this is a synthetic series spanning the same window.
        """
        if not self.has_credentials:
            if not self._allow_synthetic:
                raise WattTimeError("WattTime credentials not configured")
            window_hours = max(1, int((end - start).total_seconds() / 3600))
            return build_synthetic_forecast(region, horizon_hours=window_hours, now=start)

        with client or httpx.Client() as http_client:
            response = self._request(
                "GET",
                "/v3/historical",
                http_client,
                params={
                    "region": region,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "signal_type": signal_type,
                },
            )
            response.raise_for_status()
            historical_payload = response.json()
            self._assert_moer_units(historical_payload)
            return historical_payload.get("data", [])

    def get_signal_index(
        self,
        region: str,
        signal_type: str = "co2_moer",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """GET /v3/signal-index. Fallback for ranking; percentile, not raw MOER."""
        with client or httpx.Client() as http_client:
            response = self._request(
                "GET",
                "/v3/signal-index",
                http_client,
                params={"region": region, "signal_type": signal_type},
            )
            response.raise_for_status()
            return response.json()

    def _assert_moer_units(self, payload: dict[str, Any]) -> None:
        """Fail closed if MOER data does not have supported units for kg CO2."""
        units: str | None = payload.get("units")
        if units is None and payload.get("data"):
            units = payload["data"][0].get("units") if payload["data"] else None
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
        """Normalize a forecast envelope or a bare point list into a list of points."""
        if isinstance(payload, list):
            return list(payload)
        return list(payload.get("data") or [])
