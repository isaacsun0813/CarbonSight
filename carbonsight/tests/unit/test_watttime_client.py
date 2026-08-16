"""WattTime client: credentials, token cache, retries, endpoints, and unit gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from carbonsight_core.config import Config
from carbonsight_core.watttime import (
    LB_TO_KG,
    SUPPORTED_MOER_UNIT,
    WattTimeClient,
    WattTimeError,
)

FORECAST_POINT = {"point_time": "2026-01-01T00:00:00Z", "value": 321.0}


def _config_with_credentials(**overrides: object) -> Config:
    defaults: dict[str, object] = {
        "watttime_username": "user",
        "watttime_password": "secret",
        "max_retries": 2,
    }
    defaults.update(overrides)
    return Config(**defaults)  # type: ignore[arg-type]


def _config_without_credentials() -> Config:
    return Config(watttime_username="", watttime_password="")


class _RecordingTransport:
    """MockTransport factory that counts requests by path across many httpx clients."""

    def __init__(self, responses: dict[str, list[httpx.Response]]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.path)
        queued = self._responses.get(request.url.path)
        if not queued:
            return httpx.Response(404, json={"error": f"unstubbed {request.url.path}"})
        return queued.pop(0) if len(queued) > 1 else queued[0]

    def new_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def count(self, path: str) -> int:
        return sum(1 for call in self.calls if call == path)


def _login_ok() -> httpx.Response:
    return httpx.Response(200, json={"token": "tok-1"})


# --- credentials gate --------------------------------------------------------


def test_has_credentials_reflects_config() -> None:
    assert WattTimeClient(_config_with_credentials()).has_credentials is True
    assert WattTimeClient(_config_without_credentials()).has_credentials is False


def test_region_from_loc_has_no_synthetic_fallback() -> None:
    """Drift checks must never compare stored regions against invented ones."""
    client = WattTimeClient(_config_without_credentials())
    with pytest.raises(WattTimeError, match="credentials not configured"):
        client.region_from_loc(38.9, -77.5)


def test_forecast_without_credentials_raises() -> None:
    client = WattTimeClient(_config_without_credentials())
    with pytest.raises(WattTimeError, match="credentials not configured"):
        client.get_forecast("CAISO_NORTH")


def test_historical_without_credentials_raises() -> None:
    client = WattTimeClient(_config_without_credentials())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(WattTimeError, match="credentials not configured"):
        client.get_historical("CAISO_NORTH", start, start + timedelta(hours=2))


# --- token cache / 401 / 429 -------------------------------------------------


def test_login_uses_documented_get_with_basic_auth() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/login":
            assert request.method == "GET"
            assert request.headers.get("Authorization", "").startswith("Basic ")
            return _login_ok()
        return httpx.Response(200, json={"regions": []})

    client = WattTimeClient(_config_with_credentials())
    client.my_access(client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_login_token_is_cached_across_calls() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [
                httpx.Response(200, json={"data": [FORECAST_POINT], "units": SUPPORTED_MOER_UNIT})
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    client.get_forecast("PJM_DC", client=transport.new_client())
    client.get_forecast("PJM_DC", client=transport.new_client())

    assert transport.count("/v2/login") == 1
    assert transport.count("/v3/forecast") == 2


def test_login_is_repeated_when_token_caching_is_off() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [
                httpx.Response(200, json={"data": [FORECAST_POINT], "units": SUPPORTED_MOER_UNIT})
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials(cache_tokens=False))
    client.get_forecast("PJM_DC", client=transport.new_client())
    client.get_forecast("PJM_DC", client=transport.new_client())

    assert transport.count("/v2/login") == 2


def test_login_without_token_in_response_raises() -> None:
    transport = _RecordingTransport({"/v2/login": [httpx.Response(200, json={})]})
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(WattTimeError, match="No token in login response"):
        client.my_access(client=transport.new_client())


def test_401_triggers_one_token_refresh_then_succeeds() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok(), httpx.Response(200, json={"token": "tok-2"})],
            "/v3/my-access": [
                httpx.Response(401, json={"error": "expired"}),
                httpx.Response(200, json={"regions": ["PJM_DC"]}),
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    access = client.my_access(client=transport.new_client())

    assert access == {"regions": ["PJM_DC"]}
    assert transport.count("/v2/login") == 2
    assert transport.count("/v3/my-access") == 2


def test_repeated_401_is_not_retried_forever() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/my-access": [httpx.Response(401, json={"error": "nope"})],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(httpx.HTTPStatusError):
        client.my_access(client=transport.new_client())

    # One refresh attempt only: the second 401 is surfaced, not looped on.
    assert transport.count("/v3/my-access") == 2


def test_429_backs_off_exponentially_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(
        "carbonsight_core.watttime.client.time.sleep", lambda seconds: slept.append(seconds)
    )
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/my-access": [
                httpx.Response(429, json={"error": "slow down"}),
                httpx.Response(429, json={"error": "slow down"}),
                httpx.Response(200, json={"regions": []}),
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials(max_retries=3))
    assert client.my_access(client=transport.new_client()) == {"regions": []}
    assert slept == [1, 2]


def test_429_beyond_max_retries_surfaces_the_last_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("carbonsight_core.watttime.client.time.sleep", lambda seconds: None)
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/my-access": [httpx.Response(429, json={"error": "slow down"})],
        }
    )
    client = WattTimeClient(_config_with_credentials(max_retries=1))
    with pytest.raises(httpx.HTTPStatusError):
        client.my_access(client=transport.new_client())
    assert transport.count("/v3/my-access") == 2


def test_zero_attempt_loop_raises_rather_than_returning_none() -> None:
    """max_retries below zero must not fall off the end of _request with no response."""
    transport = _RecordingTransport({"/v2/login": [_login_ok()]})
    client = WattTimeClient(_config_with_credentials(max_retries=-1))
    with pytest.raises(WattTimeError, match="No response from WattTime"):
        client.my_access(client=transport.new_client())


# --- unit gate ---------------------------------------------------------------


def test_unsupported_units_refuse_to_compute() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [
                httpx.Response(200, json={"data": [FORECAST_POINT], "units": "g_co2_per_kwh"})
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(WattTimeError, match="Unsupported unit"):
        client.get_forecast("PJM_DC", client=transport.new_client())


def test_missing_units_refuse_to_compute() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [httpx.Response(200, json={"data": [FORECAST_POINT]})],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(WattTimeError, match="Missing 'units' field"):
        client.get_forecast("PJM_DC", client=transport.new_client())


def test_units_may_be_carried_per_point() -> None:
    point_with_units = dict(FORECAST_POINT, units=SUPPORTED_MOER_UNIT)
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [httpx.Response(200, json={"data": [point_with_units]})],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    payload = client.get_forecast("PJM_DC", client=transport.new_client())
    assert payload["data"][0]["value"] == 321.0


def test_units_may_be_carried_in_v3_meta() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [
                httpx.Response(
                    200,
                    json={
                        "data": [FORECAST_POINT],
                        "meta": {"units": SUPPORTED_MOER_UNIT},
                    },
                )
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    payload = client.get_forecast("PJM_DC", client=transport.new_client())
    assert payload["data"][0]["value"] == 321.0


def test_forecast_regions_uses_my_access_endpoint_capabilities() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/my-access": [
                httpx.Response(
                    200,
                    json={
                        "signal_types": [
                            {
                                "signal_type": "co2_moer",
                                "regions": [
                                    {
                                        "region": "FORECASTABLE",
                                        "endpoints": [{"endpoint": "v3/forecast"}],
                                    },
                                    {
                                        "region": "HISTORICAL_ONLY",
                                        "endpoints": [{"endpoint": "v3/historical"}],
                                    },
                                ],
                            }
                        ]
                    },
                )
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    assert client.forecast_regions(client=transport.new_client()) == {"FORECASTABLE"}


def test_historical_applies_the_same_unit_gate() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/historical": [
                httpx.Response(200, json={"data": [FORECAST_POINT], "units": "g_co2_per_kwh"})
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(WattTimeError, match="Unsupported unit"):
        client.get_historical("PJM_DC", start, start + timedelta(hours=1), client=transport.new_client())


def _dead_network_client() -> httpx.Client:
    def _explode(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/login":
            return _login_ok()
        raise httpx.ConnectError("network down", request=request)

    return httpx.Client(transport=httpx.MockTransport(_explode))


def test_transport_failure_raises_rather_than_fabricating_a_ranking() -> None:
    """A configured client that fails must not silently serve synthetic MOER.

    The synthetic curve is hash-derived, not physical — it ranks Sweden dirtier than
    India. Serving it during a WattTime outage would hand the user a confident,
    inverted answer with nothing marking it as fake.
    """
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(httpx.ConnectError):
        client.get_forecast("PJM_DC", client=_dead_network_client())


def test_expired_credentials_raise_rather_than_fabricating_a_ranking() -> None:
    """Credentials that exist but no longer work are a failure, not a demo."""
    transport = _RecordingTransport({"/v2/login": [httpx.Response(403, json={"error": "expired"})]})
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(httpx.HTTPStatusError):
        client.get_forecast("PJM_DC", client=transport.new_client())


def test_http_error_from_forecast_raises() -> None:
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [httpx.Response(500, json={"error": "boom"})],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    with pytest.raises(httpx.HTTPStatusError):
        client.get_forecast("PJM_DC", client=transport.new_client())


# --- small helpers -----------------------------------------------------------


def test_normalize_forecast_payload_accepts_both_shapes() -> None:
    normalize = WattTimeClient.normalize_forecast_payload
    assert normalize({"data": [FORECAST_POINT]}) == [FORECAST_POINT]
    assert normalize([FORECAST_POINT]) == [FORECAST_POINT]
    assert normalize({}) == []
    assert normalize({"data": None}) == []


def test_moer_to_kg_uses_the_pound_conversion() -> None:
    # 2000 lb CO2 in kg, hardcoded: deriving it from LB_TO_KG would pass for any constant.
    assert WattTimeClient.moer_lb_per_mwh_to_kg_co2(1000.0, 2.0) == pytest.approx(907.18474)
    assert LB_TO_KG == pytest.approx(0.45359237)


# --- flat-module import surface ----------------------------------------------


def test_package_still_exports_the_flat_module_surface() -> None:
    """`carbonsight_core.watttime` was a module; every old name must survive the package."""
    import carbonsight_core.watttime as watttime_package

    for name in ("WattTimeClient", "WattTimeError", "LB_TO_KG", "SUPPORTED_MOER_UNIT"):
        assert hasattr(watttime_package, name), name


def test_every_call_site_import_still_resolves() -> None:
    """Mirror of the imports in core, CLI, and API modules that predate the package."""
    from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
    from carbonsight_core.mapping.refresh import refresh_registry_mappings
    from carbonsight_core.mapping.validate import validate_registry_mappings
    from carbonsight_core.region_ranking import AwsRegionRankingService

    assert all(
        callable(target)
        for target in (
            JobCarbonEstimator,
            refresh_registry_mappings,
            validate_registry_mappings,
            AwsRegionRankingService,
        )
    )
