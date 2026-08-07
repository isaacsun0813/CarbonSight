"""WattTime client: token cache, 401 refresh, 429 backoff, unit gate, synthetic fallback.

The synthetic path is what lets `carbonsight advise` show a carbon lever with no
credentials, so the tests below pin both halves of that promise: it must produce
materially different MOER per region, and it must never mask a unit-gate failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from carbonsight_core.config import Config
from carbonsight_core.watttime import (
    LB_TO_KG,
    SUPPORTED_MOER_UNIT,
    SYNTHETIC_WATTTIME_REGIONS,
    WattTimeClient,
    WattTimeError,
    build_synthetic_forecast,
    build_synthetic_forecast_payload,
    synthetic_moer_for_region,
)

FORECAST_POINT = {"point_time": "2026-01-01T00:00:00Z", "value": 321.0}


def _parse_point_time(point: dict) -> datetime:
    return datetime.fromisoformat(point["point_time"].replace("Z", "+00:00"))


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


def test_forecast_without_credentials_and_synthetic_disabled_raises() -> None:
    client = WattTimeClient(_config_without_credentials(), allow_synthetic=False)
    assert client.allow_synthetic is False
    with pytest.raises(WattTimeError, match="credentials not configured"):
        client.get_forecast("CAISO_NORTH")


def test_historical_without_credentials_and_synthetic_disabled_raises() -> None:
    client = WattTimeClient(_config_without_credentials(), allow_synthetic=False)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(WattTimeError, match="credentials not configured"):
        client.get_historical("CAISO_NORTH", start, start + timedelta(hours=2))


# --- token cache / 401 / 429 -------------------------------------------------


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
    client = WattTimeClient(_config_with_credentials(), allow_synthetic=False)
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


# --- synthetic fallback ------------------------------------------------------


def test_forecast_without_credentials_returns_the_live_envelope_shape() -> None:
    """Callers do payload["data"]; the synthetic path must not hand back a bare list."""
    payload = WattTimeClient(_config_without_credentials()).get_forecast("CAISO_NORTH")
    assert isinstance(payload, dict)
    assert payload["units"] == SUPPORTED_MOER_UNIT
    assert payload["meta"]["source"] == "synthetic"
    assert len(payload["data"]) > 1
    assert payload["data"][0]["value"] > 0


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
    assert client.allow_synthetic is True
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


def test_unit_gate_is_never_masked_by_the_synthetic_fallback() -> None:
    """A bad-units response must fail closed even though allow_synthetic is on."""
    transport = _RecordingTransport(
        {
            "/v2/login": [_login_ok()],
            "/v3/forecast": [
                httpx.Response(200, json={"data": [FORECAST_POINT], "units": "g_co2_per_kwh"})
            ],
        }
    )
    client = WattTimeClient(_config_with_credentials())
    assert client.allow_synthetic is True
    with pytest.raises(WattTimeError, match="Unsupported unit"):
        client.get_forecast("PJM_DC", client=transport.new_client())


def test_synthetic_forecast_honours_the_requested_horizon() -> None:
    """run.py sizes the scheduling horizon from max-delay; ignoring it truncates the search."""
    client = WattTimeClient(_config_without_credentials())
    six_hours = client.get_forecast("PJM_DC", horizon_hours=6)["data"]
    default_horizon = client.get_forecast("PJM_DC")["data"]

    assert len(six_hours) == 6 * 12  # 5-minute resolution
    assert len(default_horizon) == 24 * 12
    span = _parse_point_time(six_hours[-1]) - _parse_point_time(six_hours[0])
    assert span == timedelta(hours=6) - timedelta(minutes=5)


def test_historical_without_credentials_spans_the_requested_window() -> None:
    start = datetime(2026, 3, 1, 6, 0, tzinfo=UTC)
    points = WattTimeClient(_config_without_credentials()).get_historical(
        "SE", start, start + timedelta(hours=3)
    )
    assert points[0]["point_time"].startswith("2026-03-01T06:00")
    assert len(points) == 3 * 12  # 3h at 5-minute resolution


def test_historical_without_credentials_tracks_the_window_length() -> None:
    """A longer window must produce a longer series, not the 24h default."""
    start = datetime(2026, 3, 1, tzinfo=UTC)
    client = WattTimeClient(_config_without_credentials())
    assert len(client.get_historical("SE", start, start + timedelta(hours=8))) == 8 * 12
    # Sub-hour windows still yield at least one hour of points rather than nothing.
    assert len(client.get_historical("SE", start, start + timedelta(minutes=20))) == 12


def test_synthetic_moer_is_deterministic_per_region() -> None:
    assert synthetic_moer_for_region("CAISO_NORTH") == synthetic_moer_for_region("CAISO_NORTH")
    assert synthetic_moer_for_region("CAISO_NORTH") != synthetic_moer_for_region("DIRTY_GRID_ZZ")


def test_synthetic_moer_is_not_flat_across_regions() -> None:
    """A flat constant everywhere would make the carbon lever invisible.

    The curves are hash-derived over an 850 lb/MWh span, so a couple of the 17
    regions can collide; what matters is the spread, not perfect uniqueness.
    """
    values = {synthetic_moer_for_region(region) for region in SYNTHETIC_WATTTIME_REGIONS}
    assert len(values) >= len(SYNTHETIC_WATTTIME_REGIONS) - 2
    assert max(values) - min(values) > 300.0


def test_synthetic_forecast_series_shape() -> None:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    points = build_synthetic_forecast("IND", horizon_hours=2, step_minutes=15, now=start)
    assert len(points) == 8
    assert points[0]["point_time"] == "2026-05-01T00:00:00Z"
    assert points[1]["point_time"] == "2026-05-01T00:15:00Z"
    assert all(point["units"] == SUPPORTED_MOER_UNIT for point in points)
    assert all(point["value"] > 0 for point in points)


def test_synthetic_forecast_varies_within_the_horizon() -> None:
    """A single repeated value would make time-weighted averaging meaningless."""
    values = {point["value"] for point in build_synthetic_forecast("DE", horizon_hours=2)}
    assert len(values) > 1


def test_synthetic_forecast_accepts_a_naive_start() -> None:
    points = build_synthetic_forecast("UK", horizon_hours=1, now=datetime(2026, 5, 1))
    assert points[0]["point_time"] == "2026-05-01T00:00:00Z"


def test_synthetic_forecast_horizon_never_yields_an_empty_series() -> None:
    assert len(build_synthetic_forecast("FR", horizon_hours=0)) == 1


@pytest.mark.parametrize("horizon_hours", [1, 3, 12, 24, 48])
def test_synthetic_payload_honours_the_requested_horizon(horizon_hours: int) -> None:
    payload = build_synthetic_forecast_payload("FR", horizon_hours=horizon_hours)
    assert len(payload["data"]) == horizon_hours * 12


def test_synthetic_payload_and_series_agree() -> None:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    payload = build_synthetic_forecast_payload("KOR", horizon_hours=1, now=start)
    assert payload["data"] == build_synthetic_forecast("KOR", horizon_hours=1, now=start)


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


def test_estimator_sees_a_per_region_lever_without_credentials() -> None:
    """The reason the envelope shape matters: carbon_model reads payload["data"].

    If the synthetic path returned a bare list, carbon_model would swallow the
    AttributeError and fall back to a flat 400 lb/MWh for every region, and the
    greenest-region ranking would be pure noise.
    """
    from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
    from carbonsight_core.models import JobSpec

    estimator = JobCarbonEstimator(WattTimeClient(_config_without_credentials()))
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
    clean = estimator.estimate_region(job, "aws", "eu-west-1", [("IE", 1.0)], 0.9)
    dirty = estimator.estimate_region(job, "aws", "eu-north-1", [("SE", 1.0)], 0.9)

    assert clean.expected_co2_kg_mean != dirty.expected_co2_kg_mean
    ratio = dirty.expected_co2_kg_mean / clean.expected_co2_kg_mean
    assert ratio > 1.5, "synthetic regions collapsed to a near-flat MOER"
