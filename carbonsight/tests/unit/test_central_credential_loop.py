"""The central-credential design, end to end.

The promise: **one** WattTime login lives on the server, and every CLI user points
at it with ``CARBONSIGHT_API_URL`` and needs no credentials of their own.

Three pieces have to line up, and until now only the two ends existed:

    worker (holds the credential)  ->  /v1/carbon/forecast  ->  ApiCarbonProvider
           fills the shared cache      serves the cache         credential-less client

These tests drive the whole chain against the real ASGI app, so a break anywhere
in it fails here rather than at someone's terminal.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from carbonsight_api.main import app
import carbonsight_core.carbon.providers as providers_module
from carbonsight_core.carbon import ApiCarbonProvider, CarbonProviderError
from carbonsight_core.config import Config
from carbonsight_core.watttime import get_global_forecast_cache, reset_global_forecast_cache
from worker import fetch_watttime as worker

SERVER_CONFIG = Config(watttime_username="server-account", watttime_password="server-secret")


class ServerSideWattTime:
    """Stands in for the one real credential the server holds."""

    def get_forecast(self, region: str, horizon_hours: int = 24, **_: object) -> Any:
        return {
            "data": [
                {"point_time": "2026-03-01T00:00:00Z", "value": 300.0 + len(region)},
                {"point_time": "2026-03-01T00:05:00Z", "value": 310.0 + len(region)},
            ],
            "units": "lbs_co2_per_mwh",
        }


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    reset_global_forecast_cache()
    yield
    reset_global_forecast_cache()


@pytest.fixture
def credentialless_client(monkeypatch: pytest.MonkeyPatch) -> ApiCarbonProvider:
    """An ApiCarbonProvider whose HTTP is routed into the app in-process.

    A sync MockTransport dispatching to TestClient: ASGITransport is async-only,
    and patching httpx.Client globally makes TestClient recurse into itself.
    """
    from fastapi.testclient import TestClient

    real_client_cls = httpx.Client
    inner = TestClient(app)

    def handler(request: httpx.Request) -> httpx.Response:
        served = inner.request(
            request.method, request.url.path, params=dict(request.url.params)
        )
        return httpx.Response(served.status_code, content=served.content,
                              headers={"content-type": "application/json"})

    def in_process(*_a: object, **_k: object) -> httpx.Client:
        return real_client_cls(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(providers_module, "httpx", type("H", (), {"Client": in_process}))
    return ApiCarbonProvider(base_url="http://api", config=Config())


def _refresh(regions: list[str]) -> worker.RefreshReport:
    return worker.fetch_all_regions(
        SERVER_CONFIG, client=ServerSideWattTime(), regions=regions
    )


class TestTheLoopIsConnected:
    def test_client_with_no_credentials_reads_worker_data(
        self, credentialless_client: ApiCarbonProvider
    ) -> None:
        assert _refresh(["SE", "IE"]).regions_refreshed == 2

        points = credentialless_client.get_forecast("SE")

        assert len(points) == 2
        # 300 + len("SE") -- proves the value came from the worker, not invented locally
        assert points[0]["value"] == 302.0
        assert credentialless_client.last_source("SE") == "cache"

    def test_the_client_really_has_no_credentials(
        self, credentialless_client: ApiCarbonProvider
    ) -> None:
        config = Config()
        assert not config.watttime_username and not config.watttime_password
        _refresh(["SE"])
        assert credentialless_client.get_forecast("SE")

    def test_mixture_weighting_works_over_the_wire(
        self, credentialless_client: ApiCarbonProvider
    ) -> None:
        from datetime import UTC, datetime

        _refresh(["SE", "IE"])
        start = datetime(2026, 3, 1, tzinfo=UTC)
        end = datetime(2026, 3, 1, 0, 10, tzinfo=UTC)
        blended = credentialless_client.get_moer_lb_per_mwh(
            [("SE", 0.5), ("IE", 0.5)], start, end
        )
        assert 300.0 < blended < 315.0


class TestHonestFailureModes:
    def test_a_region_the_worker_has_not_reached_is_503_not_invented(
        self, credentialless_client: ApiCarbonProvider
    ) -> None:
        """A cold cache must say 'not yet', never hand back a synthetic curve."""
        with pytest.raises(CarbonProviderError):
            credentialless_client.get_forecast("SE")

    def test_the_503_explains_how_to_fix_it(self) -> None:
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            response = client.get("/v1/carbon/forecast", params={"region": "SE"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "refresh worker" in detail and "WATTTIME_USERNAME" in detail

    def test_a_failed_region_does_not_become_a_fabricated_response(self) -> None:
        """Worker fails on SE -> the API must have nothing to serve for SE."""

        class PartlyBroken(ServerSideWattTime):
            def get_forecast(self, region: str, horizon_hours: int = 24, **_: object) -> Any:
                if region == "SE":
                    raise RuntimeError("upstream down")
                return super().get_forecast(region, horizon_hours)

        worker.fetch_all_regions(SERVER_CONFIG, client=PartlyBroken(), regions=["SE", "IE"])
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            assert client.get("/v1/carbon/forecast", params={"region": "SE"}).status_code == 503
            assert client.get("/v1/carbon/forecast", params={"region": "IE"}).status_code == 200


class TestTheRouteItself:
    def test_expired_data_is_served_but_labelled_stale(self) -> None:
        from fastapi.testclient import TestClient

        cache = get_global_forecast_cache()
        cache.set("SE", [{"point_time": "2026-03-01T00:00:00Z", "value": 99.0}])
        # force expiry without waiting 15 minutes
        expires_at, points = cache._store["SE"]  # noqa: SLF001
        cache._store["SE"] = (expires_at - 10_000.0, points)  # noqa: SLF001
        assert cache.get_or_expired("SE") is not None  # present but stale

        with TestClient(app) as client:
            response = client.get("/v1/carbon/forecast", params={"region": "SE"})
        assert response.status_code == 200
        assert response.json()["source"] == "stale"

    def test_regions_endpoint_reports_what_is_warm(self) -> None:
        from fastapi.testclient import TestClient

        _refresh(["SE"])
        with TestClient(app) as client:
            body = client.get("/v1/carbon/regions").json()
        assert body["cached_count"] == 1
        warm = [r["wt_region"] for r in body["regions"] if r["cached"]]
        assert warm == ["SE"]

    def test_the_route_never_calls_watttime_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Serving must stay cheap: one HTTP request must not fan out upstream."""
        from fastapi.testclient import TestClient

        def explode(*_a: object, **_k: object) -> None:
            raise AssertionError("the forecast route must not fetch on the request path")

        monkeypatch.setattr(
            "carbonsight_core.watttime.client.WattTimeClient.get_forecast", explode
        )
        _refresh(["SE"])
        with TestClient(app) as client:
            assert client.get("/v1/carbon/forecast", params={"region": "SE"}).status_code == 200
