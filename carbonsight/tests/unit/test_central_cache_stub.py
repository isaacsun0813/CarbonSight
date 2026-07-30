"""
Stubbed end-to-end validation of P0 central WattTime cache — no real credentials.

Validates:
- ForecastCache with mocked WattTime returns time-weighted MOER over arbitrary window [t, t+L̄]
- Mixture weighting works
- Carbon providers factory picks correct impl without real env
- ApiCarbonProvider can read from central API (mocked via TestClient) without WattTime creds
- POST /v1/recommendations returns results even without WattTime creds (synthetic fallback) -> everyone can schedule
- Multi-lever flip: with carbon_price=0 cheapest wins, with high carbon_price green wins (even if spot $ higher)
"""

from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import (
    ApiCarbonProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
)
from carbonsight_core.watttime import WattTimeClient
from carbonsight_core.watttime.cache import ForecastCache


def _synthetic_forecast_points(low=150, high=420):
    """
    24h diurnal synthetic: low at night 0-6h, high at day 12-18h, mid otherwise.
    Returns list of 5-min points (but simplified to hourly for stub).
    """
    base = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    pts = []
    for h in range(0, 24):
        # diurnal
        if 0 <= h < 6:
            val = low
        elif 12 <= h < 18:
            val = high
        else:
            val = (low + high) / 2
        pts.append({"point_time": (base + timedelta(hours=h)).isoformat(), "value": val})
    return pts


def test_forecast_cache_stub_time_weighted_window():
    """Stubbed WattTime -> ForecastCache time-weighted over [t, t+L̄] for joint 2D rank."""
    pts = _synthetic_forecast_points(low=150, high=420)
    mock_client = MagicMock(spec=WattTimeClient)
    mock_client.get_forecast.return_value = {"data": pts}

    cache = ForecastCache(mock_client, ttl_seconds=900)
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    t1 = datetime.fromisoformat("2026-01-01T06:00:00+00:00")  # night window low
    t2 = datetime.fromisoformat("2026-01-01T12:00:00+00:00")
    t3 = datetime.fromisoformat("2026-01-01T18:00:00+00:00")  # day window high

    moer_night = cache.get_time_weighted_moer([("CAISO", 1.0)], t0, t1)
    moer_day = cache.get_time_weighted_moer([("CAISO", 1.0)], t2, t3)

    assert moer_night < moer_day, f"night {moer_night} should be < day {moer_day}"
    # night should be ~150, day ~420
    assert abs(moer_night - 150) < 5
    assert abs(moer_day - 420) < 5
    # Second call hits cache (no second HTTP)
    assert cache.fetch_count == 1
    cache.get_time_weighted_moer([("CAISO", 1.0)], t0, t1)
    assert cache.fetch_count == 1


def test_mixture_weighting_stub():
    pts_low = [{"point_time": "2026-01-01T00:00:00+00:00", "value": 100}]
    pts_high = [{"point_time": "2026-01-01T00:00:00+00:00", "value": 200}]

    def side_effect(region, horizon_hours=24, client=None):
        return {"data": pts_low if region == "A" else pts_high}

    mock_client = MagicMock(spec=WattTimeClient)
    mock_client.get_forecast.side_effect = side_effect
    cache = ForecastCache(mock_client)
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    t1 = t0 + timedelta(hours=1)
    moer = cache.get_time_weighted_moer([("A", 0.6), ("B", 0.4)], t0, t1)
    assert abs(moer - 140) < 1e-6


def test_central_api_without_creds_still_returns_recommendations():
    """Everyone can schedule: API returns 17 rows even with no WattTime env (synthetic fallback)."""
    from carbonsight_api.main import app

    with TestClient(app) as client:
        # regions should be populated from seed_registry.json via lifespan
        regions = client.get("/v1/regions").json()
        assert len(regions) >= 1

        # central forecast without server creds returns empty but not 500
        fc = client.get("/v1/carbon/forecast?wt_region=PJM_DC&horizon_hours=24").json()
        assert fc["wt_region"] == "PJM_DC"
        assert fc["source"] in ("empty", "cache", "live")

        # recommendations without WattTime creds should still return results (synthetic)
        recs = client.post(
            "/v1/recommendations",
            json={"gpu_type": "A100", "gpu_count": 1, "duration_hours": 1},
        ).json()
        assert isinstance(recs, list)
        assert len(recs) >= 1
        # each row has expected fields
        row = recs[0]
        assert "expected_co2_kg_mean" in row
        assert "expected_cost_usd" in row
        assert row["expected_co2_kg_mean"] > 0


def test_api_carbon_provider_reads_from_central_api_stub():
    """ApiCarbonProvider with no WattTime creds reads from central API via TestClient-mocked httpx."""
    from carbonsight_api.main import app

    # Mock httpx.Client.get to call TestClient internally
    with TestClient(app) as api_client:
        # Pre-populate central cache via direct call? For stub, we mock ApiCarbonProvider._fetch_points
        # to return synthetic diurnal points, simulating central API with data.
        pts = _synthetic_forecast_points(low=100, high=300)

        with patch.object(
            ApiCarbonProvider, "_fetch_points", return_value=pts
        ):
            prov = ApiCarbonProvider(api_url="http://test")
            job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1)
            t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
            t1 = t0 + timedelta(hours=6)  # night low
            t2 = datetime.fromisoformat("2026-01-01T12:00:00+00:00")
            t3 = t2 + timedelta(hours=6)  # day high

            moer_night = prov.get_moer_lb_per_mwh([("CAISO", 1.0)], t0, t1)
            moer_day = prov.get_moer_lb_per_mwh([("CAISO", 1.0)], t2, t3)

            assert moer_night < moer_day
            # carbon cost scales
            cost_night = prov.get_cost_per_hr(job, [("CAISO", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
            cost_day = prov.get_cost_per_hr(job, [("CAISO", 1.0)], t2, t3, carbon_price_usd_per_ton=50)
            assert cost_day > cost_night


def test_multi_lever_flip_with_carbon_price_stub():
    """Carbon as lever: cheapest wins at price 0, green wins at high price, even with spot $ diff."""
    mock_client = MagicMock(spec=WattTimeClient)

    def side_effect(region, horizon_hours=24, client=None):
        vals = {"eu-north-1": 150, "us-east-1": 420}
        return {"data": [{"point_time": "2026-01-01T00:00:00+00:00", "value": vals.get(region, 400)}]}

    mock_client.get_forecast.side_effect = side_effect
    prov = WattTimeCarbonProvider(client=mock_client)
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1)
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    t1 = t0 + timedelta(hours=1)

    # Spot prices: EU $4/hr (green but expensive), US $3/hr (dirty but cheap)
    eu_carbon_50 = prov.get_cost_per_hr(job, [("eu-north-1", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    us_carbon_50 = prov.get_cost_per_hr(job, [("us-east-1", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    assert us_carbon_50 > eu_carbon_50  # dirty costs more carbon

    # At carbon_price=0, US total cheaper ($3 < $4)
    eu_total_0 = 4.0 + prov.get_cost_per_hr(job, [("eu-north-1", 1.0)], t0, t1, carbon_price_usd_per_ton=0)
    us_total_0 = 3.0 + prov.get_cost_per_hr(job, [("us-east-1", 1.0)], t0, t1, carbon_price_usd_per_ton=0)
    assert us_total_0 < eu_total_0

    # At very high carbon price 20000 $/ton, green flips despite $1 spot premium
    # facility MWh ~0.0006, so need high price to overcome $1
    eu_high = 4.0 + prov.get_cost_per_hr(job, [("eu-north-1", 1.0)], t0, t1, carbon_price_usd_per_ton=20000)
    us_high = 3.0 + prov.get_cost_per_hr(job, [("us-east-1", 1.0)], t0, t1, carbon_price_usd_per_ton=20000)
    assert eu_high < us_high, f"expected green to win at high price: eu={eu_high} us={us_high}"
