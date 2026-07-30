"""Unit tests for CarbonIntensityProvider factory + multi-lever carbon as $ lever."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import (
    ApiCarbonProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)
from carbonsight_core.watttime import WattTimeClient


def test_factory_no_env_returns_synthetic() -> None:
    os.environ.pop("CARBONSIGHT_API_URL", None)
    os.environ.pop("WATTTIME_USERNAME", None)
    os.environ.pop("WATTTIME_PASSWORD", None)
    prov = get_carbon_provider()
    assert isinstance(prov, SyntheticCarbonProvider)


def test_factory_api_url_returns_api() -> None:
    os.environ["CARBONSIGHT_API_URL"] = "https://api.carbonsight.test"
    os.environ.pop("WATTTIME_USERNAME", None)
    os.environ.pop("WATTTIME_PASSWORD", None)
    prov = get_carbon_provider()
    assert isinstance(prov, ApiCarbonProvider)
    os.environ.pop("CARBONSIGHT_API_URL")


def test_factory_watttime_creds_returns_watttime() -> None:
    os.environ.pop("CARBONSIGHT_API_URL", None)
    os.environ["WATTTIME_USERNAME"] = "u"
    os.environ["WATTTIME_PASSWORD"] = "p"
    mock_client = MagicMock(spec=WattTimeClient)
    prov = get_carbon_provider(watt_client=mock_client)
    assert isinstance(prov, WattTimeCarbonProvider)
    os.environ.pop("WATTTIME_USERNAME")
    os.environ.pop("WATTTIME_PASSWORD")


def test_synthetic_provider_positive() -> None:
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1)
    prov = SyntheticCarbonProvider()
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    end = start + timedelta(hours=1)
    kg = prov.get_kg_per_hr(job, [("CAISO", 1.0)], start, end)
    cost = prov.get_cost_per_hr(job, [("CAISO", 1.0)], start, end, carbon_price_usd_per_ton=50)
    assert kg > 0
    assert cost > 0
    # cost scales with carbon_price
    cost_0 = prov.get_cost_per_hr(job, [("CAISO", 1.0)], start, end, carbon_price_usd_per_ton=0)
    assert cost_0 == 0


def test_watttime_provider_green_vs_dirty_lever() -> None:
    mock_client = MagicMock(spec=WattTimeClient)

    def side_effect(region, horizon_hours=24, client=None):
        vals = {"eu": 150, "us": 420}
        return {"data": [{"point_time": "2026-01-01T00:00:00+00:00", "value": vals.get(region, 400)}]}

    mock_client.get_forecast.side_effect = side_effect
    prov = WattTimeCarbonProvider(client=mock_client)
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1)
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    t1 = t0 + timedelta(hours=1)
    eu_cost = prov.get_cost_per_hr(job, [("eu", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    us_cost = prov.get_cost_per_hr(job, [("us", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    assert us_cost > eu_cost  # dirty costs more

    # Total cost lever: spot $ + carbon $
    # Note: facility MWh ~0.0006, so carbon $/hr is tiny ~0.002-0.007 at $50/ton
    # At $50, spot diff $1 dominates. At very high carbon price ~20000 $/ton, flip occurs.
    eu_total = 4.0 + eu_cost
    us_total = 3.0 + us_cost
    # With price 0, us cheaper (spot $3 vs $4)
    eu_total_0 = 4.0 + prov.get_cost_per_hr(job, [("eu", 1.0)], t0, t1, carbon_price_usd_per_ton=0)
    us_total_0 = 3.0 + prov.get_cost_per_hr(job, [("us", 1.0)], t0, t1, carbon_price_usd_per_ton=0)
    assert us_total_0 < eu_total_0
    # With very high carbon price 20000 $/ton (~social cost extreme), green region wins despite $1 spot premium
    eu_high = 4.0 + prov.get_cost_per_hr(job, [("eu", 1.0)], t0, t1, carbon_price_usd_per_ton=20000)
    us_high = 3.0 + prov.get_cost_per_hr(job, [("us", 1.0)], t0, t1, carbon_price_usd_per_ton=20000)
    assert eu_high < us_high
    # Also test with equal spot price, green always wins when carbon_price>0
    eu_eq = 3.0 + prov.get_cost_per_hr(job, [("eu", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    us_eq = 3.0 + prov.get_cost_per_hr(job, [("us", 1.0)], t0, t1, carbon_price_usd_per_ton=50)
    assert eu_eq < us_eq
