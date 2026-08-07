"""Unit tests for live AWS on-demand pricing, location map, and fixture parser."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from carbonsight_core.config import Config
from carbonsight_core.cloud.aws.ondemand_pricing import (
    OnDemandPriceProvider,
    parse_ondemand_instance_price_usd,
)
from carbonsight_core.cloud.aws.pricing_locations import (
    pricing_location_for_region,
)
from carbonsight_core.estimator.pricing import (
    configure_pricing,
    estimate_cost_usd,
    estimate_job_cost,
    reset_pricing_state,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "aws_get_products_p4d_us_east_1.json"
_FIXTURE_INSTANCE_HR = 32.7726


@pytest.fixture(autouse=True)
def _reset_pricing() -> None:
    reset_pricing_state()
    yield
    reset_pricing_state()


@pytest.fixture
def get_products_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


class TestPricingLocations:
    def test_eu_west_1_maps_to_ireland(self) -> None:
        assert pricing_location_for_region("eu-west-1") == "Europe (Ireland)"

    def test_unknown_region_returns_none(self) -> None:
        assert pricing_location_for_region("unknown-region-99") is None


class TestParseOndemandInstancePrice:
    def test_parses_fixture_price_list_entry(self, get_products_fixture: dict) -> None:
        entry = get_products_fixture["PriceList"][0]
        assert parse_ondemand_instance_price_usd(entry) == pytest.approx(_FIXTURE_INSTANCE_HR)


class TestOnDemandPriceProvider:
    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_get_products_returns_per_gpu_hour(
        self, mock_boto3: MagicMock, get_products_fixture: dict,
    ) -> None:
        pricing = MagicMock()
        pricing.get_products.return_value = get_products_fixture
        mock_boto3.Session.return_value.client.return_value = pricing

        provider = OnDemandPriceProvider(Config())
        price = provider.ondemand_price_per_gpu_hour("A100", "us-east-1")
        assert price == pytest.approx(_FIXTURE_INSTANCE_HR / 8)


class TestEstimateJobCostOndemand:
    def test_live_ondemand_when_enabled(self) -> None:
        mock_provider = MagicMock()
        mock_provider.ondemand_price_per_gpu_hour.return_value = 5.0
        configure_pricing(live_aws_pricing=True)
        from carbonsight_core.estimator import pricing as pricing_mod

        pricing_mod._ondemand_provider = mock_provider
        cost = estimate_job_cost("A100", 2, 1.0, "us-east-1", use_spot=False)
        assert cost.usd == pytest.approx(10.0)
        assert "cost:live_ondemand" in cost.notes

    def test_static_ondemand_fallback_on_api_failure(self) -> None:
        mock_provider = MagicMock()
        mock_provider.ondemand_price_per_gpu_hour.return_value = None
        configure_pricing(live_aws_pricing=True)
        from carbonsight_core.estimator import pricing as pricing_mod

        pricing_mod._ondemand_provider = mock_provider
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=False)
        static = estimate_cost_usd("A100", 1, 1.0, "us-east-1", use_spot=False)
        assert cost.usd == pytest.approx(static)
        assert "cost:static_ondemand_fallback" in cost.notes

    def test_static_ondemand_when_live_disabled(self) -> None:
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=False)
        assert cost.usd == pytest.approx(4.10)
        assert cost.notes == []

    def test_spot_fallback_uses_live_ondemand_when_spot_api_fails(self) -> None:
        mock_spot = MagicMock()
        mock_spot.spot_price_per_gpu_hour.return_value = None
        mock_od = MagicMock()
        mock_od.ondemand_price_per_gpu_hour.return_value = 4.0
        configure_pricing(live_aws_pricing=True)
        from carbonsight_core.estimator import pricing as pricing_mod

        pricing_mod._spot_provider = mock_spot
        pricing_mod._ondemand_provider = mock_od
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert cost.usd == pytest.approx(4.0 * 0.35)
        assert "cost:static_spot_fallback" in cost.notes
