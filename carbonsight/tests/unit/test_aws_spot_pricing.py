"""Unit tests for live AWS spot pricing and gpu_catalog."""

from unittest.mock import MagicMock, patch

import pytest

from carbonsight_core.config import Config
from carbonsight_core.estimator.aws_spot_pricing import SpotPriceProvider
from carbonsight_core.estimator.gpu_catalog import gpus_per_instance, instance_type_for_gpu
from carbonsight_core.estimator.pricing import (
    SPOT_PRICE_FRACTION,
    configure_pricing,
    estimate_cost_usd,
    estimate_job_cost,
    reset_pricing_state,
)


@pytest.fixture(autouse=True)
def _reset_pricing() -> None:
    reset_pricing_state()
    yield
    reset_pricing_state()


class TestGpuCatalog:
    def test_a100_maps_to_p4d(self) -> None:
        assert instance_type_for_gpu("A100") == "p4d.24xlarge"
        assert gpus_per_instance("A100") == 8

    def test_unknown_gpu_returns_none(self) -> None:
        assert instance_type_for_gpu("MI300") is None


class TestSpotPriceProvider:
    @patch("carbonsight_core.estimator.aws_spot_pricing.boto3")
    def test_min_across_azs_per_gpu_hour(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [
                {"SpotPrice": "4.00", "InstanceType": "p4d.24xlarge"},
                {"SpotPrice": "3.20", "InstanceType": "p4d.24xlarge"},
            ],
        }
        mock_boto3.Session.return_value.client.return_value = ec2

        provider = SpotPriceProvider(Config())
        price = provider.spot_price_per_gpu_hour("A100", "us-east-1")
        assert price == pytest.approx(3.20 / 8)


class TestEstimateJobCost:
    def test_live_spot_when_enabled(self) -> None:
        mock_provider = MagicMock()
        mock_provider.spot_price_per_gpu_hour.return_value = 0.5
        configure_pricing(live_aws_pricing=True)
        from carbonsight_core.estimator import pricing as pricing_mod

        pricing_mod._spot_provider = mock_provider
        cost = estimate_job_cost("A100", 2, 1.0, "us-east-1", use_spot=True)
        assert cost.usd == pytest.approx(1.0)
        assert "cost:live_spot" in cost.notes

    def test_static_spot_fallback_on_api_failure(self) -> None:
        mock_provider = MagicMock()
        mock_provider.spot_price_per_gpu_hour.return_value = None
        configure_pricing(live_aws_pricing=True)
        from carbonsight_core.estimator import pricing as pricing_mod

        pricing_mod._spot_provider = mock_provider
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=True)
        static = estimate_cost_usd("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert cost.usd == pytest.approx(static)
        assert "cost:static_spot_fallback" in cost.notes

    def test_static_spot_when_live_disabled(self) -> None:
        on_demand = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert cost.usd == pytest.approx(on_demand * SPOT_PRICE_FRACTION)
        assert cost.notes == []

    def test_live_enabled_no_boto3_falls_back(self) -> None:
        with patch("carbonsight_core.estimator.aws_spot_pricing._HAS_BOTO", False):
            configure_pricing(live_aws_pricing=True)
            cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert "cost:static_spot_fallback" in cost.notes
