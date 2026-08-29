"""Unit tests for EnabledRegionsProvider and ranking skip hook."""

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.mapping.registry import CloudRegionEntry, Registry
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.region_ranking import AwsRegionRankingService


def _two_region_registry() -> Registry:
    reg = Registry()
    reg._regions = {
        ("aws", "us-east-1"): CloudRegionEntry(
            provider="aws",
            region_code="us-east-1",
            display_name="US East",
            wt_regions=[("PJM", 1.0)],
        ),
        ("aws", "eu-north-1"): CloudRegionEntry(
            provider="aws",
            region_code="eu-north-1",
            display_name="Stockholm",
            wt_regions=[("SE", 1.0)],
        ),
    }
    return reg


class TestEnabledRegionsProvider:
    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_returns_enabled_region_codes(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_regions.return_value = {
            "Regions": [
                {"RegionName": "us-east-1"},
                {"RegionName": "us-west-2"},
            ],
        }
        session = MagicMock()
        session.client.return_value = ec2
        mock_boto3.Session.return_value = session

        provider = EnabledRegionsProvider()
        codes = provider.enabled_region_codes()
        assert codes == frozenset({"us-east-1", "us-west-2"})
        ec2.describe_regions.assert_called_once_with(AllRegions=False)

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_empty_regions_returns_empty_frozenset(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_regions.return_value = {"Regions": []}
        session = MagicMock()
        session.client.return_value = ec2
        mock_boto3.Session.return_value = session

        provider = EnabledRegionsProvider()
        assert provider.enabled_region_codes() == frozenset()

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_client_error_fails_open(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_regions.side_effect = ClientError(
            {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
            "DescribeRegions",
        )
        session = MagicMock()
        session.client.return_value = ec2
        mock_boto3.Session.return_value = session

        provider = EnabledRegionsProvider()
        assert provider.enabled_region_codes() is None

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_cache_avoids_repeat_api_calls(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_regions.return_value = {
            "Regions": [{"RegionName": "us-east-1"}],
        }
        session = MagicMock()
        session.client.return_value = ec2
        mock_boto3.Session.return_value = session

        provider = EnabledRegionsProvider()
        provider.enabled_region_codes()
        provider.enabled_region_codes()
        assert ec2.describe_regions.call_count == 1


class TestRankingEnabledRegionSkip:
    @patch("carbonsight_core.estimator.carbon_model.JobCarbonEstimator.estimate_region")
    def test_collect_estimates_skips_disabled_regions(
        self, mock_estimate: MagicMock,
    ) -> None:
        mock_estimate.return_value = EstimateResult(
            cloud="aws",
            cloud_region="us-east-1",
            watttime_regions=[("PJM", 1.0)],
            mapping_confidence=0.9,
        )

        provider = MagicMock(spec=EnabledRegionsProvider)
        provider.enabled_region_codes.return_value = frozenset({"us-east-1"})

        watt_time = MagicMock()
        ranking = AwsRegionRankingService(
            _two_region_registry(),
            watt_time,
            enabled_regions_provider=provider,
        )

        skips: list[tuple[str, str]] = []
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        estimates = ranking.collect_estimates(
            job,
            on_enabled_region_skip=lambda region, reason: skips.append((region, reason)),
        )

        assert len(estimates) == 1
        assert estimates[0].cloud_region == "us-east-1"
        assert len(skips) == 1
        assert skips[0][0] == "eu-north-1"
        assert "not enabled" in skips[0][1]
        mock_estimate.assert_called_once()
        provider.enabled_region_codes.assert_called_once()
