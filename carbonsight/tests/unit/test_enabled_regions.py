"""Unit tests for EnabledRegionsProvider and ranking skip hook."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.mapping.registry import CloudRegionEntry, Registry
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.region_ranking import AwsRegionRankingService


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


@pytest.mark.parametrize("rejected_by", ["account", "quota", "offering", "estimate"])
@patch("carbonsight_core.estimator.carbon_model.JobCarbonEstimator.estimate_region")
def test_ranking_reports_rejected_region_and_keeps_valid_region(mock_estimate, rejected_by):
    registry = Registry()
    registry._regions = {
        ("aws", code): CloudRegionEntry(
            provider="aws", region_code=code, display_name=code, wt_regions=[("GRID", 1.0)],
        )
        for code in ("us-east-1", "eu-north-1")
    }
    result = EstimateResult(
        cloud="aws", cloud_region="eu-north-1", watttime_regions=[("GRID", 1.0)],
        mapping_confidence=0.9,
    )
    mock_estimate.return_value = result
    if rejected_by == "estimate":
        mock_estimate.side_effect = [RuntimeError("forecast unavailable"), result]
    provider, quota, offering = MagicMock(), MagicMock(), MagicMock()
    provider.enabled_region_codes.return_value = frozenset(
        ["eu-north-1"] if rejected_by == "account" else ["us-east-1", "eu-north-1"]
    )
    quota.check_gpu_quota.side_effect = lambda region, *args, **kwargs: SimpleNamespace(
        allowed=not (rejected_by == "quota" and region == "us-east-1"), reason="quota exceeded",
    )
    offering.check_instance_offering.side_effect = lambda region, gpu: SimpleNamespace(
        available=not (rejected_by == "offering" and region == "us-east-1"), reason="not offered",
    )
    ranking = AwsRegionRankingService(
        registry, MagicMock(), enabled_regions_provider=provider,
        quota_checker=quota, availability_checker=offering,
    )
    skips, errors = [], []
    assert ranking.collect_estimates(
        JobSpec(gpu_type="A100"), use_spot=True,
        on_region_skip=lambda region, reason: skips.append((region, reason)),
        on_estimate_error=lambda region, err: errors.append((region, str(err))),
    ) == [result]
    reason = {
        "account": "region not enabled for this AWS account", "quota": "quota exceeded",
        "offering": "not offered", "estimate": "forecast unavailable",
    }[rejected_by]
    assert (errors if rejected_by == "estimate" else skips) == [("us-east-1", reason)]
    assert (skips if rejected_by == "estimate" else errors) == []
    provider.enabled_region_codes.assert_called_once()
    assert quota.check_gpu_quota.call_count == (1 if rejected_by == "account" else 2)
    assert offering.check_instance_offering.call_count == (1 if rejected_by in {"account", "quota"} else 2)
    assert mock_estimate.call_count == (2 if rejected_by == "estimate" else 1)
    assert quota.check_gpu_quota.call_args.kwargs["use_spot"] is True
