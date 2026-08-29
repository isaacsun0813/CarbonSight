"""Unit tests for InstanceAvailabilityChecker and ranking skip hook."""

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker
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
        ("aws", "ap-south-1"): CloudRegionEntry(
            provider="aws",
            region_code="ap-south-1",
            display_name="Mumbai",
            wt_regions=[("INDIA", 1.0)],
        ),
    }
    return reg


def _mock_ec2_clients(us_offerings: list, ap_offerings: list) -> MagicMock:
    ec2_us = MagicMock()
    ec2_us.describe_instance_type_offerings.return_value = {"InstanceTypeOfferings": us_offerings}
    ec2_ap = MagicMock()
    ec2_ap.describe_instance_type_offerings.return_value = {"InstanceTypeOfferings": ap_offerings}

    session = MagicMock()
    def client(service: str, region_name: str | None = None) -> MagicMock:
        if region_name == "us-east-1":
            return ec2_us
        return ec2_ap

    session.client = client
    return session


class TestInstanceAvailabilityChecker:
    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_offering_present_is_available(self, mock_boto3: MagicMock) -> None:
        mock_boto3.Session.return_value = _mock_ec2_clients(
            [{"InstanceType": "p4d.24xlarge"}], [],
        )
        checker = InstanceAvailabilityChecker()
        result = checker.check_instance_offering("us-east-1", "A100")
        assert result.available is True
        assert result.instance_type == "p4d.24xlarge"

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_empty_offerings_not_available(self, mock_boto3: MagicMock) -> None:
        mock_boto3.Session.return_value = _mock_ec2_clients([], [])
        checker = InstanceAvailabilityChecker()
        result = checker.check_instance_offering("ap-south-1", "A100")
        assert result.available is False
        assert "p4d.24xlarge" in result.reason
        assert "ap-south-1" in result.reason

    def test_unknown_gpu_fails_open(self) -> None:
        checker = InstanceAvailabilityChecker()
        result = checker.check_instance_offering("us-east-1", "MI300")
        assert result.available is True
        assert "unknown GPU" in result.reason

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_client_error_fails_open(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        ec2.describe_instance_type_offerings.side_effect = ClientError(
            {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
            "DescribeInstanceTypeOfferings",
        )
        session = MagicMock()
        session.client.return_value = ec2
        mock_boto3.Session.return_value = session

        checker = InstanceAvailabilityChecker()
        result = checker.check_instance_offering("us-east-1", "A100")
        assert result.available is True
        assert "UnauthorizedOperation" in result.reason

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_cache_avoids_repeat_api_calls(self, mock_boto3: MagicMock) -> None:
        session = _mock_ec2_clients([{"InstanceType": "p4d.24xlarge"}], [])
        mock_boto3.Session.return_value = session

        checker = InstanceAvailabilityChecker()
        checker.check_instance_offering("us-east-1", "A100")
        checker.check_instance_offering("us-east-1", "A100")

        ec2_us = session.client("ec2", region_name="us-east-1")
        assert ec2_us.describe_instance_type_offerings.call_count == 1


class TestRankingAvailabilitySkip:
    @patch("carbonsight_core.estimator.carbon_model.JobCarbonEstimator.estimate_region")
    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_collect_estimates_skips_unavailable_regions(
        self, mock_boto3: MagicMock, mock_estimate: MagicMock,
    ) -> None:
        mock_boto3.Session.return_value = _mock_ec2_clients(
            [{"InstanceType": "p4d.24xlarge"}],
            [],
        )
        mock_estimate.return_value = EstimateResult(
            cloud="aws",
            cloud_region="us-east-1",
            watttime_regions=[("PJM", 1.0)],
            mapping_confidence=0.9,
        )

        watt_time = MagicMock()
        ranking = AwsRegionRankingService(
            _two_region_registry(),
            watt_time,
            availability_checker=InstanceAvailabilityChecker(),
        )

        skips: list[tuple[str, str]] = []
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)

        estimates = ranking.collect_estimates(
            job,
            on_availability_skip=lambda region, reason: skips.append((region, reason)),
        )

        assert len(estimates) == 1
        assert estimates[0].cloud_region == "us-east-1"
        assert len(skips) == 1
        assert skips[0][0] == "ap-south-1"
        assert "p4d.24xlarge" in skips[0][1]
        mock_estimate.assert_called_once()
