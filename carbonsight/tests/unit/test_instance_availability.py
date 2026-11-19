from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker


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
