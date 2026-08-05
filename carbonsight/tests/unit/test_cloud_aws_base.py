"""Unit tests for BaseAWSProvider cache and client pooling."""

from unittest.mock import MagicMock, patch

from carbonsight_core.cloud.aws.base import BaseAWSProvider


class _TestProvider(BaseAWSProvider):
    pass


class TestBaseAWSProvider:
    def test_set_cached_returns_value(self) -> None:
        provider = _TestProvider(cache_ttl_seconds=60)
        assert provider._set_cached("key", {"a": 1}) == {"a": 1}
        assert provider._get_cached("key") == {"a": 1}

    def test_cache_expires_after_ttl(self) -> None:
        provider = _TestProvider(cache_ttl_seconds=1)
        provider._set_cached("key", 42)
        assert provider._get_cached("key") == 42

        expires_at, _ = provider._cache["key"]
        provider._cache["key"] = (expires_at - 2, 42)
        assert provider._get_cached("key") is None

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_client_reuse_same_service_region(self, mock_boto3: MagicMock) -> None:
        ec2 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = ec2

        provider = _TestProvider()
        first = provider._client("ec2", "us-east-1")
        second = provider._client("ec2", "us-east-1")
        assert first is second
        mock_boto3.Session.return_value.client.assert_called_once()

    @patch("carbonsight_core.cloud.aws.base.boto3")
    def test_client_separate_per_region(self, mock_boto3: MagicMock) -> None:
        us_ec2 = MagicMock()
        eu_ec2 = MagicMock()

        def client_side_effect(service: str, region_name: str | None = None) -> MagicMock:
            if region_name == "us-east-1":
                return us_ec2
            return eu_ec2

        mock_boto3.Session.return_value.client.side_effect = client_side_effect

        provider = _TestProvider()
        assert provider._client("ec2", "us-east-1") is us_ec2
        assert provider._client("ec2", "eu-west-1") is eu_ec2
