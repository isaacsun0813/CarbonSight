"""EC2 instance type offerings preflight."""

from botocore.exceptions import ClientError

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.aws.gpu_catalog import aws_instance_type_for_gpu
from carbonsight_core.cloud.base import AvailabilityResult
from carbonsight_core.config import Config

PREFLIGHT_CACHE_TTL_SECONDS = 15 * 60
OFFERINGS_MAX_RESULTS = 1


class InstanceAvailabilityChecker(BaseAWSProvider):
    """Check EC2 instance type offerings per region."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int = PREFLIGHT_CACHE_TTL_SECONDS,
    ) -> None:
        super().__init__(config, cache_ttl_seconds=cache_ttl_seconds)

    def check_instance_offering(self, region: str, gpu_type: str) -> AvailabilityResult:
        """
        Return AvailabilityResult: available=True if instance type is offered in region.

        Unknown GPU types and API errors fail open (available=True).
        """
        normalized_gpu = gpu_type.strip().upper().split(":")[0].split("-")[0]
        instance_type = aws_instance_type_for_gpu(gpu_type)

        if instance_type is None:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type="",
                gpu_type=normalized_gpu,
                reason="unknown GPU type; skipping availability check",
            )

        if not HAS_BOTO:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason="boto3 not installed; skipping availability check",
            )

        cache_key = (region, instance_type)
        cached = self._get_cached(cache_key)
        if cached is not None:
            if cached:
                return AvailabilityResult(
                    region=region,
                    available=True,
                    instance_type=instance_type,
                    gpu_type=normalized_gpu,
                )
            return AvailabilityResult(
                region=region,
                available=False,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"{instance_type} not offered in {region} (cached)",
            )

        try:
            ec2 = self._client("ec2", region)
            resp = ec2.describe_instance_type_offerings(
                LocationType="region",
                Filters=[{"Name": "instance-type", "Values": [instance_type]}],
                MaxResults=OFFERINGS_MAX_RESULTS,
            )
            offerings = resp.get("InstanceTypeOfferings") or []
            available = len(offerings) > 0
            self._set_cached(cache_key, available)
            if available:
                return AvailabilityResult(
                    region=region,
                    available=True,
                    instance_type=instance_type,
                    gpu_type=normalized_gpu,
                )
            return AvailabilityResult(
                region=region,
                available=False,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"{instance_type} not offered in {region}",
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"EC2 API error: {error_code}; skipping availability check",
            )
        except Exception as e:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"Error: {e}; skipping availability check",
            )
