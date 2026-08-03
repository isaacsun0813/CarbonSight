"""
EC2 instance type offerings preflight.

Uses describe_instance_type_offerings to skip regions where the representative
GPU instance type is not sold. Fails open on missing boto3 or API errors.
"""

import time
from dataclasses import dataclass
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.estimator.gpu_catalog import instance_type_for_gpu

try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

CACHE_TTL_SECONDS = 15 * 60  # 15 minutes, match QuotaChecker
OFFERINGS_MAX_RESULTS = 1


@dataclass
class AvailabilityResult:
    """Result of an instance-type offering check for one region."""

    region: str
    available: bool
    instance_type: str
    gpu_type: str
    reason: str = ""


class InstanceAvailabilityChecker:
    """Check EC2 instance type offerings per region; cache results for 15 min."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._config = config or Config.from_env()
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, bool]] = {}
        self._ec2_clients: dict[str, Any] = {}
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            if not _HAS_BOTO:
                raise RuntimeError("boto3 not installed")
            profile = self._config.aws_profile
            self._session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return self._session

    def _ec2_client(self, region: str) -> Any:
        if region not in self._ec2_clients:
            self._ec2_clients[region] = self._get_session().client("ec2", region_name=region)
        return self._ec2_clients[region]

    def _get_cached(self, region: str, instance_type: str) -> bool | None:
        key = (region, instance_type)
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, region: str, instance_type: str, available: bool) -> None:
        self._cache[(region, instance_type)] = (time.time() + self._cache_ttl, available)

    def check_instance_offering(self, region: str, gpu_type: str) -> AvailabilityResult:
        """
        Return AvailabilityResult: available=True if instance type is offered in region.

        Unknown GPU types and API errors fail open (available=True).
        """
        normalized_gpu = gpu_type.strip().upper().split(":")[0].split("-")[0]
        instance_type = instance_type_for_gpu(gpu_type)

        if instance_type is None:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type="",
                gpu_type=normalized_gpu,
                reason="unknown GPU type; skipping availability check",
            )

        if not _HAS_BOTO:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason="boto3 not installed; skipping availability check",
            )

        cached = self._get_cached(region, instance_type)
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
            ec2 = self._ec2_client(region)
            resp = ec2.describe_instance_type_offerings(
                LocationType="region",
                Filters=[{"Name": "instance-type", "Values": [instance_type]}],
                MaxResults=OFFERINGS_MAX_RESULTS,
            )
            offerings = resp.get("InstanceTypeOfferings") or []
            available = len(offerings) > 0
            self._set_cached(region, instance_type, available)
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
            code = e.response.get("Error", {}).get("Code", "Unknown")
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"EC2 API error: {code}; skipping availability check",
            )
        except Exception as e:
            return AvailabilityResult(
                region=region,
                available=True,
                instance_type=instance_type,
                gpu_type=normalized_gpu,
                reason=f"Error: {e}; skipping availability check",
            )
