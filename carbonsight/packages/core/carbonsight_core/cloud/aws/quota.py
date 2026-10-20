"""
AWS Service Quotas preflight per design doc.
If GPU quota is 0 or below requested GPUs, mark region "blocked by quota".
Cache per (account, region, quota_code) for ~15 minutes.
"""

from dataclasses import dataclass

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider, ClientError
from carbonsight_core.config import Config

CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# AWS Service Quotas codes (vCPU quotas) per GPU family, for on-demand and spot launches.
# Find yours: aws service-quotas list-service-quotas --service-code ec2
_ON_DEMAND_QUOTA_CODES: dict[str, str] = {
    "T4":   "L-DB2E81BA",  # Running On-Demand G and VT instances (g4dn)
    "A10G": "L-DB2E81BA",  # g5 (same G-and-VT bucket as T4)
    "L4":   "L-DB2E81BA",  # g6
    "A100": "L-417A185B",  # Running On-Demand P instances (p4d / p4de)
    "V100": "L-417A185B",  # p3 / p3dn
    "H100": "L-417A185B",  # p5
}
_SPOT_QUOTA_CODES: dict[str, str] = {
    "T4":   "L-3819A6DF",  # All G and VT Spot Instance Requests
    "A10G": "L-3819A6DF",
    "L4":   "L-3819A6DF",
    "A100": "L-7212CCBC",  # All P Spot Instance Requests
    "V100": "L-7212CCBC",
    "H100": "L-7212CCBC",
}
_DEFAULT_ON_DEMAND_QUOTA_CODE = "L-DB2E81BA"  # G and VT - safe fallback
_DEFAULT_SPOT_QUOTA_CODE = "L-3819A6DF"
DEFAULT_QUOTA_SERVICE_CODE = "ec2"


@dataclass
class QuotaResult:
    """Result of a quota check for one region."""

    region: str
    allowed: bool
    current_quota: float
    requested: int
    quota_code: str
    reason: str = ""


class QuotaChecker(BaseAWSProvider):
    """Check AWS GPU-related quota per region; cache results for 15 min."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        super().__init__(config, cache_ttl_seconds=cache_ttl_seconds)

    @staticmethod
    def quota_code_for_gpu(gpu_type: str, *, use_spot: bool = False) -> str:
        """Return the AWS Service Quotas code for the GPU type (spot or on-demand family)."""
        if use_spot:
            return _SPOT_QUOTA_CODES.get(gpu_type.upper(), _DEFAULT_SPOT_QUOTA_CODE)
        return _ON_DEMAND_QUOTA_CODES.get(gpu_type.upper(), _DEFAULT_ON_DEMAND_QUOTA_CODE)

    def check_gpu_quota(
        self,
        region: str,
        requested_gpus: int,
        *,
        gpu_type: str = "T4",
        service_code: str = DEFAULT_QUOTA_SERVICE_CODE,
        use_spot: bool = False,
    ) -> QuotaResult:
        """
        Return QuotaResult: allowed=True if current vCPU quota >= requested_gpus.
        AWS quota is measured in vCPUs, not GPU count — we treat them as equivalent
        for the purpose of a go/no-go preflight (conservative).
        Fails open: if boto3 is missing or the API errors, region is allowed.
        """
        quota_code = self.quota_code_for_gpu(gpu_type, use_spot=use_spot)

        if not HAS_BOTO:
            return QuotaResult(
                region=region, allowed=True, current_quota=float("inf"),
                requested=requested_gpus, quota_code=quota_code,
                reason="boto3 not installed; skipping quota check",
            )

        cached = self._get_cached((region, quota_code))
        if cached is not None:
            allowed = cached >= requested_gpus
            return QuotaResult(
                region=region, allowed=allowed, current_quota=cached,
                requested=requested_gpus, quota_code=quota_code,
                reason="" if allowed else f"quota {cached} < requested {requested_gpus} ({quota_code}, cached)",
            )

        try:
            sq = self._client("service-quotas", region)
            resp = sq.get_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
            value = float(resp.get("Quota", {}).get("Value", 0))
            self._set_cached((region, quota_code), value)
            allowed = value >= requested_gpus
            return QuotaResult(
                region=region, allowed=allowed, current_quota=value,
                requested=requested_gpus, quota_code=quota_code,
                reason="" if allowed else f"quota {value} < requested {requested_gpus} ({quota_code})",
            )
        except ClientError as e:
            return QuotaResult(
                region=region, allowed=True, current_quota=0.0,
                requested=requested_gpus, quota_code=quota_code,
                reason=f"ServiceQuotas API error: {e.response.get('Error', {}).get('Code', 'Unknown')}; skipping",
            )
        except Exception as e:
            return QuotaResult(
                region=region, allowed=True, current_quota=0.0,
                requested=requested_gpus, quota_code=quota_code,
                reason=f"Error: {e}; skipping",
            )
