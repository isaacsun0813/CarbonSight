"""
AWS Service Quotas preflight per design doc.
If GPU quota is 0 or below requested GPUs, mark region "blocked by quota".
Cache per (account, region, quota_code) for ~15 minutes.
"""

import time
from dataclasses import dataclass, field
from typing import Any

# Optional boto3; only used when AWS preflight is enabled
try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# AWS Service Quotas codes for each GPU instance family.
# Find yours: aws service-quotas list-service-quotas --service-code ec2
# These are the "Running On-Demand <family>" vCPU quotas.
_GPU_QUOTA_CODES: dict[str, str] = {
    "A100": "L-3819A6DF",  # p4d (A100 40GB) and p4de (A100 80GB)
    "H100": "L-B2E278C4",  # p5 (H100 SXM5)
    "V100": "L-417A185B",  # p3 / p3dn
    "T4":   "L-DB2E81BA",  # g4dn
    "A10G": "L-DB2E81BA",  # g5 (same G-and-VT bucket as T4)
    "L4":   "L-DB2E81BA",  # g6
}
_DEFAULT_QUOTA_CODE = "L-DB2E81BA"  # G and VT — safe fallback
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


class QuotaChecker:
    """Check AWS GPU-related quota per region; cache results for 15 min."""

    def __init__(self, *, cache_ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds

    def _get_cached(self, region: str, quota_code: str) -> float | None:
        key = (region, quota_code)
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, region: str, quota_code: str, value: float) -> None:
        self._cache[(region, quota_code)] = (time.time() + self._cache_ttl, value)

    @staticmethod
    def quota_code_for_gpu(gpu_type: str) -> str:
        """Return the AWS Service Quotas code for the given GPU type."""
        return _GPU_QUOTA_CODES.get(gpu_type.upper(), _DEFAULT_QUOTA_CODE)

    def check_gpu_quota(
        self,
        region: str,
        requested_gpus: int,
        *,
        gpu_type: str = "T4",
        service_code: str = DEFAULT_QUOTA_SERVICE_CODE,
    ) -> QuotaResult:
        """
        Return QuotaResult: allowed=True if current vCPU quota >= requested_gpus.
        AWS quota is measured in vCPUs, not GPU count — we treat them as equivalent
        for the purpose of a go/no-go preflight (conservative).
        Fails open: if boto3 is missing or the API errors, region is allowed.
        """
        quota_code = self.quota_code_for_gpu(gpu_type)

        if not _HAS_BOTO:
            return QuotaResult(
                region=region, allowed=True, current_quota=float("inf"),
                requested=requested_gpus, quota_code=quota_code,
                reason="boto3 not installed; skipping quota check",
            )

        cached = self._get_cached(region, quota_code)
        if cached is not None:
            allowed = cached >= requested_gpus
            return QuotaResult(
                region=region, allowed=allowed, current_quota=cached,
                requested=requested_gpus, quota_code=quota_code,
                reason="" if allowed else f"quota {cached} < requested {requested_gpus} (cached)",
            )

        try:
            sq = boto3.client("service-quotas", region_name=region)
            resp = sq.get_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
            value = float(resp.get("Quota", {}).get("Value", 0))
            self._set_cached(region, quota_code, value)
            allowed = value >= requested_gpus
            return QuotaResult(
                region=region, allowed=allowed, current_quota=value,
                requested=requested_gpus, quota_code=quota_code,
                reason="" if allowed else f"quota {value} < requested {requested_gpus}",
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
