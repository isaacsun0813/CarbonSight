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
# EC2 quota for "Running On-Demand G and VT instances" (example); varies by account
DEFAULT_QUOTA_SERVICE_CODE = "ec2"
DEFAULT_QUOTA_CODE = "L-DB2E81BA"  # On-Demand G and VT (GPU) instances - example


@dataclass
class QuotaResult:
    """Result of a quota check for one region."""

    region: str
    allowed: bool  # True if quota >= requested
    current_quota: float
    requested: int
    quota_code: str
    reason: str = ""


class QuotaChecker:
    """Check AWS GPU-related quota per region; cache results."""

    def __init__(self, *, cache_ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}  # (region, quota_code) -> (expires_at, value)
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

    def check_gpu_quota(
        self,
        region: str,
        requested_gpus: int,
        *,
        quota_code: str = DEFAULT_QUOTA_CODE,
        service_code: str = DEFAULT_QUOTA_SERVICE_CODE,
    ) -> QuotaResult:
        """
        Return QuotaResult: allowed=True only if current quota >= requested_gpus.
        Uses Service Quotas API; caches for 15 min.
        """
        if not _HAS_BOTO:
            return QuotaResult(
                region=region,
                allowed=True,
                current_quota=float("inf"),
                requested=requested_gpus,
                quota_code=quota_code,
                reason="boto3 not installed; skipping quota check",
            )
        cached = self._get_cached(region, quota_code)
        if cached is not None:
            allowed = cached >= requested_gpus
            return QuotaResult(
                region=region,
                allowed=allowed,
                current_quota=cached,
                requested=requested_gpus,
                quota_code=quota_code,
                reason="" if allowed else f"quota {cached} < requested {requested_gpus} (cached)",
            )
        try:
            sq = boto3.client("service-quotas", region_name=region)
            # ListServiceQuotas for ec2 to find the quota; or GetServiceQuota with quota code
            resp = sq.get_service_quota(
                ServiceCode=service_code,
                QuotaCode=quota_code,
            )
            value = float(resp.get("Quota", {}).get("Value", 0))
            self._set_cached(region, quota_code, value)
            allowed = value >= requested_gpus
            return QuotaResult(
                region=region,
                allowed=allowed,
                current_quota=value,
                requested=requested_gpus,
                quota_code=quota_code,
                reason="" if allowed else f"quota {value} < requested {requested_gpus}",
            )
        except ClientError as e:
            # Quota not found or no permission -> allow (fail open for preflight)
            return QuotaResult(
                region=region,
                allowed=True,
                current_quota=0.0,
                requested=requested_gpus,
                quota_code=quota_code,
                reason=f"Service Quotas error: {e.response.get('Error', {}).get('Code', 'Unknown')}; skipping",
            )
        except Exception as e:
            return QuotaResult(
                region=region,
                allowed=True,
                current_quota=0.0,
                requested=requested_gpus,
                quota_code=quota_code,
                reason=f"Error: {e}; skipping",
            )
