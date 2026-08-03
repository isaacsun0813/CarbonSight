"""
Account-enabled AWS regions via describe_regions.

One global EC2 call intersected with the mapping registry before per-region preflight.
Fails open: returns None on missing boto3 or API errors so ranking proceeds unfiltered.
"""

import time
from typing import Any

from carbonsight_core.config import Config

try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

CACHE_TTL_SECONDS = 15 * 60  # 15 minutes, match other preflight checks
EC2_DESCRIBE_REGIONS_HOME = "us-east-1"


class EnabledRegionsProvider:
    """Fetch account-enabled region codes once; cache for 15 min."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._config = config or Config.from_env()
        self._cache_ttl = cache_ttl_seconds
        self._cache_expires_at: float | None = None
        self._cached_regions: frozenset[str] | None = None
        self._session: Any = None
        self._ec2_home: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            if not _HAS_BOTO:
                raise RuntimeError("boto3 not installed")
            profile = self._config.aws_profile
            self._session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return self._session

    def _get_ec2_home_client(self) -> Any:
        if self._ec2_home is None:
            self._ec2_home = self._get_session().client(
                "ec2", region_name=EC2_DESCRIBE_REGIONS_HOME,
            )
        return self._ec2_home

    def enabled_region_codes(self) -> frozenset[str] | None:
        """
        Return enabled region names for this account, or None to skip filtering (fail open).
        """
        if not _HAS_BOTO:
            return None

        if self._cached_regions is not None and self._cache_expires_at is not None:
            if time.time() <= self._cache_expires_at:
                return self._cached_regions

        try:
            resp = self._get_ec2_home_client().describe_regions(AllRegions=False)
            regions = resp.get("Regions") or []
            codes = frozenset(r["RegionName"] for r in regions if r.get("RegionName"))
            self._cached_regions = codes
            self._cache_expires_at = time.time() + self._cache_ttl
            return codes
        except ClientError:
            return None
        except Exception:
            return None
