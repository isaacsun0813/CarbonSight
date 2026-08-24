"""Account-enabled AWS regions via describe_regions."""

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider, ClientError
from carbonsight_core.config import Config

PREFLIGHT_CACHE_TTL_SECONDS = 15 * 60
EC2_DESCRIBE_REGIONS_HOME = "us-east-1"
_CACHE_KEY = "enabled_regions"


class EnabledRegionsProvider(BaseAWSProvider):
    """Fetch account-enabled region codes once; cache for 15 min."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int = PREFLIGHT_CACHE_TTL_SECONDS,
    ) -> None:
        super().__init__(config, cache_ttl_seconds=cache_ttl_seconds)

    def enabled_region_codes(self) -> frozenset[str] | None:
        """Return enabled region names for this account, or None to skip filtering (fail open)."""
        if not HAS_BOTO:
            return None

        cached = self._get_cached(_CACHE_KEY)
        if cached is not None:
            return cached

        try:
            client = self._client("ec2", EC2_DESCRIBE_REGIONS_HOME)
            resp = client.describe_regions(AllRegions=False)
            regions = resp.get("Regions") or []
            codes = frozenset(r["RegionName"] for r in regions if r.get("RegionName"))
            return self._set_cached(_CACHE_KEY, codes)
        except ClientError:
            return None
        except Exception:
            return None
