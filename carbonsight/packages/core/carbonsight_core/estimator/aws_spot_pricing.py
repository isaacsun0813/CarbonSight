"""
Live EC2 spot pricing via describe_spot_price_history.

Fails open: returns None on missing boto3, unknown GPU, or API errors so callers can use static fallback.
"""

import time
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.estimator.gpu_catalog import gpu_instance_spec

try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

DEFAULT_CACHE_TTL_SECONDS = 3600
SPOT_HISTORY_MAX_RESULTS = 50


class SpotPriceProvider:
    """Fetch regional spot $/GPU/hr from EC2 spot price history; cache per (region, instance_type)."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._cache_ttl = cache_ttl_seconds or self._config.pricing_cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}
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

    def _get_cached(self, region: str, instance_type: str) -> float | None:
        key = (region, instance_type)
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, region: str, instance_type: str, price_per_gpu_hr: float) -> None:
        self._cache[(region, instance_type)] = (time.time() + self._cache_ttl, price_per_gpu_hr)

    def spot_price_per_gpu_hour(self, gpu_type: str, region: str) -> float | None:
        """
        Return minimum recent spot $/GPU/hr in the region, or None if unavailable.

        Uses min SpotPrice across AZs in the API response (conservative for cost ranking).
        """
        if not _HAS_BOTO:
            return None

        spec = gpu_instance_spec(gpu_type)
        if spec is None:
            return None

        instance_type = spec.instance_type
        cached = self._get_cached(region, instance_type)
        if cached is not None:
            return cached

        try:
            ec2 = self._ec2_client(region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=SPOT_HISTORY_MAX_RESULTS,
            )
            history = resp.get("SpotPriceHistory", [])
            if not history:
                return None

            min_instance_hr = min(float(entry["SpotPrice"]) for entry in history)
            price_per_gpu = min_instance_hr / spec.gpus_per_instance
            self._set_cached(region, instance_type, price_per_gpu)
            return price_per_gpu
        except ClientError:
            return None
        except Exception:
            return None
