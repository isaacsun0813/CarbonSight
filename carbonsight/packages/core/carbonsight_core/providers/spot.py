"""
Isolated spot price provider abstraction.

This module is the canonical location for SpotPriceProvider protocol and its
implementations, kept separate from estimator/pricing.py to avoid merge
conflicts with the dynamic pricing wrapper branch.

Design:
- Protocol lives here, not in spot/unified_model.py
- StaticSpotPriceProvider is a read-only wrapper around estimator/pricing.py
  constants (copies dicts, no mutation)
- Boto3SpotPriceProvider is a stub/interface placeholder for the wrapper
  branch that will fetch live EC2 spot price history via boto3 and cache it.

unified_model.py should import from here instead of defining duplicates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class SpotPriceProvider(Protocol):
    """
    Minimal protocol for spot/OD price lookup.

    Implemented by:
    - StaticSpotPriceProvider (read-only dicts)
    - Boto3SpotPriceProvider (live AWS, cache)
    - Any dynamic pricing wrapper branch class
    """

    def get_spot_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        """Return spot $/GPU/hr for region×gpu_type."""
        ...

    def get_od_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        """Return on-demand $/GPU/hr for region×gpu_type."""
        ...

    def get_egress_cost_per_gb(self, src: str, dst: str) -> float:
        """
        Optional: return egress $/GB from src→dst.

        Default implementations may return a flat rate (e.g. 0.02) or
        region-pair specific pricing. Return 0.0 if src==dst.

        This method is optional in the sense that simple providers may
        not need geographic cost; but it is part of the protocol so
        unified_model/migration estimators can query it uniformly.
        """
        ...


# ---------------------------------------------------------------------------
# Static fallback — read-only wrapper around estimator/pricing.py
# ---------------------------------------------------------------------------


class StaticSpotPriceProvider:
    """
    Read-only fallback that uses existing pricing.py constants without editing them.

    Implements SpotPriceProvider protocol:
        od   = base_price[gpu] * region_multiplier
        spot = od * SPOT_PRICE_FRACTION

    Copies dicts on construction to guarantee no mutation of the source
    module's globals. Lazy-imported to avoid hard circular dependencies.

    Args:
        default_gpu_type: used when caller omits gpu_type (kept for convenience)

    Example:
        provider = StaticSpotPriceProvider()
        od   = provider.get_od_price_usd_per_gpu_hr("us-west-2", "A100")
        spot = provider.get_spot_price_usd_per_gpu_hr("us-west-2", "A100")
    """

    def __init__(self, default_gpu_type: str = "A100"):
        self._default_gpu_type = default_gpu_type
        # Lazy import to avoid circular dependency; we explicitly do NOT mutate dicts.
        from carbonsight_core.estimator.pricing import (  # noqa: WPS433
            _DEFAULT_GPU_PRICE,
            _DEFAULT_MULTIPLIER,
            _GPU_BASE_PRICE_USD_PER_HR,
            _REGION_MULTIPLIER,
            SPOT_PRICE_FRACTION,
        )

        self._base: Dict[str, float] = dict(_GPU_BASE_PRICE_USD_PER_HR)  # copy read-only
        self._region_mult: Dict[str, float] = dict(_REGION_MULTIPLIER)
        self._default_price: float = _DEFAULT_GPU_PRICE
        self._default_mult: float = _DEFAULT_MULTIPLIER
        self._spot_frac: float = SPOT_PRICE_FRACTION
        self._egress_usd_per_gb: float = 0.02  # AWS inter-region default, $/GB

    def _od_per_gpu_hr(self, region: str, gpu_type: str) -> float:
        base = self._base.get(gpu_type.upper(), self._default_price)
        mult = self._region_mult.get(region, self._default_mult)
        return base * mult

    def get_od_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        gpu_type = gpu_type or self._default_gpu_type
        return self._od_per_gpu_hr(region, gpu_type)

    def get_spot_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        gpu_type = gpu_type or self._default_gpu_type
        return self._od_per_gpu_hr(region, gpu_type) * self._spot_frac

    def get_egress_cost_per_gb(self, src: str, dst: str) -> float:
        """Flat $/GB; 0 if same region."""
        if not src or not dst or src == dst:
            return 0.0
        return self._egress_usd_per_gb

    # Convenience: total $/hr for n GPUs (not part of protocol but widely used)
    def get_od_price_usd_per_hr(self, region: str, gpu_type: str, gpu_count: int = 1) -> float:
        return self.get_od_price_usd_per_gpu_hr(region, gpu_type) * gpu_count

    def get_spot_price_usd_per_hr_total(
        self, region: str, gpu_type: str, gpu_count: int = 1
    ) -> float:
        return self.get_spot_price_usd_per_gpu_hr(region, gpu_type) * gpu_count


# ---------------------------------------------------------------------------
# Boto3 / live AWS stub — to be implemented by dynamic pricing wrapper branch
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    value: float
    ts: float  # epoch seconds


@dataclass
class Boto3SpotPriceProvider:
    """
    Stub / interface placeholder for live EC2 spot price fetching.

    Intent:
        The dynamic pricing wrapper branch should replace the placeholder
        _fetch_* methods with real boto3 calls that:
        - Map gpu_type → EC2 instance families (e.g. A100→p4d.24xlarge, H100→p5.48xlarge,
          V100→p3.16xlarge, T4→g4dn.xlarge, A10G→g5.xlarge)
        - Call ec2.describe_spot_price_history (or pricing API) per region×instance_type
          with lookback (e.g. 1h-24h) and compute median/min p50
        - Divide by GPUs per instance to get $/GPU/hr
        - Cache result with TTL to avoid throttling

    How the wrapper branch should implement:

    1. Inject a boto3 EC2 client (or Session) via __init__ or setter.
       - To keep core dependency-free, do NOT import boto3 at module import time;
         import lazily inside _fetch_* methods or accept client via constructor.

    2. Implement _fetch_spot_price_usd_per_gpu_hr(region, gpu_type):
        - Resolve gpu_type.upper() → list of instance types
        - GPUs per instance: {"p4d.24xlarge":8, "p5.48xlarge":8, "p3.16xlarge":8,
          "g4dn.xlarge":1, "g5.xlarge":1, ...}
        - Call client.describe_spot_price_history(
              InstanceTypes=[...], ProductDescriptions=["Linux/UNIX"],
              StartTime=..., EndTime=..., MaxResults=100)
        - Take recent spot price median, / gpus_per_instance
        - Return float USD

    3. Implement _fetch_od_price_usd_per_gpu_hr(region, gpu_type):
        - Option A: use AWS Pricing API get_products with filters
          (ServiceCode=AmazonEC2, location, instanceType, operatingSystem, preInstalledSw=NA ...)
        - Option B: reuse static _GPU_BASE_PRICE_USD_PER_HR * multiplier as fallback
          if pricing API not available.

    4. Caching:
        - This stub already provides _cache dict keyed by (kind, region, gpu_type)
          and TTL check. Wrapper only needs to call self._get_cached / self._set_cached.

    5. Thread safety: if used from worker threads, wrap cache access with a lock
       (not done in stub to keep deps minimal).

    6. Error handling:
        - On boto3 error / empty history, fallback to static provider values
          (self._static fallback) and log warning.
        - Raise only if no fallback configured and allow caller to handle.

    This stub is functional for imports but will raise NotImplementedError
    unless methods are overridden or ec2_client is supplied with complete logic.
    It is deliberately NOT editing estimator/pricing.py.
    """

    default_gpu_type: str = "A100"
    cache_ttl_seconds: int = 900  # 15 min default
    egress_usd_per_gb: float = 0.02

    # Optional injection — real implementation passes boto3 client/session
    ec2_client: Optional[object] = None  # type: ignore[no-redef]
    pricing_client: Optional[object] = None

    # Internal caches
    _cache: Dict[tuple[str, str, str], _CacheEntry] = field(default_factory=dict, init=False, repr=False)
    _static: StaticSpotPriceProvider = field(default_factory=StaticSpotPriceProvider, init=False, repr=False)

    def __init__(
        self,
        default_gpu_type: str = "A100",
        cache_ttl_seconds: int = 900,
        egress_usd_per_gb: float = 0.02,
        ec2_client: Optional[object] = None,
        pricing_client: Optional[object] = None,
    ):
        self.default_gpu_type = default_gpu_type
        self.cache_ttl_seconds = cache_ttl_seconds
        self.egress_usd_per_gb = egress_usd_per_gb
        self.ec2_client = ec2_client
        self.pricing_client = pricing_client
        self._cache = {}
        self._static = StaticSpotPriceProvider(default_gpu_type=default_gpu_type)

    # --- Cache helpers ---------------------------------------------------

    def _cache_key(self, kind: str, region: str, gpu_type: str) -> tuple[str, str, str]:
        return (kind, region, gpu_type.upper())

    def _get_cached(self, kind: str, region: str, gpu_type: str) -> Optional[float]:
        key = self._cache_key(kind, region, gpu_type)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (time.time() - entry.ts) <= self.cache_ttl_seconds:
            return entry.value
        # expired
        del self._cache[key]
        return None

    def _set_cached(self, kind: str, region: str, gpu_type: str, value: float) -> None:
        key = self._cache_key(kind, region, gpu_type)
        self._cache[key] = _CacheEntry(value=value, ts=time.time())

    def invalidate_cache(self, region: Optional[str] = None, gpu_type: Optional[str] = None) -> None:
        """Invalidate all or subset of cache."""
        if region is None and gpu_type is None:
            self._cache.clear()
            return
        to_del = []
        for (kind, r, g) in self._cache.keys():
            if region is not None and r != region:
                continue
            if gpu_type is not None and g != gpu_type.upper():
                continue
            to_del.append((kind, r, g))
        for k in to_del:
            self._cache.pop(k, None)

    # --- Placeholder fetch methods — wrapper branch should override --------

    def _fetch_spot_price_usd_per_gpu_hr(self, region: str, gpu_type: str) -> float:
        """
        Fetch live spot $/GPU/hr from EC2 spot price history.

        Placeholder: wrapper branch replaces with boto3 logic.

        Example implementation sketch (not executed in stub):
            import boto3
            client = self.ec2_client or boto3.client("ec2", region_name=region)
            # gpu_type → instance types mapping
            mapping = {
              "A100": (["p4d.24xlarge"], 8),
              "H100": (["p5.48xlarge"], 8),
              "V100": (["p3.16xlarge"], 8),
              "T4":   (["g4dn.xlarge"], 1),
              "A10G": (["g5.xlarge"], 1),
            }
            instance_types, gpus_per = mapping.get(gpu_type.upper(), (["p4d.24xlarge"], 8))
            resp = client.describe_spot_price_history(
                InstanceTypes=instance_types,
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=100,
            )
            prices = [float(p["SpotPrice"]) for p in resp.get("SpotPriceHistory", [])]
            if not prices:
                raise ValueError("No spot history")
            median = sorted(prices)[len(prices)//2]
            return median / gpus_per
        """
        if self.ec2_client is None:
            raise NotImplementedError(
                "Boto3SpotPriceProvider._fetch_spot_price_usd_per_gpu_hr not implemented: "
                "provide ec2_client or override method in wrapper branch. "
                "See docstring for implementation guide. Falling back to static pricing is optional."
            )
        # If client present but subclass did not override, explicitly signal
        raise NotImplementedError(
            "ec2_client supplied but fetch logic not overridden — "
            "wrapper branch must implement _fetch_spot_price_usd_per_gpu_hr"
        )

    def _fetch_od_price_usd_per_gpu_hr(self, region: str, gpu_type: str) -> float:
        """
        Fetch live OD $/GPU/hr from AWS Pricing API.

        Placeholder: wrapper can implement with pricing_client.get_products.

        Fallback: returns static pricing value if pricing_client not available.
        """
        if self.pricing_client is None:
            # Graceful fallback to static — still useful for tests without AWS creds
            return self._static.get_od_price_usd_per_gpu_hr(region, gpu_type)
        raise NotImplementedError(
            "pricing_client supplied but OD fetch not overridden — "
            "wrapper branch must implement _fetch_od_price_usd_per_gpu_hr"
        )

    # --- Protocol implementation -----------------------------------------

    def get_od_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        gpu_type = (gpu_type or self.default_gpu_type).upper()
        cached = self._get_cached("od", region, gpu_type)
        if cached is not None:
            return cached
        try:
            value = self._fetch_od_price_usd_per_gpu_hr(region, gpu_type)
        except NotImplementedError:
            raise
        except Exception:
            # Best-effort fallback to static if live fetch fails
            value = self._static.get_od_price_usd_per_gpu_hr(region, gpu_type)
        self._set_cached("od", region, gpu_type, value)
        return value

    def get_spot_price_usd_per_gpu_hr(self, region: str, gpu_type: str = "A100") -> float:
        gpu_type = (gpu_type or self.default_gpu_type).upper()
        cached = self._get_cached("spot", region, gpu_type)
        if cached is not None:
            return cached
        try:
            value = self._fetch_spot_price_usd_per_gpu_hr(region, gpu_type)
        except NotImplementedError:
            raise
        except Exception:
            value = self._static.get_spot_price_usd_per_gpu_hr(region, gpu_type)
        self._set_cached("spot", region, gpu_type, value)
        return value

    def get_egress_cost_per_gb(self, src: str, dst: str) -> float:
        """
        Return egress $/GB.

        Default: flat rate, 0 if src==dst.
        Wrapper may override with region-pair table:
          e.g. same continent 0.01, cross-continent 0.02, etc.
        """
        if not src or not dst or src == dst:
            return 0.0
        return self.egress_usd_per_gb

    # Convenience total-hr helpers (mirrors static provider)
    def get_od_price_usd_per_hr(self, region: str, gpu_type: str, gpu_count: int = 1) -> float:
        return self.get_od_price_usd_per_gpu_hr(region, gpu_type) * gpu_count

    def get_spot_price_usd_per_hr_total(
        self, region: str, gpu_type: str, gpu_count: int = 1
    ) -> float:
        return self.get_spot_price_usd_per_gpu_hr(region, gpu_type) * gpu_count


__all__ = [
    "SpotPriceProvider",
    "StaticSpotPriceProvider",
    "Boto3SpotPriceProvider",
]
