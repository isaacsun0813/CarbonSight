"""
Spot price providers — isolated from estimator/pricing.py mutation surface.

StaticSpotPriceProvider copies base dicts at init (read-only snapshot).
Boto3SpotPriceProvider is a structural stub for parallel EC2 spot-history work.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from carbonsight_core.estimator import pricing as _pricing

# Snapshot copies — never mutate the module-level dicts in pricing.py
_GPU_BASE_SNAPSHOT: dict[str, float] = dict(_pricing._GPU_BASE_PRICE_USD_PER_HR)
_REGION_MULT_SNAPSHOT: dict[str, float] = dict(_pricing._REGION_MULTIPLIER)
_SPOT_FRACTION: float = float(_pricing.SPOT_PRICE_FRACTION)
_DEFAULT_GPU = float(getattr(_pricing, "_DEFAULT_GPU_PRICE", 4.10))
_DEFAULT_MULT = float(getattr(_pricing, "_DEFAULT_MULTIPLIER", 1.20))


@runtime_checkable
class SpotPriceProvider(Protocol):
    """Protocol: spot USD per GPU-hour for (region, instance/gpu type)."""

    def get_spot_price_usd_per_gpu_hr(self, region: str, instance_type: str) -> float:
        ...


class StaticSpotPriceProvider:
    """Read-only copy of pricing tables × SPOT_PRICE_FRACTION.

    Does not import pricing for mutation — snapshots at construction time.
    """

    def __init__(
        self,
        *,
        gpu_base: dict[str, float] | None = None,
        region_multiplier: dict[str, float] | None = None,
        spot_fraction: float | None = None,
    ) -> None:
        self._gpu_base = dict(gpu_base) if gpu_base is not None else dict(_GPU_BASE_SNAPSHOT)
        self._region_mult = (
            dict(region_multiplier) if region_multiplier is not None else dict(_REGION_MULT_SNAPSHOT)
        )
        self._spot_fraction = float(spot_fraction) if spot_fraction is not None else _SPOT_FRACTION
        self._default_gpu = _DEFAULT_GPU
        self._default_mult = _DEFAULT_MULT

    def get_spot_price_usd_per_gpu_hr(self, region: str, instance_type: str) -> float:
        # instance_type may be "A100" or "p4d.24xlarge" — try GPU key first
        key = instance_type.upper()
        base = self._gpu_base.get(key)
        if base is None:
            # strip common prefixes / try bare GPU token
            for gpu in self._gpu_base:
                if gpu in key:
                    base = self._gpu_base[gpu]
                    break
        if base is None:
            base = self._default_gpu
        mult = self._region_mult.get(region, self._default_mult)
        return float(base) * float(mult) * self._spot_fraction

    def get_on_demand_usd_per_gpu_hr(self, region: str, instance_type: str) -> float:
        spot = self.get_spot_price_usd_per_gpu_hr(region, instance_type)
        if self._spot_fraction <= 0:
            return spot
        return spot / self._spot_fraction


class Boto3SpotPriceProvider:
    """Stub for live EC2 spot price history (parallel workstream).

    Structure exists so pricing stays behind SpotPriceProvider Protocol.
    Until wired, falls back to StaticSpotPriceProvider.
    """

    def __init__(self, cache_ttl: int = 3600, *, use_fallback: bool = True) -> None:
        self._cache_ttl = int(cache_ttl)
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}  # key -> (expires, price)
        self._fallback = StaticSpotPriceProvider() if use_fallback else None

    def get_spot_price_usd_per_gpu_hr(self, region: str, instance_type: str) -> float:
        key = (region, instance_type)
        now = time.time()
        hit = self._cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        try:
            price = self._fetch_from_boto3(region, instance_type)
        except NotImplementedError:
            if self._fallback is None:
                raise
            price = self._fallback.get_spot_price_usd_per_gpu_hr(region, instance_type)
        self._cache[key] = (now + self._cache_ttl, price)
        return price

    def _fetch_from_boto3(self, region: str, instance_type: str) -> float:
        """Placeholder for boto3 EC2 describe_spot_price_history."""
        raise NotImplementedError(
            "Boto3SpotPriceProvider._fetch_from_boto3 is a stub; "
            "use StaticSpotPriceProvider or enable fallback."
        )


def get_spot_price_provider(*, live: bool = False) -> SpotPriceProvider:
    """Factory: static by default; live selects boto3 stub (with static fallback)."""
    if live:
        return Boto3SpotPriceProvider(use_fallback=True)
    return StaticSpotPriceProvider()
