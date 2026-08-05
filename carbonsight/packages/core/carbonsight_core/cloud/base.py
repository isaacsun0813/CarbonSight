"""Cloud provider Protocol interfaces (structural typing for future multi-cloud)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from carbonsight_core.preflight.availability import AvailabilityResult


class OnDemandPricingProvider(Protocol):
    def ondemand_price_per_gpu_hour(self, gpu_type: str, region_code: str) -> float | None: ...


class SpotPricingProvider(Protocol):
    def spot_price_per_gpu_hour(self, gpu_type: str, region: str) -> float | None: ...


class AvailabilityProvider(Protocol):
    def check_instance_offering(self, region: str, gpu_type: str) -> AvailabilityResult: ...
