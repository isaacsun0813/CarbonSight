"""Shared cloud provider interfaces and AWS infrastructure."""

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.base import (
    AvailabilityProvider,
    AvailabilityResult,
    OnDemandPricingProvider,
    SpotPricingProvider,
)

__all__ = [
    "AvailabilityProvider",
    "AvailabilityResult",
    "BaseAWSProvider",
    "HAS_BOTO",
    "OnDemandPricingProvider",
    "SpotPricingProvider",
]
