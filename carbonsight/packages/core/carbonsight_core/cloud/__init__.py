"""Shared cloud provider interfaces and AWS infrastructure."""

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.base import (
    AvailabilityProvider,
    OnDemandPricingProvider,
    SpotPricingProvider,
)

__all__ = [
    "AvailabilityProvider",
    "BaseAWSProvider",
    "HAS_BOTO",
    "OnDemandPricingProvider",
    "SpotPricingProvider",
]
