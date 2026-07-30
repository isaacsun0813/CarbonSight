"""Providers: abstractions for carbon intensity and spot pricing."""

from .carbon import (
    ApiCarbonProvider,
    CarbonIntensityProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)
from .spot import (
    Boto3SpotPriceProvider,
    SpotPriceProvider,
    StaticSpotPriceProvider,
)

__all__ = [
    # Carbon
    "CarbonIntensityProvider",
    "WattTimeCarbonProvider",
    "ApiCarbonProvider",
    "SyntheticCarbonProvider",
    "get_carbon_provider",
    # Spot pricing — isolated abstraction to avoid conflict with dynamic wrapper branch
    "SpotPriceProvider",
    "StaticSpotPriceProvider",
    "Boto3SpotPriceProvider",
]
