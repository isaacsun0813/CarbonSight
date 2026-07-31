"""Carbon and spot price providers."""

from carbonsight_core.providers.carbon import (
    DEFAULT_CARBON_REGIONS,
    ApiCarbonProvider,
    CarbonIntensityProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)
from carbonsight_core.providers.spot import (
    Boto3SpotPriceProvider,
    SpotPriceProvider,
    StaticSpotPriceProvider,
    get_spot_price_provider,
)

__all__ = [
    "ApiCarbonProvider",
    "Boto3SpotPriceProvider",
    "CarbonIntensityProvider",
    "DEFAULT_CARBON_REGIONS",
    "SpotPriceProvider",
    "StaticSpotPriceProvider",
    "SyntheticCarbonProvider",
    "WattTimeCarbonProvider",
    "get_carbon_provider",
    "get_spot_price_provider",
]
