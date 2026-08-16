"""Carbon intensity: how dirty is the power, per grid region, over a window.

The sibling of ``cloud/``: that one answers what compute costs and whether you can
get it; this one answers what running it emits.
"""

from carbonsight_core.carbon.base import (
    CarbonIntensityProvider,
    CarbonProviderError,
    ForecastBackedProvider,
    as_utc,
)
from carbonsight_core.carbon.providers import (
    ApiCarbonProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)
from carbonsight_core.carbon.synthetic import (
    SYNTHETIC_WATTTIME_REGIONS,
    build_synthetic_forecast,
    synthetic_moer_for_region,
)

__all__ = [
    "ApiCarbonProvider",
    "CarbonIntensityProvider",
    "CarbonProviderError",
    "ForecastBackedProvider",
    "SyntheticCarbonProvider",
    "SYNTHETIC_WATTTIME_REGIONS",
    "WattTimeCarbonProvider",
    "as_utc",
    "get_carbon_provider",
    "build_synthetic_forecast",
    "synthetic_moer_for_region",
]
