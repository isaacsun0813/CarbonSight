"""Carbon intensity: how dirty is the power, per grid region, over a window.

The sibling of ``cloud/``: that one answers what compute costs and whether you can
get it; this one answers what running it emits.
"""

from carbonsight_core.carbon.base import (
    CarbonIntensityProvider,
    CarbonProviderError,
    ForecastBackedProvider,
    as_utc,
    facility_mwh_per_hr,
)
from carbonsight_core.carbon.providers import (
    ApiCarbonProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)

__all__ = [
    "ApiCarbonProvider",
    "CarbonIntensityProvider",
    "CarbonProviderError",
    "ForecastBackedProvider",
    "SyntheticCarbonProvider",
    "WattTimeCarbonProvider",
    "as_utc",
    "facility_mwh_per_hr",
    "get_carbon_provider",
]
