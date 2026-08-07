"""Carbon intensity: how dirty is the power, per grid region, over a window.

The sibling of ``cloud/``: that one answers what compute costs and whether you can
get it; this one answers what running it emits.
"""

from carbonsight_core.carbon.base import (
    FALLBACK_MOER_LB_PER_MWH,
    CarbonIntensityProvider,
    ForecastBackedProvider,
    as_utc,
    facility_mwh_per_hr,
)
from carbonsight_core.carbon.providers import (
    ApiCarbonProvider,
    CarbonProviderError,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)

__all__ = [
    "FALLBACK_MOER_LB_PER_MWH",
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
