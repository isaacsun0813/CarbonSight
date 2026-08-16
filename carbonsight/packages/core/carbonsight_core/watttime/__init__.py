"""WattTime client and the central forecast cache.

Re-exports the flat-module surface (``WattTimeClient``, ``WattTimeError``, ``LB_TO_KG``,
``SUPPORTED_MOER_UNIT``) so ``from carbonsight_core.watttime import ...`` keeps working.
"""

from carbonsight_core.watttime.cache import (
    ForecastCache,
    get_global_forecast_cache,
    mixture_weighted_moer,
    reset_global_forecast_cache,
    time_weighted_moer,
)
from carbonsight_core.watttime.client import (
    LB_TO_KG,
    SUPPORTED_MOER_UNIT,
    WATTTIME_BASE,
    WattTimeClient,
    WattTimeError,
)

__all__ = [
    "ForecastCache",
    "LB_TO_KG",
    "SUPPORTED_MOER_UNIT",
    "WATTTIME_BASE",
    "WattTimeClient",
    "WattTimeError",
    "get_global_forecast_cache",
    "mixture_weighted_moer",
    "reset_global_forecast_cache",
    "time_weighted_moer",
]
