"""WattTime client, synthetic MOER fallback, and the central forecast cache.

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
    SYNTHETIC_WATTTIME_REGIONS,
    WATTTIME_BASE,
    WattTimeClient,
    WattTimeError,
    build_synthetic_forecast,
    build_synthetic_forecast_payload,
    synthetic_moer_for_region,
)

__all__ = [
    "ForecastCache",
    "LB_TO_KG",
    "SUPPORTED_MOER_UNIT",
    "SYNTHETIC_WATTTIME_REGIONS",
    "WATTTIME_BASE",
    "WattTimeClient",
    "WattTimeError",
    "build_synthetic_forecast",
    "build_synthetic_forecast_payload",
    "get_global_forecast_cache",
    "mixture_weighted_moer",
    "reset_global_forecast_cache",
    "synthetic_moer_for_region",
    "time_weighted_moer",
]
