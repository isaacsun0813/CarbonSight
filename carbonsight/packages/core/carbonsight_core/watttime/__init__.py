"""
WattTime package: client + cache.

Keep backward compat: `from carbonsight_core.watttime import WattTimeClient` still works.
"""
from .client import (
    LB_TO_KG,
    SUPPORTED_MOER_UNIT,
    WATTTIME_BASE,
    WattTimeClient,
    WattTimeError,
)

# Re-export cache for convenience, but avoid circular import at top-level
try:
    from .cache import ForecastCache, _time_weighted_moer  # noqa: F401
except Exception:
    # cache may import client, but client already loaded, so this should succeed
    # If it fails in some edge, don't break client import
    pass

__all__ = [
    "WattTimeClient",
    "WattTimeError",
    "WATTTIME_BASE",
    "SUPPORTED_MOER_UNIT",
    "LB_TO_KG",
    "ForecastCache",
]
