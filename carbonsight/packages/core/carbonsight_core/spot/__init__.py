"""Spot scheduling subpackage (SkyNomad-inspired)."""

# Availability tracking (Sec 4.3) — always available
try:
    from carbonsight_core.spot.availability import (  # noqa: F401
        AvailabilityTracker,
        SpotObservation,
        VirtualInstance,
    )

    _avail_exports = ["AvailabilityTracker", "SpotObservation", "VirtualInstance"]
except ImportError:
    _avail_exports = []

__all__ = list(_avail_exports)

# Lifetime / hazard analytics are optional; they may be provided by a parallel
# implementation. Import best-effort so missing module does not break the package.
try:
    from carbonsight_core.spot.lifetime import (  # type: ignore # noqa: F401
        LifetimeStats,
        compute_at_risk,
        compute_cumulative_hazard,
        compute_gamma,
        compute_gamma_star,
        compute_hazard,
        compute_survival,
        compute_volatility_adjusted_survival,
        expected_remaining,
        predict_remaining_lifetime,
    )

    __all__ += [
        "LifetimeStats",
        "compute_at_risk",
        "compute_hazard",
        "compute_cumulative_hazard",
        "compute_survival",
        "expected_remaining",
        "compute_gamma",
        "compute_gamma_star",
        "compute_volatility_adjusted_survival",
        "predict_remaining_lifetime",
    ]
except ImportError:
    # Lifetime module not present in this checkout — availability still works.
    pass

# Progress value V(t), thrifty, safety net (Sec 4.5)
try:
    from carbonsight_core.spot.progress import (  # type: ignore # noqa: F401
        ODCandidate,
        ProgressState,
        compute_safety_net_total_cost,
        safety_net_selection_for_state,
        select_cheapest_od_region,
    )

    __all__ += [
        "ProgressState",
        "ODCandidate",
        "compute_safety_net_total_cost",
        "select_cheapest_od_region",
        "safety_net_selection_for_state",
    ]
except ImportError:
    pass

# Unified model with carbon lever (Sec 4.6 extended)
try:
    from carbonsight_core.spot.unified_model import (  # type: ignore # noqa: F401
        Boto3SpotPriceProvider,
        CandidateState,
        MigrationCostEstimator,
        SpotPriceProvider,
        StaticSpotPriceProvider,
        candidate_utility,
        carbon_cost_per_hr,
        effectiveness,
        rank_candidates,
        total_cost_per_hr,
        utility,
    )

    __all__ += [
        "SpotPriceProvider",
        "StaticSpotPriceProvider",
        "Boto3SpotPriceProvider",
        "MigrationCostEstimator",
        "CandidateState",
        "carbon_cost_per_hr",
        "total_cost_per_hr",
        "effectiveness",
        "utility",
        "candidate_utility",
        "rank_candidates",
    ]
except ImportError:
    pass

# Policy (Algo1) – decision loop
try:
    from carbonsight_core.spot.policy import (  # type: ignore # noqa: F401
        Action,
        PolicyState,
        SkyNomadPolicy,
    )

    __all__ += [
        "PolicyState",
        "Action",
        "SkyNomadPolicy",
    ]
except ImportError:
    pass

# Also re-export canonical providers from providers/spot.py for convenience
# (avoids circular: spot/__init__ tries unified_model first, then direct fallback)
try:
    from carbonsight_core.providers.spot import (  # type: ignore # noqa: F401
        Boto3SpotPriceProvider as _Boto3SpotPriceProvider,
        SpotPriceProvider as _SpotPriceProvider,
        StaticSpotPriceProvider as _StaticSpotPriceProvider,
    )

    # Ensure they are in __all__ if unified_model import failed
    for _name in ("SpotPriceProvider", "StaticSpotPriceProvider", "Boto3SpotPriceProvider"):
        if _name not in __all__:
            __all__.append(_name)
except ImportError:
    pass
