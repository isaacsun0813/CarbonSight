"""
Unified cost model with carbon lever, migration cost, effectiveness, utility.

Per SkyNomad Sec 4.6 extended with carbon:

- Migration cost: E = e · ckpt_size
  where e = egress $/GB (default 0.02 $/GB), ckpt_size in GB.
  Optionally + egress carbon · price, if provided.

- Effectiveness: η = max(0, L̄ - d) / L̄
  where L̄ = expected remaining lifetime (mean lifetime hr),
        d = cold start / restore delay (hours).
  For OD: L̄ → ∞ ⇒ η → 1.

- Candidate state:
  CandidateState(region, mode, L̄, price, carbon_kg_per_hr, migration_cost)

- Carbon $ conversion (same as providers/carbon.py):
  carbon_$ /hr = carbon_kg_per_hr /1000 · price_per_ton · weight

- Total cost: C_total = spot_price $/hr + carbon $/hr

- Utility: U_s = V·η - C_total - E/L̄
  where V = future progress value from progress.py,
        E/L̄ = amortized migration cost per hour of lifetime.
  For OD with L̄=∞, η=1, E/L̄ →0 ⇒ U_od = V - C_od_total
  For idle: U=0

Spot price provider protocol + minimal static fallback (no edit of pricing.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

# Canonical spot provider abstraction lives in providers/spot.py to avoid
# conflict with dynamic pricing wrapper branch. Re-export here for
# backward compatibility so existing imports from unified_model continue to work.
from carbonsight_core.providers.spot import (
    Boto3SpotPriceProvider,  # also re-exported for convenience
    SpotPriceProvider,
    StaticSpotPriceProvider,
)


# ---------------------------------------------------------------------------
# Migration cost estimator: E = e * ckpt_size
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MigrationCostEstimator:
    """
    Estimates migration cost E = e · ckpt_size_gb.

    Attributes:
        egress_usd_per_gb: e in $/GB (default 0.02 AWS inter-region egress)
        egress_carbon_kg_per_gb: optional network carbon kg/GB (adds to $ via price)
        carbon_price_usd_per_ton: price for converting egress carbon to $
        carbon_weight: multiplier
    """

    egress_usd_per_gb: float = 0.02
    egress_carbon_kg_per_gb: float = 0.0
    carbon_price_usd_per_ton: float = 50.0
    carbon_weight: float = 1.0

    def estimate(self, ckpt_size_gb: float) -> float:
        """
        Pure $ egress cost: E = e·ckpt_size

        Args:
            ckpt_size_gb: checkpoint size in GB

        Returns:
            $ cost of migration
        """
        if ckpt_size_gb <= 0:
            return 0.0
        dollar = self.egress_usd_per_gb * ckpt_size_gb
        if self.egress_carbon_kg_per_gb:
            carbon_kg = self.egress_carbon_kg_per_gb * ckpt_size_gb
            dollar += carbon_kg / 1000.0 * self.carbon_price_usd_per_ton * self.carbon_weight
        return dollar

    def estimate_breakdown(self, ckpt_size_gb: float) -> Tuple[float, float, float]:
        """
        Returns (egress_$, carbon_kg, total_$)
        """
        if ckpt_size_gb <= 0:
            return 0.0, 0.0, 0.0
        egress = self.egress_usd_per_gb * ckpt_size_gb
        carbon_kg = self.egress_carbon_kg_per_gb * ckpt_size_gb
        carbon_dollar = carbon_kg / 1000.0 * self.carbon_price_usd_per_ton * self.carbon_weight
        return egress, carbon_kg, egress + carbon_dollar


# ---------------------------------------------------------------------------
# Candidate & utility
# ---------------------------------------------------------------------------

Mode = Literal["spot", "on_demand", "od", "idle"]


@dataclass(frozen=True, slots=True)
class CandidateState:
    """
    Region×mode candidate for unified ranking.

    Attributes:
        region: cloud region (e.g. us-east-1)
        mode: spot | on_demand (or od) | idle
        mean_lifetime_hr: L̄ predicted remaining lifetime (hours).
                          For OD use math.inf; for idle irrelevant.
        price_per_hr: spot $/hr or OD $/hr (already includes gpu_count scaling)
        carbon_kg_per_hr: operational carbon intensity kgCO2/hr (facility MWh·MOER)
        migration_cost: E_{r0→r} $ (from MigrationCostEstimator)
    """

    region: str
    mode: Mode
    mean_lifetime_hr: float
    price_per_hr: float
    carbon_kg_per_hr: float
    migration_cost: float = 0.0

    @property
    def is_spot(self) -> bool:
        return self.mode == "spot"

    @property
    def is_od(self) -> bool:
        return self.mode in ("on_demand", "od")

    @property
    def is_idle(self) -> bool:
        return self.mode == "idle"


def carbon_cost_per_hr(
    carbon_kg_per_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """
    Convert kgCO2/hr to $/hr: kg/1000·price·weight

    Mirrors providers/carbon.py CarbonIntensityProvider.get_cost_per_hr
    """
    if carbon_kg_per_hr <= 0 or carbon_price_usd_per_ton <= 0:
        return 0.0
    return carbon_kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight


def total_cost_per_hr(
    price_per_hr: float,
    carbon_kg_per_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """
    C_total = spot $ + carbon $   (multi-lever $)

    Args:
        price_per_hr: spot or OD $/hr
        carbon_kg_per_hr: kg/hr
        carbon_price_usd_per_ton: $/ton (configurable, default 50 EPA SCC)
        carbon_weight: lambda weight
    """
    return price_per_hr + carbon_cost_per_hr(
        carbon_kg_per_hr, carbon_price_usd_per_ton, carbon_weight
    )


def effectiveness(mean_lifetime_hr: float, cold_start_hr: float) -> float:
    """
    η = max(0, L̄ - d) / L̄

    Fraction of lifetime that does useful work after cold start.

    Args:
        mean_lifetime_hr: L̄ expected remaining lifetime
        cold_start_hr: d cold start + restore delay

    Returns:
        η ∈ [0,1].  1 for OD (L̄=inf), 0 if L̄≤d or L̄≤0.
    """
    if math.isinf(mean_lifetime_hr):
        return 1.0
    if mean_lifetime_hr <= 1e-12:
        return 0.0
    eff = (mean_lifetime_hr - cold_start_hr) / mean_lifetime_hr
    return max(0.0, eff)


def utility(
    V: float,
    eta: float,
    c_total_per_hr: float,
    migration_cost: float,
    mean_lifetime_hr: float,
) -> float:
    """
    Unified utility: U_s = V·η - C_total - E/L̄

    Args:
        V: future progress value (from ProgressState.future_progress_value)
        eta: effectiveness = max(0,L̄-d)/L̄
        c_total_per_hr: C_total = spot$ + carbon$ per hr
        migration_cost: E (total $)
        mean_lifetime_hr: L̄ (inf for OD → E/L̄=0)

    Returns:
        Utility $/hr equivalent. Higher is better. Negative means
        candidate costs more than progress value (should idle).
        For idle mode caller should return 0.
    """
    if math.isinf(V):
        # Infinite deadline pressure: any feasible candidate huge positive
        # if it can make progress (eta>0)
        return math.inf if eta > 0 else -math.inf

    amortized = 0.0
    if math.isinf(mean_lifetime_hr):
        amortized = 0.0
    elif mean_lifetime_hr > 1e-12:
        amortized = migration_cost / mean_lifetime_hr
    else:
        # L̄=0 → migration infinitely amortized negative
        return -math.inf

    return V * eta - c_total_per_hr - amortized


def candidate_utility(
    candidate: CandidateState,
    V: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """
    Compute U_s for a given CandidateState.

    Args:
        candidate: region×mode with L̄, price, carbon_kg/hr, E
        V: future progress value (V(t) from progress.py)
        cold_start_hr: d
        carbon_price_usd_per_ton: configurable $/ton
        carbon_weight: lambda carbon weight

    Returns:
        Utility float. For idle returns 0.
    """
    if candidate.is_idle:
        return 0.0

    eta = effectiveness(candidate.mean_lifetime_hr, cold_start_hr)
    c_total = total_cost_per_hr(
        candidate.price_per_hr,
        candidate.carbon_kg_per_hr,
        carbon_price_usd_per_ton,
        carbon_weight,
    )
    return utility(V, eta, c_total, candidate.migration_cost, candidate.mean_lifetime_hr)


def rank_candidates(
    candidates: Iterable[CandidateState],
    V: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> List[Tuple[CandidateState, float]]:
    """
    Rank candidates by utility descending.

    Returns list of (candidate, U) sorted high→low.
    """
    scored: List[Tuple[CandidateState, float]] = []
    for c in candidates:
        u = candidate_utility(c, V, cold_start_hr, carbon_price_usd_per_ton, carbon_weight)
        scored.append((c, u))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def carbon_aware_flip_example() -> str:
    """
    Tiny doc for test: demonstrates multi-lever flip.

    Region A (green, expensive): spot $4/hr, carbon 0.1 kg/hr
    Region B (dirty, cheap):    spot $3/hr, carbon 0.8 kg/hr
    At carbon_price=0, B cheaper total → wins.
    At carbon_price=20000 $/ton (≈ extreme SCC), A wins despite $1 premium.
    This matches existing tests/test_carbon_provider.py pattern.
    """
    return "see docs"


__all__ = [
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
