"""
SkyNomad Sec 4.6 — unified region x mode utility, extended with a carbon lever.

    E        = e * ckpt_gb                       migration cost ($, egress)
    eta      = max(0, Lbar - d) / Lbar           effectiveness (1 for on-demand)
    C_total  = price $/hr + carbon $/hr          both levers in one unit
    U        = V * eta - C_total - E / Lbar      utility, $/hr throughout

``V`` comes from ``ProgressState.future_progress_value``; ``Lbar`` from the
Nelson-Aalen fit in ``spot/lifetime.py``. On-demand has ``Lbar = inf`` so
``eta = 1`` and ``E/Lbar = 0``, leaving ``U_od = V - C_total_od``. Idle scores 0,
which is what makes "wait" a real option rather than a special case.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from carbonsight_core.spot.progress import carbon_usd_per_hr

Mode = Literal["spot", "on_demand", "idle"]


@dataclass(frozen=True, slots=True)
class MigrationCostEstimator:
    """E = e * ckpt_gb, optionally plus the dollarised carbon of the transfer."""

    egress_usd_per_gb: float = 0.02
    egress_carbon_kg_per_gb: float = 0.0
    carbon_price_usd_per_ton: float = 50.0
    carbon_weight: float = 1.0

    def estimate(self, ckpt_size_gb: float) -> float:
        if ckpt_size_gb <= 0:
            return 0.0
        return self.egress_usd_per_gb * ckpt_size_gb + carbon_usd_per_hr(
            self.egress_carbon_kg_per_gb * ckpt_size_gb,
            self.carbon_price_usd_per_ton,
            self.carbon_weight,
        )


@dataclass(frozen=True, slots=True)
class CandidateState:
    """One region x mode option. Prices are totals for the job's GPU count."""

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
        return self.mode == "on_demand"

    @property
    def is_idle(self) -> bool:
        return self.mode == "idle"


def total_cost_per_hr(
    price_per_hr: float,
    carbon_kg_per_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """C_total = compute $/hr + carbon $/hr."""
    return price_per_hr + carbon_usd_per_hr(
        carbon_kg_per_hr, carbon_price_usd_per_ton, carbon_weight
    )


def effectiveness(mean_lifetime_hr: float, cold_start_hr: float) -> float:
    """eta = max(0, Lbar - d) / Lbar — the fraction of a lifetime spent working."""
    if math.isinf(mean_lifetime_hr):
        return 1.0
    if mean_lifetime_hr <= 1e-12:
        return 0.0
    return max(0.0, (mean_lifetime_hr - cold_start_hr) / mean_lifetime_hr)


def utility(
    v: float,
    eta: float,
    c_total_per_hr: float,
    migration_cost: float,
    mean_lifetime_hr: float,
) -> float:
    """U = V*eta - C_total - E/Lbar, all in $/hr."""
    if math.isinf(v):
        return math.inf if eta > 0 else -math.inf
    if math.isinf(mean_lifetime_hr):
        amortized = 0.0
    elif mean_lifetime_hr > 1e-12:
        amortized = migration_cost / mean_lifetime_hr
    else:
        return -math.inf
    return v * eta - c_total_per_hr - amortized


def candidate_utility(
    candidate: CandidateState,
    v: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """U for one candidate; idle is the zero baseline."""
    if candidate.is_idle:
        return 0.0
    return utility(
        v,
        effectiveness(candidate.mean_lifetime_hr, cold_start_hr),
        total_cost_per_hr(
            candidate.price_per_hr,
            candidate.carbon_kg_per_hr,
            carbon_price_usd_per_ton,
            carbon_weight,
        ),
        candidate.migration_cost,
        candidate.mean_lifetime_hr,
    )


def rank_candidates(
    candidates: Iterable[CandidateState],
    v: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> list[tuple[CandidateState, float]]:
    """``(candidate, U)`` sorted best-first."""
    scored = [
        (c, candidate_utility(c, v, cold_start_hr, carbon_price_usd_per_ton, carbon_weight))
        for c in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


__all__ = [
    "CandidateState",
    "MigrationCostEstimator",
    "Mode",
    "candidate_utility",
    "effectiveness",
    "rank_candidates",
    "total_cost_per_hr",
    "utility",
]
