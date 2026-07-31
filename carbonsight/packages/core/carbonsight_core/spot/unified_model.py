"""
SkyNomad Sec 4.6 — Unified multi-lever model (cost + carbon + time + availability).

U_s = V * eta - C_total - E / Lbar

where:
  V(t) = C_od * theta / theta_tilde     value of finishing
  eta  = instance efficiency (perf / cost proxy)
  C_total = spot_cost + carbon_dollarized
  E = eviction_penalty * (1 - survival)
  Lbar = expected remaining lifetime hours
"""

from __future__ import annotations

from dataclasses import dataclass, field

from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.progress import ProgressState

# Re-export canonical provider (single definition lives in providers/spot.py)
__all__ = [
    "ODCandidate",
    "StaticSpotPriceProvider",
    "compute_utility",
    "rank_candidates",
    "facility_mwh_for_job",
]

LB_TO_KG = 0.45359237
KG_PER_TON = 1000.0

# Default facility power draw proxy (kW) per GPU for tiny carbon estimates
_DEFAULT_GPU_KW = 0.3
_DEFAULT_PUE = 1.2


@dataclass
class ODCandidate:
    """One on-demand/spot placement candidate for joint ranking."""

    region: str
    instance_type: str
    spot_price: float  # USD per GPU-hour
    moer: float  # lb CO2 / MWh (time-weighted over Lbar)
    survival: float = 0.9
    lbar: float = 12.0  # expected remaining hours
    on_demand_price: float | None = None  # USD per GPU-hour; default spot/0.35
    gpu_count: int = 1
    efficiency: float = 1.0  # eta
    notes: list[str] = field(default_factory=list)

    # Filled by compute_utility
    utility: float = 0.0
    c_total: float = 0.0
    carbon_kg: float = 0.0
    value_v: float = 0.0


def facility_mwh_for_job(
    *,
    gpu_count: int,
    lbar_hours: float,
    gpu_kw: float = _DEFAULT_GPU_KW,
    pue: float = _DEFAULT_PUE,
) -> float:
    """Rough facility MWh over remaining lifetime: kW * h * PUE / 1000."""
    return max(0.0, gpu_count * gpu_kw * lbar_hours * pue / 1000.0)


def compute_utility(
    cand: ODCandidate,
    progress: ProgressState,
    *,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
    eviction_penalty: float = 50.0,
    theta_tilde: float | None = None,
    gpu_kw: float = _DEFAULT_GPU_KW,
    pue: float = _DEFAULT_PUE,
) -> float:
    """Compute U_s for one candidate and mutate cand with intermediates."""
    theta = progress.theta()
    # Baseline rate: finish remaining work over full original deadline window proxy
    if theta_tilde is None or theta_tilde <= 0:
        # Use a gentle baseline so V stays well-scaled
        theta_tilde = max(theta, 1e-6) * 0.5 if theta > 0 else 1e-3
        theta_tilde = max(theta_tilde, 1e-6)

    c_od = cand.on_demand_price
    if c_od is None or c_od <= 0:
        # Invert default spot fraction 0.35
        c_od = cand.spot_price / 0.35 if cand.spot_price > 0 else 1.0

    # V(t) = C_od * theta / theta_tilde  (value of finishing under urgency)
    v = c_od * (theta / theta_tilde)
    eta = max(cand.efficiency, 1e-9)
    lbar = max(cand.lbar, 1e-6)

    # Spot compute cost over expected remaining
    spot_cost = cand.spot_price * cand.gpu_count * lbar

    # Carbon kg = MWh * MOER_lb * lb_to_kg
    mwh = facility_mwh_for_job(
        gpu_count=cand.gpu_count, lbar_hours=lbar, gpu_kw=gpu_kw, pue=pue
    )
    carbon_kg = mwh * cand.moer * LB_TO_KG
    carbon_usd = (carbon_kg / KG_PER_TON) * carbon_price_usd_per_ton * carbon_weight

    c_total = spot_cost + carbon_usd

    # Eviction penalty scaled by failure probability, amortized by Lbar
    e = eviction_penalty * (1.0 - max(0.0, min(1.0, cand.survival)))
    u = v * eta - c_total - (e / lbar)

    cand.utility = u
    cand.c_total = c_total
    cand.carbon_kg = carbon_kg
    cand.value_v = v
    return u


def rank_candidates(
    candidates: list[ODCandidate],
    progress: ProgressState,
    *,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
    eviction_penalty: float = 50.0,
    theta_tilde: float | None = None,
) -> list[ODCandidate]:
    """Rank candidates by U_s descending (higher is better)."""
    for c in candidates:
        compute_utility(
            c,
            progress,
            carbon_price_usd_per_ton=carbon_price_usd_per_ton,
            carbon_weight=carbon_weight,
            eviction_penalty=eviction_penalty,
            theta_tilde=theta_tilde,
        )
    return sorted(candidates, key=lambda c: c.utility, reverse=True)
