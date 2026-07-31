"""
SkyNomad Sec 4.5 — progress, deadline pressure, and future-progress value V(t).

``ProgressState(p, P, t, T)`` — all four in hours:

    theta       = (P - p) / (T - t)       deadline pressure (work left / time left)
    theta_tilde = p / t                   achieved rate, falls back to P/T at t=0
    V(t)        = C_od * theta / theta_tilde

At t=0 the fallback makes theta == theta_tilde == P/T, so V == C_od: equilibrium
anchoring, i.e. accept any spot cheaper than on-demand and reject on-demand
itself. A job behind schedule has theta > theta_tilde and bids more for an hour
of progress; a job ahead of schedule bids less.

Two short-circuits sit in front of the ranking:

    thrifty     p >= P                  job is done, release the instance
    safety net  T - t < P - p + 2d      spot restarts can no longer make the
                                        deadline; take the cheapest on-demand
                                        region, argmin_r C_od(r)*(P-p+d)
                                        + E(r) + carbon$(r)*(P-p+d)

Carbon is dollarised the same way everywhere: kg/hr / 1000 * $/ton * weight.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def carbon_usd_per_hr(
    carbon_kg_per_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """kgCO2/hr -> $/hr at the given social cost of carbon."""
    if carbon_kg_per_hr <= 0 or carbon_price_usd_per_ton <= 0:
        return 0.0
    return carbon_kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight


@dataclass(frozen=True, slots=True)
class ProgressState:
    """Progress snapshot: p done of P, at elapsed t against deadline T (all hours)."""

    p: float
    P: float
    t: float
    T: float

    @property
    def remaining_work(self) -> float:
        return self.P - self.p

    @property
    def remaining_time(self) -> float:
        return self.T - self.t

    @property
    def target_rate(self) -> float:
        """P/T — the rate that finishes exactly on the deadline from a standing start."""
        return self.P / self.T if self.T > 0 else 0.0

    @property
    def avg_progress(self) -> float:
        """theta_tilde = p/t, falling back to P/T at t=0 so V anchors at C_od."""
        if self.t > 1e-12:
            return self.p / self.t
        return self.target_rate

    @property
    def deadline_pressure(self) -> float:
        """theta = (P-p)/(T-t). Zero when done, inf when the deadline has passed."""
        if self.remaining_work <= 0:
            return 0.0
        if self.remaining_time <= 1e-12:
            return math.inf
        return self.remaining_work / self.remaining_time

    def future_progress_value(self, c_od_per_hr: float) -> float:
        """V(t) = C_od * theta / theta_tilde — what an hour of progress is worth."""
        theta = self.deadline_pressure
        if math.isinf(theta):
            return math.inf
        if theta <= 0:
            return 0.0
        tilde = self.avg_progress
        if tilde > 1e-12:
            return c_od_per_hr * theta / tilde
        # theta_tilde == 0. At t=0 that can only mean a degenerate P or T, since
        # avg_progress returns P/T there — anchor at C_od. At t>0 it means the
        # job has genuinely done nothing in the time it has had, which is maximal
        # pressure, not "on plan": falling back to P/T here would let a stalled
        # job bid the same as a healthy one.
        return c_od_per_hr if self.t <= 1e-12 else math.inf

    def is_thrifty(self) -> bool:
        """p >= P: nothing left to do."""
        return self.p >= self.P

    def is_safety_net(self, cold_start_hr: float) -> bool:
        """T - t < P - p + 2d: no room left for another spot restart."""
        return self.remaining_time < self.remaining_work + 2.0 * cold_start_hr


@dataclass(frozen=True, slots=True)
class ODCandidate:
    """On-demand fallback option considered by the safety net."""

    region: str
    od_price_per_hr: float
    migration_cost: float = 0.0
    carbon_kg_per_hr: float = 0.0


def compute_safety_net_total_cost(
    candidate: ODCandidate,
    remaining_work: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> float:
    """Cost(r) = (C_od(r) + carbon$(r)) * (P-p+d) + E(r)."""
    duration = max(0.0, remaining_work + cold_start_hr)
    per_hr = candidate.od_price_per_hr + carbon_usd_per_hr(
        candidate.carbon_kg_per_hr, carbon_price_usd_per_ton, carbon_weight
    )
    return per_hr * duration + candidate.migration_cost


def select_cheapest_od_region(
    candidates: Iterable[ODCandidate],
    remaining_work: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> tuple[ODCandidate, float] | None:
    """argmin_r total cost to finish on on-demand. None when there are no candidates."""
    best: tuple[ODCandidate, float] | None = None
    for cand in candidates:
        cost = compute_safety_net_total_cost(
            cand, remaining_work, cold_start_hr, carbon_price_usd_per_ton, carbon_weight
        )
        if best is None or cost < best[1]:
            best = (cand, cost)
    return best


__all__ = [
    "ODCandidate",
    "ProgressState",
    "carbon_usd_per_hr",
    "compute_safety_net_total_cost",
    "select_cheapest_od_region",
]
