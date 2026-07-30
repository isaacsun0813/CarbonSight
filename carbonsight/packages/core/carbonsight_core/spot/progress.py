"""
Progress tracking and deadline-aware future-progress value V(t).

Per SkyNomad Sec 4.5 (extended with carbon lever):

- ProgressState(p,P,t,T): p progress done, P total required, t elapsed, T deadline.
  All units in hours (or same time unit) for simplicity.  p,P can also be abstract
  progress counters; math stays identical.

- Deadline pressure: θ(t) = (P-p)/(T-t)   remaining work / remaining time
- Avg progress:      θ̃(t) = p/t  (with fallback to P/T at t=0)
- Future progress value: V(t) = C_od · θ / θ̃
  where C_od = cheapest on-demand $/hr (or C_total_od including carbon$).
  Equilibrium anchoring: when θ = θ̃ = P/T → V = C_od (accept any spot, reject OD).
  Monotonic: higher deadline pressure ⇒ higher V.
  Scale-invariant: scaling time preserves ratio.

- Thrifty: if p ≥ P → job done, safe to idle / release spot.

- Safety net: if T-t < P-p + 2d → deadline infeasible with spot restarts,
  switch to cheapest OD region:
      argmin_r [ C_od(r)·(P-p+d) + E(r0→r) + carbon_cost(r) ]

Carbon lever extension: carbon cost added to safety-net total as
    carbon_$ = carbon_kg_per_hr / 1000 · price_per_ton · (P-p+d)
so dirty OD regions penalized even in safety net.

Uses same $ conversion as providers/carbon.py:
    cost_per_hr = kg_per_hr / 1000 * price_per_ton * carbon_weight
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# Core progress state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgressState:
    """
    Progress snapshot at current time.

    Attributes:
        p: progress completed so far (e.g. hours of useful work done)
        P: total progress required to finish job (e.g. total compute hours)
        t: elapsed time since job start (hours)
        T: deadline absolute time since start (hours).  Remaining = T-t.
    """

    p: float
    P: float
    t: float
    T: float

    # -- derived quantities -------------------------------------------------

    @property
    def remaining_work(self) -> float:
        """P - p"""
        return self.P - self.p

    @property
    def remaining_time(self) -> float:
        """T - t"""
        return self.T - self.t

    @property
    def target_rate(self) -> float:
        """Average rate needed if perfectly on schedule from start: P/T."""
        if self.T <= 0:
            return 0.0
        return self.P / self.T

    @property
    def avg_progress(self) -> float:
        """
        Average progress rate so far: θ̃ = p/t.
        Falls back to target_rate P/T when t<=0 to avoid div0
        and to preserve anchoring V=C_od at start.
        """
        if self.t > 1e-12:
            return self.p / self.t
        # t==0 → use target rate
        return self.target_rate

    @property
    def deadline_pressure(self) -> float:
        """
        Deadline pressure θ(t) = (P-p)/(T-t).

        Returns inf if remaining_time <=0 and remaining_work>0 (deadline missed),
        0 if remaining_work<=0.
        """
        rem_w = self.remaining_work
        rem_t = self.remaining_time
        if rem_w <= 0:
            return 0.0
        if rem_t <= 1e-12:
            return math.inf
        return rem_w / rem_t

    # -- value function V(t) ------------------------------------------------

    def future_progress_value(self, c_od_per_hr: float) -> float:
        """
        Future progress value V(t) = C_od · θ / θ̃

        Args:
            c_od_per_hr: cheapest on-demand total cost per hour (C_total_od_min)
                         already includes spot $ + carbon $ if carbon-aware.
                         When balanced, θ=θ̃ ⇒ V=C_od per equilibrium anchoring.

        Returns:
            V(t) ≥0, or inf if deadline pressure infinite.
        """
        theta = self.deadline_pressure
        if math.isinf(theta):
            # Deadline exceeded but work remains: infinite pressure
            return math.inf if c_od_per_hr >= 0 else -math.inf
        if theta <= 0:
            return 0.0

        tilde = self.avg_progress
        if tilde <= 1e-12:
            tilde = self.target_rate
        if tilde <= 1e-12:
            # No target defined (P=0 or T=0) → avoid div0, anchor to C_od
            return c_od_per_hr

        return c_od_per_hr * theta / tilde

    # -- thrifty & safety net ------------------------------------------------

    def is_thrifty(self) -> bool:
        """Thrifty rule: p ≥ P ⇒ job done, can idle / release spot."""
        return self.p >= self.P

    # Alias for readability, matches paper phrasing p≥P => done
    @property
    def is_done(self) -> bool:
        return self.is_thrifty()

    def is_safety_net(self, cold_start_hr: float) -> bool:
        """
        Safety net trigger: T-t < P-p + 2d

        If true, scheduler must switch to cheapest on-demand region until done.

        Args:
            cold_start_hr: d = cold start + checkpoint restore time (hours)
                           typically ~6min =0.1h, + ckpt_size/bw.

        Returns:
            True if deadline infeasible with further spot restarts.
        """
        return self.remaining_time < self.remaining_work + 2.0 * cold_start_hr

    # ------------------------------------------------------------------
    # Convenience for ranking / carbon-aware V
    # ------------------------------------------------------------------

    def carbon_aware_future_value(
        self,
        c_od_per_hr: float,
        c_od_carbon_per_hr: float,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        """
        Carbon-aware variant: C_od_total = C_od + carbon_$/hr.

        Mirrors unified_model C_total = spot $ + carbon $.
        """
        carbon_dollar_per_hr = (
            c_od_carbon_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight
            if c_od_carbon_per_hr
            else 0.0
        )
        c_total = c_od_per_hr + carbon_dollar_per_hr
        return self.future_progress_value(c_total)


# ---------------------------------------------------------------------------
# Safety-net cheapest OD region selection (carbon-aware)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ODCandidate:
    """
    Candidate for safety-net OD fallback.

    Attributes:
        region: cloud region identifier
        od_price_per_hr: on-demand $/hr (already multiplied by gpu count if needed)
        migration_cost: E_{r0→r} = e·ckpt_size + ingress/egress fees
        carbon_kg_per_hr: operational carbon intensity kgCO2/hr (facility)
        carbon_price_usd_per_ton: overridden per candidate or use global default
    """

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
    """
    Compute total cost for safety-net OD option:

        Cost(r) = C_od(r)·(P-p + d) + E(r0→r) + carbon_cost(r)

    where carbon_cost = carbon_kg_per_hr/1000·price·(P-p+d)

    Args:
        candidate: ODCandidate with price, migration, carbon rate
        remaining_work: P-p (hours)
        cold_start_hr: d
        carbon_price_usd_per_ton: $/ton CO2 (configurable)
        carbon_weight: multiplier (lambda) for carbon weighting

    Returns:
        Total $ cost to finish remaining work on this OD region.
    """
    duration = remaining_work + cold_start_hr
    if duration < 0:
        duration = 0.0

    dollar = candidate.od_price_per_hr * duration + candidate.migration_cost

    # carbon $ lever: same conversion as providers/carbon.py get_cost_per_hr
    if candidate.carbon_kg_per_hr and carbon_price_usd_per_ton:
        carbon_dollar_per_hr = (
            candidate.carbon_kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight
        )
        dollar += carbon_dollar_per_hr * duration

    return dollar


def select_cheapest_od_region(
    candidates: Iterable[ODCandidate],
    remaining_work: float,
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> Optional[Tuple[ODCandidate, float]]:
    """
    Safety-net decision: argmin_r [C·(P-p+d)+E+carbon_cost]

    Args:
        candidates: iterable of OD regions with price, E, carbon
        remaining_work: P-p
        cold_start_hr: d
        carbon_price_usd_per_ton: configurable social cost
        carbon_weight: carbon weighting factor

    Returns:
        (best_candidate, best_cost) or None if empty.

    Example:
        >>> cands = [ODCandidate("us-east-1", 4.1, 0.2, 0.5),
        ...          ODCandidate("eu-north-1", 4.5, 0.3, 0.1)]
        >>> best, cost = select_cheapest_od_region(cands, 5.0, 0.1, 50)[0:2]
    """
    best: Optional[ODCandidate] = None
    best_cost = math.inf

    for cand in candidates:
        cost = compute_safety_net_total_cost(
            cand, remaining_work, cold_start_hr, carbon_price_usd_per_ton, carbon_weight
        )
        if cost < best_cost:
            best_cost = cost
            best = cand

    if best is None:
        return None
    return best, best_cost


def safety_net_selection_for_state(
    state: ProgressState,
    candidates: Iterable[ODCandidate],
    cold_start_hr: float,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
) -> Optional[Tuple[ODCandidate, float]]:
    """
    Wrapper that extracts remaining_work = P-p from ProgressState.

    Matches spec: T-t < P-p+2d -> cheapest OD region argmin[...]
    Caller should first check state.is_safety_net(d); if True, call this.
    """
    return select_cheapest_od_region(
        candidates,
        state.remaining_work,
        cold_start_hr,
        carbon_price_usd_per_ton,
        carbon_weight,
    )


__all__ = [
    "ProgressState",
    "ODCandidate",
    "compute_safety_net_total_cost",
    "select_cheapest_od_region",
    "safety_net_selection_for_state",
]
