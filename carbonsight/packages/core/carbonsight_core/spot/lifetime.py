"""
Lifetime prediction via Nelson-Aalen hazard estimator.

Per SkyNomad Sec 4.4:

- e(l): number of preemptions (events) observed at lifetime l
- c(l): number of censored observations (proactive migrations) at lifetime l
- n(l) = Σ_{x ≥ l} (e(x)+c(x))  — at-risk set size at age l
- h(l) = e(l) / n(l)            — Nelson-Aalen hazard increment
- H(l) = Σ_{l_i ≤ l} h(l_i)     — cumulative hazard
- S(l) = exp(-H(l))             — survival function
- L̄(a) = 1/S(a) Σ_{l_i > a} S(l_i)  — expected remaining lifetime given age a
- Volatility adjustment:
    γ_W = e_W / Σ_{t ∈ W} h(a(t))
    γ*  = max_{W ending now} γ_W
    S̃(l) = exp(γ* · -H(l)) = S(l)^{γ*}

All math is pure Python, no boto3, no pricing dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple, Union


# Type aliases for clarity
Lifetime = float
Count = int
StatsDict = Dict[Lifetime, Tuple[Count, Count]]
HazardDict = Dict[Lifetime, float]
CumHazardDict = Dict[Lifetime, float]
SurvivalDict = Dict[Lifetime, float]


def _normalize_stats(
    stats: Union["LifetimeStats", StatsDict, Mapping[Lifetime, Tuple[Count, Count]]],
) -> StatsDict:
    """Accept LifetimeStats or plain dict and return a plain dict copy."""
    if isinstance(stats, LifetimeStats):
        return dict(stats.stats)
    # Mapping but not LifetimeStats
    return {float(k): (int(v[0]), int(v[1])) for k, v in stats.items()}


@dataclass
class LifetimeStats:
    """
    Aggregation of observed virtual-instance lifetimes.

    Internal representation:
        dict[ lifetime -> (e(l) preemptions, c(l) censored) ]

    Lifetimes are typically in hours (float or int).  Counts are non-negative ints.
    """

    stats: StatsDict = field(default_factory=dict)

    def __init__(self, stats: Union[StatsDict, Mapping[Lifetime, Tuple[Count, Count]], None] = None):
        if stats is None:
            self.stats = {}
        elif isinstance(stats, dict):
            # copy and normalize
            self.stats = {float(k): (int(v[0]), int(v[1])) for k, v in stats.items()}
        else:
            # Mapping
            self.stats = {float(k): (int(v[0]), int(v[1])) for k, v in stats.items()}

    def add(self, lifetime: Lifetime, preempted: bool = True) -> None:
        """Record a single observation at `lifetime`.

        Args:
            lifetime: observed lifetime (e.g. hours until preemption or censoring)
            preempted: True if preemption event (e), False if censored (c)
        """
        l = float(lifetime)
        e, c = self.stats.get(l, (0, 0))
        if preempted:
            e += 1
        else:
            c += 1
        self.stats[l] = (e, c)

    def add_batch(self, lifetimes: Iterable[Tuple[Lifetime, bool]]) -> None:
        """Add many observations: iterable of (lifetime, preempted)."""
        for l, preempted in lifetimes:
            self.add(l, preempted)

    def total_events(self) -> int:
        return sum(e for e, _ in self.stats.values())

    def total_censored(self) -> int:
        return sum(c for _, c in self.stats.values())

    def total_observations(self) -> int:
        return self.total_events() + self.total_censored()

    def sorted_lifetimes(self) -> list[Lifetime]:
        return sorted(self.stats.keys())

    def to_dict(self) -> StatsDict:
        return dict(self.stats)

    @classmethod
    def from_observations(
        cls, observations: Iterable[Tuple[Lifetime, bool]]
    ) -> "LifetimeStats":
        """Build from iterable of (lifetime, preempted)."""
        obj = cls()
        obj.add_batch(observations)
        return obj

    def __len__(self) -> int:
        return len(self.stats)

    def __repr__(self) -> str:
        return f"LifetimeStats({self.stats})"


def compute_at_risk(
    stats: Union[LifetimeStats, StatsDict, Mapping[Lifetime, Tuple[Count, Count]]],
) -> Dict[Lifetime, int]:
    """
    Compute at-risk set n(l) = Σ_{x ≥ l} (e(x)+c(x)).

    Returns dict lifetime -> n(l) sorted by lifetime key.
    """
    norm = _normalize_stats(stats)
    if not norm:
        return {}
    sorted_ls = sorted(norm.keys())
    # suffix sum
    total_by_l = {l: e + c for l, (e, c) in norm.items()}
    n: Dict[Lifetime, int] = {}
    running = 0
    # iterate descending
    for l in reversed(sorted_ls):
        running += total_by_l[l]
        n[l] = running
    # return in ascending order for determinism
    return {l: n[l] for l in sorted_ls}


def compute_hazard(
    stats: Union[LifetimeStats, StatsDict, Mapping[Lifetime, Tuple[Count, Count]]],
) -> HazardDict:
    """
    Nelson-Aalen hazard increment: h(l) = e(l)/n(l)

    Args:
        stats: dict lifetime -> (e, c) or LifetimeStats

    Returns:
        dict lifetime -> hazard h(l) in [0,1]
    """
    norm = _normalize_stats(stats)
    if not norm:
        return {}
    at_risk = compute_at_risk(norm)
    hazard: HazardDict = {}
    for l in sorted(norm.keys()):
        e, _ = norm[l]
        n_l = at_risk.get(l, 0)
        if n_l <= 0:
            hazard[l] = 0.0
        else:
            hazard[l] = e / n_l
    return hazard


def compute_cumulative_hazard(hazard: Mapping[Lifetime, float]) -> CumHazardDict:
    """
    H(l) = Σ_{l_i ≤ l} h(l_i)

    Monotonically non-decreasing if hazard >=0.

    Returns dict lifetime -> cumulative H(l)
    """
    if not hazard:
        return {}
    sorted_ls = sorted(hazard.keys())
    cum: CumHazardDict = {}
    running = 0.0
    for l in sorted_ls:
        running += float(hazard[l])
        cum[l] = running
    return cum


def compute_survival(cumulative_hazard: Mapping[Lifetime, float]) -> SurvivalDict:
    """
    S(l) = exp(-H(l))

    Returns dict lifetime -> survival probability in (0,1]
    Monotonically non-increasing as H increases.
    """
    if not cumulative_hazard:
        return {}
    survival: SurvivalDict = {}
    for l, H in cumulative_hazard.items():
        # Clamp large H to avoid underflow to 0, but math.exp handles up to ~709
        try:
            s = math.exp(-float(H))
        except OverflowError:
            s = 0.0
        survival[l] = s
    # preserve sorted order
    return {l: survival[l] for l in sorted(survival.keys())}


def expected_remaining(
    survival: Mapping[Lifetime, float],
    age: float,
) -> float:
    """
    Expected remaining lifetime given survival to age a:

        L̄(a) = 1/S(a) Σ_{l_i > a} S(l_i)

    Args:
        survival: dict lifetime -> S(l), assumed sorted ascending (will sort).
        age: current age a (float). If a < min(lifetime), S(a)=1. If a >= max,
              no future mass -> returns 0.

    Returns:
        Expected remaining lifetime (same time unit as keys). 0 if no future prob.

    Note:
        For discrete lifetimes, this sums step-wise S.  For continuous interpretation,
        sum approximates integral ∫_{a}^{∞} S(t)/S(a) dt  with unit steps.
        If you have non-unit spacing, the caller can weight by Δl externally or
        interpret sum as count of future intervals.
    """
    if not survival:
        return 0.0
    sorted_ls = sorted(survival.keys())
    # Determine S(a)
    if age < sorted_ls[0]:
        s_a = 1.0
    else:
        # Last lifetime <= age
        # find greatest l <= age
        # linear scan (small n) – binary search could be used for large n
        last_le = None
        for l in sorted_ls:
            if l <= age:
                last_le = l
            else:
                break
        if last_le is None:
            s_a = 1.0
        else:
            s_a = float(survival[last_le])

    if s_a <= 0.0:
        return 0.0

    # sum of S(l_i) for l_i > age
    sum_future = 0.0
    for l in sorted_ls:
        if l > age:
            sum_future += float(survival[l])

    return sum_future / s_a


def compute_gamma(num_preemptions_in_window: int, hazard_sum: float) -> float:
    """
    Volatility window gamma:

        γ_W = e_W / Σ_{t ∈ W} h(a(t))

    Args:
        num_preemptions_in_window: e_W  (count of preemptions in window)
        hazard_sum: Σ h(a(t)) over observation times in window

    Returns:
        γ_W.  Returns 1.0 (neutral) if hazard_sum <=0 (no info).
        Can be >1 in volatile periods, <1 in stable periods.
    """
    if hazard_sum <= 0.0:
        # Neutral – no hazard mass to compare to
        return 1.0
    return float(num_preemptions_in_window) / float(hazard_sum)


def compute_gamma_star(gammas: Iterable[float]) -> float:
    """
    γ* = max_{W ending now} γ_W

    Args:
        gammas: iterable of γ_W values for windows ending now

    Returns:
        max gamma, or 1.0 if empty (neutral).
    """
    max_g = 1.0
    found = False
    for g in gammas:
        if not found:
            max_g = float(g)
            found = True
        else:
            if float(g) > max_g:
                max_g = float(g)
    return max_g if found else 1.0


def compute_volatility_adjusted_survival(
    cumulative_hazard: Mapping[Lifetime, float],
    gamma_star: float,
) -> SurvivalDict:
    """
    Adjusted survival: S̃(l) = exp(γ* · -H(l)) = S(l)^{γ*}

    Args:
        cumulative_hazard: H(l)
        gamma_star: γ* volatility factor (≥0 typically). γ*>1 reduces survival
                     (more pessimistic), γ*<1 increases survival.

    Returns:
        dict lifetime -> adjusted survival S̃
    """
    if not cumulative_hazard:
        return {}
    adj: SurvivalDict = {}
    for l in sorted(cumulative_hazard.keys()):
        H = float(cumulative_hazard[l])
        try:
            adj[l] = math.exp(-float(gamma_star) * H)
        except OverflowError:
            adj[l] = 0.0
    return adj


def compute_volatility_adjusted_survival_from_survival(
    survival: Mapping[Lifetime, float],
    gamma_star: float,
) -> SurvivalDict:
    """
    Alternate entry: given S, compute S̃ = S^{γ*}.

    Equivalent to exp(γ*·-H) when S = exp(-H). Useful if H not directly available.
    """
    if not survival:
        return {}
    adj: SurvivalDict = {}
    for l in sorted(survival.keys()):
        s = float(survival[l])
        if s <= 0.0:
            adj[l] = 0.0
        else:
            # s^{gamma*} = exp(gamma* * log(s))
            try:
                adj[l] = math.exp(float(gamma_star) * math.log(s))
            except (ValueError, OverflowError):
                adj[l] = 0.0
    return adj


def predict_remaining_lifetime(
    stats: Union[LifetimeStats, StatsDict, Mapping[Lifetime, Tuple[Count, Count]]],
    age: float,
    gamma_star: float | None = None,
) -> float:
    """
    High-level convenience: predict expected remaining lifetime.

    Steps:
        1. h(l) = e(l)/n(l)
        2. H(l) = Σ h
        3. S(l) = exp(-H)  or S̃(l)=exp(-γ* H) if gamma_star provided
        4. L̄(a) = 1/S(a) Σ_{l_i>a} S(l_i)

    Args:
        stats: lifetime aggregation
        age: current age a of the running instance
        gamma_star: optional volatility factor. If None, use unadjusted S.
                    If provided, survival is volatility-adjusted.

    Returns:
        Expected remaining lifetime (float, ≥0).  Returns 0 if stats empty or
        age beyond last observed lifetime.
    """
    hazard = compute_hazard(stats)
    if not hazard:
        return 0.0
    cum_hazard = compute_cumulative_hazard(hazard)
    if gamma_star is None:
        survival = compute_survival(cum_hazard)
    else:
        survival = compute_volatility_adjusted_survival(cum_hazard, gamma_star)
    return expected_remaining(survival, age)


# ---------------------------------------------------------------------------
# Optional helper for constructing LifetimeStats from raw virtual-instance traces
# ---------------------------------------------------------------------------

def lifetimes_from_virtual_instances(
    instances: Iterable[Tuple[Lifetime, bool]],
) -> LifetimeStats:
    """
    Helper to build LifetimeStats from list of (lifetime, preempted).

    Args:
        instances: iterable of (lifetime, is_preemption)
                   is_preemption True = e, False = censored c

    Returns:
        LifetimeStats
    """
    return LifetimeStats.from_observations(instances)
