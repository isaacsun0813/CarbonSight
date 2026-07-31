"""
SkyNomad Sec 4.4 — lifetime prediction via the Nelson-Aalen hazard estimator.

Observations are aggregated as ``lifetime -> (e, c)``: ``e`` preemptions and
``c`` censored (proactively migrated / still alive) instances at that lifetime.

    n(l) = sum_{x >= l} (e(x) + c(x))            at-risk set
    h(l) = e(l) / n(l)                           hazard increment
    H(l) = sum_{l_i <= l} h(l_i)                 cumulative hazard
    S(l) = exp(-H(l))                            survival
    L(a) = 1/S(a) * integral_a^l_max S(t) dt     expected remaining lifetime at age a

Volatility adjustment for a burst of preemptions in a recent window W:

    gamma_W = e_W / sum_{t in W} h(a(t))
    gamma*  = max over windows ending now
    S~(l)   = exp(-gamma* * H(l)) = S(l) ** gamma*

S is a right-continuous step function — S(t) = S(l_i) for t in [l_i, l_{i+1}) —
so that integral is an exact sum of rectangles whose widths are the *gaps*
between consecutive observed lifetimes. Weighting each S(l_i) by an implicit
width of 1 instead makes L(a) depend on how many distinct lifetimes were
observed rather than how long they were: [1,2,3,4] and [1,2,3,400] both come out
at 1.7998. An hourly probe grid makes lifetimes integer-*valued*, not
*contiguous*, so it does not rescue the unit-width form.

The integral is truncated at the largest observed lifetime (a restricted mean
survival time). With censored observations S never reaches 0, so an untruncated
tail would diverge; the restriction makes L(a) a documented lower bound.

Pure Python — no numpy, no pricing, no AWS.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

Stats = Mapping[float, tuple[int, int]]

# Floor applied by callers so a region with a degenerate fit cannot divide by ~0
# in E/Lbar or drive eta to 0. Shared so the scheduler and the backtest agree.
MIN_LIFETIME_HR = 0.25


def _normalize(stats: LifetimeStats | Stats) -> dict[float, tuple[int, int]]:
    raw = stats.stats if isinstance(stats, LifetimeStats) else stats
    return {float(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


@dataclass
class LifetimeStats:
    """Observed lifetimes aggregated as ``{lifetime: (preemptions, censored)}``."""

    stats: dict[float, tuple[int, int]] = field(default_factory=dict)

    def __init__(self, stats: Stats | None = None) -> None:
        self.stats = _normalize(stats) if stats else {}

    def add(self, lifetime: float, preempted: bool = True) -> None:
        e, c = self.stats.get(float(lifetime), (0, 0))
        self.stats[float(lifetime)] = (e + 1, c) if preempted else (e, c + 1)

    def add_batch(self, observations: Iterable[tuple[float, bool]]) -> None:
        for lifetime, preempted in observations:
            self.add(lifetime, preempted)

    @classmethod
    def from_observations(cls, observations: Iterable[tuple[float, bool]]) -> LifetimeStats:
        obj = cls()
        obj.add_batch(observations)
        return obj

    def total_events(self) -> int:
        return sum(e for e, _ in self.stats.values())

    def total_censored(self) -> int:
        return sum(c for _, c in self.stats.values())

    def total_observations(self) -> int:
        return self.total_events() + self.total_censored()

    def sorted_lifetimes(self) -> list[float]:
        return sorted(self.stats)

    def to_dict(self) -> dict[float, tuple[int, int]]:
        return dict(self.stats)

    def __len__(self) -> int:
        return len(self.stats)

    def __repr__(self) -> str:
        return f"LifetimeStats({self.stats})"


def compute_at_risk(stats: LifetimeStats | Stats) -> dict[float, int]:
    """n(l) = sum_{x >= l} (e(x) + c(x)), ascending by lifetime."""
    norm = _normalize(stats)
    if not norm:
        return {}
    running = 0
    suffix: dict[float, int] = {}
    for lifetime in sorted(norm, reverse=True):
        e, c = norm[lifetime]
        running += e + c
        suffix[lifetime] = running
    return {lifetime: suffix[lifetime] for lifetime in sorted(norm)}


def compute_hazard(stats: LifetimeStats | Stats) -> dict[float, float]:
    """h(l) = e(l) / n(l), in [0, 1]."""
    norm = _normalize(stats)
    at_risk = compute_at_risk(norm)
    return {
        lifetime: (norm[lifetime][0] / at_risk[lifetime] if at_risk[lifetime] > 0 else 0.0)
        for lifetime in sorted(norm)
    }


def compute_cumulative_hazard(hazard: Mapping[float, float]) -> dict[float, float]:
    """H(l) = sum_{l_i <= l} h(l_i)."""
    running = 0.0
    cum: dict[float, float] = {}
    for lifetime in sorted(hazard):
        running += float(hazard[lifetime])
        cum[lifetime] = running
    return cum


def compute_survival(cumulative_hazard: Mapping[float, float]) -> dict[float, float]:
    """S(l) = exp(-H(l))."""
    return {
        lifetime: math.exp(-float(cumulative_hazard[lifetime]))
        for lifetime in sorted(cumulative_hazard)
    }


def expected_remaining(survival: Mapping[float, float], age: float) -> float:
    """L(a) = (1/S(a)) * integral from a to l_max of S(t) dt.

    Exact for the step function, not a unit-width approximation: each rectangle
    is S at the left edge times the gap to the next observed lifetime. Zero once
    ``age`` reaches the last observed lifetime, where the restricted integral is
    empty. Negative ages are clamped to 0.
    """
    if not survival:
        return 0.0
    age = max(0.0, age)
    lifetimes = sorted(survival)
    if age >= lifetimes[-1]:
        return 0.0

    # S at the left edge of the first interval: survival at the newest lifetime
    # already passed, or 1 when the job has not reached the first one.
    s_left = 1.0
    for lifetime in lifetimes:
        if lifetime > age:
            break
        s_left = float(survival[lifetime])
    if s_left <= 0.0:
        return 0.0

    s_age = s_left
    area = 0.0
    left = age
    for lifetime in lifetimes:
        if lifetime <= age:
            continue
        area += s_left * (lifetime - left)
        left = lifetime
        s_left = float(survival[lifetime])
    return area / s_age


def compute_gamma(num_preemptions_in_window: int, hazard_sum: float) -> float:
    """gamma_W = e_W / sum h(a(t)); 1.0 (neutral) when there is no hazard mass."""
    if hazard_sum <= 0.0:
        return 1.0
    return num_preemptions_in_window / hazard_sum


def compute_gamma_star(gammas: Iterable[float]) -> float:
    """gamma* = max over windows ending now; 1.0 (neutral) when empty."""
    values = [float(g) for g in gammas]
    return max(values) if values else 1.0


def compute_volatility_adjusted_survival(
    cumulative_hazard: Mapping[float, float], gamma_star: float
) -> dict[float, float]:
    """S~(l) = exp(-gamma* * H(l)). gamma* > 1 is pessimistic, < 1 optimistic."""
    return {
        lifetime: math.exp(-float(gamma_star) * float(cumulative_hazard[lifetime]))
        for lifetime in sorted(cumulative_hazard)
    }


def compute_volatility_adjusted_survival_from_survival(
    survival: Mapping[float, float], gamma_star: float
) -> dict[float, float]:
    """S~(l) = S(l) ** gamma*, for when H is not to hand."""
    out: dict[float, float] = {}
    for lifetime in sorted(survival):
        s = float(survival[lifetime])
        out[lifetime] = math.exp(float(gamma_star) * math.log(s)) if s > 0.0 else 0.0
    return out


def predict_remaining_lifetime(
    stats: LifetimeStats | Stats,
    age: float,
    gamma_star: float | None = None,
) -> float:
    """h -> H -> S (optionally volatility-adjusted) -> L(age). Zero with no data."""
    hazard = compute_hazard(stats)
    if not hazard:
        return 0.0
    cum = compute_cumulative_hazard(hazard)
    survival = (
        compute_survival(cum)
        if gamma_star is None
        else compute_volatility_adjusted_survival(cum, gamma_star)
    )
    return expected_remaining(survival, age)


def lifetimes_from_virtual_instances(
    instances: Iterable[tuple[float, bool]],
) -> LifetimeStats:
    """Build ``LifetimeStats`` from ``(lifetime_hours, preempted)`` pairs."""
    return LifetimeStats.from_observations(instances)


__all__ = [
    "LifetimeStats",
    "compute_at_risk",
    "compute_cumulative_hazard",
    "compute_gamma",
    "compute_gamma_star",
    "compute_hazard",
    "compute_survival",
    "compute_volatility_adjusted_survival",
    "compute_volatility_adjusted_survival_from_survival",
    "expected_remaining",
    "lifetimes_from_virtual_instances",
    "predict_remaining_lifetime",
]
