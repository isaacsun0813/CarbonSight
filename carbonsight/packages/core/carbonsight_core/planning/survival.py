"""
Spot instance survival model (SkyNomad §4.4.1–4.4.2 barebones).

Uses Nelson–Aalen hazard estimation with optional volatility adjustment.
When no observations exist, falls back to exponential prior with
MEAN_SPOT_LIFETIME_HOURS (same default as dashboard preemption heuristic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

MEAN_SPOT_LIFETIME_HOURS = 4.0


@dataclass(frozen=True, slots=True)
class SpotObservation:
    """One availability observation for a virtual instance."""

    time_utc: datetime
    available: bool


@dataclass
class NelsonAalenModel:
    """
    Non-parametric survival model from complete (preemption) and censored lifetimes.

    Lifetimes are in hours on a discrete grid (rounded to 0.25h buckets).
    """

    complete_counts: dict[float, int] = field(default_factory=dict)
    censored_counts: dict[float, int] = field(default_factory=dict)

    def record_complete(self, lifetime_hours: float) -> None:
        key = _bucket(lifetime_hours)
        self.complete_counts[key] = self.complete_counts.get(key, 0) + 1

    def record_censored(self, lifetime_hours: float) -> None:
        key = _bucket(lifetime_hours)
        self.censored_counts[key] = self.censored_counts.get(key, 0) + 1

    def _sorted_lifetimes(self) -> list[float]:
        keys = set(self.complete_counts) | set(self.censored_counts)
        return sorted(keys)

    def hazard(self, lifetime_hours: float) -> float:
        """Nelson–Aalen hazard h(l) = e(l) / n(l)."""
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return 1.0 / MEAN_SPOT_LIFETIME_HOURS
        l = _bucket(lifetime_hours)
        e_l = self.complete_counts.get(l, 0)
        n_l = sum(
            self.complete_counts.get(x, 0) + self.censored_counts.get(x, 0)
            for x in lifetimes
            if x >= l
        )
        if n_l <= 0:
            return 0.0
        return e_l / n_l

    def cumulative_hazard(self, lifetime_hours: float) -> float:
        """H(l) = sum of h(l_i) for l_i <= l."""
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return lifetime_hours / MEAN_SPOT_LIFETIME_HOURS
        total = 0.0
        for l_i in lifetimes:
            if l_i <= lifetime_hours:
                total += self.hazard(l_i)
        return total

    def survival(self, lifetime_hours: float, *, gamma_star: float = 1.0) -> float:
        """S(l) = exp(-H(l)); volatility-adjusted when gamma_star > 1."""
        h = self.cumulative_hazard(lifetime_hours)
        return math.exp(-gamma_star * h)

    def expected_remaining_lifetime(self, age_hours: float, *, gamma_star: float = 1.0) -> float:
        """
        E[L - a | L > a] ≈ sum_{l_i > a} S(l_i) / S(a) on discrete grid.

        Falls back to exponential prior when no data.
        """
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return max(MEAN_SPOT_LIFETIME_HOURS - age_hours, 0.25)
        age = _bucket(age_hours)
        s_a = self.survival(age, gamma_star=gamma_star)
        if s_a <= 1e-12:
            return 0.25
        tail = [l for l in lifetimes if l > age]
        if not tail:
            return 0.25
        weighted = sum(self.survival(l, gamma_star=gamma_star) for l in tail)
        return max(weighted / s_a, 0.25)


def _bucket(hours: float) -> float:
    return round(max(hours, 0.0) / 0.25) * 0.25


def volatility_ratio(
    observations: list[SpotObservation],
    model: NelsonAalenModel,
    *,
    age_at: list[tuple[datetime, float]],
) -> float:
    """
    gamma_W = observed preemptions / expected preemptions in window.

    age_at: (observation time, instance age hours) for each observation in window.
    """
    if not age_at:
        return 1.0
    observed = sum(1 for o in observations if not o.available)
    expected = sum(model.hazard(age) for _, age in age_at)
    if expected <= 0:
        return 1.0 if observed == 0 else float(observed)
    return observed / expected


def conservative_volatility(observations: list[SpotObservation], model: NelsonAalenModel) -> float:
    """gamma* = max gamma_W over suffix windows ending at the latest observation."""
    if len(observations) < 2:
        return 1.0
    sorted_obs = sorted(observations, key=lambda o: o.time_utc)
    ages: list[tuple[datetime, float]] = []
    streak_start: datetime | None = None
    gamma_max = 1.0
    for i, obs in enumerate(sorted_obs):
        if obs.available:
            if streak_start is None:
                streak_start = obs.time_utc
            age_h = (obs.time_utc - streak_start).total_seconds() / 3600.0
        else:
            streak_start = None
            age_h = 0.0
        ages.append((obs.time_utc, age_h))
        window = sorted_obs[: i + 1]
        window_ages = ages[: i + 1]
        gamma_w = volatility_ratio(window, model, age_at=window_ages)
        gamma_max = max(gamma_max, gamma_w)
    return gamma_max


@dataclass
class VirtualInstanceTracker:
    """Per-region virtual spot instance for lifetime prediction."""

    region: str
    model: NelsonAalenModel = field(default_factory=NelsonAalenModel)
    observations: list[SpotObservation] = field(default_factory=list)
    streak_start_utc: datetime | None = None

    def record_probe(self, time_utc: datetime, available: bool) -> None:
        self.observations.append(SpotObservation(time_utc, available))
        if available:
            if self.streak_start_utc is None:
                self.streak_start_utc = time_utc
        else:
            if self.streak_start_utc is not None:
                lifetime = (time_utc - self.streak_start_utc).total_seconds() / 3600.0
                self.model.record_complete(lifetime)
            self.streak_start_utc = None

    def record_proactive_migration(self, time_utc: datetime) -> None:
        """Right-censored lifetime when migrating away while spot was up."""
        if self.streak_start_utc is not None:
            lifetime = (time_utc - self.streak_start_utc).total_seconds() / 3600.0
            self.model.record_censored(lifetime)
            self.streak_start_utc = None

    def current_age_hours(self, now_utc: datetime) -> float:
        if self.streak_start_utc is None:
            return 0.0
        return (now_utc - self.streak_start_utc).total_seconds() / 3600.0

    def predict_lifetime_hours(self, now_utc: datetime) -> float:
        gamma_star = conservative_volatility(self.observations, self.model)
        age = self.current_age_hours(now_utc)
        return self.model.expected_remaining_lifetime(age, gamma_star=gamma_star)
