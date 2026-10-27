"""
Spot instance survival model (SkyNomad §4.4.1–4.4.2 barebones).

Uses Nelson–Aalen hazard estimation with optional volatility adjustment.

Lifetimes come from *observations* we recorded ourselves (launch results, observed
preemptions, optional launch probes). With no observations the model falls back to an
exponential prior with ``ASSUMED_SPOT_MEAN_LIFETIME_HOURS``. That number is a
placeholder assumption, not a measurement of real AWS Spot interruption behavior.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

# Placeholder prior used only when there are no lifetime samples. Not measured AWS data.
ASSUMED_SPOT_MEAN_LIFETIME_HOURS = 4.0
# Backwards-compatible alias (the dashboard heuristic uses the same placeholder).
MEAN_SPOT_LIFETIME_HOURS = ASSUMED_SPOT_MEAN_LIFETIME_HOURS

DEFAULT_BUCKET_HOURS = 0.25


@dataclass(frozen=True, slots=True)
class SpotObservation:
    """One availability observation for a virtual instance."""

    time_utc: datetime
    available: bool
    # Where the observation came from: probe | launch | preemption | migration (| sps, reserved).
    source: str = "probe"
    # Run that produced it; used to avoid chaining streaks across separate runs.
    run_id: str | None = None


@dataclass
class NelsonAalenModel:
    """
    Non-parametric survival model from complete (preemption) and censored lifetimes.

    Lifetimes are in hours on a discrete grid (rounded to ``bucket_hours`` buckets). The
    grid cannot be finer than the observation spacing: probes every 2h only know a lifetime
    to within ~2h, so callers should set ``bucket_hours`` to the probe interval.
    """

    complete_counts: dict[float, int] = field(default_factory=dict)
    censored_counts: dict[float, int] = field(default_factory=dict)
    bucket_hours: float = DEFAULT_BUCKET_HOURS
    # Raw (unbucketed) samples, kept for analytics and prior labeling.
    complete_samples: list[float] = field(default_factory=list)
    censored_samples: list[float] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.complete_samples) + len(self.censored_samples)

    def _bucket(self, hours: float) -> float:
        return _bucket(hours, self.bucket_hours)

    def record_complete(self, lifetime_hours: float) -> None:
        key = self._bucket(lifetime_hours)
        self.complete_counts[key] = self.complete_counts.get(key, 0) + 1
        self.complete_samples.append(max(lifetime_hours, 0.0))

    def record_censored(self, lifetime_hours: float) -> None:
        key = self._bucket(lifetime_hours)
        self.censored_counts[key] = self.censored_counts.get(key, 0) + 1
        self.censored_samples.append(max(lifetime_hours, 0.0))

    def _sorted_lifetimes(self) -> list[float]:
        keys = set(self.complete_counts) | set(self.censored_counts)
        return sorted(keys)

    def hazard(self, lifetime_hours: float) -> float:
        """Nelson–Aalen hazard h(l) = e(l) / n(l)."""
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return 1.0 / ASSUMED_SPOT_MEAN_LIFETIME_HOURS
        l = self._bucket(lifetime_hours)
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
            return lifetime_hours / ASSUMED_SPOT_MEAN_LIFETIME_HOURS
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

        Falls back to the assumed exponential prior when there are no samples.
        """
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return max(ASSUMED_SPOT_MEAN_LIFETIME_HOURS - age_hours, 0.25)
        age = self._bucket(age_hours)
        s_a = self.survival(age, gamma_star=gamma_star)
        if s_a <= 1e-12:
            return 0.25
        tail = [l for l in lifetimes if l > age]
        if not tail:
            return 0.25
        weighted = sum(self.survival(l, gamma_star=gamma_star) for l in tail)
        return max(weighted / s_a, 0.25)


def _bucket(hours: float, bucket_hours: float = DEFAULT_BUCKET_HOURS) -> float:
    return round(max(hours, 0.0) / bucket_hours) * bucket_hours


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
    # Called with each *new* observation (never for replayed history), e.g. to persist it.
    on_observation: Callable[[SpotObservation], None] | None = None

    def _apply(self, obs: SpotObservation) -> None:
        self.observations.append(obs)
        if obs.available:
            if self.streak_start_utc is None:
                self.streak_start_utc = obs.time_utc
        else:
            if self.streak_start_utc is not None:
                lifetime = (obs.time_utc - self.streak_start_utc).total_seconds() / 3600.0
                self.model.record_complete(lifetime)
            self.streak_start_utc = None

    def record_probe(
        self,
        time_utc: datetime,
        available: bool,
        source: str = "probe",
        run_id: str | None = None,
    ) -> None:
        obs = SpotObservation(time_utc, available, source, run_id)
        self._apply(obs)
        if self.on_observation is not None:
            self.on_observation(obs)

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

    def prior_source(self) -> str:
        """Human-readable origin of the lifetime estimate (never implies AWS measurements)."""
        n_obs = len(self.observations)
        if self.model.n_samples:
            return (
                f"history: {n_obs} observations, {self.model.n_samples} lifetime samples "
                f"(bucket {self.model.bucket_hours:g}h)"
            )
        if n_obs:
            return (
                f"assumed default {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h "
                f"({n_obs} observations, no lifetime samples yet)"
            )
        return f"assumed default {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h (no observations)"


def replay_observations(
    tracker: VirtualInstanceTracker,
    observations: Iterable[SpotObservation],
    *,
    max_gap_hours: float | None = None,
    close_open_streak: bool = True,
) -> None:
    """
    Replay stored observations (time order) into ``tracker`` without firing callbacks.

    A streak is closed as *right-censored* at its last observation, instead of chained into the
    next one, when the run changes or the gap exceeds ``max_gap_hours`` — otherwise a lifetime
    spanning days between runs would be fabricated. ``close_open_streak`` closes the streak left
    open at the end of history (it belongs to a past run).
    """
    def close_streak(at: datetime) -> None:
        # A streak seen only once has no measurable length: drop it rather than record 0h.
        if tracker.streak_start_utc is not None and at > tracker.streak_start_utc:
            tracker.record_proactive_migration(at)
        else:
            tracker.streak_start_utc = None

    prev: SpotObservation | None = None
    for obs in sorted(observations, key=lambda o: o.time_utc):
        if prev is not None and tracker.streak_start_utc is not None:
            run_changed = (
                prev.run_id is not None and obs.run_id is not None and prev.run_id != obs.run_id
            )
            gap_hours = (obs.time_utc - prev.time_utc).total_seconds() / 3600.0
            gap_too_big = max_gap_hours is not None and gap_hours > max_gap_hours
            if run_changed or gap_too_big:
                close_streak(prev.time_utc)
        tracker._apply(obs)
        prev = obs
    if close_open_streak and prev is not None:
        close_streak(prev.time_utc)
