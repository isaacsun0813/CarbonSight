"""
Spot instance survival model (SkyNomad §4.4.1–4.4.2 barebones).

Uses Nelson–Aalen survival estimates over observed ages and an assumed exponential tail.

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

# Assumed mean for unobserved lifetime ages, including the no-history case.
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.bucket_hours) or self.bucket_hours <= 0:
            raise ValueError("bucket_hours must be finite and positive")

    def _bucket(self, hours: float) -> float:
        return _bucket(hours, self.bucket_hours)

    def record_complete(self, lifetime_hours: float) -> None:
        _validate_hours(lifetime_hours)
        key = self._bucket(lifetime_hours)
        self.complete_counts[key] = self.complete_counts.get(key, 0) + 1
        self.complete_samples.append(max(lifetime_hours, 0.0))

    def record_censored(self, lifetime_hours: float) -> None:
        _validate_hours(lifetime_hours)
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
        event_age = self._bucket(lifetime_hours)
        e_l = self.complete_counts.get(event_age, 0)
        n_l = sum(
            self.complete_counts.get(x, 0) + self.censored_counts.get(x, 0)
            for x in lifetimes
            if x >= event_age
        )
        if n_l <= 0:
            return 0.0
        return e_l / n_l

    def cumulative_hazard(self, lifetime_hours: float) -> float:
        """Cumulative interruption hazard; unobserved ages use the assumed exponential tail."""
        _validate_hours(lifetime_hours)
        lifetimes = self._sorted_lifetimes()
        if not lifetimes:
            return lifetime_hours / ASSUMED_SPOT_MEAN_LIFETIME_HOURS
        measured = sum(self.hazard(age) for age in lifetimes if age <= lifetime_hours)
        tail = max(0.0, lifetime_hours - lifetimes[-1])
        return measured + tail / ASSUMED_SPOT_MEAN_LIFETIME_HOURS

    def survival(self, lifetime_hours: float) -> float:
        return math.exp(-self.cumulative_hazard(lifetime_hours))

    def expected_remaining_lifetime(self, age_hours: float, *, horizon_hours: float) -> float:
        """Expected usable hours within the next horizon, conditional on survival to age.

        Integrate S(age + t) / S(age) over 0 <= t <= horizon. Beyond the last observed
        age, assume an exponential tail with the labeled placeholder mean.
        """
        _validate_hours(age_hours)
        _validate_hours(horizon_hours)
        lifetimes = self._sorted_lifetimes()
        support = lifetimes[-1] if lifetimes else 0.0
        if age_hours >= support:
            return -ASSUMED_SPOT_MEAN_LIFETIME_HOURS * math.expm1(
                -horizon_hours / ASSUMED_SPOT_MEAN_LIFETIME_HOURS
            )
        end = min(age_hours + horizon_hours, support)
        baseline = self.cumulative_hazard(age_hours)
        start = age_hours
        area = 0.0
        for boundary in [t for t in lifetimes if age_hours < t < end] + [end]:
            area += (boundary - start) * math.exp(
                -(self.cumulative_hazard(start) - baseline)
            )
            start = boundary
        tail_hours = max(0.0, horizon_hours - (end - age_hours))
        tail_probability = math.exp(-(self.cumulative_hazard(end) - baseline))
        return area + tail_probability * -ASSUMED_SPOT_MEAN_LIFETIME_HOURS * math.expm1(
            -tail_hours / ASSUMED_SPOT_MEAN_LIFETIME_HOURS
        )


def _validate_hours(hours: float) -> None:
    if not math.isfinite(hours) or hours < 0:
        raise ValueError("hours must be finite and nonnegative")


def _bucket(hours: float, bucket_hours: float = DEFAULT_BUCKET_HOURS) -> float:
    return round(hours / bucket_hours) * bucket_hours


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
        if obs.source == "migration":
            self.record_proactive_migration(obs.time_utc)
            return
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

    def predict_lifetime_hours(self, now_utc: datetime, *, horizon_hours: float) -> float:
        return self.model.expected_remaining_lifetime(
            self.current_age_hours(now_utc), horizon_hours=horizon_hours,
        )

    def prior_source(self) -> str:
        if self.model.n_samples:
            support = max(self.model._sorted_lifetimes())
            return (
                f"history: {len(self.observations)} observations, {self.model.n_samples} "
                f"lifetime samples (support {support:g}h, bucket {self.model.bucket_hours:g}h); "
                f"assumed exponential tail mean {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h"
            )
        return f"assumed exponential mean {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h (no lifetime samples)"


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
