"""Unit tests for spot survival model."""

from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.planning.survival import (
    ASSUMED_SPOT_MEAN_LIFETIME_HOURS,
    MEAN_SPOT_LIFETIME_HOURS,
    NelsonAalenModel,
    SpotObservation,
    VirtualInstanceTracker,
    conservative_volatility,
    replay_observations,
)

UTC = timezone.utc


class TestNelsonAalenModel:
    def test_default_expected_lifetime(self) -> None:
        model = NelsonAalenModel()
        remaining = model.expected_remaining_lifetime(0.0)
        assert remaining == pytest.approx(MEAN_SPOT_LIFETIME_HOURS, rel=0.01)

    def test_complete_lifetime_lowers_remaining(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(2.0)
        model.record_complete(2.0)
        remaining = model.expected_remaining_lifetime(2.0)
        assert remaining < MEAN_SPOT_LIFETIME_HOURS

    def test_survival_decreases_with_age(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1.0)
        s0 = model.survival(0.0)
        s2 = model.survival(2.0)
        assert s2 < s0

    def test_volatility_adjustment_shrinks_survival(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1.0)
        s1 = model.survival(1.0, gamma_star=1.0)
        s2 = model.survival(1.0, gamma_star=2.0)
        assert s2 < s1


class TestVirtualInstanceTracker:
    def test_predict_lifetime_positive(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        tracker = VirtualInstanceTracker(region="us-east-1")
        tracker.record_probe(t0, available=True)
        assert tracker.predict_lifetime_hours(t0) > 0

    def test_preemption_resets_streak(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        tracker = VirtualInstanceTracker(region="us-east-1")
        tracker.record_probe(t0, available=True)
        tracker.record_probe(t0 + timedelta(hours=2), available=False)
        assert tracker.current_age_hours(t0 + timedelta(hours=3)) == 0.0

    def test_conservative_volatility_default(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        obs = [SpotObservation(t0, True)]
        model = NelsonAalenModel()
        assert conservative_volatility(obs, model) == 1.0


class TestBucketHours:
    def test_default_bucket_is_quarter_hour(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1.1)
        assert list(model.complete_counts) == [1.0]

    def test_bucket_follows_probe_interval(self) -> None:
        model = NelsonAalenModel(bucket_hours=2.0)
        model.record_complete(3.1)
        model.record_censored(0.4)
        assert list(model.complete_counts) == [4.0]
        assert list(model.censored_counts) == [0.0]
        # raw samples stay unbucketed for analytics
        assert model.complete_samples == [3.1]


class TestReplayObservations:
    T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def _obs(self, hours: float, available: bool, run_id: str | None = None) -> SpotObservation:
        return SpotObservation(self.T0 + timedelta(hours=hours), available, "probe", run_id)

    def test_replay_yields_complete_lifetime(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        replay_observations(
            tracker, [self._obs(0, True), self._obs(1, True), self._obs(3, False)]
        )
        assert tracker.model.complete_samples == [3.0]
        assert tracker.model.censored_samples == []

    def test_gap_closes_streak_as_censored_not_chained(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        replay_observations(
            tracker,
            [self._obs(0, True), self._obs(2, True), self._obs(50, True), self._obs(51, False)],
            max_gap_hours=6.0,
        )
        # first streak censored at its last observation (2h); second is a fresh 1h lifetime
        assert tracker.model.censored_samples == [2.0]
        assert tracker.model.complete_samples == [1.0]

    def test_run_boundary_closes_streak_as_censored(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        replay_observations(
            tracker,
            [self._obs(0, True, "a"), self._obs(2, True, "a"), self._obs(30, True, "b")],
        )
        assert tracker.model.censored_samples == [2.0]

    def test_lone_observation_is_not_a_zero_hour_sample(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        replay_observations(tracker, [self._obs(0, True)])
        assert tracker.model.n_samples == 0
        assert tracker.streak_start_utc is None

    def test_replay_never_fires_callback(self) -> None:
        seen: list[SpotObservation] = []
        tracker = VirtualInstanceTracker(region="r", on_observation=seen.append)
        replay_observations(tracker, [self._obs(0, True), self._obs(1, False)])
        assert seen == []


class TestPriorSourceAndCallback:
    T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def test_no_observations_is_labeled_assumed(self) -> None:
        label = VirtualInstanceTracker(region="r").prior_source()
        assert label == "assumed default 4h (no observations)"

    def test_observations_without_samples_still_assumed(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        tracker.record_probe(self.T0, available=True)
        assert tracker.prior_source().startswith("assumed default")

    def test_history_label_once_lifetime_recorded(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        tracker.record_probe(self.T0, available=True)
        tracker.record_probe(self.T0 + timedelta(hours=2), available=False)
        assert tracker.prior_source().startswith("history: 2 observations")

    def test_source_tag_and_callback(self) -> None:
        seen: list[SpotObservation] = []
        tracker = VirtualInstanceTracker(region="r", on_observation=seen.append)
        tracker.record_probe(self.T0, available=True, source="launch")
        tracker.record_probe(self.T0, available=True)
        assert [o.source for o in seen] == ["launch", "probe"]
        assert tracker.observations == seen

    def test_assumed_constant_alias(self) -> None:
        assert MEAN_SPOT_LIFETIME_HOURS == ASSUMED_SPOT_MEAN_LIFETIME_HOURS
