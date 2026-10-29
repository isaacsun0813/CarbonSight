"""Unit tests for spot survival model."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.planning.survival import (
    ASSUMED_SPOT_MEAN_LIFETIME_HOURS,
    MEAN_SPOT_LIFETIME_HOURS,
    NelsonAalenModel,
    SpotObservation,
    VirtualInstanceTracker,
    replay_observations,
)

UTC = timezone.utc


class TestNelsonAalenModel:
    def test_default_exponential_is_age_independent_and_bounded(self) -> None:
        model = NelsonAalenModel()
        expected = 4 * (1 - math.exp(-0.5))
        for age in (0.0, 3.0, 20.0):
            assert model.expected_remaining_lifetime(age, horizon_hours=2) == pytest.approx(expected)
        assert model.expected_remaining_lifetime(0, horizon_hours=100) == pytest.approx(4)

    def test_integrates_nonuniform_intervals_and_scales_with_time(self) -> None:
        for scale in (1, 10):
            model = NelsonAalenModel()
            model.record_complete(1 * scale)
            model.record_complete(3 * scale)
            expected = scale * (1 + 2 * math.exp(-0.5))
            assert model.expected_remaining_lifetime(0, horizon_hours=3 * scale) == pytest.approx(expected)

    def test_conditions_at_exact_age_and_clips_intervals(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1)
        model.record_complete(3)
        assert model.expected_remaining_lifetime(1, horizon_hours=1.5) == pytest.approx(1.5)
        assert model.expected_remaining_lifetime(0.5, horizon_hours=1) == pytest.approx(
            0.5 + 0.5 * math.exp(-0.5)
        )

    def test_tied_censored_samples_contribute_to_population(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1)
        model.record_censored(1)
        model.record_complete(3)
        assert model.hazard(1) == pytest.approx(1 / 3)
        assert model.expected_remaining_lifetime(0, horizon_hours=3) == pytest.approx(
            1 + 2 * math.exp(-1 / 3)
        )

    def test_unobserved_tail_uses_explicit_exponential_assumption(self) -> None:
        model = NelsonAalenModel()
        model.record_complete(1)
        tail = 4 * (1 - math.exp(-0.5))
        assert model.expected_remaining_lifetime(0, horizon_hours=3) == pytest.approx(
            1 + math.exp(-1) * tail
        )
        assert model.expected_remaining_lifetime(8, horizon_hours=2) == pytest.approx(tail)

    def test_censored_history_preserves_support_without_inventing_events(self) -> None:
        model = NelsonAalenModel()
        model.record_censored(2)
        assert model.cumulative_hazard(2) == 0
        assert model.expected_remaining_lifetime(0, horizon_hours=3) == pytest.approx(
            2 + 4 * (1 - math.exp(-0.25))
        )

    def test_zero_horizon(self) -> None:
        assert NelsonAalenModel().expected_remaining_lifetime(0, horizon_hours=0) == 0

    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
    def test_rejects_invalid_query_and_sample_hours(self, value) -> None:
        model = NelsonAalenModel()
        for operation in (
            lambda: model.expected_remaining_lifetime(value, horizon_hours=1),
            lambda: model.expected_remaining_lifetime(0, horizon_hours=value),
            lambda: model.record_complete(value),
            lambda: model.record_censored(value),
        ):
            with pytest.raises(ValueError):
                operation()

    @pytest.mark.parametrize("bucket", [0, -1, float("nan"), float("inf")])
    def test_rejects_invalid_bucket(self, bucket) -> None:
        with pytest.raises(ValueError):
            NelsonAalenModel(bucket_hours=bucket)


class TestVirtualInstanceTracker:
    def test_predict_lifetime_positive(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        tracker = VirtualInstanceTracker(region="us-east-1")
        tracker.record_probe(t0, available=True)
        assert tracker.predict_lifetime_hours(t0, horizon_hours=2) > 0

    def test_preemption_resets_streak(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        tracker = VirtualInstanceTracker(region="us-east-1")
        tracker.record_probe(t0, available=True)
        tracker.record_probe(t0 + timedelta(hours=2), available=False)
        assert tracker.current_age_hours(t0 + timedelta(hours=3)) == 0.0



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

    def test_migration_is_censored_and_repeated_failures_are_not_events(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        observations = [
            self._obs(0, True),
            SpotObservation(self.T0 + timedelta(hours=2), False, "migration"),
            self._obs(3, False),
            self._obs(4, False),
        ]
        replay_observations(tracker, observations)
        assert tracker.model.censored_samples == [2]
        assert tracker.model.complete_samples == []


class TestPriorSourceAndCallback:
    T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def test_no_observations_is_labeled_assumed(self) -> None:
        label = VirtualInstanceTracker(region="r").prior_source()
        assert label == "assumed exponential mean 4h (no lifetime samples)"

    def test_observations_without_samples_still_assumed(self) -> None:
        tracker = VirtualInstanceTracker(region="r")
        tracker.record_probe(self.T0, available=True)
        assert tracker.prior_source().startswith("assumed exponential")

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
