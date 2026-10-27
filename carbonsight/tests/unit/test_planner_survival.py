"""Unit tests for spot survival model."""

from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.planning.survival import (
    ASSUMED_SPOT_MEAN_LIFETIME_HOURS,
    MEAN_SPOT_LIFETIME_HOURS,
    MIN_EVENTS_FOR_NELSON_AALEN,
    NelsonAalenModel,
    SpotObservation,
    VirtualInstanceTracker,
    advisor_band_weights,
    conservative_volatility,
    pooled_hazard_scale,
    refresh_hybrid_priors,
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


class TestHybridPrior:
    T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    @staticmethod
    def _trackers(*regions: str) -> dict[str, VirtualInstanceTracker]:
        return {r: VirtualInstanceTracker(region=r) for r in regions}

    def test_weights_normalized_to_mean_one(self) -> None:
        weights = advisor_band_weights({"a": 0, "b": 4})
        assert sum(weights.values()) / 2 == pytest.approx(1.0)
        assert weights["b"] / weights["a"] == pytest.approx(10.0)  # 0.25 / 0.025

    def test_same_band_everywhere_is_neutral(self) -> None:
        assert advisor_band_weights({"a": 4, "b": 4}) == {"a": 1.0, "b": 1.0}

    def test_invalid_bands_are_ignored(self) -> None:
        assert advisor_band_weights({"a": 9, "b": -1}) == {}

    def test_no_data_scale_is_assumed_default(self) -> None:
        scale, events = pooled_hazard_scale(self._trackers("a").values())
        assert events == 0
        assert scale == pytest.approx(1.0 / ASSUMED_SPOT_MEAN_LIFETIME_HOURS)

    def test_advisor_tilts_regions_but_keeps_mean_hazard(self) -> None:
        trackers = self._trackers("calm", "busy")
        refresh_hybrid_priors(trackers, {"calm": 0, "busy": 4})
        calm = trackers["calm"].predict_lifetime_hours(self.T0)
        busy = trackers["busy"].predict_lifetime_hours(self.T0)
        assert calm > busy
        # Weights average to 1, so mean hazard is still 1 / assumed mean.
        mean_hazard = (trackers["calm"].prior_hazard + trackers["busy"].prior_hazard) / 2
        assert mean_hazard == pytest.approx(1.0 / ASSUMED_SPOT_MEAN_LIFETIME_HOURS)

    def test_equal_bands_reduce_to_assumed_default(self) -> None:
        trackers = self._trackers("a", "b")
        refresh_hybrid_priors(trackers, {"a": 4, "b": 4})
        assert trackers["a"].predict_lifetime_hours(self.T0) == pytest.approx(
            ASSUMED_SPOT_MEAN_LIFETIME_HOURS
        )

    def test_gamma_exponential_posterior(self) -> None:
        trackers = self._trackers("a")
        trackers["a"].model.record_complete(1.0)
        trackers["a"].model.record_complete(3.0)
        refresh_hybrid_priors(trackers, {"a": 4})
        # scale = (1+2)/(1*4+4) = 0.375; posterior = (1+2)/(1/0.375 + 4)
        expected_hazard = 3.0 / (1.0 / 0.375 + 4.0)
        assert trackers["a"].predict_lifetime_hours(self.T0) == pytest.approx(1.0 / expected_hazard)

    def test_censored_exposure_lengthens_lifetime(self) -> None:
        plain = self._trackers("a")
        censored = self._trackers("a")
        censored["a"].model.record_censored(10.0)
        refresh_hybrid_priors(plain, {"a": 4})
        refresh_hybrid_priors(censored, {"a": 4})
        assert censored["a"].predict_lifetime_hours(self.T0) > plain["a"].predict_lifetime_hours(
            self.T0
        )

    def test_switches_to_nelson_aalen_at_enough_events(self) -> None:
        hybrid = self._trackers("a")
        plain = self._trackers("a")
        for tracker in (hybrid["a"], plain["a"]):
            for hours in (1.0, 2.0, 2.0, 3.0, 4.0):
                tracker.model.record_complete(hours)
        assert len(hybrid["a"].model.complete_samples) >= MIN_EVENTS_FOR_NELSON_AALEN
        refresh_hybrid_priors(hybrid, {"a": 0})
        assert hybrid["a"].predict_lifetime_hours(self.T0) == pytest.approx(
            plain["a"].predict_lifetime_hours(self.T0)
        )

    def test_no_prior_keeps_legacy_behavior(self) -> None:
        tracker = VirtualInstanceTracker(region="a")
        assert tracker.prior_hazard is None
        assert tracker.predict_lifetime_hours(self.T0) == pytest.approx(
            NelsonAalenModel().expected_remaining_lifetime(0.0)
        )

    def test_label_tiers(self) -> None:
        trackers = self._trackers("a", "b")
        refresh_hybrid_priors(trackers, {"a": 1})
        assert trackers["a"].prior_source() == "assumed default 4h x advisor band 5-10%"
        assert trackers["b"].prior_source() == (
            "assumed default 4h (no advisor band for this region)"
        )
        trackers["a"].model.record_complete(2.0)
        refresh_hybrid_priors(trackers, {"a": 1})
        assert trackers["a"].prior_source().startswith("history: 1 events, 2.0h exposure")
        assert trackers["b"].prior_source().startswith("pooled history (1 events)")
