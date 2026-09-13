"""Unit tests for spot survival model."""

from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.planning.survival import (
    MEAN_SPOT_LIFETIME_HOURS,
    NelsonAalenModel,
    SpotObservation,
    VirtualInstanceTracker,
    conservative_volatility,
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
