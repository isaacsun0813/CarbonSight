"""Spot availability tracking (SkyNomad Sec 4.3).

Probe traces are on a fixed hourly grid, so ``T(h)`` below reads as "hour h" and
an inferred lifetime is a count of hours.
"""

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.spot.availability import (
    AvailabilityTracker,
    SpotObservation,
    VirtualInstance,
    synthetic_probe_trace,
    synthetic_uptime,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def t(hours: float) -> datetime:
    """Hour ``hours`` of the trace."""
    return BASE + timedelta(hours=hours)


def record(tracker: AvailabilityTracker, region: str, seq: list[tuple[float, int]]) -> None:
    for hour, outcome in seq:
        tracker.record_observation(t(hour), region, outcome)


class TestSpotObservation:
    def test_valid_outcomes(self):
        assert SpotObservation(t=t(0), region="us-east-1", outcome=1).outcome == 1
        assert SpotObservation(t=t(0), region="us-east-1", outcome=0).outcome == 0

    @pytest.mark.parametrize("bad", [2, -1, 10])
    def test_invalid_outcome_raises(self, bad):
        with pytest.raises(ValueError):
            SpotObservation(t=t(0), region="us-east-1", outcome=bad)

    def test_empty_region_raises(self):
        with pytest.raises(ValueError):
            SpotObservation(t=t(0), region="", outcome=1)

    def test_non_datetime_timestamp_raises(self):
        with pytest.raises(TypeError):
            SpotObservation(t=0, region="us-east-1", outcome=1)

    def test_naive_timestamp_is_read_as_utc(self):
        obs = SpotObservation(t=datetime(2026, 1, 1), region="us-east-1", outcome=1)
        assert obs.t == BASE

    def test_to_from_dict_roundtrip(self):
        obs = SpotObservation(t=t(3.5), region="eu-west-1", outcome=1)
        back = SpotObservation.from_dict(obs.to_dict())
        assert (back.t, back.region, back.outcome) == (obs.t, obs.region, obs.outcome)


class TestVirtualInstance:
    def test_lifetime_hours(self):
        vi = VirtualInstance("us-east-1-1", "us-east-1", t(0), t(10), "preemption")
        assert vi.lifetime_hours == pytest.approx(10.0)
        assert vi.preempted is True

    def test_censored_is_not_preempted(self):
        vi = VirtualInstance("x-1", "us-east-1", t(0), t(2), "censored")
        assert vi.preempted is False

    def test_invalid_end_reason(self):
        with pytest.raises(ValueError):
            VirtualInstance("x-1", "r", t(0), t(1), "invalid")  # type: ignore[arg-type]

    def test_to_from_dict_roundtrip(self):
        vi = VirtualInstance("x-1", "r", t(0), t(4), "preemption")
        back = VirtualInstance.from_dict(vi.to_dict())
        assert back == vi


class TestAvailabilityTracker:
    def test_record_and_get(self):
        tracker = AvailabilityTracker()
        tracker.record_observation(SpotObservation(t=t(0), region="us-east-1", outcome=0))
        tracker.record_observation(t(1), "us-east-1", 1)
        tracker.record_observation(t_or_obs=t(2), region="us-east-1", outcome=1)
        assert [o.outcome for o in tracker.get_observations("us-east-1")] == [0, 1, 1]
        assert len(tracker) == 3

    def test_mixing_obs_and_kwargs_raises(self):
        tracker = AvailabilityTracker()
        with pytest.raises(ValueError):
            tracker.record_observation(
                SpotObservation(t=t(0), region="r", outcome=1), region="r", outcome=1
            )

    def test_extract_single_preemption(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (2, 1), (3, 0)])
        (vi,) = tracker.extract_virtual_instances("us-east-1")
        assert (vi.start_t, vi.end_t, vi.end_reason) == (t(1), t(3), "preemption")
        assert vi.lifetime_hours == pytest.approx(2.0)

    def test_extract_censored(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (2, 1)])
        (vi,) = tracker.extract_virtual_instances("us-east-1")
        assert (vi.start_t, vi.end_t, vi.end_reason) == (t(1), t(2), "censored")
        assert vi.region == "us-east-1"

    def test_trace_starting_available(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 1), (1, 1), (2, 0)])
        (vi,) = tracker.extract_virtual_instances("us-east-1")
        assert (vi.start_t, vi.end_t, vi.end_reason) == (t(0), t(2), "preemption")

    def test_multiple_cycles(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)])
        first, second = tracker.extract_virtual_instances("us-east-1")
        assert (first.start_t, first.end_t, first.end_reason) == (t(1), t(2), "preemption")
        assert (second.start_t, second.end_t, second.end_reason) == (t(3), t(4), "preemption")

    def test_multiple_regions(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (2, 0)])
        record(tracker, "eu-west-1", [(0, 0), (5, 1)])
        assert len(tracker.extract_virtual_instances()) == 2
        assert tracker.extract_virtual_instances("us-east-1")[0].end_reason == "preemption"
        assert tracker.extract_virtual_instances("eu-west-1")[0].end_reason == "censored"

    def test_only_zeroes_yields_nothing(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(h, 0) for h in range(5)])
        assert tracker.extract_virtual_instances("us-east-1") == []

    def test_probe_trace_lifetimes(self):
        """Down, up at 5 through 15, preempted at 20, up again at 25, still up at 30."""
        tracker = AvailabilityTracker()
        record(
            tracker,
            "ap-south-1",
            [(0, 0), (5, 1), (10, 1), (15, 1), (20, 0), (25, 1), (30, 1)],
        )
        first, second = tracker.extract_virtual_instances("ap-south-1")
        assert (first.start_t, first.end_t, first.end_reason) == (t(5), t(20), "preemption")
        assert first.lifetime_hours == pytest.approx(15.0)
        assert (second.start_t, second.end_t, second.end_reason) == (t(25), t(30), "censored")
        assert second.lifetime_hours == pytest.approx(5.0)

    def test_observed_lifetimes_pairs(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (3, 0), (4, 1)])
        assert tracker.observed_lifetimes("us-east-1") == [(2.0, True), (0.0, False)]

    def test_is_available(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (10, 1), (20, 0)])
        assert tracker.is_available("us-east-1", t(5)) is False
        assert tracker.is_available("us-east-1", t(10)) is True
        assert tracker.is_available("us-east-1", t(15)) is True
        assert tracker.is_available("us-east-1", t(20)) is False
        assert tracker.is_available("us-east-1", t(25)) is False
        assert tracker.is_available("eu-west-1", t(10)) is False

    def test_unsorted_insertion_is_sorted_on_read(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(2, 0), (0, 0), (1, 1)])
        (vi,) = tracker.extract_virtual_instances("us-east-1")
        assert (vi.start_t, vi.end_t) == (t(1), t(2))
        assert [o.t for o in tracker.get_observations("us-east-1")] == [t(0), t(1), t(2)]

    def test_to_dict_from_dict(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 0), (1, 1), (2, 0)])
        record(tracker, "eu-west-1", [(0, 1)])
        restored = AvailabilityTracker.from_dict(tracker.to_dict())
        assert len(restored) == len(tracker)
        assert restored.extract_virtual_instances() == tracker.extract_virtual_instances()
        assert restored.is_available("us-east-1", t(1)) is True
        assert restored.is_available("us-east-1", t(2)) is False

    def test_empty_tracker(self):
        tracker = AvailabilityTracker()
        assert tracker.extract_virtual_instances() == []
        assert tracker.extract_virtual_instances("us-east-1") == []
        assert tracker.is_available("us-east-1", t(0)) is False
        assert tracker.to_dict()["observations"] == []
        assert len(AvailabilityTracker.from_dict(tracker.to_dict())) == 0

    def test_clear(self):
        tracker = AvailabilityTracker()
        record(tracker, "us-east-1", [(0, 1)])
        tracker.clear()
        assert len(tracker) == 0 and tracker.regions() == []


class TestSyntheticTrace:
    def test_uptime_deterministic_and_bounded(self):
        for region in ("us-east-1", "sa-east-1", "af-south-1"):
            u = synthetic_uptime(region)
            assert 0.45 <= u <= 0.95
            assert u == synthetic_uptime(region)

    def test_uptime_varies_across_regions(self):
        values = {synthetic_uptime(r) for r in ("us-east-1", "eu-north-1", "sa-east-1")}
        assert len(values) == 3

    def test_trace_is_reproducible_and_hourly(self):
        a = synthetic_probe_trace("us-east-1", hours=24)
        b = synthetic_probe_trace("us-east-1", hours=24)
        assert [(o.t, o.outcome) for o in a] == [(o.t, o.outcome) for o in b]
        assert len(a) == 24
        assert (a[1].t - a[0].t) == timedelta(hours=1)

    def test_trace_yields_integer_hour_lifetimes(self):
        tracker = AvailabilityTracker(synthetic_probe_trace("eu-west-1", hours=72))
        lifetimes = [lt for lt, _ in tracker.observed_lifetimes("eu-west-1")]
        assert lifetimes
        assert all(lt == pytest.approx(round(lt)) for lt in lifetimes)

    def test_high_uptime_gives_longer_lifetimes(self):
        low = AvailabilityTracker(synthetic_probe_trace("r", hours=500, uptime=0.4))
        high = AvailabilityTracker(synthetic_probe_trace("r", hours=500, uptime=0.95))
        mean_low = sum(lt for lt, _ in low.observed_lifetimes("r")) / len(
            low.observed_lifetimes("r")
        )
        mean_high = sum(lt for lt, _ in high.observed_lifetimes("r")) / len(
            high.observed_lifetimes("r")
        )
        assert mean_high > mean_low
