"""Unit tests for spot availability tracking (SkyNomad Sec 4.3)."""

from datetime import datetime, timezone, timedelta

import pytest

from carbonsight_core.spot.availability import (
    AvailabilityTracker,
    SpotObservation,
    VirtualInstance,
)


# ---------------------------------------------------------------------------
# SpotObservation
# ---------------------------------------------------------------------------


class TestSpotObservation:
    def test_valid_outcomes(self):
        o = SpotObservation(t=0, region="us-east-1", outcome=1)
        assert o.outcome == 1
        o0 = SpotObservation(t=0, region="us-east-1", outcome=0)
        assert o0.outcome == 0

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValueError):
            SpotObservation(t=0, region="us-east-1", outcome=2)
        with pytest.raises(ValueError):
            SpotObservation(t=0, region="us-east-1", outcome=-1)

    def test_to_from_dict_roundtrip(self):
        o = SpotObservation(t=123.5, region="eu-west-1", outcome=1)
        d = o.to_dict()
        o2 = SpotObservation.from_dict(d)
        assert o2.t == o.t
        assert o2.region == o.region
        assert o2.outcome == o.outcome

    def test_datetime_t_roundtrip(self):
        ts = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        o = SpotObservation(t=ts, region="us-west-2", outcome=1)
        d = o.to_dict()
        o2 = SpotObservation.from_dict(d)
        assert isinstance(o2.t, datetime)
        assert o2.t == ts


# ---------------------------------------------------------------------------
# VirtualInstance
# ---------------------------------------------------------------------------


class TestVirtualInstance:
    def test_lifetime_float(self):
        vi = VirtualInstance(
            id="us-east-1-1", region="us-east-1", start_t=0, end_t=10, end_reason="preemption"
        )
        assert vi.lifetime == pytest.approx(10.0)

    def test_lifetime_datetime(self):
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=2)
        vi = VirtualInstance(
            id="x-1", region="us-east-1", start_t=t0, end_t=t1, end_reason="preemption"
        )
        assert vi.lifetime == pytest.approx(7200.0)

    def test_invalid_end_reason(self):
        with pytest.raises(ValueError):
            VirtualInstance(
                id="x-1", region="r", start_t=0, end_t=1, end_reason="invalid"  # type: ignore
            )


# ---------------------------------------------------------------------------
# AvailabilityTracker core
# ---------------------------------------------------------------------------


class TestAvailabilityTracker:
    def test_record_and_get(self):
        tracker = AvailabilityTracker()
        tracker.record_observation(SpotObservation(t=0, region="us-east-1", outcome=0))
        tracker.record_observation(1, "us-east-1", 1)
        tracker.record_observation(t_or_obs=2, region="us-east-1", outcome=1)
        obs = tracker.get_observations("us-east-1")
        assert len(obs) == 3
        assert [o.outcome for o in obs] == [0, 1, 1]

    def test_extract_single_preemption(self):
        tracker = AvailabilityTracker()
        # probe seq: 0,1,1,0 => virtual instance from t=1 to t=3, preemption
        seq = [(0, 0), (1, 1), (2, 1), (3, 0)]
        for t, out in seq:
            tracker.record_observation(t, "us-east-1", out)

        vis = tracker.extract_virtual_instances("us-east-1")
        assert len(vis) == 1
        vi = vis[0]
        assert vi.start_t == 1
        assert vi.end_t == 3
        assert vi.end_reason == "preemption"
        assert vi.lifetime == pytest.approx(2.0)

    def test_extract_censored(self):
        tracker = AvailabilityTracker()
        seq = [(0, 0), (1, 1), (2, 1)]
        for t, out in seq:
            tracker.record_observation(t, "us-east-1", out)

        vis = tracker.extract_virtual_instances("us-east-1")
        assert len(vis) == 1
        vi = vis[0]
        assert vi.start_t == 1
        assert vi.end_t == 2
        assert vi.end_reason == "censored"
        assert vi.region == "us-east-1"

    def test_probe_seq_starts_with_1(self):
        tracker = AvailabilityTracker()
        seq = [(0, 1), (1, 1), (2, 0)]
        for t, out in seq:
            tracker.record_observation(t, "us-east-1", out)

        vis = tracker.extract_virtual_instances("us-east-1")
        assert len(vis) == 1
        assert vis[0].start_t == 0
        assert vis[0].end_t == 2
        assert vis[0].end_reason == "preemption"

    def test_multiple_cycles(self):
        tracker = AvailabilityTracker()
        # 0,1,0,1,0 => two preemptions
        seq = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
        for t, out in seq:
            tracker.record_observation(t, "us-east-1", out)

        vis = tracker.extract_virtual_instances("us-east-1")
        assert len(vis) == 2
        assert vis[0].start_t == 1 and vis[0].end_t == 2
        assert vis[0].end_reason == "preemption"
        assert vis[1].start_t == 3 and vis[1].end_t == 4
        assert vis[1].end_reason == "preemption"

    def test_multiple_regions(self):
        tracker = AvailabilityTracker()
        tracker.record_observation(0, "us-east-1", 0)
        tracker.record_observation(1, "us-east-1", 1)
        tracker.record_observation(2, "us-east-1", 0)

        tracker.record_observation(0, "eu-west-1", 0)
        tracker.record_observation(5, "eu-west-1", 1)

        vis_all = tracker.extract_virtual_instances()
        assert len(vis_all) == 2  # one per region

        vis_us = tracker.extract_virtual_instances("us-east-1")
        assert len(vis_us) == 1
        assert vis_us[0].region == "us-east-1"
        assert vis_us[0].end_reason == "preemption"

        vis_eu = tracker.extract_virtual_instances("eu-west-1")
        assert len(vis_eu) == 1
        assert vis_eu[0].end_reason == "censored"

    def test_is_available(self):
        tracker = AvailabilityTracker()
        tracker.record_observation(0, "us-east-1", 0)
        tracker.record_observation(10, "us-east-1", 1)
        tracker.record_observation(20, "us-east-1", 0)

        assert tracker.is_available("us-east-1", at=5) is False
        assert tracker.is_available("us-east-1", at=10) is True
        assert tracker.is_available("us-east-1", at=15) is True
        assert tracker.is_available("us-east-1", at=20) is False
        assert tracker.is_available("us-east-1", at=25) is False

        # No observations for region
        assert tracker.is_available("eu-west-1", at=10) is False

    def test_is_available_datetime(self):
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=1)
        t2 = t0 + timedelta(hours=2)

        tracker = AvailabilityTracker()
        tracker.record_observation(t0, "us-east-1", 0)
        tracker.record_observation(t1, "us-east-1", 1)

        assert tracker.is_available("us-east-1", at=t0) is False
        assert tracker.is_available("us-east-1", at=t1) is True
        assert tracker.is_available("us-east-1", at=t2) is True

    def test_unsorted_insertion(self):
        tracker = AvailabilityTracker()
        # Insert out of order
        tracker.record_observation(2, "us-east-1", 0)
        tracker.record_observation(0, "us-east-1", 0)
        tracker.record_observation(1, "us-east-1", 1)

        vis = tracker.extract_virtual_instances("us-east-1")
        assert len(vis) == 1
        assert vis[0].start_t == 1
        assert vis[0].end_t == 2

        # Observations should be returned sorted
        obs = tracker.get_observations("us-east-1")
        assert [o.t for o in obs] == [0, 1, 2]

    def test_to_dict_from_dict(self):
        tracker = AvailabilityTracker()
        tracker.record_observation(0, "us-east-1", 0)
        tracker.record_observation(1, "us-east-1", 1)
        tracker.record_observation(2, "us-east-1", 0)
        tracker.record_observation(0, "eu-west-1", 1)

        d = tracker.to_dict()
        tracker2 = AvailabilityTracker.from_dict(d)

        assert len(tracker2) == len(tracker)
        vis1 = tracker.extract_virtual_instances()
        vis2 = tracker2.extract_virtual_instances()
        assert len(vis1) == len(vis2)
        # Compare by region and reason
        vis1_sorted = sorted([(v.region, v.start_t, v.end_t, v.end_reason) for v in vis1])
        vis2_sorted = sorted([(v.region, v.start_t, v.end_t, v.end_reason) for v in vis2])
        assert vis1_sorted == vis2_sorted

        # Also check is_available preserved
        assert tracker2.is_available("us-east-1", 1) is True
        assert tracker2.is_available("us-east-1", 2) is False

    def test_empty_tracker(self):
        tracker = AvailabilityTracker()
        assert tracker.extract_virtual_instances() == []
        assert tracker.extract_virtual_instances("us-east-1") == []
        assert tracker.is_available("us-east-1", at=0) is False
        d = tracker.to_dict()
        assert d["observations"] == []
        tracker2 = AvailabilityTracker.from_dict(d)
        assert len(tracker2) == 0

    def test_only_zeroes_no_instances(self):
        tracker = AvailabilityTracker()
        for t in range(5):
            tracker.record_observation(t, "us-east-1", 0)
        vis = tracker.extract_virtual_instances("us-east-1")
        assert vis == []

    def test_probe_seq_virtual_lifetime_accuracy(self):
        """Validate probe seq -> virtual instance lifetime mapping per spec."""
        tracker = AvailabilityTracker()
        # Simulate: unavailable at 0, becomes available at 5, stays till 15, then preempted at 20, then available again at 25 still alive at 30
        probes = [
            (0, 0),
            (5, 1),
            (10, 1),
            (15, 1),
            (20, 0),
            (25, 1),
            (30, 1),
        ]
        for t, o in probes:
            tracker.record_observation(t, "ap-south-1", o)

        vis = tracker.extract_virtual_instances("ap-south-1")
        assert len(vis) == 2
        # First instance: 5 -> 20 preemption
        assert vis[0].start_t == 5
        assert vis[0].end_t == 20
        assert vis[0].end_reason == "preemption"
        assert vis[0].lifetime == pytest.approx(15)

        # Second instance: 25 -> 30 censored
        assert vis[1].start_t == 25
        assert vis[1].end_t == 30
        assert vis[1].end_reason == "censored"
        assert vis[1].lifetime == pytest.approx(5)

    def test_censored_vs_preemption_distinction(self):
        """Explicit test that censored and preemption are distinguished correctly."""
        tracker = AvailabilityTracker()
        # Preemption case
        tracker.record_observation(0, "us-east-1", 0)
        tracker.record_observation(1, "us-east-1", 1)
        tracker.record_observation(2, "us-east-1", 0)
        # Censored case different region
        tracker.record_observation(0, "eu-north-1", 0)
        tracker.record_observation(1, "eu-north-1", 1)

        vis = {v.region: v for v in tracker.extract_virtual_instances()}
        assert vis["us-east-1"].end_reason == "preemption"
        assert vis["eu-north-1"].end_reason == "censored"
