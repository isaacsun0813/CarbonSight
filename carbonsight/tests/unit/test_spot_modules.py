"""Additional SkyNomad spot module coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.spot.availability import AvailabilityTracker
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.policy import Action, SkyNomadPolicy
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import ODCandidate, facility_mwh_for_job, rank_candidates


def test_facility_mwh_scales_with_gpus():
    a = facility_mwh_for_job(gpu_count=1, lbar_hours=10)
    b = facility_mwh_for_job(gpu_count=4, lbar_hours=10)
    assert b == pytest.approx(4 * a)


def test_policy_run_loop_completes():
    tracker = AvailabilityTracker()
    life = LifetimeStats.from_exponential(8.0, n=20)
    ps = ProgressState.from_deadline_hours(total_progress=1.0, current_progress=0.0, deadline_hours=5)
    policy = SkyNomadPolicy(tracker, life, ps, probe_interval_hours=1.0)

    def cands(_now):
        return [
            ODCandidate("r1", "A100", 1.0, 200.0, 0.95, 4.0, on_demand_price=3.0),
            ODCandidate("r2", "A100", 1.1, 180.0, 0.9, 4.0, on_demand_price=3.2),
        ]

    decisions = policy.run_loop(cands, max_steps=30, step_hours=0.5, work_per_hour=0.2)
    assert decisions
    assert any(d.action == Action.TERMINATE for d in decisions) or ps.is_complete()


def test_migrate_on_delta():
    tracker = AvailabilityTracker()
    life = LifetimeStats.from_exponential(20.0)
    ps = ProgressState.from_deadline_hours(deadline_hours=30)
    policy = SkyNomadPolicy(tracker, life, ps, delta_utility=0.01, carbon_price_usd_per_ton=100.0)
    cheap_green = ODCandidate("green", "A100", 0.5, 50.0, 0.99, 20.0, on_demand_price=2.0)
    expensive = ODCandidate("dirty", "A100", 3.0, 900.0, 0.5, 5.0, on_demand_price=2.0)
    policy.current = expensive
    # rank both so utilities set
    rank_candidates([cheap_green, expensive], ps, carbon_price_usd_per_ton=100.0)
    d = policy.decide([cheap_green, expensive])
    assert d.action in (Action.MIGRATE, Action.RUN, Action.PROBE)


def test_lifetime_empty_defaults():
    s = LifetimeStats()
    assert s.compute_survival(5.0) <= 1.0
    assert s.expected_remaining(0.0) >= 1.0


@pytest.mark.parametrize(
    "p,P,hours,expected_theta",
    [
        (0, 10, 5, 2.0),
        (5, 10, 5, 1.0),
        (9, 10, 2, 0.5),
    ],
)
def test_theta_cases(p, P, hours, expected_theta):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ps = ProgressState(P=P, p=p, T=now + timedelta(hours=hours), t=now)
    assert ps.theta() == pytest.approx(expected_theta)
