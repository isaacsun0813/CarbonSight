"""Unit tests for dynamic scheduler skeleton."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from carbonsight_core.models import JobSpec
from carbonsight_core.planning import (
    JobMode,
    PlanConstraints,
    SchedulerCandidate,
    make_runtime_state,
    pick_next_state,
    safety_net_triggered,
)

UTC = timezone.utc


def _constraints(t0: datetime, deadline_hours: float, total_work: float = 4.0) -> PlanConstraints:
    return PlanConstraints(
        deadline_utc=t0 + timedelta(hours=deadline_hours),
        now_utc=t0,
        cold_start_hours=0.5,
    )


class TestSafetyNet:
    def test_triggered_when_slack_low(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        constraints = _constraints(t0, deadline_hours=4.5)
        state = replace(
            make_runtime_state(
                job,
                constraints,
                region="us-east-1",
                mode=JobMode.SPOT,
                progress_hours=0.0,
            ),
            now_utc=t0 + timedelta(hours=0.5),
        )
        assert safety_net_triggered(state, constraints)

    def test_not_triggered_with_slack(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        constraints = _constraints(t0, deadline_hours=12.0)
        state = make_runtime_state(job, constraints, region="us-east-1", mode=JobMode.SPOT)
        assert not safety_net_triggered(state, constraints)


class TestPickNextState:
    def test_safety_net_forces_ondemand(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        constraints = _constraints(t0, deadline_hours=4.5)
        state = replace(
            make_runtime_state(job, constraints, region="us-east-1", mode=JobMode.SPOT),
            now_utc=t0 + timedelta(hours=0.5),
        )
        candidates = [
            SchedulerCandidate(
                region="us-east-1",
                mode=JobMode.ON_DEMAND,
                cost_rate_usd_per_hr=8.0,
                carbon_rate_kg_per_hr=2.0,
                expected_lifetime_hours=4.0,
            ),
            SchedulerCandidate(
                region="us-west-2",
                mode=JobMode.ON_DEMAND,
                cost_rate_usd_per_hr=9.0,
                carbon_rate_kg_per_hr=0.5,
                migration_usd=2.0,
                expected_lifetime_hours=4.0,
            ),
        ]
        decision = pick_next_state(state, constraints, candidates, c_od_min=8.0)
        assert decision.safety_net
        assert decision.chosen is not None
        assert decision.chosen.mode == JobMode.ON_DEMAND
        assert decision.chosen.region == "us-east-1"

    def test_fallback_picks_cheaper_ondemand(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        constraints = _constraints(t0, deadline_hours=4.5)
        state = replace(
            make_runtime_state(job, constraints, region="us-west-2", mode=JobMode.IDLE),
            now_utc=t0 + timedelta(hours=0.5),
        )
        candidates = [
            SchedulerCandidate(
                region="us-east-1",
                mode=JobMode.ON_DEMAND,
                cost_rate_usd_per_hr=4.0,
                carbon_rate_kg_per_hr=3.0,
                migration_usd=1.0,
                expected_lifetime_hours=4.0,
            ),
            SchedulerCandidate(
                region="us-west-2",
                mode=JobMode.ON_DEMAND,
                cost_rate_usd_per_hr=9.0,
                carbon_rate_kg_per_hr=0.5,
                expected_lifetime_hours=4.0,
            ),
        ]
        decision = pick_next_state(state, constraints, candidates, c_od_min=4.0)
        assert decision.safety_net
        assert decision.chosen is not None
        assert decision.chosen.region == "us-east-1"

    def test_idle_picks_upgrade_when_utility_positive(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        constraints = _constraints(t0, deadline_hours=4.0)
        state = make_runtime_state(job, constraints, region="", mode=JobMode.IDLE)
        candidates = [
            SchedulerCandidate(
                region="us-west-2",
                mode=JobMode.SPOT,
                cost_rate_usd_per_hr=1.0,
                carbon_rate_kg_per_hr=0.1,
                expected_lifetime_hours=2.0,
            ),
        ]
        decision = pick_next_state(state, constraints, candidates, c_od_min=4.0)
        assert not decision.safety_net
        assert decision.chosen is not None
        assert decision.chosen.mode == JobMode.SPOT

    def test_no_change_when_no_improvement(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        constraints = _constraints(t0, deadline_hours=12.0)
        state = make_runtime_state(
            job,
            constraints,
            region="us-east-1",
            mode=JobMode.SPOT,
            progress_hours=0.5,
        )
        candidates = [
            SchedulerCandidate(
                region="us-east-1",
                mode=JobMode.SPOT,
                cost_rate_usd_per_hr=100.0,
                carbon_rate_kg_per_hr=50.0,
                expected_lifetime_hours=2.0,
            ),
        ]
        decision = pick_next_state(state, constraints, candidates, c_od_min=4.0)
        assert decision.chosen is None
