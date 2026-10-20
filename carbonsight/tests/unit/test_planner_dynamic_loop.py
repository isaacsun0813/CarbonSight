"""Unit tests for plan_dynamic_job simulation loop."""

from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.models import JobSpec
from carbonsight_core.planning import (
    DynamicEventType,
    DynamicJob,
    JobMode,
    PlanConstraints,
    RegionPlanInput,
    SchedulerCandidate,
    SchedulerDecision,
    plan_dynamic_job,
)

UTC = timezone.utc


def _forecast(t0: datetime, values: list[float], step_min: int = 60) -> list[dict]:
    return [
        {"point_time": (t0 + timedelta(minutes=i * step_min)).isoformat(), "value": v}
        for i, v in enumerate(values)
    ]


def _constraints(t0: datetime, deadline_hours: float, **kwargs) -> PlanConstraints:
    return PlanConstraints(
        deadline_utc=t0 + timedelta(hours=deadline_hours),
        now_utc=t0,
        cold_start_hours=0.0,
        **kwargs,
    )


class TestPlanDynamicJob:
    def test_completes_with_feasible_deadline(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-west-2",
                watttime_regions=[("PACW", 1.0)],
                forecast_points=_forecast(t0, [200.0] * 12),
            ),
        ]
        result = plan_dynamic_job(job, _constraints(t0, 8.0), regions, tick_hours=1.0)
        assert result.deadline_met
        assert result.initial_launch is not None
        assert result.final_state.progress_hours >= job.duration_hours

    def test_missed_deadline_when_impossible(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=8.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [400.0] * 4),
            ),
        ]
        result = plan_dynamic_job(job, _constraints(t0, 2.0), regions, tick_hours=1.0)
        assert not result.deadline_met
        assert any(e.event_type == DynamicEventType.MISSED_DEADLINE for e in result.events)

    def test_initial_launch_spot_by_default(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-west-2",
                watttime_regions=[("PACW", 1.0)],
                forecast_points=_forecast(t0, [100.0] * 8),
            ),
        ]
        result = plan_dynamic_job(job, _constraints(t0, 6.0), regions, tick_hours=1.0)
        assert result.initial_launch is not None
        assert result.initial_launch.mode == JobMode.SPOT

    def test_safety_net_uses_ondemand_when_tight(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [400.0] * 6),
            ),
        ]
        constraints = PlanConstraints(
            deadline_utc=t0 + timedelta(hours=4.6),
            now_utc=t0,
            cold_start_hours=0.5,
        )
        state = plan_dynamic_job(job, constraints, regions, tick_hours=0.5)
        assert any(e.event_type == DynamicEventType.SAFETY_NET for e in state.events) or state.deadline_met

    def test_carbon_budget_can_fail_plan(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [5000.0] * 8),
            ),
        ]
        result = plan_dynamic_job(
            job,
            _constraints(t0, 8.0, carbon_budget_kg=0.01),
            regions,
            tick_hours=1.0,
        )
        assert not result.deadline_met

    def test_empty_regions(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        result = plan_dynamic_job(job, _constraints(t0, 4.0), [])
        assert not result.deadline_met
        assert result.initial_launch is None


class TestDynamicJob:
    def _regions(self, t0: datetime) -> list[RegionPlanInput]:
        return [
            RegionPlanInput(
                cloud_region="us-west-2",
                watttime_regions=[("PACW", 1.0)],
                forecast_points=_forecast(t0, [200.0] * 12),
            ),
        ]

    def test_methods_append_to_caller_events(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        events: list = []
        dyn = DynamicJob(job, _constraints(t0, 8.0), self._regions(t0), events)

        state = dyn.make_initial_state()
        chosen = SchedulerCandidate(
            region="us-west-2",
            mode=JobMode.SPOT,
            cost_rate_usd_per_hr=1.0,
            carbon_rate_kg_per_hr=0.1,
            expected_lifetime_hours=4.0,
        )
        decision = SchedulerDecision(chosen=chosen, safety_net=False, utility=1.0, reason="test")
        state = dyn.launch_instance(state, decision)
        assert state.mode == JobMode.SPOT
        assert dyn.initial_launch == chosen
        assert dyn.checkpoint == "us-west-2"
        assert [e.event_type for e in events] == [DynamicEventType.LAUNCH]

        state = dyn.advance_work(state, tick_hours=1.0)
        assert state.progress_hours >= job.duration_hours
        assert events[-1].event_type == DynamicEventType.COMPLETE
        assert dyn.events is events

        result = dyn.finalize(state)
        assert result.deadline_met
        assert result.events is events

    def test_finalize_missed_deadline(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        dyn = DynamicJob(job, _constraints(t0, 8.0), self._regions(t0), [])
        result = dyn.finalize(dyn.make_initial_state())
        assert not result.deadline_met
        assert result.events[-1].event_type == DynamicEventType.MISSED_DEADLINE

    def _launched(self, **kwargs):
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        dyn = DynamicJob(job, _constraints(t0, 12.0), self._regions(t0), [], **kwargs)
        chosen = SchedulerCandidate(
            region="us-west-2",
            mode=JobMode.SPOT,
            cost_rate_usd_per_hr=1.0,
            carbon_rate_kg_per_hr=0.1,
            expected_lifetime_hours=0.5,
        )
        decision = SchedulerDecision(chosen=chosen, safety_net=False, utility=1.0, reason="test")
        return dyn, dyn.launch_instance(dyn.make_initial_state(), decision), job

    def test_simulated_preemption_by_default(self) -> None:
        dyn, state, _ = self._launched()
        state = dyn.advance_work(state, tick_hours=1.0)  # 0.5h lifetime budget: runs 0.5h
        assert state.progress_hours == 0.5
        state = dyn.advance_work(state, tick_hours=1.0)
        assert state.mode == JobMode.IDLE
        assert dyn.events[-1].event_type == DynamicEventType.PREEMPT

    def test_no_simulated_preemption_when_observing_real_cluster(self) -> None:
        dyn, state, _ = self._launched(simulate_preemption=False)
        state = dyn.advance_work(state, tick_hours=1.0)
        state = dyn.advance_work(state, tick_hours=0.5)
        assert state.mode == JobMode.SPOT
        assert state.progress_hours == 1.5
        assert all(e.event_type != DynamicEventType.PREEMPT for e in dyn.events)

    def test_observed_preemption_returns_to_checkpoint_region(self) -> None:
        dyn, state, _ = self._launched(simulate_preemption=False)
        state = dyn.advance_work(state, tick_hours=0.5)
        progress = state.progress_hours
        state = dyn.advance_work(state, tick_hours=0.5, preempted=True)
        assert state.mode == JobMode.IDLE
        assert state.region == "us-west-2"
        assert state.progress_hours == progress
        assert dyn.events[-1].event_type == DynamicEventType.PREEMPT
        assert "observed" in dyn.events[-1].notes

    def test_observed_completion_clamps_progress(self) -> None:
        dyn, state, job = self._launched(simulate_preemption=False)
        state = dyn.advance_work(state, tick_hours=0.25, completed=True)
        assert state.progress_hours == job.duration_hours
        assert state.mode == JobMode.IDLE
        assert dyn.events[-1].event_type == DynamicEventType.COMPLETE
        # only the 0.25h that actually ran is charged
        assert state.cumulative_cost_usd < 1.0 + 0.25 * 100

    def test_fallback_without_run_to_completion_only_launches(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        # slack (2.5h - 2h) < 2 * cold_start (1h): the Safety Net fires immediately
        constraints = PlanConstraints(
            deadline_utc=t0 + timedelta(hours=2.5), now_utc=t0, cold_start_hours=0.5,
        )
        dyn = DynamicJob(job, constraints, self._regions(t0), [])
        state = dyn.make_initial_state()
        decision = dyn.decide(state)
        assert decision.safety_net and decision.chosen is not None
        state = dyn.fallback_to_ondemand(state, decision, run_to_completion=False)
        assert state.mode == JobMode.ON_DEMAND
        assert state.progress_hours == 0.0
        assert [e.event_type for e in dyn.events] == [DynamicEventType.SAFETY_NET]
        assert dyn.initial_launch == decision.chosen

    def test_wait_tick_advances_time_and_records_idle(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = DynamicJob(job, _constraints(t0, 8.0), self._regions(t0), [])
        state = dyn.make_initial_state()
        decision = SchedulerDecision(chosen=None, safety_net=False, utility=0.0, reason="wait")
        state = dyn.wait_tick(state, decision, tick_hours=0.5)
        assert state.now_utc == t0 + timedelta(hours=0.5)
        assert dyn.events[-1].event_type == DynamicEventType.IDLE
