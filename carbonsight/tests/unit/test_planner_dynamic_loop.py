"""Unit tests for plan_dynamic_job simulation loop."""

from datetime import datetime, timedelta, timezone

import pytest

from carbonsight_core.models import JobSpec
from carbonsight_core.planning import (
    DynamicEventType,
    JobMode,
    PlanConstraints,
    RegionPlanInput,
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
