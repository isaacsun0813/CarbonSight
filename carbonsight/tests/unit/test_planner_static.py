"""Unit tests for static job planner."""

from datetime import datetime, timedelta, timezone

from carbonsight_core.models import JobSpec
from carbonsight_core.planning import (
    PlanConstraints,
    RegionPlanInput,
    plan_static_job,
)

UTC = timezone.utc


def _forecast(t0: datetime, values: list[float], step_min: int = 60) -> list[dict]:
    return [
        {"point_time": (t0 + timedelta(minutes=i * step_min)).isoformat(), "value": v}
        for i, v in enumerate(values)
    ]


def _constraints(
    t0: datetime,
    deadline_hours: float,
    *,
    carbon_budget_kg: float | None = None,
    max_cost_premium: float = 0.20,
    lambda_co2: float = 0.0,
) -> PlanConstraints:
    return PlanConstraints(
        deadline_utc=t0 + timedelta(hours=deadline_hours),
        now_utc=t0,
        carbon_budget_kg=carbon_budget_kg,
        max_cost_premium=max_cost_premium,
        lambda_co2_usd_per_kg=lambda_co2,
        cold_start_hours=0.0,
    )


class TestStaticPlannerDeadline:
    def test_rejects_late_finish(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=4.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [400.0] * 10),
            ),
        ]
        constraints = _constraints(t0, deadline_hours=3.0)
        result = plan_static_job(job, constraints, regions)
        assert result.best is None
        assert len(result.filtered_out["deadline"]) > 0

    def test_finds_feasible_within_deadline(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [400.0] * 8),
            ),
        ]
        constraints = _constraints(t0, deadline_hours=8.0)
        result = plan_static_job(job, constraints, regions)
        assert result.best is not None
        assert result.best.finish_utc <= constraints.deadline_utc


class TestStaticPlannerCarbonBudget:
    def test_carbon_budget_filters_dirty_now(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dirty = RegionPlanInput(
            cloud_region="us-east-1",
            watttime_regions=[("PJM_DC", 1.0)],
            forecast_points=_forecast(t0, [2000.0] * 8),
        )
        clean = RegionPlanInput(
            cloud_region="us-west-2",
            watttime_regions=[("PACW", 1.0)],
            forecast_points=_forecast(t0, [100.0] * 8),
        )
        constraints = _constraints(t0, deadline_hours=8.0, carbon_budget_kg=5.0)
        result = plan_static_job(job, constraints, [dirty, clean])
        assert result.best is not None
        assert result.best.cloud_region == "us-west-2"


class TestStaticPlannerCostPremium:
    def test_cost_premium_filters_expensive_region(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        cheap = RegionPlanInput(
            cloud_region="us-east-1",
            watttime_regions=[("PJM_DC", 1.0)],
            forecast_points=_forecast(t0, [400.0] * 4),
        )
        expensive = RegionPlanInput(
            cloud_region="eu-west-2",
            watttime_regions=[("UK", 1.0)],
            forecast_points=_forecast(t0, [400.0] * 4),
        )
        constraints = _constraints(t0, deadline_hours=4.0, max_cost_premium=0.0)
        result = plan_static_job(job, constraints, [cheap, expensive])
        assert result.best is not None
        assert result.best.cloud_region == "us-east-1"


class TestStaticPlannerWait:
    def test_waiting_wins_when_moer_drops(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        values = [900.0, 900.0, 900.0, 200.0, 200.0, 200.0, 200.0]
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, values),
            ),
        ]
        constraints = _constraints(t0, deadline_hours=8.0, lambda_co2=1.0)
        result = plan_static_job(job, constraints, regions)
        assert result.best is not None
        assert result.best.start_utc > t0


class TestStaticPlannerLambda:
    def test_lambda_shifts_toward_greener_region(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dirty_cheap = RegionPlanInput(
            cloud_region="us-east-1",
            watttime_regions=[("PJM_DC", 1.0)],
            forecast_points=_forecast(t0, [1500.0] * 4),
        )
        clean_costly = RegionPlanInput(
            cloud_region="us-west-2",
            watttime_regions=[("PACW", 1.0)],
            forecast_points=_forecast(t0, [100.0] * 4),
        )
        constraints_low = _constraints(t0, deadline_hours=4.0, max_cost_premium=1.0, lambda_co2=0.0)
        constraints_high = _constraints(t0, deadline_hours=4.0, max_cost_premium=1.0, lambda_co2=50.0)
        r_low = plan_static_job(job, constraints_low, [dirty_cheap, clean_costly])
        r_high = plan_static_job(job, constraints_high, [dirty_cheap, clean_costly])
        assert r_low.best is not None
        assert r_high.best is not None
        assert r_high.best.cloud_region == "us-west-2"


class TestStaticPlannerEmpty:
    def test_empty_feasible_returns_none(self) -> None:
        t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=10.0)
        regions = [
            RegionPlanInput(
                cloud_region="us-east-1",
                watttime_regions=[("PJM_DC", 1.0)],
                forecast_points=_forecast(t0, [400.0] * 2),
            ),
        ]
        constraints = _constraints(t0, deadline_hours=2.0)
        result = plan_static_job(job, constraints, regions)
        assert result.best is None
