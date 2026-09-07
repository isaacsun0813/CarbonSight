"""Static pre-launch planner: brute-force over region x start x mode."""

from carbonsight_core.estimator.pricing import estimate_job_cost
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.constraints import (
    filter_by_priority,
    finish_time,
    latest_feasible_start,
    pick_best_candidate,
)
from carbonsight_core.planning.types import (
    JobMode,
    PlanCandidate,
    PlanConstraints,
    PlanResult,
    RegionPlanInput,
)
from carbonsight_core.planning.utility import (
    carbon_rate_kg_per_hr,
    deadline_pressure,
    estimate_window_carbon_kg,
    parse_forecast_utc,
    progress_value,
    calc_utility,
)


def _migration_cost(
    constraints: PlanConstraints,
    from_region: str | None,
    to_region: str,
) -> float:
    if from_region is None or from_region == to_region:
        return 0.0
    return constraints.migration_cost_usd.get((from_region, to_region), 0.0)


def _cheapest_ondemand_rate(regions: list[RegionPlanInput], job: JobSpec) -> float:
    rates: list[float] = []
    for region in regions:
        est = estimate_job_cost(
            job.gpu_type, job.gpu_count, 1.0, region.cloud_region, use_spot=False,
        )
        rates.append(est.usd)
    return min(rates) if rates else 0.0


def enumerate_candidates(
    job: JobSpec,
    constraints: PlanConstraints,
    regions: list[RegionPlanInput],
    *,
    checkpoint_region: str | None = None,
) -> list[PlanCandidate]:
    """Build all (region, start_slot, mode) candidates with scores."""
    c_od_min = _cheapest_ondemand_rate(regions, job)
    candidates: list[PlanCandidate] = []

    for region in regions:
        if not region.forecast_points:
            continue
        sorted_pts = region.forecast_points
        latest_start = latest_feasible_start(
            constraints.deadline_utc,
            job.duration_hours,
            constraints.cold_start_hours,
        )
        migration_usd = _migration_cost(constraints, checkpoint_region, region.cloud_region)

        for p in sorted_pts:
            t_s = parse_forecast_utc(p["point_time"])
            before_planning_window = t_s < constraints.now_utc
            if before_planning_window:
                continue
            after_latest_start = t_s > latest_start
            if after_latest_start:
                break

            t_finish = finish_time(t_s, job.duration_hours, constraints.cold_start_hours)
            carbon_kg, moer_lb = estimate_window_carbon_kg(
                job, sorted_pts, t_s, t_finish,
            )
            remaining_time = (constraints.deadline_utc - t_s).total_seconds() / 3600.0
            theta = deadline_pressure(job.duration_hours, remaining_time)
            elapsed = (t_s - constraints.now_utc).total_seconds() / 3600.0
            theta_bar = job.duration_hours / max(elapsed + job.duration_hours, 1e-9)
            v_t = progress_value(theta, theta_bar, c_od_min)
            carb_rate = carbon_rate_kg_per_hr(job, moer_lb)

            for mode in (JobMode.SPOT, JobMode.ON_DEMAND):
                use_spot = mode == JobMode.SPOT
                cost_est = estimate_job_cost(
                    job.gpu_type,
                    job.gpu_count,
                    job.duration_hours,
                    region.cloud_region,
                    use_spot=use_spot,
                )
                cost_usd = cost_est.usd + migration_usd
                cost_rate = cost_usd / job.duration_hours if job.duration_hours > 0 else 0.0
                utility = calc_utility(
                    v_t,
                    cost_rate,
                    carb_rate,
                    mode,
                    expected_lifetime_hours=job.duration_hours,
                    cold_start_hours=constraints.cold_start_hours,
                    migration_usd=migration_usd,
                    lambda_co2_usd_per_kg=constraints.lambda_co2_usd_per_kg,
                )
                candidates.append(
                    PlanCandidate(
                        cloud_region=region.cloud_region,
                        mode=mode,
                        start_utc=t_s,
                        finish_utc=t_finish,
                        cost_usd=cost_usd,
                        carbon_kg=carbon_kg,
                        utility=utility,
                        moer_lb_per_mwh=moer_lb,
                        migration_usd=migration_usd,
                    )
                )

    return candidates


def plan_static_job(
    job: JobSpec,
    constraints: PlanConstraints,
    regions: list[RegionPlanInput],
    *,
    checkpoint_region: str | None = None,
) -> PlanResult:
    """Enumerate, filter by priority, return best candidate."""
    all_candidates = enumerate_candidates(
        job, constraints, regions, checkpoint_region=checkpoint_region,
    )
    feasible, filtered_out = filter_by_priority(all_candidates, constraints)
    best = pick_best_candidate(feasible)
    return PlanResult(best=best, feasible=feasible, filtered_out=filtered_out)
