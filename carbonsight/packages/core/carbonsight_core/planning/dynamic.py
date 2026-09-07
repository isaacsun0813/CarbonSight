"""Dynamic scheduler: single-step decisions and full simulation loop."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from carbonsight_core.estimator.pricing import estimate_job_cost
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.survival import VirtualInstanceTracker
from carbonsight_core.planning.types import (
    DynamicEventType,
    DynamicPlanEvent,
    DynamicPlanResult,
    JobMode,
    JobRuntimeState,
    PlanConstraints,
    RegionPlanInput,
    SchedulerCandidate,
    SchedulerDecision,
)
from carbonsight_core.planning.utility import (
    carbon_rate_kg_per_hr,
    estimate_window_carbon_kg,
    progress_value,
    calc_utility,
    deadline_pressure,
)


def safety_net_triggered(
    state: JobRuntimeState,
    constraints: PlanConstraints,
) -> bool:
    """SkyNomad Safety Net: slack < remaining work + 2*cold_start."""
    remaining_work = max(0.0, state.total_work_hours - state.progress_hours)
    remaining_time = (constraints.deadline_utc - state.now_utc).total_seconds() / 3600.0
    slack = remaining_time - remaining_work
    return slack < 2.0 * constraints.cold_start_hours


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _moer_at_time(forecast_points: list[dict], at: datetime) -> float:
    if not forecast_points:
        return 400.0
    sorted_pts = sorted(forecast_points, key=lambda p: p["point_time"])
    chosen = sorted_pts[0]
    for p in sorted_pts:
        t = _parse_utc(p["point_time"])
        if t <= at:
            chosen = p
        else:
            break
    return float(chosen.get("value", 400.0))


def _migration_cost(
    constraints: PlanConstraints,
    from_region: str,
    to_region: str,
) -> float:
    if not from_region or from_region == to_region:
        return 0.0
    return constraints.migration_cost_usd.get((from_region, to_region), 0.0)


def _cheapest_ondemand_rate(regions: list[RegionPlanInput], job: JobSpec) -> float:
    rates = [
        estimate_job_cost(job.gpu_type, job.gpu_count, 1.0, r.cloud_region, use_spot=False).usd
        for r in regions
    ]
    return min(rates) if rates else 0.0


def _build_scheduler_candidates(
    job: JobSpec,
    constraints: PlanConstraints,
    regions: list[RegionPlanInput],
    trackers: dict[str, VirtualInstanceTracker],
    now_utc: datetime,
    checkpoint_region: str,
) -> list[SchedulerCandidate]:
    candidates: list[SchedulerCandidate] = []
    for region in regions:
        moer = _moer_at_time(region.forecast_points, now_utc)
        carb_rate = carbon_rate_kg_per_hr(job, moer)
        migration = _migration_cost(constraints, checkpoint_region, region.cloud_region)
        spot_cost = estimate_job_cost(
            job.gpu_type, job.gpu_count, 1.0, region.cloud_region, use_spot=True,
        ).usd
        od_cost = estimate_job_cost(
            job.gpu_type, job.gpu_count, 1.0, region.cloud_region, use_spot=False,
        ).usd
        lifetime = trackers[region.cloud_region].predict_lifetime_hours(now_utc)
        candidates.append(
            SchedulerCandidate(
                region=region.cloud_region,
                mode=JobMode.SPOT,
                cost_rate_usd_per_hr=spot_cost,
                carbon_rate_kg_per_hr=carb_rate,
                migration_usd=migration,
                expected_lifetime_hours=lifetime,
            )
        )
        candidates.append(
            SchedulerCandidate(
                region=region.cloud_region,
                mode=JobMode.ON_DEMAND,
                cost_rate_usd_per_hr=od_cost,
                carbon_rate_kg_per_hr=carb_rate,
                migration_usd=migration,
                expected_lifetime_hours=max(job.duration_hours, 1.0),
            )
        )
    return candidates


def _current_utility(
    state: JobRuntimeState,
    constraints: PlanConstraints,
    candidate: SchedulerCandidate,
    progress_val: float,
) -> float:
    if state.mode == JobMode.IDLE:
        return 0.0
    return calc_utility(
        progress_val,
        candidate.cost_rate_usd_per_hr,
        candidate.carbon_rate_kg_per_hr,
        state.mode,
        expected_lifetime_hours=candidate.expected_lifetime_hours,
        cold_start_hours=constraints.cold_start_hours,
        migration_usd=0.0,
        lambda_co2_usd_per_kg=constraints.lambda_co2_usd_per_kg,
    )


def _fallback_ondemand_cost(
    state: JobRuntimeState,
    constraints: PlanConstraints,
    candidate: SchedulerCandidate,
    remaining_work_hours: float,
) -> float:
    finish_hours = remaining_work_hours + constraints.cold_start_hours
    compute = candidate.cost_rate_usd_per_hr * finish_hours
    carbon = (
        constraints.lambda_co2_usd_per_kg
        * candidate.carbon_rate_kg_per_hr
        * finish_hours
    )
    return compute + candidate.migration_usd + carbon


def pick_next_state(
    state: JobRuntimeState,
    constraints: PlanConstraints,
    candidates: list[SchedulerCandidate],
    *,
    c_od_min: float,
) -> SchedulerDecision:
    """One scheduling step: Safety Net forces on-demand fallback, else pick best utility > current."""
    remaining_work = max(0.0, state.total_work_hours - state.progress_hours)
    remaining_time = (constraints.deadline_utc - state.now_utc).total_seconds() / 3600.0
    theta = deadline_pressure(remaining_work, remaining_time)
    elapsed = (state.now_utc - constraints.now_utc).total_seconds() / 3600.0
    theta_bar = (
        state.progress_hours / max(elapsed, 1e-9) if elapsed > 0 else state.total_work_hours
    )
    v_t = progress_value(theta, theta_bar, c_od_min)

    if safety_net_triggered(state, constraints):
        od_candidates = [c for c in candidates if c.mode == JobMode.ON_DEMAND]
        if not od_candidates:
            return SchedulerDecision(
                chosen=None,
                safety_net=True,
                utility=0.0,
                reason="safety_net triggered but no on-demand candidates",
            )
        best = min(
            od_candidates,
            key=lambda c: _fallback_ondemand_cost(state, constraints, c, remaining_work),
        )
        return SchedulerDecision(
            chosen=best,
            safety_net=True,
            utility=_fallback_ondemand_cost(state, constraints, best, remaining_work),
            reason="safety_net forced on-demand fallback",
        )

    current_candidate = SchedulerCandidate(
        region=state.region,
        mode=state.mode,
        cost_rate_usd_per_hr=0.0,
        carbon_rate_kg_per_hr=0.0,
    )
    if state.mode != JobMode.IDLE:
        current_candidate = next(
            (c for c in candidates if c.region == state.region and c.mode == state.mode),
            current_candidate,
        )
    u_current = _current_utility(state, constraints, current_candidate, v_t)

    for cand in sorted(
        candidates,
        key=lambda c: calc_utility(
            v_t,
            c.cost_rate_usd_per_hr,
            c.carbon_rate_kg_per_hr,
            c.mode,
            expected_lifetime_hours=c.expected_lifetime_hours,
            cold_start_hours=constraints.cold_start_hours,
            migration_usd=c.migration_usd,
            lambda_co2_usd_per_kg=constraints.lambda_co2_usd_per_kg,
        ),
        reverse=True,
    ):
        u = calc_utility(
            v_t,
            cand.cost_rate_usd_per_hr,
            cand.carbon_rate_kg_per_hr,
            cand.mode,
            expected_lifetime_hours=cand.expected_lifetime_hours,
            cold_start_hours=constraints.cold_start_hours,
            migration_usd=c.migration_usd,
            lambda_co2_usd_per_kg=constraints.lambda_co2_usd_per_kg,
        )
        if u > u_current:
            return SchedulerDecision(
                chosen=cand,
                safety_net=False,
                utility=u,
                reason="higher utility than current state",
            )

    return SchedulerDecision(
        chosen=None,
        safety_net=False,
        utility=u_current,
        reason="no candidate improves utility",
    )


def _run_ondemand_to_completion(
    state: JobRuntimeState,
    constraints: PlanConstraints,
    chosen: SchedulerCandidate,
    region_input: RegionPlanInput,
    job: JobSpec,
    events: list[DynamicPlanEvent],
    *,
    event_type: DynamicEventType = DynamicEventType.SAFETY_NET,
) -> JobRuntimeState:
    remaining = max(0.0, state.total_work_hours - state.progress_hours)
    now = state.now_utc
    cost_delta = chosen.migration_usd
    carbon_delta = 0.0
    progress = state.progress_hours

    if state.mode == JobMode.IDLE:
        cold_end = now + timedelta(hours=constraints.cold_start_hours)
        cost_delta += chosen.cost_rate_usd_per_hr * constraints.cold_start_hours
        _, moer = estimate_window_carbon_kg(job, region_input.forecast_points, now, cold_end)
        carbon_delta += carbon_rate_kg_per_hr(job, moer) * constraints.cold_start_hours
        now = cold_end
        events.append(
            DynamicPlanEvent(
                time_utc=now,
                event_type=event_type,
                region=chosen.region,
                mode=JobMode.ON_DEMAND,
                progress_hours=progress,
                cost_usd_delta=cost_delta,
                carbon_kg_delta=carbon_delta,
                notes="launch on-demand",
            )
        )

    run_hours = remaining
    run_end = now + timedelta(hours=run_hours)
    run_cost = chosen.cost_rate_usd_per_hr * run_hours
    run_carbon, _ = estimate_window_carbon_kg(job, region_input.forecast_points, now, run_end)
    progress += run_hours
    now = run_end
    events.append(
        DynamicPlanEvent(
            time_utc=now,
            event_type=DynamicEventType.COMPLETE,
            region=chosen.region,
            mode=JobMode.ON_DEMAND,
            progress_hours=progress,
            cost_usd_delta=run_cost,
            carbon_kg_delta=run_carbon,
            notes="work complete on-demand",
        )
    )
    return JobRuntimeState(
        region=chosen.region,
        mode=JobMode.IDLE,
        progress_hours=progress,
        now_utc=now,
        cumulative_cost_usd=state.cumulative_cost_usd + cost_delta + run_cost,
        cumulative_carbon_kg=state.cumulative_carbon_kg + carbon_delta + run_carbon,
        total_work_hours=state.total_work_hours,
    )


def plan_dynamic_job(
    job: JobSpec,
    constraints: PlanConstraints,
    regions: list[RegionPlanInput],
    *,
    tick_hours: float = 1.0,
    checkpoint_region: str | None = None,
) -> DynamicPlanResult:
    """
    Simulate greedy spot + wait + Safety Net until work completes or deadline missed.

    Returns advisory trajectory and first launch decision for CLI integration.
    """
    if not regions:
        empty = make_runtime_state(job, constraints)
        return DynamicPlanResult(
            deadline_met=False,
            events=[],
            initial_launch=None,
            final_state=empty,
            total_cost_usd=0.0,
            total_carbon_kg=0.0,
        )

    region_map = {r.cloud_region: r for r in regions}
    trackers = {r.cloud_region: VirtualInstanceTracker(region=r.cloud_region) for r in regions}
    for tracker in trackers.values():
        tracker.record_probe(constraints.now_utc, available=True)

    c_od_min = _cheapest_ondemand_rate(regions, job)
    checkpoint = checkpoint_region or ""
    state = make_runtime_state(job, constraints)
    events: list[DynamicPlanEvent] = []
    initial_launch: SchedulerCandidate | None = None
    spot_run_start: datetime | None = None
    spot_lifetime_budget: float = 0.0

    max_iters = int(
        (constraints.deadline_utc - constraints.now_utc).total_seconds() / 3600.0 / tick_hours
    ) + 1000
    iters = 0

    while (
        state.progress_hours < job.duration_hours
        and state.now_utc < constraints.deadline_utc
        and iters < max_iters
    ):
        iters += 1
        remaining = job.duration_hours - state.progress_hours

        if state.mode != JobMode.IDLE:
            region_input = region_map[state.region]
            candidate = next(
                (
                    c
                    for c in _build_scheduler_candidates(
                        job, constraints, [region_input], trackers, state.now_utc, checkpoint,
                    )
                    if c.region == state.region and c.mode == state.mode
                ),
                None,
            )
            if candidate is None:
                state = replace(state, mode=JobMode.IDLE)
                continue

            run_hours = min(tick_hours, remaining)
            if state.mode == JobMode.SPOT and spot_run_start is not None:
                elapsed_spot = (state.now_utc - spot_run_start).total_seconds() / 3600.0
                run_hours = min(run_hours, max(spot_lifetime_budget - elapsed_spot, 0.0))
                if run_hours <= 1e-9:
                    trackers[state.region].record_probe(state.now_utc, available=False)
                    events.append(
                        DynamicPlanEvent(
                            time_utc=state.now_utc,
                            event_type=DynamicEventType.PREEMPT,
                            region=state.region,
                            mode=JobMode.SPOT,
                            progress_hours=state.progress_hours,
                            cost_usd_delta=0.0,
                            carbon_kg_delta=0.0,
                            notes="spot lifetime exceeded",
                        )
                    )
                    state = replace(state, mode=JobMode.IDLE, region=checkpoint or state.region)
                    spot_run_start = None
                    continue

            run_end = state.now_utc + timedelta(hours=run_hours)
            run_cost = candidate.cost_rate_usd_per_hr * run_hours
            run_carbon, _ = estimate_window_carbon_kg(
                job, region_input.forecast_points, state.now_utc, run_end,
            )
            state = JobRuntimeState(
                region=state.region,
                mode=state.mode,
                progress_hours=state.progress_hours + run_hours,
                now_utc=run_end,
                cumulative_cost_usd=state.cumulative_cost_usd + run_cost,
                cumulative_carbon_kg=state.cumulative_carbon_kg + run_carbon,
                total_work_hours=state.total_work_hours,
            )
            if state.progress_hours >= job.duration_hours:
                events.append(
                    DynamicPlanEvent(
                        time_utc=state.now_utc,
                        event_type=DynamicEventType.COMPLETE,
                        region=state.region,
                        mode=state.mode,
                        progress_hours=state.progress_hours,
                        cost_usd_delta=run_cost,
                        carbon_kg_delta=run_carbon,
                        notes="work complete",
                    )
                )
                state = replace(state, mode=JobMode.IDLE)
                break
            continue

        candidates = _build_scheduler_candidates(
            job, constraints, regions, trackers, state.now_utc, checkpoint,
        )
        decision = pick_next_state(state, constraints, candidates, c_od_min=c_od_min)

        if decision.safety_net and decision.chosen:
            if initial_launch is None:
                initial_launch = decision.chosen
            region_input = region_map[decision.chosen.region]
            state = _run_ondemand_to_completion(
                state, constraints, decision.chosen, region_input, job, events,
            )
            break

        if decision.chosen is None:
            next_time = state.now_utc + timedelta(hours=tick_hours)
            events.append(
                DynamicPlanEvent(
                    time_utc=next_time,
                    event_type=DynamicEventType.IDLE,
                    region=state.region,
                    mode=JobMode.IDLE,
                    progress_hours=state.progress_hours,
                    cost_usd_delta=0.0,
                    carbon_kg_delta=0.0,
                    notes=decision.reason,
                )
            )
            state = replace(state, now_utc=next_time)
            continue

        chosen = decision.chosen
        if initial_launch is None:
            initial_launch = chosen
        region_input = region_map[chosen.region]
        launch_cost = chosen.migration_usd + chosen.cost_rate_usd_per_hr * constraints.cold_start_hours
        cold_end = state.now_utc + timedelta(hours=constraints.cold_start_hours)
        cold_carbon, _ = estimate_window_carbon_kg(
            job, region_input.forecast_points, state.now_utc, cold_end,
        )
        events.append(
            DynamicPlanEvent(
                time_utc=cold_end,
                event_type=DynamicEventType.LAUNCH,
                region=chosen.region,
                mode=chosen.mode,
                progress_hours=state.progress_hours,
                cost_usd_delta=launch_cost,
                carbon_kg_delta=cold_carbon,
                notes=decision.reason,
            )
        )
        checkpoint = chosen.region
        trackers[chosen.region].record_probe(cold_end, available=True)
        spot_run_start = cold_end if chosen.mode == JobMode.SPOT else None
        spot_lifetime_budget = chosen.expected_lifetime_hours if chosen.mode == JobMode.SPOT else 0.0
        state = JobRuntimeState(
            region=chosen.region,
            mode=chosen.mode,
            progress_hours=state.progress_hours,
            now_utc=cold_end,
            cumulative_cost_usd=state.cumulative_cost_usd + launch_cost,
            cumulative_carbon_kg=state.cumulative_carbon_kg + cold_carbon,
            total_work_hours=state.total_work_hours,
        )

    deadline_met = state.progress_hours >= job.duration_hours
    if not deadline_met:
        events.append(
            DynamicPlanEvent(
                time_utc=state.now_utc,
                event_type=DynamicEventType.MISSED_DEADLINE,
                region=state.region,
                mode=state.mode,
                progress_hours=state.progress_hours,
                cost_usd_delta=0.0,
                carbon_kg_delta=0.0,
                notes="deadline missed before work complete",
            )
        )

    if constraints.carbon_budget_kg is not None and state.cumulative_carbon_kg > constraints.carbon_budget_kg:
        deadline_met = False

    return DynamicPlanResult(
        deadline_met=deadline_met,
        events=events,
        initial_launch=initial_launch,
        final_state=state,
        total_cost_usd=state.cumulative_cost_usd,
        total_carbon_kg=state.cumulative_carbon_kg,
    )


def make_runtime_state(
    job: JobSpec,
    constraints: PlanConstraints,
    *,
    region: str = "",
    mode: JobMode = JobMode.IDLE,
    progress_hours: float = 0.0,
    now_utc: datetime | None = None,
) -> JobRuntimeState:
    """Helper to build JobRuntimeState from a JobSpec."""
    return JobRuntimeState(
        region=region,
        mode=mode,
        progress_hours=progress_hours,
        now_utc=now_utc or constraints.now_utc,
        total_work_hours=job.duration_hours,
    )
