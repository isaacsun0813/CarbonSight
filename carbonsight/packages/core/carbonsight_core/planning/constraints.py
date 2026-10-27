"""Constraint filtering with lexicographic priority: deadline, carbon, cost."""

from datetime import datetime, timedelta

from carbonsight_core.planning.types import PlanCandidate, PlanConstraints


def finish_time(
    start_utc: datetime,
    duration_hours: float,
    cold_start_hours: float,
) -> datetime:
    """Wall-clock finish time including cold start."""
    total_hours = cold_start_hours + duration_hours
    return start_utc + timedelta(hours=total_hours)


def is_deadline_feasible(finish_utc: datetime, deadline_utc: datetime) -> bool:
    return finish_utc <= deadline_utc


def latest_feasible_start(
    deadline_utc: datetime,
    duration_hours: float,
    cold_start_hours: float,
) -> datetime:
    """Latest start time that can still finish by deadline."""
    total_hours = cold_start_hours + duration_hours
    return deadline_utc - timedelta(hours=total_hours)


def filter_by_priority(
    candidates: list[PlanCandidate],
    constraints: PlanConstraints,
) -> tuple[list[PlanCandidate], dict[str, list[PlanCandidate]]]:
    """
    Apply lexicographic filters: deadline (hard), carbon budget (hard), cost premium (soft).

    Returns (survivors, filtered_out_by_reason).
    """
    filtered_out: dict[str, list[PlanCandidate]] = {
        "deadline": [],
        "carbon_budget": [],
        "cost_premium": [],
    }

    survivors: list[PlanCandidate] = []
    carbon_cap = constraints.carbon_budget_kg
    for c in candidates:
        misses_deadline = not is_deadline_feasible(c.finish_utc, constraints.deadline_utc)
        if misses_deadline:
            filtered_out["deadline"].append(c)
            continue
        over_carbon_cap = carbon_cap is not None and c.carbon_kg > carbon_cap
        if over_carbon_cap:
            filtered_out["carbon_budget"].append(c)
            continue
        survivors.append(c)

    if not survivors:
        return [], filtered_out

    min_cost = min(c.cost_usd for c in survivors)
    cost_ceiling = min_cost * (1.0 + constraints.max_cost_premium)
    tier3: list[PlanCandidate] = []
    over_ceiling: list[PlanCandidate] = []
    for c in survivors:
        within_cost_ceiling = c.cost_usd <= cost_ceiling
        if within_cost_ceiling:
            tier3.append(c)
        else:
            over_ceiling.append(c)

    if not tier3:
        tier3 = [min(survivors, key=lambda c: (c.cost_usd, c.carbon_kg, c.start_utc))]
    else:
        filtered_out["cost_premium"].extend(over_ceiling)

    return tier3, filtered_out


def pick_best_candidate(candidates: list[PlanCandidate]) -> PlanCandidate | None:
    """Argmax utility; tie-break lower carbon, lower cost, earlier start."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (c.utility, -c.carbon_kg, -c.cost_usd, -c.start_utc.timestamp()),
    )
