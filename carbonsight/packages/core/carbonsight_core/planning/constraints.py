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

    tier1: list[PlanCandidate] = []
    for c in candidates:
        if is_deadline_feasible(c.finish_utc, constraints.deadline_utc):
            tier1.append(c)
        else:
            filtered_out["deadline"].append(c)

    tier2: list[PlanCandidate] = []
    if constraints.carbon_budget_kg is not None:
        for c in tier1:
            if c.carbon_kg <= constraints.carbon_budget_kg:
                tier2.append(c)
            else:
                filtered_out["carbon_budget"].append(c)
    else:
        tier2 = tier1

    if not tier2:
        return [], filtered_out

    min_cost = min(c.cost_usd for c in tier2)
    cost_ceiling = min_cost * (1.0 + constraints.max_cost_premium)
    tier3 = [c for c in tier2 if c.cost_usd <= cost_ceiling]
    if not tier3:
        cheapest = min(tier2, key=lambda c: (c.cost_usd, c.carbon_kg, c.start_utc))
        tier3 = [cheapest]
    else:
        for c in tier2:
            if c not in tier3:
                filtered_out["cost_premium"].append(c)

    return tier3, filtered_out


def pick_best_candidate(candidates: list[PlanCandidate]) -> PlanCandidate | None:
    """Argmax utility; tie-break lower carbon, lower cost, earlier start."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (c.utility, -c.carbon_kg, -c.cost_usd, -c.start_utc.timestamp()),
    )
