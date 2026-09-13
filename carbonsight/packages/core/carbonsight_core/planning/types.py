"""Planning types for static pre-launch and dynamic scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class JobMode(str, Enum):
    """Instance execution mode."""

    SPOT = "spot"
    ON_DEMAND = "on_demand"
    IDLE = "idle"


class DynamicEventType(str, Enum):
    """Events recorded during dynamic plan simulation."""

    IDLE = "idle"
    LAUNCH = "launch"
    TERMINATE = "terminate"
    PREEMPT = "preempt"
    SAFETY_NET = "safety_net"
    COMPLETE = "complete"
    MISSED_DEADLINE = "missed_deadline"


@dataclass(frozen=True, slots=True)
class PlanConstraints:
    """Hard and soft constraints for job planning."""

    deadline_utc: datetime
    now_utc: datetime
    carbon_budget_kg: float | None = None
    max_cost_premium: float = 0.20
    lambda_co2_usd_per_kg: float = 0.0
    cold_start_hours: float = 0.5
    migration_cost_usd: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def from_finish_by(
        cls,
        finish_by_utc: datetime,
        now_utc: datetime,
        *,
        carbon_budget_kg: float | None = None,
        max_cost_premium: float = 0.20,
        lambda_co2_usd_per_kg: float = 0.0,
        cold_start_hours: float = 0.5,
        migration_cost_usd: dict[tuple[str, str], float] | None = None,
    ) -> PlanConstraints:
        """Build constraints with deadline = finish_by."""
        return cls(
            deadline_utc=finish_by_utc,
            now_utc=now_utc,
            carbon_budget_kg=carbon_budget_kg,
            max_cost_premium=max_cost_premium,
            lambda_co2_usd_per_kg=lambda_co2_usd_per_kg,
            cold_start_hours=cold_start_hours,
            migration_cost_usd=migration_cost_usd or {},
        )


@dataclass(frozen=True, slots=True)
class RegionPlanInput:
    """Per-region inputs for static planning (caller supplies forecasts)."""

    cloud_region: str
    watttime_regions: list[tuple[str, float]]
    forecast_points: list[dict]
    mapping_confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class PlanCandidate:
    """One scored (region, start, mode) option."""

    cloud_region: str
    mode: JobMode
    start_utc: datetime
    finish_utc: datetime
    cost_usd: float
    carbon_kg: float
    utility: float
    moer_lb_per_mwh: float
    migration_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Outcome of static planning."""

    best: PlanCandidate | None
    feasible: list[PlanCandidate]
    filtered_out: dict[str, list[PlanCandidate]]


@dataclass(frozen=True, slots=True)
class JobRuntimeState:
    """Runtime state for dynamic single-step scheduling."""

    region: str
    mode: JobMode
    progress_hours: float
    now_utc: datetime
    cumulative_carbon_kg: float = 0.0
    cumulative_cost_usd: float = 0.0
    total_work_hours: float = 0.0


@dataclass(frozen=True, slots=True)
class SchedulerCandidate:
    """One candidate state for dynamic scheduling."""

    region: str
    mode: JobMode
    cost_rate_usd_per_hr: float
    carbon_rate_kg_per_hr: float
    migration_usd: float = 0.0
    expected_lifetime_hours: float = 1.0


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """Result of one dynamic scheduling step."""

    chosen: SchedulerCandidate | None
    safety_net: bool
    utility: float
    reason: str


@dataclass(frozen=True, slots=True)
class DynamicPlanEvent:
    """One event in a simulated dynamic execution trajectory."""

    time_utc: datetime
    event_type: DynamicEventType
    region: str
    mode: JobMode
    progress_hours: float
    cost_usd_delta: float
    carbon_kg_delta: float
    notes: str = ""


@dataclass(frozen=True, slots=True)
class DynamicPlanResult:
    """Outcome of dynamic plan simulation."""

    deadline_met: bool
    events: list[DynamicPlanEvent]
    initial_launch: SchedulerCandidate | None
    final_state: JobRuntimeState
    total_cost_usd: float
    total_carbon_kg: float
