"""Core types per CarbonSight Technical Design: JobSpec, EstimateResult."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobSpec(BaseModel):
    """Specification for an ML job used for carbon and cost estimation."""

    gpu_type: str = Field(..., description="e.g. T4, A100, H100")
    gpu_count: int = Field(1, ge=0)
    duration_hours: float = Field(..., gt=0)
    cpu_count: int | None = None
    mem_gib: float | None = None
    data_in_gb: float = Field(0.0, ge=0)
    data_out_gb: float = Field(0.0, ge=0)
    start_time_utc: datetime | None = None  # None => "now"
    constraints: dict[str, Any] = Field(default_factory=dict)
    gpu_utilization: float | None = Field(
        default=None,
        description=(
            "Observed GPU utilization in [0, 1] (e.g. from nvidia-smi). "
            "When set, GPU draw uses this value instead of sampling utilization."
        ),
    )
    # SkyNomad / carbon-aware scheduling extensions
    deadline_hours: float = Field(
        48.0,
        gt=0,
        description="Wall-clock deadline from now (hours) for multi-lever scheduling.",
    )
    checkpoint_size_gb: float = Field(
        0.0,
        ge=0,
        description="Checkpoint size used to estimate migration cost/time.",
    )
    cold_start_minutes: float = Field(
        5.0,
        ge=0,
        description="Cold-start overhead when launching or migrating (minutes).",
    )
    carbon_price_usd_per_ton: float = Field(
        50.0,
        ge=0,
        description="Social cost of carbon ($/metric ton) for joint U ranking.",
    )
    carbon_weight: float = Field(
        1.0,
        ge=0,
        description="Lambda weight on the carbon lever relative to dollars.",
    )
    progress_hours_done: float = Field(
        0.0,
        ge=0,
        description="Compute-hours already completed (p in the deadline-pressure model).",
    )
    current_region: str = Field(
        "",
        description="Region holding the current checkpoint (r0); migration cost is 0 for it.",
    )

    @field_validator("gpu_utilization")
    @classmethod
    def _clamp_gpu_utilization(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, v))


class EstimateResult(BaseModel):
    """Result of estimating carbon and cost for one region option."""

    cloud: str
    cloud_region: str
    watttime_regions: list[tuple[str, float]] = Field(
        ..., description="(region_code, weight); weights sum to 1.0"
    )
    expected_cost_usd: float = 0.0
    expected_co2_kg_mean: float = 0.0
    expected_co2_kg_p10: float = 0.0
    expected_co2_kg_p90: float = 0.0
    mapping_confidence: float = Field(..., ge=0, le=1)
    notes: list[str] = Field(default_factory=list)
    # Optional joint-ranking extras (schedule / recommendations synthetic path)
    utility_score: float | None = None
    moer_lb_per_mwh: float | None = None
    spot_price_usd_per_gpu_hr: float | None = None
    survival: float | None = None
    lbar_hours: float | None = None


class ActualRunResult(BaseModel):
    """Actual CO₂, cost, and duration measured after a job completes."""

    cloud: str
    cloud_region: str
    watttime_regions: list[tuple[str, float]] = Field(
        ..., description="(region_code, weight); weights sum to 1.0"
    )
    actual_start_utc: datetime
    actual_end_utc: datetime
    actual_duration_hours: float
    actual_co2_kg: float
    actual_cost_usd: float
    estimated_co2_kg: float = Field(..., description="Pre-run estimate (mean) for comparison")
    co2_savings_vs_estimate_pct: float = Field(
        ..., description="(estimated - actual) / estimated * 100; positive = used less than predicted"
    )
