"""Core types per CarbonSight Technical Design: JobSpec, EstimateResult."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class JobSpec(BaseModel):
    """Specification for an ML job used for carbon and cost estimation.

    Extended for deadline-aware carbon scheduling:
    - deadline_hours (T): hours from now by which job must complete
    - deadline_utc: absolute deadline timestamp
    - checkpoint_size_gb (S_ckpt): checkpoint size in GB for migration cost
    - cold_start_minutes (d): cold start delay for migrations
    - carbon_price_usd_per_ton: $/ton CO2 for cost conversion
    - carbon_weight: weight for carbon vs cost in utility (0..1)
    - progress_hours_done (p): already completed hours
    - current_region (r0): current placement region
    """

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

    # --- Deadline-aware carbon scheduling extensions ---

    deadline_hours: float | None = Field(
        default=None,
        description="T: hours from now by which job must complete; None means no deadline",
    )
    deadline_utc: datetime | None = Field(
        default=None,
        description="Absolute deadline timestamp (UTC); alternative to deadline_hours",
    )
    checkpoint_size_gb: float = Field(
        default=0.0,
        ge=0,
        description="S_ckpt: checkpoint size in GB used to estimate migration cost/time",
    )
    cold_start_minutes: float = Field(
        default=6.0,
        ge=0,
        description="d: cold start / startup overhead in minutes per migration",
    )
    carbon_price_usd_per_ton: float = Field(
        default=50.0,
        ge=0,
        description="Carbon price in USD per ton CO2e for monetizing emissions",
    )
    carbon_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for carbon in utility function (0..1, or >=0); 1.0 = full carbon cost",
    )
    progress_hours_done: float = Field(
        default=0.0,
        ge=0,
        description="p: already completed execution hours (for preempted/migrating jobs)",
    )
    current_region: str | None = Field(
        default=None,
        description="r0: current region placement, if job is already running",
    )

    @field_validator("gpu_utilization")
    @classmethod
    def _clamp_gpu_utilization(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, v))

    @field_validator("deadline_hours")
    @classmethod
    def _validate_deadline_hours(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v <= 0:
            raise ValueError("deadline_hours must be > 0 (T from now)")
        return v

    @field_validator("deadline_utc")
    @classmethod
    def _validate_deadline_utc(cls, v: datetime | None) -> datetime | None:
        # Allow None; keep datetime as provided (naive or aware)
        return v

    @field_validator("checkpoint_size_gb")
    @classmethod
    def _validate_checkpoint_size(cls, v: float) -> float:
        if v < 0:
            raise ValueError("checkpoint_size_gb (S_ckpt) must be >= 0")
        return v

    @field_validator("cold_start_minutes")
    @classmethod
    def _validate_cold_start(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cold_start_minutes (d) must be >= 0")
        return v

    @field_validator("carbon_price_usd_per_ton")
    @classmethod
    def _validate_carbon_price(cls, v: float) -> float:
        if v < 0:
            raise ValueError("carbon_price_usd_per_ton must be >= 0")
        return v

    @field_validator("carbon_weight")
    @classmethod
    def _validate_carbon_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError("carbon_weight must be >= 0")
        return v

    @field_validator("progress_hours_done")
    @classmethod
    def _validate_progress(cls, v: float) -> float:
        if v < 0:
            raise ValueError("progress_hours_done (p) must be >= 0")
        return v

    @field_validator("current_region")
    @classmethod
    def _validate_current_region(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @model_validator(mode="after")
    def _validate_progress_vs_duration(self) -> "JobSpec":
        # Optional sanity: progress should not massively exceed duration; allow but don't hard fail?
        # We enforce progress <= duration * 2 as a loose check, otherwise raise if progress > duration and duration >0
        # For backward compat we only warn via capping? Instead we allow any >=0 but raise if progress > duration
        # and duration is set. However we allow equality for completed jobs, so only error if > duration.
        # To stay backward compatible, we allow progress up to duration, but if > duration raise.
        if self.progress_hours_done > self.duration_hours:
            # Allow slight floating tolerance, but error if clearly over
            if self.progress_hours_done > self.duration_hours + 1e-9:
                raise ValueError(
                    f"progress_hours_done ({self.progress_hours_done}) cannot exceed "
                    f"duration_hours ({self.duration_hours})"
                )
        return self


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

    # --- Optional extended fields for deadline-aware / spot scheduling ---

    expected_migration_cost_usd: float | None = Field(
        default=None,
        ge=0,
        description="Expected extra cost due to checkpoint migration",
    )
    expected_lifetime_hours: float | None = Field(
        default=None,
        ge=0,
        description="Expected spot instance lifetime in hours",
    )
    utility_U: float | None = Field(
        default=None,
        description="Utility U combining carbon * price + cost, weighted by carbon_weight",
    )
    effectiveness_eta: float | None = Field(
        default=None,
        description="Effectiveness eta metric (e.g., carbon savings / cost)",
    )
    forecast_moer_lb_per_mwh: float | None = Field(
        default=None,
        ge=0,
        description="Forecast MOER used for this estimate (lb/MWh), optional for debugging",
    )

    @field_validator("expected_migration_cost_usd", "expected_lifetime_hours", "forecast_moer_lb_per_mwh")
    @classmethod
    def _validate_optional_non_negative(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v < 0:
            raise ValueError("value must be >= 0")
        return v


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
