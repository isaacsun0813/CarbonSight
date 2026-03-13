"""Core types per CarbonSight Technical Design: JobSpec, EstimateResult."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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
