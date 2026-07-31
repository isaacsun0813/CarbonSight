"""POST/GET /v1/recommendations — joint U rank with synthetic 17-row fallback."""

from __future__ import annotations

from pathlib import Path

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.spot.scheduler_service import schedule_job
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class RecommendationRequest(BaseModel):
    gpu_type: str = "A100"
    gpu_count: int = 1
    duration_hours: float = 1.0
    cpu_count: int | None = None
    mem_gib: float | None = None
    gpu_utilization: float | None = Field(
        default=None,
        description="Optional GPU utilization in [0, 1] to fix the power model (same as CLI --gpu-util).",
    )
    max_cost_premium: float = Field(
        0.20,
        description="Max fractional cost above cheapest region (0.2 = 20%%); same as CLI --max-cost-premium.",
    )
    deadline_hours: float = Field(48.0, description="SkyNomad deadline horizon (hours).")
    checkpoint_size_gb: float = 0.0
    cold_start_minutes: float = 5.0
    carbon_price_usd_per_ton: float = Field(50.0, description="Social cost of carbon for joint U.")


def _load_registry(request: Request) -> Registry | None:
    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return None
    reg = Registry()
    reg.load_json(rpath)
    return reg


def _run_recommendations(req: RecommendationRequest, request: Request) -> list[dict]:
    job = JobSpec(
        gpu_type=req.gpu_type,
        gpu_count=req.gpu_count,
        duration_hours=req.duration_hours,
        cpu_count=req.cpu_count,
        mem_gib=req.mem_gib,
        gpu_utilization=req.gpu_utilization,
        deadline_hours=req.deadline_hours,
        checkpoint_size_gb=req.checkpoint_size_gb,
        cold_start_minutes=req.cold_start_minutes,
        carbon_price_usd_per_ton=req.carbon_price_usd_per_ton,
    )
    reg = _load_registry(request)
    result = schedule_job(job, registry=reg, config=Config.from_env())
    rows = [e.model_dump(mode="json") for e in result.estimates]
    # Guarantee synthetic fallback never returns []
    if not rows:
        # Absolute fallback: 17 empty-ish synthetic rows
        from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS

        for i, wt in enumerate(DEFAULT_CARBON_REGIONS):
            rows.append(
                {
                    "cloud": "aws",
                    "cloud_region": f"synthetic-{i}",
                    "watttime_regions": [[wt, 1.0]],
                    "expected_cost_usd": 1.0,
                    "expected_co2_kg_mean": float(i + 1),
                    "expected_co2_kg_p10": float(i + 1) * 0.8,
                    "expected_co2_kg_p90": float(i + 1) * 1.2,
                    "mapping_confidence": 0.5,
                    "notes": ["synthetic_fallback"],
                }
            )
    return rows


@router.post("/recommendations")
def post_recommendations(req: RecommendationRequest, request: Request) -> list[dict]:
    """Rank regions by joint U_s (cost + carbon + availability); always returns rows."""
    return _run_recommendations(req, request)


@router.get("/recommendations")
def get_recommendations(request: Request) -> list[dict]:
    """GET convenience: default job, 17 synthetic-capable rows."""
    return _run_recommendations(RecommendationRequest(), request)
