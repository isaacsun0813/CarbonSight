"""POST /v1/recommendations -> ranked list (same rules as CLI advise)."""

from pathlib import Path

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.watttime import WattTimeClient
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


@router.post("/recommendations")
def post_recommendations(req: RecommendationRequest, request: Request) -> list[dict]:
    """Rank regions greenest-first, filtered by cost premium vs cheapest (matches `carbonsight advise`)."""
    job = JobSpec(
        gpu_type=req.gpu_type,
        gpu_count=req.gpu_count,
        duration_hours=req.duration_hours,
        cpu_count=req.cpu_count,
        mem_gib=req.mem_gib,
        gpu_utilization=req.gpu_utilization,
    )
    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return []

    reg = Registry()
    reg.load_json(rpath)
    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        return []

    watt_time = WattTimeClient(config)
    ranking = AwsRegionRankingService(reg, watt_time)
    estimates = ranking.collect_estimates(job)
    if not estimates:
        return []

    visible, _, _ = AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(
        estimates, req.max_cost_premium
    )
    return [r.model_dump(mode="json") for r in visible]
