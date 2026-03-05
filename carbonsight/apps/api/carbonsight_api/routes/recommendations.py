"""POST /v1/recommendations -> ranked list."""

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import estimate_option
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime import WattTimeClient, WattTimeError

router = APIRouter()


class RecommendationRequest(BaseModel):
    gpu_type: str = "A100"
    gpu_count: int = 1
    duration_hours: float = 1.0
    cpu_count: int | None = None
    mem_gib: float | None = None


@router.post("/recommendations")
def post_recommendations(req: RecommendationRequest, request: Request) -> list[dict]:
    """Return ranked regions with carbon/cost estimates and confidence."""
    job = JobSpec(
        gpu_type=req.gpu_type,
        gpu_count=req.gpu_count,
        duration_hours=req.duration_hours,
        cpu_count=req.cpu_count,
        mem_gib=req.mem_gib,
    )
    reg = Registry()
    rpath: Path = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return []
    reg.load_json(rpath)
    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        return []
    wt = WattTimeClient(config)
    results = []
    for entry in reg.all_regions():
        if entry.wt_regions and entry.provider.lower() == "aws":
            try:
                res = estimate_option(
                    job,
                    entry.provider,
                    entry.region_code,
                    entry.wt_regions,
                    0.35 * entry.s_source + 0.25 * entry.s_geo + 0.25 * entry.s_wt_stability + 0.15 * entry.s_recency,
                    wt,
                )
                results.append(res.model_dump(mode="json"))
            except (WattTimeError, Exception):
                pass
    results.sort(key=lambda r: r["expected_co2_kg_mean"])
    return results
