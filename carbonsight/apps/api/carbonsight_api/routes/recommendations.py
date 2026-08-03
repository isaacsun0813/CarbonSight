"""POST/GET /v1/recommendations — joint U rank over the registry's real regions."""

from __future__ import annotations

import math
from pathlib import Path

from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.spot.scheduler_service import schedule_job
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class RecommendationRequest(BaseModel):
    """Mirrors the constraints on ``JobSpec``.

    Without them the JobSpec was built inside the handler, so a bad
    ``duration_hours`` surfaced as a plain-text 500 (or an OverflowError for
    1e400) instead of a 422 naming the field.
    """

    gpu_type: str = Field("A100", min_length=1)
    gpu_count: int = Field(1, ge=0, le=4096)
    duration_hours: float = Field(1.0, gt=0, le=8760, allow_inf_nan=False)
    cpu_count: int | None = Field(None, ge=0)
    mem_gib: float | None = Field(None, ge=0, allow_inf_nan=False)
    gpu_utilization: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Optional GPU utilization in [0, 1] to fix the power model (CLI --gpu-util).",
    )
    deadline_hours: float = Field(
        48.0, gt=0, le=8760, allow_inf_nan=False, description="Deadline horizon T (hours)."
    )
    checkpoint_size_gb: float = Field(0.0, ge=0, allow_inf_nan=False)
    cold_start_minutes: float = Field(5.0, ge=0, allow_inf_nan=False)
    carbon_price_usd_per_ton: float = Field(
        50.0, ge=0, allow_inf_nan=False, description="Social cost of carbon for joint U."
    )
    carbon_weight: float = Field(
        1.0, ge=0, allow_inf_nan=False, description="Weight on the carbon lever vs dollars."
    )
    progress_hours_done: float = Field(
        0.0, ge=0, allow_inf_nan=False, description="Compute-hours already done (p)."
    )
    elapsed_hours: float = Field(
        0.0, ge=0, allow_inf_nan=False, description="Wall-clock hours since job start (t)."
    )
    current_region: str = Field("", description="Region holding the checkpoint (r0).")


def _load_registry(request: Request) -> Registry | None:
    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return None
    reg = Registry()
    reg.load_json(rpath)
    return reg


def _run_recommendations(req: RecommendationRequest, request: Request) -> dict:
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
        carbon_weight=req.carbon_weight,
        progress_hours_done=req.progress_hours_done,
        current_region=req.current_region,
    )
    result = schedule_job(job, registry=_load_registry(request), elapsed_hours=req.elapsed_hours)

    decision: dict | None = None
    if result.decision is not None:
        decision = {
            "kind": result.decision.kind,
            "rule": result.decision.rule,
            "region": result.decision.region,
            "mode": result.decision.mode,
            "reason": result.decision.reason,
            "estimated_total_cost_usd": result.decision.estimated_total_cost_usd,
        }

    # V is infinite once the deadline has actually passed, which is not
    # representable in JSON — send null and let action/decision carry it.
    # deadline_passed is read off the progress state, not inferred from V, so a
    # healthy job can never be reported as out of time.
    value_v = result.value_v if math.isfinite(result.value_v) else None

    return {
        "action": result.action,
        "decision": decision,
        "value_v": value_v,
        "deadline_passed": result.progress.deadline_passed,
        "regions": [e.model_dump(mode="json") for e in result.estimates],
    }


@router.post("/recommendations")
def post_recommendations(req: RecommendationRequest, request: Request) -> dict:
    """Rank the registry's regions by joint U, with the policy's decision attached.

    ``action`` is ``rank`` | ``thrifty`` | ``safety_net`` | ``empty``. When the
    job is out of slack the ranking is still returned for context, but
    ``decision`` is what the caller should act on — a bare list of spot rows
    would tell a job past its deadline to keep gambling on spot.
    """
    return _run_recommendations(req, request)


@router.get("/recommendations")
def get_recommendations(request: Request) -> dict:
    """GET convenience: the same ranking for a default 1h A100 job."""
    return _run_recommendations(RecommendationRequest(), request)
