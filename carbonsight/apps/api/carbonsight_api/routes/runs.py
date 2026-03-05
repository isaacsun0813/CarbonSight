"""POST /v1/runs, GET /v1/runs/{id}."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# In-memory store for MVP; replace with Postgres when DB is wired
_runs_store: dict[str, dict] = {}

router = APIRouter()


class RunCreate(BaseModel):
    job_spec: dict
    chosen_region: str | None = None
    estimated_co2_kg: float | None = None
    estimated_cost_usd: float | None = None


@router.post("/runs")
def post_runs(body: RunCreate) -> dict:
    """Record a run; return run_id."""
    run_id = str(uuid.uuid4())
    _runs_store[run_id] = {
        "run_id": run_id,
        "job_spec_json": body.job_spec,
        "chosen_region": body.chosen_region,
        "estimated_co2_kg": body.estimated_co2_kg,
        "estimated_cost_usd": body.estimated_cost_usd,
        "status": "recorded",
    }
    return {"run_id": run_id}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Return run status and estimates."""
    if run_id not in _runs_store:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs_store[run_id]
