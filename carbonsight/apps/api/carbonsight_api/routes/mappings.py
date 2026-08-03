"""POST /v1/mappings/revalidate -> run mapping drift validation."""

from pathlib import Path

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.mapping.validate import validate_registry_mappings, validate_result_to_dict
from carbonsight_core.watttime import WattTimeClient
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/mappings/revalidate")
def post_mappings_revalidate(request: Request) -> dict:
    """Run mapping drift checks (region-from-loc vs stored registry)."""
    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        raise HTTPException(
            status_code=503,
            detail="WattTime credentials required (WATTTIME_USERNAME, WATTTIME_PASSWORD).",
        )

    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        raise HTTPException(status_code=404, detail="Registry file not found.")

    reg = Registry()
    reg.load_json(rpath)
    wt = WattTimeClient(config)
    result = validate_registry_mappings(reg, wt)
    return validate_result_to_dict(result)
