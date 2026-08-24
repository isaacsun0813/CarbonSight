"""POST /v1/mappings/revalidate -> run mapping drift validation."""

from pathlib import Path

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.mapping.validate import drift_report_to_dict, validate_registry_mappings
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

    registry_path: Path | None = getattr(request.app.state, "registry_path", None)
    if not registry_path or not registry_path.exists():
        raise HTTPException(status_code=404, detail="Registry file not found.")

    registry = Registry()
    registry.load_json(registry_path)
    watt_time = WattTimeClient(config)
    drift_report = validate_registry_mappings(registry, watt_time)
    return drift_report_to_dict(drift_report)
