"""GET /v1/regions -> the cloud regions the mapping registry actually knows about."""

from pathlib import Path

from carbonsight_core.mapping.registry import (
    Registry,
    mapping_confidence,
    mapping_confidence_label,
)
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/regions")
def get_regions(request: Request) -> list[dict]:
    """Registry contents only. Empty list when no registry is configured."""
    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return []

    reg = Registry()
    reg.load_json(rpath)
    out: list[dict] = []
    for entry in reg.all_regions():
        conf = mapping_confidence(
            entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency
        )
        out.append(
            {
                "provider": entry.provider,
                "region_code": entry.region_code,
                "display_name": entry.display_name,
                "country": entry.country,
                "confidence": conf,
                "label": mapping_confidence_label(conf),
                "wt_regions": [{"wt_region": r, "weight": w} for r, w in entry.wt_regions],
            }
        )
    return out
