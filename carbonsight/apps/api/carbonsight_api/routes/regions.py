"""GET /v1/regions -> cloud regions + mapping confidence (synthetic fallback)."""

from pathlib import Path

from carbonsight_core.mapping.registry import mapping_confidence, mapping_confidence_label
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/regions")
def get_regions(request: Request) -> list[dict]:
    """Return cloud regions with mapping confidence; never empty (17-row fallback)."""
    rpath: Path | None = getattr(request.app.state, "registry_path", None)
    out: list[dict] = []
    if rpath and rpath.exists():
        from carbonsight_core.mapping.registry import Registry

        reg = Registry()
        reg.load_json(rpath)
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
    if not out:
        # Synthetic 17 rows so CLI/API demos work without registry
        for i, wt in enumerate(DEFAULT_CARBON_REGIONS):
            out.append(
                {
                    "provider": "aws",
                    "region_code": f"synthetic-{i}",
                    "display_name": f"Synthetic {wt}",
                    "country": "",
                    "confidence": 0.5,
                    "label": "Low",
                    "wt_regions": [{"wt_region": wt, "weight": 1.0}],
                    "wt_region": wt,
                }
            )
    return out
