"""GET /v1/regions -> cloud regions + mapping confidence."""

from pathlib import Path

from fastapi import APIRouter, Request

from carbonsight_core.mapping.registry import _confidence, _confidence_label

router = APIRouter()


@router.get("/regions")
def get_regions(request: Request) -> list[dict]:
    """Return cloud regions with mapping confidence."""
    rpath: Path = getattr(request.app.state, "registry_path", None)
    if not rpath or not rpath.exists():
        return []
    from carbonsight_core.mapping.registry import Registry

    reg = Registry()
    reg.load_json(rpath)
    out = []
    for entry in reg.all_regions():
        conf = _confidence(entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency)
        out.append({
            "provider": entry.provider,
            "region_code": entry.region_code,
            "display_name": entry.display_name,
            "country": entry.country,
            "confidence": conf,
            "label": _confidence_label(conf),
            "wt_regions": [{"wt_region": r, "weight": w} for r, w in entry.wt_regions],
        })
    return out
