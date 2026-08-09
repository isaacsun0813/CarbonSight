"""
Three-layer mapping registry per design doc:
1) Cloud region identity: provider + region_code + human name
2) Physical site candidates: lat/lon per region
3) Grid mapping: (wt_region, weight) per region; mixture when multiple WT regions.
Confidence: 0.35*S_source + 0.25*S_geo + 0.25*S_wt_stability + 0.15*S_recency
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class CloudSite(BaseModel):
    """One physical site (lat/lon) for a cloud region."""

    site_id: str
    provider: str
    region_code: str
    lat: float
    lon: float
    source_type: str = "official"  # official | country_only | inferred
    last_verified_at: str | None = None


class GridMappingEntry(BaseModel):
    """WattTime region and weight for one site."""

    wt_region: str
    weight: float = 1.0


class CloudRegionEntry(BaseModel):
    """Cloud region with sites and resolved grid mapping (mixture)."""

    provider: str
    region_code: str
    display_name: str
    country: str = ""
    source_url: str = ""
    sites: list[CloudSite] = Field(default_factory=list)
    # Resolved: list of (wt_region, weight); weights sum to 1.0
    wt_regions: list[tuple[str, float]] = Field(default_factory=list)
    # Confidence sub-scores [0,1] per design
    s_source: float = 0.8
    s_geo: float = 0.8
    s_wt_stability: float = 1.0
    s_recency: float = 0.8


class MappingResult(BaseModel):
    """Result of resolve(provider, region_code): wt_regions and confidence."""

    cloud: str
    cloud_region: str
    watttime_regions: list[tuple[str, float]]
    confidence: float
    label: str  # High | Medium | Low


def mapping_confidence(s_source: float, s_geo: float, s_wt_stability: float, s_recency: float) -> float:
    """Overall mapping confidence: 0.35*S_source + 0.25*S_geo + 0.25*S_wt_stability + 0.15*S_recency."""
    return 0.35 * s_source + 0.25 * s_geo + 0.25 * s_wt_stability + 0.15 * s_recency


def mapping_confidence_label(c: float) -> str:
    """High >= 0.85, Medium 0.60-0.84, Low < 0.60."""
    if c >= 0.85:
        return "High"
    if c >= 0.60:
        return "Medium"
    return "Low"


class Registry:
    """Load from JSON; resolve(cloud, region_code) -> MappingResult."""

    def __init__(self) -> None:
        self._regions: dict[tuple[str, str], CloudRegionEntry] = {}

    def load_json(self, path: Path | str) -> None:
        """Load registry from JSON. Format: list of CloudRegionEntry or dict with 'regions' key."""
        path = Path(path)
        data = json.loads(path.read_text())
        if isinstance(data, list):
            regions = data
        else:
            regions = data.get("regions", data.get("cloud_regions", []))
        for r in regions:
            if isinstance(r, dict):
                entry = CloudRegionEntry(
                    provider=r["provider"],
                    region_code=r["region_code"],
                    display_name=r.get("display_name", r["region_code"]),
                    country=r.get("country", ""),
                    source_url=r.get("source_url", ""),
                    sites=[CloudSite(**s) for s in r.get("sites", [])],
                    wt_regions=[(x["wt_region"], x.get("weight", 1.0)) for x in r.get("wt_regions", [])],
                    s_source=float(r.get("s_source", 0.8)),
                    s_geo=float(r.get("s_geo", 0.8)),
                    s_wt_stability=float(r.get("s_wt_stability", 1.0)),
                    s_recency=float(r.get("s_recency", 0.8)),
                )
            else:
                entry = CloudRegionEntry.model_validate(r)
            self._regions[(entry.provider, entry.region_code)] = entry

    def resolve(self, cloud: str, region_code: str) -> MappingResult | None:
        """Return MappingResult for (cloud, region_code) or None if not found."""
        key = (cloud.lower(), region_code.lower())
        entry = self._regions.get(key)
        if not entry:
            return None
        conf = mapping_confidence(entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency)
        return MappingResult(
            cloud=entry.provider,
            cloud_region=entry.region_code,
            watttime_regions=entry.wt_regions or [],
            confidence=conf,
            label=mapping_confidence_label(conf),
        )

    def all_regions(self) -> list[CloudRegionEntry]:
        """Return all cloud regions (for listing)."""
        return list(self._regions.values())

    def grid_regions(self) -> list[str]:
        """Every WattTime grid region any cloud region maps to, sorted and deduped.

        This is the set the refresh worker must keep warm: a grid region referenced
        here but never fetched means the cloud regions that map to it have no MOER,
        so they drop out of every ranking. Deriving the worker's list from the
        registry rather than a hardcoded tuple is what keeps the two from drifting.
        """
        return sorted(
            {wt_region for entry in self._regions.values() for wt_region, _ in entry.wt_regions}
        )


def bundled_registry_path() -> Path:
    """Path to the ``seed_registry.json`` shipped inside this package."""
    return Path(__file__).resolve().parent / "seed_registry.json"


def default_grid_regions() -> list[str]:
    """Grid regions the bundled registry references.

    The worker refreshes exactly these and the API reports exactly these, so the
    two cannot drift from what the registry actually needs. A grid region the
    registry maps to but nobody fetches leaves its cloud regions with no MOER.

    Raises if the registry is missing or empty rather than substituting the
    synthetic list: quietly refreshing the wrong set is how those four regions
    went unnoticed in the first place.
    """
    registry = Registry()
    registry.load_json(bundled_registry_path())
    grid_regions = registry.grid_regions()
    if not grid_regions:
        raise ValueError(
            f"{bundled_registry_path()} declares no wt_regions; there is nothing to refresh.",
        )
    return grid_regions


def site_to_json_dict(site: CloudSite) -> dict:
    """Serialize one site for seed_registry.json round-trip."""
    data: dict = {
        "site_id": site.site_id,
        "provider": site.provider,
        "region_code": site.region_code,
        "lat": site.lat,
        "lon": site.lon,
        "source_type": site.source_type,
    }
    if site.last_verified_at is not None:
        data["last_verified_at"] = site.last_verified_at
    return data


def entry_to_json_dict(entry: CloudRegionEntry) -> dict:
    """Serialize one cloud region entry for seed_registry.json round-trip."""
    return {
        "provider": entry.provider,
        "region_code": entry.region_code,
        "display_name": entry.display_name,
        "country": entry.country,
        "source_url": entry.source_url,
        "sites": [site_to_json_dict(site) for site in entry.sites],
        "wt_regions": [{"wt_region": wt, "weight": weight} for wt, weight in entry.wt_regions],
        "s_source": entry.s_source,
        "s_geo": entry.s_geo,
        "s_wt_stability": entry.s_wt_stability,
        "s_recency": entry.s_recency,
    }


def registry_to_json_dict(registry: Registry) -> dict:
    """Serialize full registry as {\"regions\": [...]}."""
    return {"regions": [entry_to_json_dict(entry) for entry in registry.all_regions()]}


def write_registry_json(registry: Registry, path: Path | str) -> None:
    """Write registry JSON to path (indent=2, trailing newline)."""
    out = Path(path)
    out.write_text(json.dumps(registry_to_json_dict(registry), indent=2) + "\n")
