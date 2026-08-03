"""Read-only drift checks: stored wt_regions vs live region-from-loc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carbonsight_core.mapping.refresh import (
    build_wt_regions_from_site_codes,
    parse_wt_region_code,
)
from carbonsight_core.mapping.registry import (
    Registry,
    mapping_confidence,
    mapping_confidence_label,
)
from carbonsight_core.watttime import WattTimeClient, WattTimeError


@dataclass
class SiteValidateDetail:
    """Per-site live region-from-loc outcome."""

    site_id: str
    live_wt_region: str | None = None
    error: str | None = None


@dataclass
class RegionValidateResult:
    """One cloud region drift check outcome."""

    provider: str
    region_code: str
    stored_wt_regions: list[tuple[str, float]]
    live_wt_regions: list[tuple[str, float]]
    drift: bool
    mapping_confidence: float
    confidence_label: str
    site_details: list[SiteValidateDetail] = field(default_factory=list)


@dataclass
class ValidateResult:
    """Aggregate outcome of validate_registry_mappings."""

    regions: list[RegionValidateResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regions_ok: int = 0
    regions_drift: int = 0
    has_drift: bool = False


def _wt_regions_to_json(wt_regions: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [{"wt_region": wt, "weight": weight} for wt, weight in wt_regions]


def validate_result_status(result: ValidateResult) -> str:
    """Return ok or drift for a completed validate run."""
    return "drift" if result.has_drift else "ok"


def validate_result_to_dict(result: ValidateResult) -> dict[str, Any]:
    """Serialize ValidateResult for CLI --json and API responses."""
    return {
        "status": validate_result_status(result),
        "regions_ok": result.regions_ok,
        "regions_drift": result.regions_drift,
        "warnings": list(result.warnings),
        "regions": [
            {
                "provider": region.provider,
                "region_code": region.region_code,
                "stored_wt_regions": _wt_regions_to_json(region.stored_wt_regions),
                "live_wt_regions": _wt_regions_to_json(region.live_wt_regions),
                "drift": region.drift,
                "mapping_confidence": region.mapping_confidence,
                "confidence_label": region.confidence_label,
                "sites": [
                    {
                        "site_id": site.site_id,
                        "live_wt_region": site.live_wt_region,
                        **({"error": site.error} if site.error else {}),
                    }
                    for site in region.site_details
                ],
            }
            for region in result.regions
        ],
    }


def skipped_validate_result_to_dict(message: str) -> dict[str, Any]:
    """Serialize a skipped validate run (no WattTime credentials)."""
    return {
        "status": "skipped",
        "regions_ok": 0,
        "regions_drift": 0,
        "warnings": [message],
        "regions": [],
    }


def validate_registry_mappings(registry: Registry, watt_time: WattTimeClient) -> ValidateResult:
    """
    Compare stored wt_regions to live region-from-loc per site; read-only (no registry mutation).

    Per-site failures append warnings and are skipped. If no site in a region succeeds,
    that region is not counted as drift (wt_regions unchanged semantics).
    """
    result = ValidateResult()

    for entry in registry.all_regions():
        if not entry.sites:
            result.warnings.append(f"{entry.provider}/{entry.region_code}: no sites; skipped")
            continue

        site_codes: list[str] = []
        site_details: list[SiteValidateDetail] = []

        for site in entry.sites:
            try:
                data = watt_time.region_from_loc(site.lat, site.lon, signal_type="co2_moer")
                code = parse_wt_region_code(data)
                site_codes.append(code)
                site_details.append(SiteValidateDetail(site_id=site.site_id, live_wt_region=code))
            except WattTimeError as err:
                msg = f"{entry.region_code} {site.site_id}: {err}"
                result.warnings.append(msg)
                site_details.append(SiteValidateDetail(site_id=site.site_id, error=str(err)))
            except Exception as err:
                msg = f"{entry.region_code} {site.site_id}: {err}"
                result.warnings.append(msg)
                site_details.append(SiteValidateDetail(site_id=site.site_id, error=str(err)))

        if not site_codes:
            result.warnings.append(
                f"{entry.provider}/{entry.region_code}: no successful region-from-loc; wt_regions unchanged",
            )
            continue

        if len(site_codes) != len(entry.sites):
            result.warnings.append(
                f"{entry.provider}/{entry.region_code}: partial site success "
                f"({len(site_codes)}/{len(entry.sites)}); mixture from successful sites only",
            )

        live_wt_regions = build_wt_regions_from_site_codes(site_codes)
        stored_wt_regions = list(entry.wt_regions)
        drift = live_wt_regions != stored_wt_regions
        conf = mapping_confidence(
            entry.s_source,
            entry.s_geo,
            entry.s_wt_stability,
            entry.s_recency,
        )

        result.regions.append(
            RegionValidateResult(
                provider=entry.provider,
                region_code=entry.region_code,
                stored_wt_regions=stored_wt_regions,
                live_wt_regions=live_wt_regions,
                drift=drift,
                mapping_confidence=conf,
                confidence_label=mapping_confidence_label(conf),
                site_details=site_details,
            )
        )
        if drift:
            result.regions_drift += 1
            result.has_drift = True
        else:
            result.regions_ok += 1

    return result
