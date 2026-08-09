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
class SiteDriftDetail:
    """Per-site live region-from-loc outcome."""

    site_id: str
    live_wt_region: str | None = None
    error: str | None = None


@dataclass
class RegionDriftResult:
    """One cloud region drift check outcome."""

    provider: str
    region_code: str
    stored_wt_regions: list[tuple[str, float]]
    live_wt_regions: list[tuple[str, float]]
    drift: bool
    mapping_confidence: float
    confidence_label: str
    site_details: list[SiteDriftDetail] = field(default_factory=list)


@dataclass
class DriftReport:
    """Aggregate outcome of validate_registry_mappings."""

    regions: list[RegionDriftResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regions_ok: int = 0
    regions_drift: int = 0
    has_drift: bool = False


def _wt_regions_to_json(wt_regions: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [{"wt_region": wt, "weight": weight} for wt, weight in wt_regions]


def drift_report_status(drift_report: DriftReport) -> str:
    """Return ok or drift for a completed validate run."""
    return "drift" if drift_report.has_drift else "ok"


def drift_report_to_dict(drift_report: DriftReport) -> dict[str, Any]:
    """Serialize DriftReport for CLI --json and API responses."""
    return {
        "status": drift_report_status(drift_report),
        "regions_ok": drift_report.regions_ok,
        "regions_drift": drift_report.regions_drift,
        "warnings": list(drift_report.warnings),
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
            for region in drift_report.regions
        ],
    }


def skipped_drift_report_to_dict(message: str) -> dict[str, Any]:
    """Serialize a skipped validate run (no WattTime credentials)."""
    return {
        "status": "skipped",
        "regions_ok": 0,
        "regions_drift": 0,
        "warnings": [message],
        "regions": [],
    }


def validate_registry_mappings(registry: Registry, watt_time: WattTimeClient) -> DriftReport:
    """
    Compare stored wt_regions to live region-from-loc per site; read-only (no registry mutation).

    Per-site failures append warnings and are skipped. If no site in a region succeeds,
    that region is not counted as drift (wt_regions unchanged semantics).
    """
    drift_report = DriftReport()

    for entry in registry.all_regions():
        if not entry.sites:
            drift_report.warnings.append(f"{entry.provider}/{entry.region_code}: no sites; skipped")
            continue

        site_wt_regions: list[str] = []
        site_details: list[SiteDriftDetail] = []

        for site in entry.sites:
            try:
                loc_response = watt_time.region_from_loc(
                    site.lat, site.lon, signal_type="co2_moer",
                )
                wt_region = parse_wt_region_code(loc_response)
                site_wt_regions.append(wt_region)
                site_details.append(
                    SiteDriftDetail(site_id=site.site_id, live_wt_region=wt_region),
                )
            except WattTimeError as err:
                msg = f"{entry.region_code} {site.site_id}: {err}"
                drift_report.warnings.append(msg)
                site_details.append(SiteDriftDetail(site_id=site.site_id, error=str(err)))
            except Exception as err:  # noqa: BLE001 - one bad site must not stop the region
                msg = f"{entry.region_code} {site.site_id}: {err}"
                drift_report.warnings.append(msg)
                site_details.append(SiteDriftDetail(site_id=site.site_id, error=str(err)))

        if not site_wt_regions:
            drift_report.warnings.append(
                f"{entry.provider}/{entry.region_code}: no successful region-from-loc; wt_regions unchanged",
            )
            continue

        if len(site_wt_regions) != len(entry.sites):
            drift_report.warnings.append(
                f"{entry.provider}/{entry.region_code}: partial site success "
                f"({len(site_wt_regions)}/{len(entry.sites)}); mixture from successful sites only",
            )

        live_wt_regions = build_wt_regions_from_site_codes(site_wt_regions)
        stored_wt_regions = list(entry.wt_regions)
        drift = live_wt_regions != stored_wt_regions
        confidence = mapping_confidence(
            entry.s_source,
            entry.s_geo,
            entry.s_wt_stability,
            entry.s_recency,
        )

        drift_report.regions.append(
            RegionDriftResult(
                provider=entry.provider,
                region_code=entry.region_code,
                stored_wt_regions=stored_wt_regions,
                live_wt_regions=live_wt_regions,
                drift=drift,
                mapping_confidence=confidence,
                confidence_label=mapping_confidence_label(confidence),
                site_details=site_details,
            )
        )
        if drift:
            drift_report.regions_drift += 1
            drift_report.has_drift = True
        else:
            drift_report.regions_ok += 1

    return drift_report
