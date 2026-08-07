"""Refresh mapping registry WattTime regions from site coordinates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from carbonsight_core.mapping.registry import Registry
from carbonsight_core.watttime import WattTimeClient, WattTimeError


@dataclass
class RegionRefreshChange:
    """One cloud region row before/after refresh."""

    provider: str
    region_code: str
    old_wt_regions: list[tuple[str, float]]
    new_wt_regions: list[tuple[str, float]]
    changed: bool


@dataclass
class RefreshReport:
    """Aggregate outcome of refresh_registry_mappings."""

    changes: list[RegionRefreshChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regions_updated: int = 0
    regions_unchanged: int = 0


def parse_wt_region_code(loc_response: dict) -> str:
    """Extract WattTime region code from region-from-loc JSON."""
    wt_region = loc_response.get("region") or loc_response.get("abbrev")
    if not wt_region:
        raise ValueError("region-from-loc response missing region/abbrev")
    return str(wt_region)


def build_wt_regions_from_site_codes(codes: list[str]) -> list[tuple[str, float]]:
    """Build mixture weights from per-site WattTime region codes."""
    if not codes:
        return []
    counts = Counter(codes)
    site_count = len(codes)
    if len(counts) == 1:
        return [(next(iter(counts)), 1.0)]
    return [(region, count / site_count) for region, count in sorted(counts.items())]


def refresh_registry_mappings(registry: Registry, watt_time: WattTimeClient) -> RefreshReport:
    """
    Re-resolve wt_regions from WattTime region-from-loc for each site; mutate registry in place.

    Per-site failures append warnings and are skipped. If no site in a region succeeds,
    wt_regions for that region are left unchanged.
    """
    today = date.today().isoformat()
    refresh_report = RefreshReport()

    for entry in registry.all_regions():
        if not entry.sites:
            refresh_report.warnings.append(f"{entry.provider}/{entry.region_code}: no sites; skipped")
            continue

        site_codes: list[str] = []
        succeeded_site_ids: set[str] = set()

        for site in entry.sites:
            try:
                loc_response = watt_time.region_from_loc(
                    site.lat, site.lon, signal_type="co2_moer",
                )
                site_codes.append(parse_wt_region_code(loc_response))
                succeeded_site_ids.add(site.site_id)
            except WattTimeError as err:
                refresh_report.warnings.append(f"{entry.region_code} {site.site_id}: {err}")
            except Exception as err:  # noqa: BLE001 - one bad site must not stop the region
                refresh_report.warnings.append(f"{entry.region_code} {site.site_id}: {err}")

        if not site_codes:
            refresh_report.warnings.append(
                f"{entry.provider}/{entry.region_code}: no successful region-from-loc; wt_regions unchanged",
            )
            continue

        if len(site_codes) != len(entry.sites):
            refresh_report.warnings.append(
                f"{entry.provider}/{entry.region_code}: partial site success "
                f"({len(site_codes)}/{len(entry.sites)}); mixture from successful sites only",
            )

        new_wt_regions = build_wt_regions_from_site_codes(site_codes)
        old_wt_regions = list(entry.wt_regions)
        changed = old_wt_regions != new_wt_regions

        for site in entry.sites:
            if site.site_id in succeeded_site_ids:
                site.last_verified_at = today

        entry.wt_regions = new_wt_regions
        if len(site_codes) == len(entry.sites):
            entry.s_recency = 1.0

        refresh_report.changes.append(
            RegionRefreshChange(
                provider=entry.provider,
                region_code=entry.region_code,
                old_wt_regions=old_wt_regions,
                new_wt_regions=new_wt_regions,
                changed=changed,
            )
        )
        if changed:
            refresh_report.regions_updated += 1
        else:
            refresh_report.regions_unchanged += 1

    return refresh_report
