"""
Build ranked ODCandidate lists for schedule / recommendations.

Uses carbon + spot providers, availability, lifetime — single orchestration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry, mapping_confidence
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.providers.carbon import (
    DEFAULT_CARBON_REGIONS,
    CarbonIntensityProvider,
    get_carbon_provider,
)
from carbonsight_core.providers.spot import SpotPriceProvider, StaticSpotPriceProvider, get_spot_price_provider
from carbonsight_core.spot.availability import AvailabilityTracker
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import ODCandidate, rank_candidates

# Cloud region -> primary WattTime region (subset used when registry missing)
_FALLBACK_CLOUD_TO_WT: list[tuple[str, str]] = [
    ("us-east-1", "PJM_DC"),
    ("us-east-2", "PJM_EASTERN_OH"),
    ("us-west-1", "CAISO_NORTH"),
    ("us-west-2", "PACW"),
    ("ca-central-1", "HQ"),
    ("eu-west-1", "IE"),
    ("eu-west-2", "UK"),
    ("eu-west-3", "FR"),
    ("eu-central-1", "DE"),
    ("eu-north-1", "SE"),
    ("eu-south-1", "IT"),
    ("ap-northeast-1", "JP_TK"),
    ("ap-northeast-2", "KOR"),
    ("ap-southeast-1", "SGP"),
    ("ap-southeast-2", "NEM_NSW"),
    ("ap-south-1", "IND"),
    ("sa-east-1", "BRA"),
]


@dataclass
class ScheduleResult:
    ranked: list[ODCandidate]
    progress: ProgressState
    estimates: list[EstimateResult]


def _pairs_from_registry(registry: Registry | None) -> list[tuple[str, str, list[tuple[str, float]], float]]:
    """Return list of (cloud, region, wt_regions, confidence)."""
    out: list[tuple[str, str, list[tuple[str, float]], float]] = []
    if registry is not None:
        for entry in registry.all_regions():
            if entry.provider.lower() != "aws" or not entry.wt_regions:
                continue
            conf = mapping_confidence(
                entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency
            )
            out.append((entry.provider, entry.region_code, list(entry.wt_regions), conf))
    if not out:
        for cloud_region, wt in _FALLBACK_CLOUD_TO_WT:
            out.append(("aws", cloud_region, [(wt, 1.0)], 0.7))
    # Cap / pad toward 17 rows for stable synthetic API behavior
    if len(out) > 17:
        out = out[:17]
    while len(out) < 17:
        # pad from DEFAULT_CARBON_REGIONS
        idx = len(out)
        wt = DEFAULT_CARBON_REGIONS[idx % len(DEFAULT_CARBON_REGIONS)]
        out.append(("aws", f"synthetic-{idx}", [(wt, 1.0)], 0.5))
    return out


def build_candidates(
    job: JobSpec,
    *,
    carbon: CarbonIntensityProvider | None = None,
    spot: SpotPriceProvider | None = None,
    tracker: AvailabilityTracker | None = None,
    lifetime: LifetimeStats | None = None,
    registry: Registry | None = None,
    progress: ProgressState | None = None,
) -> tuple[list[ODCandidate], ProgressState]:
    """Construct ODCandidate list and ProgressState for a job."""
    carbon = carbon or get_carbon_provider()
    spot = spot or get_spot_price_provider()
    tracker = tracker or AvailabilityTracker()
    lifetime = lifetime or LifetimeStats.from_exponential(24.0)
    progress = progress or ProgressState.from_deadline_hours(
        total_progress=1.0,
        current_progress=0.0,
        deadline_hours=job.deadline_hours,
    )

    pairs = _pairs_from_registry(registry)
    # Seed availability if empty
    if not tracker.observations:
        tracker.seed_defaults([(r, job.gpu_type) for _, r, _, _ in pairs], default_availability=0.85)

    lbar = lifetime.expected_remaining(0.0)
    survival = lifetime.compute_survival(0.0)
    candidates: list[ODCandidate] = []

    static = spot if isinstance(spot, StaticSpotPriceProvider) else StaticSpotPriceProvider()

    for cloud, region, wt_regions, _conf in pairs:
        # Mixture MOER
        moer_acc = 0.0
        w_sum = 0.0
        for wt_region, w in wt_regions:
            try:
                m = carbon.get_moer(wt_region, lbar_hours=lbar)
            except Exception:
                m = 400.0
            moer_acc += w * m
            w_sum += w
        moer = moer_acc / w_sum if w_sum else 400.0

        spot_px = spot.get_spot_price_usd_per_gpu_hr(region, job.gpu_type)
        try:
            od_px = static.get_on_demand_usd_per_gpu_hr(region, job.gpu_type)
        except Exception:
            od_px = spot_px / 0.35 if spot_px else 1.0

        avail = tracker.get_availability(region, job.gpu_type)
        # Blend survival with availability
        surv = max(0.05, min(1.0, 0.5 * survival + 0.5 * avail))

        candidates.append(
            ODCandidate(
                region=region,
                instance_type=job.gpu_type,
                spot_price=spot_px,
                moer=moer,
                survival=surv,
                lbar=lbar,
                on_demand_price=od_px,
                gpu_count=job.gpu_count,
                efficiency=1.0,
                notes=[f"cloud={cloud}", f"wt={wt_regions}"],
            )
        )

    ranked = rank_candidates(
        candidates,
        progress,
        carbon_price_usd_per_ton=job.carbon_price_usd_per_ton,
    )
    return ranked, progress


def candidates_to_estimates(
    ranked: list[ODCandidate],
    job: JobSpec,
    registry: Registry | None = None,
) -> list[EstimateResult]:
    """Map ranked candidates to EstimateResult rows (API / advise shape)."""
    # Build quick lookup for wt regions
    wt_map: dict[str, list[tuple[str, float]]] = {}
    conf_map: dict[str, float] = {}
    if registry is not None:
        for entry in registry.all_regions():
            conf_map[entry.region_code] = mapping_confidence(
                entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency
            )
            wt_map[entry.region_code] = list(entry.wt_regions)

    out: list[EstimateResult] = []
    for c in ranked:
        wt = wt_map.get(c.region) or [(c.region, 1.0)]
        conf = conf_map.get(c.region, 0.7)
        cost = c.spot_price * c.gpu_count * max(c.lbar, job.duration_hours)
        out.append(
            EstimateResult(
                cloud="aws",
                cloud_region=c.region,
                watttime_regions=wt,
                expected_cost_usd=cost,
                expected_co2_kg_mean=c.carbon_kg,
                expected_co2_kg_p10=c.carbon_kg * 0.8,
                expected_co2_kg_p90=c.carbon_kg * 1.2,
                mapping_confidence=conf,
                notes=list(c.notes) + [f"U_s={c.utility:.4f}"],
                utility_score=c.utility,
                moer_lb_per_mwh=c.moer,
                spot_price_usd_per_gpu_hr=c.spot_price,
                survival=c.survival,
                lbar_hours=c.lbar,
            )
        )
    return out


def schedule_job(
    job: JobSpec,
    *,
    registry: Registry | None = None,
    config: Config | None = None,
    top_n: int | None = None,
) -> ScheduleResult:
    """Full schedule: providers -> candidates -> rank by U_s."""
    _ = config  # reserved for future wiring
    ranked, progress = build_candidates(job, registry=registry)
    if top_n is not None:
        ranked = ranked[:top_n]
    estimates = candidates_to_estimates(ranked, job, registry=registry)
    return ScheduleResult(ranked=ranked, progress=progress, estimates=estimates)


def warm_forecast_cache(
    carbon: CarbonIntensityProvider | None = None,
    regions: list[str] | None = None,
) -> int:
    """Prefetch forecasts into provider caches. Returns number of regions warmed."""
    carbon = carbon or get_carbon_provider()
    regs = regions or list(DEFAULT_CARBON_REGIONS)
    n = 0
    for r in regs:
        try:
            carbon.get_forecast(r)
            n += 1
        except Exception:
            continue
    return n


def ranked_as_json(result: ScheduleResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in result.ranked:
        rows.append(
            {
                "region": c.region,
                "instance_type": c.instance_type,
                "utility": c.utility,
                "spot_price_usd_per_gpu_hr": c.spot_price,
                "moer_lb_per_mwh": c.moer,
                "carbon_kg": c.carbon_kg,
                "c_total": c.c_total,
                "survival": c.survival,
                "lbar_hours": c.lbar,
                "value_v": c.value_v,
                "gpu_count": c.gpu_count,
            }
        )
    return rows
