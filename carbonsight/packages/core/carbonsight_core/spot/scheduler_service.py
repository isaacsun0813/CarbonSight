"""
Shared orchestration behind ``carbonsight schedule``, ``advise`` and the API.

One path builds the region x mode candidate set:

    registry -> (cloud_region, wt mixture)          real regions only, never padded
    availability -> lifetime -> Lbar                per region, Nelson-Aalen
    carbon provider -> kgCO2/hr                     over the window each mode occupies
    spot provider  -> $/hr                          spot and on-demand
    migration estimator -> E                        0 for the region already holding
                                                    the checkpoint
    ProgressState -> V(t)                           anchored on the cheapest OD total

and ``rank_candidates`` scores them with U = V*eta - C_total - E/Lbar.

``ScheduleResult.action`` is ``thrifty`` / ``safety_net`` / ``rank`` so callers
can render the short-circuits without re-deriving them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from carbonsight_core.mapping.registry import Registry, mapping_confidence
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.paths import packaged_registry_json_file, registry_json_path
from carbonsight_core.providers.carbon import CarbonIntensityProvider, get_carbon_provider
from carbonsight_core.providers.spot import (
    SpotPriceProvider,
    StaticSpotPriceProvider,
    get_spot_price_provider,
)
from carbonsight_core.spot.availability import AvailabilityTracker, synthetic_probe_trace
from carbonsight_core.spot.lifetime import (
    MIN_LIFETIME_HR,
    LifetimeStats,
    predict_remaining_lifetime,
)
from carbonsight_core.spot.policy import Action, PolicyState, SkyNomadPolicy
from carbonsight_core.spot.progress import ODCandidate, ProgressState, carbon_usd_per_hr
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    effectiveness,
    total_cost_per_hr,
)

DEFAULT_COLD_START_HR = 0.1
CHECKPOINT_RESTORE_HR_PER_GB = 0.002  # ~7 s/GB at 10 Gbps


@dataclass
class RegionInputs:
    """Per-region inputs to the ranking, kept so callers can explain a row."""

    cloud: str
    region: str
    wt_regions: list[tuple[str, float]]
    confidence: float
    mean_lifetime_hr: float
    moer_lb_per_mwh: float
    carbon_kg_per_hr: float
    spot_price_per_hr: float
    od_price_per_hr: float
    migration_cost: float


@dataclass
class ScheduleResult:
    """A ranking plus the ``SkyNomadPolicy`` decision that goes with it.

    ``decision`` is the authoritative Algo 1 output; ``action`` mirrors
    ``decision.rule`` (plus ``"empty"`` when the registry maps nothing) so
    callers can switch on a plain string.
    """

    ranked: list[tuple[CandidateState, float]]
    progress: ProgressState
    value_v: float
    cold_start_hr: float
    action: str
    inputs: dict[str, RegionInputs]
    decision: Action | None = None
    estimates: list[EstimateResult] = field(default_factory=list)

    @property
    def safety_net_region(self) -> str | None:
        return self.decision.region if self.action == "safety_net" and self.decision else None

    @property
    def safety_net_total_cost(self) -> float | None:
        if self.action != "safety_net" or self.decision is None:
            return None
        return self.decision.estimated_total_cost_usd


def default_registry() -> Registry | None:
    """Load the packaged seed registry; ``None`` when it is not on disk."""
    path = registry_json_path(Path(__file__).resolve().parents[4])
    if not path.exists():
        # Installed wheel: the source-relative walk above does not exist there.
        packaged = packaged_registry_json_file()
        if packaged is None:
            return None
        path = packaged
    reg = Registry()
    reg.load_json(path)
    return reg


def registry_pairs(
    registry: Registry | None,
) -> list[tuple[str, str, list[tuple[str, float]], float]]:
    """``(cloud, region_code, wt mixture, confidence)`` for every mapped AWS region."""
    reg = registry if registry is not None else default_registry()
    if reg is None:
        return []
    out = []
    for entry in reg.all_regions():
        if entry.provider.lower() != "aws" or not entry.wt_regions:
            continue
        conf = mapping_confidence(entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency)
        out.append((entry.provider, entry.region_code, list(entry.wt_regions), conf))
    return out


def synthetic_lifetime_stats(region: str) -> LifetimeStats:
    """Nelson-Aalen stats for a region from a deterministic synthetic probe trace.

    Stands in for a real availability ledger. Real ``AvailabilityTracker``
    observations go through the same ``observed_lifetimes`` -> ``LifetimeStats``
    path, so swapping the source in changes nothing downstream. Returns a fresh
    object each call: ``LifetimeStats`` is mutable and callers may ``add`` to it.
    """
    tracker = AvailabilityTracker(synthetic_probe_trace(region))
    return LifetimeStats.from_observations(tracker.observed_lifetimes(region))


@lru_cache(maxsize=256)
def _synthetic_mean_lifetime_hr(region: str) -> float:
    """Cached Lbar for the synthetic trace — a float, so nothing mutable is shared."""
    return predict_remaining_lifetime(synthetic_lifetime_stats(region), age=0.0)


def mean_lifetime_hr(region: str, tracker: AvailabilityTracker | None = None) -> float:
    """Lbar(0) for a region, from real probe data when available, else synthetic."""
    if tracker is not None and tracker.get_observations(region):
        stats = LifetimeStats.from_observations(tracker.observed_lifetimes(region))
        lbar = predict_remaining_lifetime(stats, age=0.0)
    else:
        lbar = _synthetic_mean_lifetime_hr(region)
    return max(MIN_LIFETIME_HR, lbar)


def cold_start_hours(job: JobSpec) -> float:
    """d = launch overhead + checkpoint restore time."""
    return job.cold_start_minutes / 60.0 + job.checkpoint_size_gb * CHECKPOINT_RESTORE_HR_PER_GB


def progress_from_job(job: JobSpec, *, elapsed_hours: float = 0.0) -> ProgressState:
    """p/P/t/T in hours, straight off the JobSpec."""
    return ProgressState(
        p=job.progress_hours_done,
        P=job.duration_hours,
        t=elapsed_hours,
        T=job.deadline_hours,
    )


def build_candidates(
    job: JobSpec,
    *,
    carbon: CarbonIntensityProvider | None = None,
    spot: SpotPriceProvider | None = None,
    registry: Registry | None = None,
    tracker: AvailabilityTracker | None = None,
    progress: ProgressState | None = None,
    now: datetime | None = None,
    egress_usd_per_gb: float = 0.02,
) -> tuple[list[CandidateState], list[ODCandidate], dict[str, RegionInputs], ProgressState]:
    """Region x {spot, on_demand} candidates plus an idle option, and their inputs."""
    carbon = carbon or get_carbon_provider()
    spot = spot or get_spot_price_provider()
    static = spot if isinstance(spot, StaticSpotPriceProvider) else StaticSpotPriceProvider()
    progress = progress or progress_from_job(job)
    now = now or datetime.now(UTC)
    migration = MigrationCostEstimator(
        egress_usd_per_gb=egress_usd_per_gb,
        carbon_price_usd_per_ton=job.carbon_price_usd_per_ton,
        carbon_weight=job.carbon_weight,
    )
    migration_cost_gb = migration.estimate(job.checkpoint_size_gb)

    candidates: list[CandidateState] = []
    od_options: list[ODCandidate] = []
    inputs: dict[str, RegionInputs] = {}

    for cloud, region, wt_regions, conf in registry_pairs(registry):
        lbar = mean_lifetime_hr(region, tracker)
        # Spot is priced over the window it is expected to survive; on-demand over
        # the work that is actually left.
        spot_end = now + timedelta(hours=lbar)
        od_end = now + timedelta(hours=max(progress.remaining_work, 1e-6))
        kg_spot = carbon.get_kg_per_hr(job, wt_regions, now, spot_end)
        kg_od = carbon.get_kg_per_hr(job, wt_regions, now, od_end)
        moer = carbon.get_moer_lb_per_mwh(wt_regions, now, spot_end)

        spot_px = spot.get_spot_price_usd_per_gpu_hr(region, job.gpu_type) * job.gpu_count
        od_px = static.get_on_demand_usd_per_gpu_hr(region, job.gpu_type) * job.gpu_count
        e_cost = 0.0 if region == job.current_region else migration_cost_gb

        candidates.append(
            CandidateState(region, "spot", lbar, spot_px, kg_spot, e_cost)
        )
        candidates.append(
            CandidateState(region, "on_demand", math.inf, od_px, kg_od, e_cost)
        )
        od_options.append(ODCandidate(region, od_px, e_cost, kg_od))
        inputs[region] = RegionInputs(
            cloud=cloud,
            region=region,
            wt_regions=wt_regions,
            confidence=conf,
            mean_lifetime_hr=lbar,
            moer_lb_per_mwh=moer,
            carbon_kg_per_hr=kg_spot,
            spot_price_per_hr=spot_px,
            od_price_per_hr=od_px,
            migration_cost=e_cost,
        )

    if candidates:
        candidates.append(CandidateState("idle", "idle", 0.0, 0.0, 0.0, 0.0))
    return candidates, od_options, inputs, progress


def schedule_job(
    job: JobSpec,
    *,
    registry: Registry | None = None,
    carbon: CarbonIntensityProvider | None = None,
    spot: SpotPriceProvider | None = None,
    tracker: AvailabilityTracker | None = None,
    elapsed_hours: float = 0.0,
    egress_usd_per_gb: float = 0.02,
    now: datetime | None = None,
    top_n: int | None = None,
    policy: SkyNomadPolicy | None = None,
    current_mode: str = "spot",
    current_utility: float = 0.0,
) -> ScheduleResult:
    """Providers -> candidates -> V(t) -> ``SkyNomadPolicy`` decision + ranking.

    The thrifty / safety-net / delta rules are not reimplemented here: this
    delegates to ``SkyNomadPolicy.rank_and_decide`` so there is exactly one
    Algo 1 in the tree for the CLI, the API, the backtest and any future
    supervisor loop to share.
    """
    progress = progress_from_job(job, elapsed_hours=elapsed_hours)
    candidates, od_options, inputs, progress = build_candidates(
        job,
        carbon=carbon,
        spot=spot,
        registry=registry,
        tracker=tracker,
        progress=progress,
        now=now,
        egress_usd_per_gb=egress_usd_per_gb,
    )
    cold_start_hr = cold_start_hours(job)

    c_od_min = min(
        (
            total_cost_per_hr(
                o.od_price_per_hr,
                o.carbon_kg_per_hr,
                job.carbon_price_usd_per_ton,
                job.carbon_weight,
            )
            for o in od_options
        ),
        default=0.0,
    )
    value_v = progress.future_progress_value(c_od_min)

    policy = policy or SkyNomadPolicy(
        migration_estimator=MigrationCostEstimator(
            egress_usd_per_gb=egress_usd_per_gb,
            carbon_price_usd_per_ton=job.carbon_price_usd_per_ton,
            carbon_weight=job.carbon_weight,
        ),
        carbon_price_usd_per_ton=job.carbon_price_usd_per_ton,
        carbon_weight=job.carbon_weight,
    )
    state = PolicyState(
        p=progress.p,
        P=progress.P,
        t=progress.t,
        T=progress.T,
        r0=job.current_region,
        ckpt_size_gb=job.checkpoint_size_gb,
        cold_start_hr=cold_start_hr,
        current_region=job.current_region,
        current_mode=current_mode,
        current_utility=current_utility,
    )
    ranked, decision = policy.rank_and_decide(state, candidates, progress, value_v)

    action = "empty" if not candidates else decision.rule
    if top_n is not None:
        ranked = ranked[:top_n]

    result = ScheduleResult(
        ranked=ranked,
        progress=progress,
        value_v=value_v,
        cold_start_hr=cold_start_hr,
        action=action,
        inputs=inputs,
        decision=decision,
    )
    result.estimates = candidates_to_estimates(result, job)
    return result


def candidates_to_estimates(result: ScheduleResult, job: JobSpec) -> list[EstimateResult]:
    """One ``EstimateResult`` per cloud region, greenest-first, over ``job.duration_hours``.

    Cost and CO2 are for the job as specified, not for a spot instance's expected
    lifetime — this is the row the ``advise`` and ``/v1/recommendations`` contracts
    promise.
    """
    # Past the deadline V is infinite, and so is every utility. Infinity is not
    # JSON, so it would reach an API client as null with "U_s=inf" left in the
    # notes; report it as unavailable instead and let ``action`` carry the news.
    best_utility: dict[str, float] = {}
    for cand, u in result.ranked:
        if cand.is_idle or not math.isfinite(u):
            continue
        prev = best_utility.get(cand.region)
        if prev is None or u > prev:
            best_utility[cand.region] = u
    deadline_passed = result.progress.deadline_passed

    out: list[EstimateResult] = []
    for region, inp in result.inputs.items():
        kg = inp.carbon_kg_per_hr * job.duration_hours
        cost = inp.spot_price_per_hr * job.duration_hours
        utility_score = best_utility.get(region)
        if utility_score is None:
            u_note = "U_s=n/a (deadline passed)" if deadline_passed else "U_s=n/a"
        else:
            u_note = f"U_s={utility_score:.4f}"
        out.append(
            EstimateResult(
                cloud=inp.cloud,
                cloud_region=region,
                watttime_regions=inp.wt_regions,
                expected_cost_usd=cost,
                expected_co2_kg_mean=kg,
                expected_co2_kg_p10=kg * 0.8,
                expected_co2_kg_p90=kg * 1.2,
                mapping_confidence=inp.confidence,
                notes=[
                    f"Lbar={inp.mean_lifetime_hr:.2f}h",
                    f"eta={effectiveness(inp.mean_lifetime_hr, result.cold_start_hr):.3f}",
                    u_note,
                ],
                utility_score=utility_score,
                moer_lb_per_mwh=inp.moer_lb_per_mwh,
                spot_price_usd_per_gpu_hr=(
                    inp.spot_price_per_hr / job.gpu_count if job.gpu_count else inp.spot_price_per_hr
                ),
                survival=effectiveness(inp.mean_lifetime_hr, result.cold_start_hr),
                lbar_hours=inp.mean_lifetime_hr,
            )
        )
    out.sort(key=lambda e: (e.expected_co2_kg_mean, e.expected_cost_usd))
    return out


def ranked_as_json(result: ScheduleResult, job: JobSpec) -> list[dict[str, object]]:
    """Flat rows for ``carbonsight schedule --json``."""
    rows: list[dict[str, object]] = []
    for cand, u in result.ranked:
        lbar = cand.mean_lifetime_hr
        finite = not math.isinf(lbar)
        rows.append(
            {
                "cloud_region": cand.region,
                "mode": cand.mode,
                "mean_lifetime_hr": lbar if finite else None,
                "effectiveness_eta": effectiveness(lbar, result.cold_start_hr),
                "price_per_hr_usd": cand.price_per_hr,
                "carbon_kg_per_hr": cand.carbon_kg_per_hr,
                "carbon_cost_per_hr_usd": carbon_usd_per_hr(
                    cand.carbon_kg_per_hr, job.carbon_price_usd_per_ton, job.carbon_weight
                ),
                "total_cost_per_hr_usd": total_cost_per_hr(
                    cand.price_per_hr,
                    cand.carbon_kg_per_hr,
                    job.carbon_price_usd_per_ton,
                    job.carbon_weight,
                ),
                "migration_cost_usd": cand.migration_cost,
                "amortized_migration_per_hr": (
                    cand.migration_cost / lbar if finite and lbar > 0 else 0.0
                ),
                "moer_lb_per_mwh": (
                    result.inputs[cand.region].moer_lb_per_mwh if cand.region in result.inputs
                    else None
                ),
                "utility_u": u,
                "value_v": result.value_v,
            }
        )
    return rows


def warm_forecast_cache(
    carbon: CarbonIntensityProvider | None = None,
    regions: list[str] | None = None,
) -> int:
    """Prefetch forecasts into provider caches. Returns the number of regions warmed."""
    from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS

    carbon = carbon or get_carbon_provider()
    warmed = 0
    for region in regions or DEFAULT_CARBON_REGIONS:
        try:
            carbon.get_forecast(region)
            warmed += 1
        except Exception:
            continue
    return warmed


__all__ = [
    "RegionInputs",
    "ScheduleResult",
    "build_candidates",
    "candidates_to_estimates",
    "cold_start_hours",
    "default_registry",
    "mean_lifetime_hr",
    "progress_from_job",
    "ranked_as_json",
    "registry_pairs",
    "schedule_job",
    "synthetic_lifetime_stats",
    "warm_forecast_cache",
]
