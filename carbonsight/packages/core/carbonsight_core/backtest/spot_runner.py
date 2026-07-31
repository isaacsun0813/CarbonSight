"""
Spot backtest: three policies over the same synthetic availability traces.

Regions get one of three availability patterns (paper Fig 2): generally
available, mostly available with frequent preemptions, or largely unavailable.
Probes are hourly, which is also the simulation step, so an inferred lifetime is
an integer number of hours — what the Nelson-Aalen sum in ``spot/lifetime.py``
assumes.

Policies, all with the same deadline and the same traces:

    UP-single   one region for the whole job: spot when it is up, on-demand when
                it is not. Never migrates, never stalls.
    UP-multi    cheapest *available* spot anywhere, else the cheapest on-demand.
                Migrates freely and pays egress for it.
    SkyNomad    rank region x mode by U = V*eta - C_total - E/Lbar, with the
                thrifty and safety-net rules in front. May idle when every
                candidate scores below zero, and pays for the hour it lost.

Progress only accrues on an hour actually spent running, so idling costs
deadline slack rather than being free. Reported savings are against both
baselines; a negative number means the policy lost, and is reported as such.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import SyntheticCarbonProvider
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.availability import AvailabilityTracker, SpotObservation
from carbonsight_core.spot.lifetime import LifetimeStats, predict_remaining_lifetime
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    rank_candidates,
)

DEFAULT_REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "eu-north-1",
    "ap-southeast-1",
    "ap-northeast-1",
    "us-east-2",
    "sa-east-1",
]

# Availability patterns cycled across regions, as (uptime, label).
PATTERNS = [(0.95, "generally"), (0.70, "frequent"), (0.20, "unavailable")]

COLD_START_HR = 0.1
MIN_LIFETIME_HR = 0.5


@dataclass
class SpotBacktestResult:
    n_workloads: int
    days: int
    deadline_ratio: float
    checkpoint_gb: float
    skynomad_cost_mean: float
    up_single_cost_mean: float
    up_multi_cost_mean: float
    cost_savings_vs_up_single_pct: float
    cost_savings_vs_up_multi_pct: float
    deadline_met_pct: float
    up_single_deadline_met_pct: float
    up_multi_deadline_met_pct: float
    migrations_mean: float
    egress_cost_pct_mean: float
    carbon_kg_mean: float
    up_multi_carbon_kg_mean: float
    details: list[dict] = field(default_factory=list)


def _build_traces(
    regions: list[str], days: int, seed: int, start: datetime
) -> dict[str, AvailabilityTracker]:
    """One hourly Bernoulli probe trace per region, pattern assigned round-robin."""
    trackers: dict[str, AvailabilityTracker] = {}
    for i, region in enumerate(regions):
        uptime, _label = PATTERNS[i % len(PATTERNS)]
        rng = random.Random(seed * 1000 + i)
        tracker = AvailabilityTracker()
        for hour in range(days * 24 + 1):
            tracker.record_observation(
                SpotObservation(
                    t=start + timedelta(hours=hour),
                    region=region,
                    outcome=1 if rng.random() < uptime else 0,
                )
            )
        trackers[region] = tracker
    return trackers


def _lifetimes(trackers: dict[str, AvailabilityTracker]) -> dict[str, float]:
    """Lbar(0) per region from the trace history, via Nelson-Aalen."""
    out: dict[str, float] = {}
    for region, tracker in trackers.items():
        stats = LifetimeStats.from_observations(tracker.observed_lifetimes(region))
        out[region] = max(MIN_LIFETIME_HR, predict_remaining_lifetime(stats, age=0.0))
    return out


def _run_up_single(
    region: str,
    trackers: dict[str, AvailabilityTracker],
    prices: dict[str, tuple[float, float]],
    carbon: dict[str, float],
    start: datetime,
    work_hours: float,
    deadline_hours: float,
) -> tuple[float, float, bool]:
    """Pinned to one region; on-demand covers the hours spot is down."""
    spot_px, od_px = prices[region]
    cost = 0.0
    kg = 0.0
    done = 0.0
    for hour in range(int(deadline_hours)):
        if done >= work_hours:
            break
        available = trackers[region].is_available(region, start + timedelta(hours=hour))
        cost += spot_px if available else od_px
        kg += carbon[region]
        done += 1.0
    return cost, kg, done >= work_hours


def _run_up_multi(
    regions: list[str],
    trackers: dict[str, AvailabilityTracker],
    prices: dict[str, tuple[float, float]],
    carbon: dict[str, float],
    start: datetime,
    work_hours: float,
    deadline_hours: float,
    migration_cost: float,
) -> tuple[float, float, int, bool]:
    """Cheapest available spot each hour, else cheapest on-demand. Pays egress to move."""
    cost = 0.0
    kg = 0.0
    done = 0.0
    migrations = 0
    current: str | None = None
    cheapest_od = min(regions, key=lambda r: prices[r][1])
    for hour in range(int(deadline_hours)):
        if done >= work_hours:
            break
        now = start + timedelta(hours=hour)
        available = [r for r in regions if trackers[r].is_available(r, now)]
        if available:
            chosen = min(available, key=lambda r: prices[r][0])
            cost += prices[chosen][0]
        else:
            chosen = cheapest_od
            cost += prices[chosen][1]
        if current is not None and chosen != current:
            migrations += 1
            cost += migration_cost
        current = chosen
        kg += carbon[chosen]
        done += 1.0
    return cost, kg, migrations, done >= work_hours


def _run_skynomad(
    regions: list[str],
    trackers: dict[str, AvailabilityTracker],
    prices: dict[str, tuple[float, float]],
    carbon: dict[str, float],
    lifetimes: dict[str, float],
    start: datetime,
    work_hours: float,
    deadline_hours: float,
    migration_est: MigrationCostEstimator,
    checkpoint_gb: float,
    carbon_price: float,
) -> tuple[float, float, float, int, bool]:
    """U-ranked region x mode with thrifty / safety-net short-circuits."""
    cost = 0.0
    egress = 0.0
    kg = 0.0
    done = 0.0
    migrations = 0
    current: str | None = None
    cheapest_od = min(regions, key=lambda r: prices[r][1])

    for hour in range(int(deadline_hours)):
        progress = ProgressState(p=done, P=work_hours, t=float(hour), T=deadline_hours)
        if progress.is_thrifty():
            break
        now = start + timedelta(hours=hour)

        if progress.is_safety_net(COLD_START_HR):
            cost += prices[cheapest_od][1]
            kg += carbon[cheapest_od]
            if current is not None and current != cheapest_od:
                migrations += 1
                egress += migration_est.estimate(checkpoint_gb)
            current = cheapest_od
            done += 1.0
            continue

        c_od_min = min(prices[r][1] for r in regions)
        value_v = progress.future_progress_value(c_od_min)

        candidates: list[CandidateState] = []
        for region in regions:
            move = 0.0 if current in (None, region) else migration_est.estimate(checkpoint_gb)
            if trackers[region].is_available(region, now):
                candidates.append(
                    CandidateState(
                        region, "spot", lifetimes[region], prices[region][0], carbon[region], move
                    )
                )
            candidates.append(
                CandidateState(
                    region, "on_demand", math.inf, prices[region][1], carbon[region], move
                )
            )
        candidates.append(CandidateState("idle", "idle", 0.0, 0.0, 0.0, 0.0))

        best, best_u = rank_candidates(candidates, value_v, COLD_START_HR, carbon_price)[0]
        if best.is_idle or best_u <= 0.0:
            continue  # wait: no progress this hour, and the slack is consumed

        cost += best.price_per_hr
        kg += best.carbon_kg_per_hr
        if current is not None and current != best.region:
            migrations += 1
            egress += best.migration_cost
        current = best.region
        done += 1.0

    return cost, egress, kg, migrations, done >= work_hours


def run_spot_backtest(
    n: int = 100,
    days: int = 14,
    deadline_ratio: float = 1.5,
    checkpoint_gb: float = 100.0,
    seed: int = 42,
    regions: list[str] | None = None,
    work_hours: float = 30.0,
    carbon_price_usd_per_ton: float = 50.0,
) -> SpotBacktestResult:
    """Run every policy over ``n`` workloads and report what actually happened."""
    regions = regions or DEFAULT_REGIONS
    deadline_hours = work_hours * deadline_ratio
    trace_start = datetime(2026, 1, 1, tzinfo=UTC)
    trackers = _build_traces(regions, days, seed, trace_start)
    lifetimes = _lifetimes(trackers)

    spot_provider = StaticSpotPriceProvider()
    carbon_provider = SyntheticCarbonProvider()
    migration_est = MigrationCostEstimator(egress_usd_per_gb=0.02)
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=work_hours)

    prices = {
        r: (
            spot_provider.get_spot_price_usd_per_gpu_hr(r, job.gpu_type) * job.gpu_count,
            spot_provider.get_on_demand_usd_per_gpu_hr(r, job.gpu_type) * job.gpu_count,
        )
        for r in regions
    }
    window_end = trace_start + timedelta(hours=deadline_hours)
    carbon_kg = {
        r: carbon_provider.get_kg_per_hr(job, [(r, 1.0)], trace_start, window_end) for r in regions
    }
    migration_flat = migration_est.estimate(checkpoint_gb)

    sky_costs: list[float] = []
    single_costs: list[float] = []
    multi_costs: list[float] = []
    sky_kg: list[float] = []
    multi_kg: list[float] = []
    migrations: list[int] = []
    egress_pcts: list[float] = []
    met = {"sky": 0, "single": 0, "multi": 0}
    details: list[dict] = []

    rng = random.Random(seed)
    max_offset = max(1, days * 24 - int(deadline_hours) - 1)

    for i in range(n):
        start = trace_start + timedelta(hours=rng.randint(0, max_offset))
        # UP-single rotates its pinned region so the baseline is not one lucky pattern.
        pinned = regions[i % len(regions)]

        s_cost, s_kg, s_met = _run_up_single(
            pinned, trackers, prices, carbon_kg, start, work_hours, deadline_hours
        )
        m_cost, m_kg, m_migr, m_met = _run_up_multi(
            regions, trackers, prices, carbon_kg, start, work_hours, deadline_hours, migration_flat
        )
        k_cost, k_egress, k_kg, k_migr, k_met = _run_skynomad(
            regions,
            trackers,
            prices,
            carbon_kg,
            lifetimes,
            start,
            work_hours,
            deadline_hours,
            migration_est,
            checkpoint_gb,
            carbon_price_usd_per_ton,
        )

        total_sky = k_cost + k_egress
        sky_costs.append(total_sky)
        single_costs.append(s_cost)
        multi_costs.append(m_cost)
        sky_kg.append(k_kg)
        multi_kg.append(m_kg)
        migrations.append(k_migr)
        egress_pcts.append(k_egress / total_sky * 100 if total_sky > 0 else 0.0)
        met["sky"] += int(k_met)
        met["single"] += int(s_met)
        met["multi"] += int(m_met)
        if i < 10:
            details.append(
                {
                    "skynomad": total_sky,
                    "up_single": s_cost,
                    "up_multi": m_cost,
                    "skynomad_migrations": k_migr,
                    "up_multi_migrations": m_migr,
                    "skynomad_met_deadline": k_met,
                }
            )

    def mean(values: list[float] | list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    sky_mean = mean(sky_costs)
    single_mean = mean(single_costs)
    multi_mean = mean(multi_costs)

    return SpotBacktestResult(
        n_workloads=n,
        days=days,
        deadline_ratio=deadline_ratio,
        checkpoint_gb=checkpoint_gb,
        skynomad_cost_mean=sky_mean,
        up_single_cost_mean=single_mean,
        up_multi_cost_mean=multi_mean,
        cost_savings_vs_up_single_pct=(
            (1 - sky_mean / single_mean) * 100 if single_mean else 0.0
        ),
        cost_savings_vs_up_multi_pct=((1 - sky_mean / multi_mean) * 100 if multi_mean else 0.0),
        deadline_met_pct=100.0 * met["sky"] / n if n else 0.0,
        up_single_deadline_met_pct=100.0 * met["single"] / n if n else 0.0,
        up_multi_deadline_met_pct=100.0 * met["multi"] / n if n else 0.0,
        migrations_mean=mean(migrations),
        egress_cost_pct_mean=mean(egress_pcts),
        carbon_kg_mean=mean(sky_kg),
        up_multi_carbon_kg_mean=mean(multi_kg),
        details=details,
    )
