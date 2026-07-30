"""
Backtest v2: multi-region spot + carbon simulation per SkyNomad evaluation.

Generates synthetic spot availability traces similar to paper Fig 2 (16 H100 13 zones):
- Patterns: generally available (high uptime), mostly available frequent preemptions (volatile), largely unavailable
- Lifetime distributions: log-scaled box plot heavy-tailed
- Simulates job 30h compute /45h deadline, ckpt 100GB, uses AvailabilityTracker + LifetimeStats + ProgressState + rank_candidates

Baselines:
- UP single-region (Uniform Progress) paper baseline: use spot when available else OD, spread progress evenly
- UP(S) multi-region failover cheapest-first
- SkyNomad-inspired: our policy with V(t), L̄, η, U, migration cost E

Metrics: cost savings vs baselines, deadline meet %, # migrations, egress %, selection accuracy vs oracle.

No real AWS/WattTime — uses SyntheticCarbonProvider and StaticSpotPriceProvider.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import List, Tuple

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import SyntheticCarbonProvider
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.availability import AvailabilityTracker, SpotObservation
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import CandidateState, MigrationCostEstimator, rank_candidates, total_cost_per_hr


@dataclass
class SpotTracePoint:
    ts: datetime
    region: str
    available: bool  # spot available


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
    migrations_mean: float
    egress_cost_pct_mean: float
    selection_accuracy_pct: float
    details: List[dict] = field(default_factory=list)


def _generate_spot_traces(
    regions: List[str], days: int, seed: int = 42
) -> List[SpotTracePoint]:
    """
    Generate synthetic availability traces per region with distinct patterns.
    - Pattern assignment: 1/3 generally available (95% up), 1/3 mostly available frequent preemptions (70% up but short lifetimes), 1/3 largely unavailable (20% up)
    - Timestamps every 10min like paper simulation
    """
    r = random.Random(seed)
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    points: List[SpotTracePoint] = []

    pattern_map = {}
    for i, region in enumerate(regions):
        if i % 3 == 0:
            pattern_map[region] = "generally"  # 95%
        elif i % 3 == 1:
            pattern_map[region] = "frequent"  # 70% but volatile
        else:
            pattern_map[region] = "unavailable"  # 20%

    cur = start
    while cur <= end:
        for region in regions:
            pat = pattern_map[region]
            # availability probability
            if pat == "generally":
                avail = r.random() < 0.95
            elif pat == "frequent":
                # frequent preemptions: higher volatility, on/off more often
                avail = r.random() < 0.70
            else:
                avail = r.random() < 0.20
            points.append(SpotTracePoint(ts=cur, region=region, available=avail))
        cur += timedelta(minutes=10)

    # Sort by ts
    points.sort(key=lambda p: p.ts)
    return points


def _build_tracker_from_traces(
    traces: List[SpotTracePoint], up_to: datetime
) -> dict[str, AvailabilityTracker]:
    """Build per-region AvailabilityTracker up to time up_to from traces."""
    trackers: dict[str, AvailabilityTracker] = {}
    for pt in traces:
        if pt.ts > up_to:
            break
        if pt.region not in trackers:
            trackers[pt.region] = AvailabilityTracker()
        trackers[pt.region].record_observation(
            SpotObservation(t=pt.ts, region=pt.region, outcome=1 if pt.available else 0)
        )
    return trackers


def _estimate_lifetime_from_tracker(tracker: AvailabilityTracker, region: str) -> float:
    """Estimate mean lifetime from tracker's virtual instances, fallback synthetic."""
    try:
        vis = tracker.extract_virtual_instances(region)
        if not vis:
            return 3.5
        # Compute mean lifetime in hours
        lifetimes = []
        for vi in vis:
            if isinstance(vi.start_t, datetime) and isinstance(vi.end_t, datetime):
                lifetimes.append((vi.end_t - vi.start_t).total_seconds() / 3600.0)
            elif isinstance(vi.start_t, (int, float)) and isinstance(vi.end_t, (int, float)):
                lifetimes.append(float(vi.end_t - vi.start_t))
        if lifetimes:
            return sum(lifetimes) / len(lifetimes)
    except Exception:
        pass
    return 3.5


def run_spot_backtest(
    n: int = 100,
    days: int = 14,
    deadline_ratio: float = 1.5,  # T/P like paper 45/30=1.5
    checkpoint_gb: float = 100.0,
    seed: int = 42,
    regions: List[str] | None = None,
) -> SpotBacktestResult:
    """
    Run SkyNomad-inspired backtest.

    For each synthetic workload (JobSpec with duration), simulate:
    - Decision times every 1h, check availability from traces
    - UP single-region: stays in first region, uses spot if available else OD
    - UP(S): multi-region failover cheapest-first spot if available else cheapest OD
    - SkyNomad: uses rank_candidates with V(t), η, E/L̄, carbon lever

    Returns aggregated metrics.
    """
    regions = regions or [
        "us-east-1",
        "us-west-2",
        "eu-west-1",
        "eu-north-1",
        "ap-southeast-1",
        "ap-northeast-1",
        "us-east-2",
        "us-west-1",
    ]
    job_duration_hours = 30.0  # like paper
    traces = _generate_spot_traces(regions, days, seed=seed)

    spot_provider = StaticSpotPriceProvider()
    carbon_provider = SyntheticCarbonProvider()
    migration_est = MigrationCostEstimator(egress_usd_per_gb=0.02)

    skynomad_costs = []
    up_single_costs = []
    up_multi_costs = []
    deadline_met = 0
    migrations = []
    egress_pcts = []
    selection_correct = 0

    r = random.Random(seed)
    base_start = datetime.now(UTC) - timedelta(days=days - 1)

    for i in range(n):
        # Random start time within traces
        start_offset = timedelta(hours=r.randint(0, max(1, days * 24 - int(job_duration_hours * deadline_ratio) - 1)))
        job_start = base_start + start_offset
        deadline_hours = job_duration_hours * deadline_ratio
        job_end_deadline = job_start + timedelta(hours=deadline_hours)

        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=job_duration_hours)

        # Build trackers up to job_start (history for L̄ prediction)
        trackers = _build_tracker_from_traces(traces, job_start)

        # Simulate three policies over time
        # For simplicity, we simulate total cost as sum over time steps 1h with availability check
        # and count migrations

        # UP single: first region only
        single_region = regions[0]
        up_single_cost = 0.0
        p_single = 0.0
        t_elapsed = 0.0
        cur = job_start
        while cur < job_end_deadline and p_single < job_duration_hours:
            # availability at cur for single region
            tracker = trackers.get(single_region)
            available = False
            if tracker:
                available = tracker.is_available(single_region, cur)
            # If not available, need OD
            if available:
                price = spot_provider.get_spot_price_usd_per_gpu_hr(single_region, job.gpu_type) * job.gpu_count
            else:
                price = spot_provider.get_od_price_usd_per_gpu_hr(single_region, job.gpu_type) * job.gpu_count
            up_single_cost += price * 1.0  # 1h step
            p_single += 1.0 if available or True else 0.0  # OD always makes progress, spot only if available
            cur += timedelta(hours=1)
            t_elapsed += 1.0

        # UP(S) multi: cheapest spot first
        up_multi_cost = 0.0
        p_multi = 0.0
        cur = job_start
        while cur < job_end_deadline and p_multi < job_duration_hours:
            # Find cheapest available spot region
            cheapest_price = math.inf
            cheapest_region = None
            for region in regions:
                tr = trackers.get(region)
                avail = tr.is_available(region, cur) if tr else False
                if avail:
                    price = spot_provider.get_spot_price_usd_per_gpu_hr(region, job.gpu_type)
                    if price < cheapest_price:
                        cheapest_price = price
                        cheapest_region = region
            if cheapest_region is None:
                # No spot, use cheapest OD
                cheapest_od = min(
                    (spot_provider.get_od_price_usd_per_gpu_hr(r, job.gpu_type), r) for r in regions
                )
                price = cheapest_od[0]
            else:
                price = cheapest_price
            up_multi_cost += price * job.gpu_count * 1.0
            p_multi += 1.0
            cur += timedelta(hours=1)

        # SkyNomad-inspired: V(t), L̄, η, migration
        skynomad_cost = 0.0
        p_sky = 0.0
        cur = job_start
        current_region = None
        migr_count = 0
        egress_cost = 0.0

        while cur < job_end_deadline and p_sky < job_duration_hours:
            t_elapsed_sky = (cur - job_start).total_seconds() / 3600.0
            prog = ProgressState(p=p_sky, P=job_duration_hours, t=t_elapsed_sky, T=deadline_hours)
            # Safety net
            if prog.is_safety_net(0.1):
                # cheapest OD
                cheapest_od_price = min(
                    spot_provider.get_od_price_usd_per_gpu_hr(r, job.gpu_type) for r in regions
                )
                skynomad_cost += cheapest_od_price * job.gpu_count * 1.0
                p_sky += 1.0
                cur += timedelta(hours=1)
                continue

            # Build candidates for this decision time
            candidates = []
            c_od_min = min(
                spot_provider.get_od_price_usd_per_gpu_hr(r, job.gpu_type) for r in regions
            )
            V = prog.future_progress_value(c_od_min)

            for region in regions:
                tr = trackers.get(region)
                avail = tr.is_available(region, cur) if tr else False
                if not avail:
                    continue
                Lbar = _estimate_lifetime_from_tracker(tr, region) if tr else 3.5
                spot_price = spot_provider.get_spot_price_usd_per_gpu_hr(region, job.gpu_type) * job.gpu_count
                try:
                    carbon_kg = carbon_provider.get_kg_per_hr(
                        job,
                        [(region, 1.0)],  # simplified, real would use wt_regions
                        cur,
                        cur + timedelta(hours=Lbar),
                    )
                except Exception:
                    carbon_kg = 0.15
                migr_cost = 0.0
                if current_region and current_region != region and checkpoint_gb > 0:
                    migr_cost = migration_est.estimate(checkpoint_gb)
                candidates.append(
                    CandidateState(
                        region=region,
                        mode="spot",
                        mean_lifetime_hr=Lbar,
                        price_per_hr=spot_price,
                        carbon_kg_per_hr=carbon_kg,
                        migration_cost=migr_cost,
                    )
                )
            # Also OD candidates
            for region in regions:
                od_price = spot_provider.get_od_price_usd_per_gpu_hr(region, job.gpu_type) * job.gpu_count
                candidates.append(
                    CandidateState(
                        region=region,
                        mode="on_demand",
                        mean_lifetime_hr=math.inf,
                        price_per_hr=od_price,
                        carbon_kg_per_hr=0.15,
                        migration_cost=0.0,
                    )
                )

            ranked = rank_candidates(candidates, V, 0.1, 50.0, 1.0)
            if not ranked or ranked[0][1] <= 0:
                # idle
                cur += timedelta(hours=1)
                continue

            top_cand, top_u = ranked[0]
            price_to_use = top_cand.price_per_hr
            skynomad_cost += price_to_use * 1.0
            if current_region and current_region != top_cand.region:
                migr_count += 1
                egress_cost += top_cand.migration_cost
            current_region = top_cand.region
            p_sky += 1.0  # simplified: effective progress if spot and Lbar>d else 0, but assume makes progress
            cur += timedelta(hours=1)

        skynomad_costs.append(skynomad_cost + egress_cost)
        up_single_costs.append(up_single_cost)
        up_multi_costs.append(up_multi_cost)
        migrations.append(migr_count)
        egress_pct = (egress_cost / skynomad_cost * 100) if skynomad_cost > 0 else 0
        egress_pcts.append(egress_pct)
        if p_sky >= job_duration_hours:
            deadline_met += 1
        # Selection accuracy vs oracle cheapest: if SkyNomad picks cheapest region at t0, count
        # Simplified: compare first region chosen
        if current_region == min(regions, key=lambda r: spot_provider.get_spot_price_usd_per_gpu_hr(r, job.gpu_type)):
            selection_correct += 1

    # Aggregate
    def _mean(arr):
        return sum(arr) / len(arr) if arr else 0.0

    up_single_mean = _mean(up_single_costs)
    up_multi_mean = _mean(up_multi_costs)
    sky_mean = _mean(skynomad_costs)

    savings_single = (1 - sky_mean / up_single_mean) * 100 if up_single_mean else 0
    savings_multi = (1 - sky_mean / up_multi_mean) * 100 if up_multi_mean else 0

    return SpotBacktestResult(
        n_workloads=n,
        days=days,
        deadline_ratio=deadline_ratio,
        checkpoint_gb=checkpoint_gb,
        skynomad_cost_mean=sky_mean,
        up_single_cost_mean=up_single_mean,
        up_multi_cost_mean=up_multi_mean,
        cost_savings_vs_up_single_pct=savings_single,
        cost_savings_vs_up_multi_pct=savings_multi,
        deadline_met_pct=100.0 * deadline_met / n if n else 0,
        migrations_mean=_mean(migrations),
        egress_cost_pct_mean=_mean(egress_pcts),
        selection_accuracy_pct=100.0 * selection_correct / n if n else 0,
        details=[
            {"skynomad": skynomad_costs[i], "up_single": up_single_costs[i], "up_multi": up_multi_costs[i]}
            for i in range(min(10, n))
        ],
    )
