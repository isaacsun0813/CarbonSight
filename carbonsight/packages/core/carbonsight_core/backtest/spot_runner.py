"""
Spot backtest: the SkyNomad policy against single-region and multi-region baselines.

Regions get one of three availability patterns (paper Fig 2): generally available
(95% up), mostly available with frequent preemptions (70%), or largely
unavailable (20%). Probes are hourly, which is also the simulation step.

Policies, all on the same traces, the same deadline and the same prices:

    UP-single   pinned to one region for the whole job: spot when it is up,
                on-demand when it is not. Never migrates, never stalls. Run
                once per region, so we can report both the sensible default
                pin and the best pin available with hindsight.
    UP-multi    cheapest *available* spot anywhere, else the cheapest on-demand.
                Migrates freely and pays egress for it.
    SkyNomad    ``SkyNomadPolicy`` — the same Algo 1 the CLI and API use, with
                thrifty, safety net, the probe interval and the delta
                anti-flapping rule. May idle when every candidate scores below
                zero, and loses the deadline slack when it does.

Reporting rules, because a backtest that flatters itself is worthless:

  * The headline is against ``up_single_default`` — a region a reasonable
    engineer would actually pin to (cheapest price tier, best uptime pattern) —
    and against ``up_single_best``, the best single pin chosen with hindsight.
    Beating an *average over all pins* is easy when a third of the regions are
    seeded at 20% uptime; that number is reported too, and labelled.
  * Run many seeds and report mean +/- sigma. A single seed is not a result.
  * Progress accrues only on an hour actually spent running.

Known limitation, stated so it is not mistaken for a result: carbon intensity is
held constant per region for the whole window, so this harness cannot exercise
temporal carbon shifting (waiting for a greener hour). It measures the spatial
carbon lever only. ``SpotBacktestResult.caveats`` carries this to the caller.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import SyntheticCarbonProvider
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.availability import AvailabilityTracker, SpotObservation
from carbonsight_core.spot.lifetime import (
    MIN_LIFETIME_HR,
    LifetimeStats,
    predict_remaining_lifetime,
)
from carbonsight_core.spot.policy import PolicyState, SkyNomadPolicy
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    candidate_utility,
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
DEFAULT_SEEDS = (11, 22, 33, 42, 55, 66, 77, 88)

CARBON_CAVEAT = (
    "carbon intensity is constant per region across the window, so temporal "
    "carbon shifting is not exercised — only the spatial lever"
)


@dataclass
class SpotBacktestResult:
    """One seed's run. Costs are means over the workloads in that run."""

    n_workloads: int
    days: int
    seed: int
    deadline_ratio: float
    checkpoint_gb: float
    skynomad_cost_mean: float
    up_single_default_cost_mean: float
    up_single_best_cost_mean: float
    up_single_rotating_cost_mean: float
    up_multi_cost_mean: float
    up_single_default_region: str
    up_single_best_region: str
    cost_savings_vs_up_single_default_pct: float
    cost_savings_vs_up_single_best_pct: float
    cost_savings_vs_up_multi_pct: float
    deadline_met_pct: float
    up_multi_deadline_met_pct: float
    migrations_mean: float
    egress_cost_pct_mean: float
    idle_hours_mean: float
    carbon_kg_mean: float
    up_multi_carbon_kg_mean: float
    caveats: list[str] = field(default_factory=lambda: [CARBON_CAVEAT])
    details: list[dict] = field(default_factory=list)


@dataclass
class MultiSeedBacktestResult:
    """Mean +/- sigma of each headline metric across seeds."""

    seeds: list[int]
    n_workloads: int
    days: int
    up_single_default_region: str
    metrics: dict[str, tuple[float, float]]
    per_seed: list[SpotBacktestResult] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: [CARBON_CAVEAT])

    def mean(self, key: str) -> float:
        return self.metrics[key][0]

    def stdev(self, key: str) -> float:
        return self.metrics[key][1]


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


def default_pinned_region(regions: list[str], prices: dict[str, tuple[float, float]]) -> str:
    """The region a reasonable engineer pins to without running this harness.

    Cheapest on-demand price among those on the best availability pattern —
    information available up front, unlike ``up_single_best`` which needs the
    result. Ties break on name so the choice is deterministic.
    """
    best_pattern = [r for i, r in enumerate(regions) if PATTERNS[i % len(PATTERNS)][0] == 0.95]
    pool = best_pattern or regions
    return min(pool, key=lambda r: (prices[r][1], r))


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


@dataclass
class _SkyRun:
    cost: float
    egress: float
    carbon_kg: float
    migrations: int
    idle_hours: int
    met_deadline: bool


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
    carbon_weight: float = 1.0,
    delta: float = 0.05,
    probe_interval_hr: float = 2.0,
) -> _SkyRun:
    """Drive ``SkyNomadPolicy`` hour by hour — the same Algo 1 the CLI and API use."""
    policy = SkyNomadPolicy(
        migration_estimator=migration_est,
        delta=delta,
        probe_interval_hr=probe_interval_hr,
        carbon_price_usd_per_ton=carbon_price,
        carbon_weight=carbon_weight,
    )
    cost = egress = kg = done = 0.0
    migrations = idle_hours = 0
    current_region = ""
    current_mode = "spot"

    for hour in range(int(deadline_hours)):
        progress = ProgressState(p=done, P=work_hours, t=float(hour), T=deadline_hours)
        if progress.is_thrifty():
            break
        now = start + timedelta(hours=hour)

        candidates: list[CandidateState] = []
        for region in regions:
            move = (
                0.0
                if current_region in ("", region)
                else migration_est.estimate(checkpoint_gb)
            )
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

        c_od_min = min(prices[r][1] for r in regions)
        value_v = progress.future_progress_value(c_od_min)

        # Utility of staying put, for the delta rule. Unavailable spot means the
        # current placement is not on the board, which reads as -inf and forces
        # a move — exactly right, since it is making no progress.
        current_u = -math.inf
        for cand in candidates:
            if cand.region == current_region and cand.mode == current_mode:
                current_u = candidate_utility(
                    cand, value_v, COLD_START_HR, carbon_price, carbon_weight
                )
                break

        state = PolicyState(
            p=done,
            P=work_hours,
            t=float(hour),
            T=deadline_hours,
            r0=current_region,
            ckpt_size_gb=checkpoint_gb,
            cold_start_hr=COLD_START_HR,
            current_region=current_region,
            current_mode=current_mode,
            current_utility=current_u,
        )
        ranked, action = policy.rank_and_decide(state, candidates, progress, value_v)

        if action.kind == "launch":
            chosen = next(
                (c for c, _u in ranked if c.region == action.region and c.mode == action.mode),
                None,
            )
            if chosen is None or chosen.is_idle:
                idle_hours += 1
                continue
            if current_region and current_region != chosen.region:
                migrations += 1
                egress += chosen.migration_cost
            current_region, current_mode = chosen.region, chosen.mode
        elif action.kind == "stay":
            chosen = next(
                (
                    c
                    for c, _u in ranked
                    if c.region == current_region and c.mode == current_mode
                ),
                None,
            )
            if chosen is None:
                idle_hours += 1
                continue
        else:  # idle / terminate
            idle_hours += 1
            continue

        if chosen.is_idle:
            idle_hours += 1
            continue

        cost += chosen.price_per_hr
        kg += chosen.carbon_kg_per_hr
        done += 1.0

    return _SkyRun(cost, egress, kg, migrations, idle_hours, done >= work_hours)


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
    """Run every policy over ``n`` workloads for one seed."""
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
    default_region = default_pinned_region(regions, prices)

    sky_costs: list[float] = []
    single_costs: dict[str, list[float]] = {r: [] for r in regions}
    rotating_costs: list[float] = []
    multi_costs: list[float] = []
    sky_kg: list[float] = []
    multi_kg: list[float] = []
    migrations: list[int] = []
    idle_hours: list[int] = []
    egress_pcts: list[float] = []
    met = {"sky": 0, "multi": 0}
    details: list[dict] = []

    rng = random.Random(seed)
    max_offset = max(1, days * 24 - int(deadline_hours) - 1)

    for i in range(n):
        start = trace_start + timedelta(hours=rng.randint(0, max_offset))

        # Every possible single-region pin, so the baseline is a choice we can
        # defend rather than an average over regions nobody would pick.
        for region in regions:
            cost, _kg, _met = _run_up_single(
                region, trackers, prices, carbon_kg, start, work_hours, deadline_hours
            )
            single_costs[region].append(cost)
        rotating_costs.append(single_costs[regions[i % len(regions)]][-1])

        m_cost, m_kg, m_migr, m_met = _run_up_multi(
            regions, trackers, prices, carbon_kg, start, work_hours, deadline_hours, migration_flat
        )
        sky = _run_skynomad(
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

        total_sky = sky.cost + sky.egress
        sky_costs.append(total_sky)
        multi_costs.append(m_cost)
        sky_kg.append(sky.carbon_kg)
        multi_kg.append(m_kg)
        migrations.append(sky.migrations)
        idle_hours.append(sky.idle_hours)
        egress_pcts.append(sky.egress / total_sky * 100 if total_sky > 0 else 0.0)
        met["sky"] += int(sky.met_deadline)
        met["multi"] += int(m_met)
        if i < 10:
            details.append(
                {
                    "skynomad": total_sky,
                    "up_single_default": single_costs[default_region][-1],
                    "up_multi": m_cost,
                    "skynomad_migrations": sky.migrations,
                    "skynomad_idle_hours": sky.idle_hours,
                    "skynomad_met_deadline": sky.met_deadline,
                    "up_multi_migrations": m_migr,
                }
            )

    def mean(values: list[float] | list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    def savings(baseline: float) -> float:
        return (1 - sky_mean / baseline) * 100 if baseline else 0.0

    sky_mean = mean(sky_costs)
    per_region_mean = {r: mean(v) for r, v in single_costs.items()}
    best_region = min(per_region_mean, key=lambda r: per_region_mean[r])

    return SpotBacktestResult(
        n_workloads=n,
        days=days,
        seed=seed,
        deadline_ratio=deadline_ratio,
        checkpoint_gb=checkpoint_gb,
        skynomad_cost_mean=sky_mean,
        up_single_default_cost_mean=per_region_mean[default_region],
        up_single_best_cost_mean=per_region_mean[best_region],
        up_single_rotating_cost_mean=mean(rotating_costs),
        up_multi_cost_mean=mean(multi_costs),
        up_single_default_region=default_region,
        up_single_best_region=best_region,
        cost_savings_vs_up_single_default_pct=savings(per_region_mean[default_region]),
        cost_savings_vs_up_single_best_pct=savings(per_region_mean[best_region]),
        cost_savings_vs_up_multi_pct=savings(mean(multi_costs)),
        deadline_met_pct=100.0 * met["sky"] / n if n else 0.0,
        up_multi_deadline_met_pct=100.0 * met["multi"] / n if n else 0.0,
        migrations_mean=mean(migrations),
        egress_cost_pct_mean=mean(egress_pcts),
        idle_hours_mean=mean(idle_hours),
        carbon_kg_mean=mean(sky_kg),
        up_multi_carbon_kg_mean=mean(multi_kg),
        details=details,
    )


AGGREGATED_METRICS = (
    "skynomad_cost_mean",
    "up_single_default_cost_mean",
    "up_single_best_cost_mean",
    "up_multi_cost_mean",
    "cost_savings_vs_up_single_default_pct",
    "cost_savings_vs_up_single_best_pct",
    "cost_savings_vs_up_multi_pct",
    "deadline_met_pct",
    "migrations_mean",
    "egress_cost_pct_mean",
    "idle_hours_mean",
    "carbon_kg_mean",
    "up_multi_carbon_kg_mean",
)


def run_spot_backtest_multi(
    n: int = 50,
    days: int = 14,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    **kwargs: object,
) -> MultiSeedBacktestResult:
    """Repeat the backtest across seeds and report mean +/- sigma of each metric.

    A single seed is a sample, not a result: the seed-to-seed spread on the
    UP-multi delta is wider than the delta itself.
    """
    runs = [run_spot_backtest(n=n, days=days, seed=s, **kwargs) for s in seeds]  # type: ignore[arg-type]
    metrics: dict[str, tuple[float, float]] = {}
    for key in AGGREGATED_METRICS:
        values = [getattr(r, key) for r in runs]
        metrics[key] = (
            statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
    return MultiSeedBacktestResult(
        seeds=list(seeds),
        n_workloads=n,
        days=days,
        up_single_default_region=runs[0].up_single_default_region,
        metrics=metrics,
        per_seed=runs,
    )
