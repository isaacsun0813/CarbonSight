"""
Spot-aware backtest runner.

Simulates evictions using LifetimeStats survival and includes carbon cost in metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import SyntheticCarbonProvider
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import ODCandidate, rank_candidates

DEFAULT_REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-north-1",
    "eu-west-1",
    "ap-southeast-1",
]


@dataclass
class SpotBacktestResult:
    n_workloads: int
    days: int
    carbon_savings_pct_mean: float
    cost_savings_pct_mean: float
    eviction_rate: float
    joint_utility_mean: float
    regret_mean: float
    details: list[dict] = field(default_factory=list)


def run_spot_backtest(
    n: int = 200,
    days: int = 30,
    seed: int = 42,
    regions: list[str] | None = None,
    carbon_price_usd_per_ton: float = 50.0,
) -> SpotBacktestResult:
    """Simulate workloads with eviction draws and joint U ranking."""
    regions = regions or DEFAULT_REGIONS
    rng = random.Random(seed)
    carbon = SyntheticCarbonProvider()
    spot = StaticSpotPriceProvider()
    lifetime = LifetimeStats.from_exponential(18.0, n=40)

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    carbon_savings: list[float] = []
    cost_savings: list[float] = []
    utilities: list[float] = []
    regrets: list[float] = []
    evictions = 0
    total_placements = 0
    details: list[dict] = []

    gpu_types = ["T4", "A100", "H100"]

    for i in range(n):
        gpu = rng.choice(gpu_types)
        gcount = rng.choice([1, 2, 4])
        dur = max(0.5, rng.lognormvariate(0.4, 0.6))
        t0 = start + timedelta(seconds=rng.randint(0, max(1, days * 86400)))
        job = JobSpec(
            gpu_type=gpu,
            gpu_count=gcount,
            duration_hours=dur,
            deadline_hours=dur * 2 + 6,
            carbon_price_usd_per_ton=carbon_price_usd_per_ton,
        )
        progress = ProgressState.from_deadline_hours(
            deadline_hours=job.deadline_hours, now=t0
        )
        lbar = lifetime.expected_remaining(0.0)
        surv = lifetime.compute_survival(0.0)

        cands: list[ODCandidate] = []
        for region in regions:
            # Map cloud region loosely to a synthetic WT code via hash
            wt = region  # synthetic provider accepts any string
            moer = carbon.get_moer(wt, lbar_hours=lbar, window_start=t0)
            px = spot.get_spot_price_usd_per_gpu_hr(region, gpu)
            od = spot.get_on_demand_usd_per_gpu_hr(region, gpu)
            cands.append(
                ODCandidate(
                    region=region,
                    instance_type=gpu,
                    spot_price=px,
                    moer=moer,
                    survival=surv,
                    lbar=lbar,
                    on_demand_price=od,
                    gpu_count=gcount,
                )
            )
        ranked = rank_candidates(
            cands, progress, carbon_price_usd_per_ton=carbon_price_usd_per_ton
        )
        chosen = ranked[0]
        oracle = min(ranked, key=lambda c: c.c_total)  # lowest true cost proxy

        # Eviction draw from survival
        total_placements += 1
        if rng.random() > chosen.survival:
            evictions += 1
            # pay restart penalty in cost
            chosen_cost = chosen.c_total * 1.25
        else:
            chosen_cost = chosen.c_total

        baseline = ranked[-1] if len(ranked) > 1 else chosen
        if baseline.carbon_kg > 0:
            carbon_savings.append((1 - chosen.carbon_kg / baseline.carbon_kg) * 100)
        if baseline.c_total > 0:
            cost_savings.append((1 - chosen_cost / max(baseline.c_total, 1e-9)) * 100)
        utilities.append(chosen.utility)
        regrets.append(max(0.0, chosen_cost - oracle.c_total))

        if i < 10:
            details.append(
                {
                    "chosen": chosen.region,
                    "utility": chosen.utility,
                    "carbon_kg": chosen.carbon_kg,
                    "cost": chosen_cost,
                }
            )

    return SpotBacktestResult(
        n_workloads=n,
        days=days,
        carbon_savings_pct_mean=sum(carbon_savings) / len(carbon_savings) if carbon_savings else 0.0,
        cost_savings_pct_mean=sum(cost_savings) / len(cost_savings) if cost_savings else 0.0,
        eviction_rate=evictions / total_placements if total_placements else 0.0,
        joint_utility_mean=sum(utilities) / len(utilities) if utilities else 0.0,
        regret_mean=sum(regrets) / len(regrets) if regrets else 0.0,
        details=details,
    )
