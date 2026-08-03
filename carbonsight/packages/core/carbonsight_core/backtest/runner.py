"""
Backtest: generate N synthetic workloads, decision-time score, oracle (best realized), metrics.
Per design: carbon savings %, cost savings %, regret, rank accuracy (top-1, top-3).
MVP: synthetic MOER and cost series in-memory; no parquet required.
"""

import hashlib
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from carbonsight_core.models import JobSpec

# Region options for backtest (cloud_region -> fixed synthetic MOER and price)
DEFAULT_REGIONS = ["eu-north-1", "us-east-1"]


@dataclass
class BacktestResult:
    """Aggregated backtest metrics."""

    n_workloads: int
    days: int
    carbon_savings_pct_mean: float
    cost_savings_pct_mean: float
    regret_mean: float
    regret_p95: float
    rank_top1_pct: float
    rank_top3_pct: float
    details: list[dict] = field(default_factory=list)



def _region_seed(region: str) -> int:
    """Stable per-region offset.

    ``hash(str)`` is salted per process, so the same seed gave different results
    across runs: carbon savings moved from -10.4 to -50.7 purely with
    PYTHONHASHSEED.
    """
    return int(hashlib.md5(region.encode("utf-8")).hexdigest()[:8], 16)


def _synthetic_moer(region: str, t: datetime, seed: int) -> float:
    """Synthetic MOER (lb/MWh) per region and time for reproducibility."""
    r = random.Random(seed + _region_seed(region) + t.date().toordinal())
    base = {"eu-north-1": 180, "us-east-1": 420}.get(region, 350)
    return base + r.gauss(0, 40)


def _synthetic_price(region: str, t: datetime, seed: int) -> float:
    """Synthetic $/hr per region and time."""
    r = random.Random(seed + _region_seed(region) + t.date().toordinal())
    base = {"eu-north-1": 3.2, "us-east-1": 2.8}.get(region, 3.0)
    return max(0.5, base + r.gauss(0, 0.3))


def _generate_workloads(n: int, days: int, seed: int) -> list[tuple[JobSpec, datetime]]:
    """Generate n synthetic workloads: JobSpec + start_time over the last `days` days."""
    r = random.Random(seed)
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    workloads = []
    gpu_types = ["T4", "A100", "H100"]
    weights = [0.2, 0.5, 0.3]
    for _ in range(n):
        gpu_type = r.choices(gpu_types, weights=weights)[0]
        gpu_count = r.choice([1, 2, 4])
        duration_hours = max(0.5, r.lognormvariate(0.5, 0.8))
        start_time = start + timedelta(seconds=r.randint(0, max(1, days * 86400)))
        job = JobSpec(gpu_type=gpu_type, gpu_count=gpu_count, duration_hours=duration_hours)
        workloads.append((job, start_time))
    return workloads


def run_backtest(
    n: int = 1000,
    days: int = 180,
    seed: int = 42,
    regions: list[str] | None = None,
) -> BacktestResult:
    """
    Run backtest: for each workload, score regions at decision time (synthetic MOER/price),
    choose best; compute oracle (best realized over job window); then regret and savings vs baseline.
    """
    regions = regions or DEFAULT_REGIONS
    workloads = _generate_workloads(n, days, seed)
    baseline_region = regions[0]
    results = []
    regrets = []
    carbon_savings_pct = []
    cost_savings_pct = []
    rank_top1 = 0
    rank_top3 = 0

    for job, start_time in workloads:
        # Decision-time view: synthetic MOER and price at start_time
        scores_at_t0 = []
        for r in regions:
            moer = _synthetic_moer(r, start_time, seed)
            price = _synthetic_price(r, start_time, seed)
            # Score = normalized carbon + normalized cost (lower better)
            carbon_kg = (job.gpu_count * 300 / 1000) * job.duration_hours * (moer / 1000) * 0.45359237
            cost_usd = price * job.duration_hours
            scores_at_t0.append((r, carbon_kg, cost_usd, carbon_kg + cost_usd / 10))  # simple combined score

        # Chosen = argmin score at t0
        scores_at_t0.sort(key=lambda x: x[3])
        chosen_region = scores_at_t0[0][0]
        chosen_carbon = scores_at_t0[0][1]
        chosen_cost = scores_at_t0[0][2]
        chosen_score = scores_at_t0[0][3]

        # Oracle: best realized over job window (we use same synthetic for simplicity)
        realized = []
        for r in regions:
            moer = _synthetic_moer(r, start_time, seed)
            price = _synthetic_price(r, start_time, seed)
            carbon_kg = (job.gpu_count * 300 / 1000) * job.duration_hours * (moer / 1000) * 0.45359237
            cost_usd = price * job.duration_hours
            realized.append((r, carbon_kg, cost_usd, carbon_kg + cost_usd / 10))
        realized.sort(key=lambda x: x[3])
        oracle_score = realized[0][3]
        oracle_region = realized[0][0]
        regret = chosen_score - oracle_score
        regrets.append(regret)

        # Baseline (first region) carbon and cost
        base_carbon = next(x[1] for x in realized if x[0] == baseline_region)
        base_cost = next(x[2] for x in realized if x[0] == baseline_region)
        if base_carbon > 0:
            carbon_savings_pct.append((1 - chosen_carbon / base_carbon) * 100)
        if base_cost > 0:
            cost_savings_pct.append((1 - chosen_cost / base_cost) * 100)

        chosen_rank = next(i for i, x in enumerate(realized) if x[0] == chosen_region)
        if chosen_rank == 0:
            rank_top1 += 1
        if chosen_rank < 3:
            rank_top3 += 1

        results.append({
            "chosen_region": chosen_region,
            "oracle_region": oracle_region,
            "regret": regret,
            "carbon_kg_chosen": chosen_carbon,
            "cost_usd_chosen": chosen_cost,
        })

    regrets.sort()
    return BacktestResult(
        n_workloads=n,
        days=days,
        carbon_savings_pct_mean=sum(carbon_savings_pct) / len(carbon_savings_pct) if carbon_savings_pct else 0,
        cost_savings_pct_mean=sum(cost_savings_pct) / len(cost_savings_pct) if cost_savings_pct else 0,
        regret_mean=sum(regrets) / len(regrets) if regrets else 0,
        regret_p95=regrets[int(0.95 * len(regrets))] if regrets else 0,
        rank_top1_pct=100 * rank_top1 / n,
        rank_top3_pct=100 * rank_top3 / n,
        details=results[:10],
    )
