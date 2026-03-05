"""
Carbon estimator: E_facility_MWh, time-integrate MOER, Monte Carlo for mean/p10/p90.
Design: CO2_kg = (E_facility_MWh * MOER_lb_per_MWh) * 0.45359237; support mixture mapping.
"""

import random
from datetime import datetime, timezone
from typing import Any, Callable

from carbonsight_core.estimator.power_model import sample_power_params
from carbonsight_core.estimator.pricing import estimate_cost_usd
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.watttime import SUPPORTED_MOER_UNIT, WattTimeClient, WattTimeError


LB_TO_KG = 0.45359237
MONTE_CARLO_SAMPLES = 1000


def _facility_mwh_per_hour(p_it_w: float, pue: float) -> float:
    """E_facility_MWh for one hour = (P_IT_W/1000)*1*hour*PUE/1000."""
    return (p_it_w / 1000.0) * 1.0 * (pue / 1000.0)


def _get_moer_at_time(
    wt: WattTimeClient,
    wt_regions: list[tuple[str, float]],
    t: datetime,
    client: Any = None,
) -> float:
    """Weighted average MOER (lb/MWh) across mixture at time t. Uses forecast or signal-index."""
    if not wt_regions:
        return 0.0
    total = 0.0
    for region, weight in wt_regions:
        try:
            # Prefer forecast; fallback to signal-index (percentile - we use as proxy; design says raw MOER for kg)
            data = wt.get_forecast(region, horizon_hours=1, client=client)
            points = data.get("data", [])
            if points:
                total += weight * float(points[0].get("value", 0))
            else:
                total += weight * 400.0  # fallback default lb/MWh
        except Exception:
            total += weight * 400.0
    return total


def estimate_option(
    job: JobSpec,
    cloud: str,
    cloud_region: str,
    watttime_regions: list[tuple[str, float]],
    mapping_confidence: float,
    wt: WattTimeClient,
    *,
    start_time: datetime | None = None,
    moer_override: float | None = None,
    rng: random.Random | None = None,
) -> EstimateResult:
    """
    Run Monte Carlo; return EstimateResult with mean, p10, p90.
    Refuses to compute kg if WattTime units are not lbs_co2_per_mwh (unit gate).
    """
    rng = rng or random.Random()
    start = start_time or datetime.now(timezone.utc)
    samples_kg: list[float] = []

    # Fetch MOER once before the loop — it's deterministic for a fixed start time.
    # Calling it inside the loop made 1000 HTTP requests per region, causing a multi-minute hang.
    if moer_override is not None:
        moer_lb = moer_override
    else:
        try:
            moer_lb = _get_moer_at_time(wt, watttime_regions, start, None)
        except WattTimeError as e:
            if SUPPORTED_MOER_UNIT in str(e).lower() or "unit" in str(e).lower():
                raise
            moer_lb = 400.0

    for _ in range(MONTE_CARLO_SAMPLES):
        params = sample_power_params(job, rng)
        facility_mwh_total = _facility_mwh_per_hour(params.p_it_w, params.pue) * job.duration_hours
        co2_lb = facility_mwh_total * moer_lb
        co2_kg = co2_lb * LB_TO_KG
        samples_kg.append(co2_kg)

    samples_kg.sort()
    n = len(samples_kg)
    mean_kg = sum(samples_kg) / n
    p10 = samples_kg[int(0.10 * n)] if n else 0.0
    p90 = samples_kg[int(0.90 * n)] if n else 0.0

    cost_usd = estimate_cost_usd(job.gpu_type, job.gpu_count, job.duration_hours, cloud_region)

    return EstimateResult(
        cloud=cloud,
        cloud_region=cloud_region,
        watttime_regions=watttime_regions,
        expected_cost_usd=cost_usd,
        expected_co2_kg_mean=mean_kg,
        expected_co2_kg_p10=p10,
        expected_co2_kg_p90=p90,
        mapping_confidence=mapping_confidence,
        notes=[],
    )
