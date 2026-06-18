"""
Carbon estimator: E_facility_MWh, time-integrate MOER, Monte Carlo for mean/p10/p90.
Design: CO2_kg = (E_facility_MWh * MOER_lb_per_MWh) * 0.45359237; support mixture mapping.
"""

import random
from datetime import UTC, datetime
from typing import Any

from carbonsight_core.estimator.power_model import sample_power_params
from carbonsight_core.estimator.pricing import estimate_cost_usd
from carbonsight_core.models import ActualRunResult, EstimateResult, JobSpec
from carbonsight_core.watttime import SUPPORTED_MOER_UNIT, WattTimeClient, WattTimeError

LB_TO_KG = 0.45359237
MONTE_CARLO_SAMPLES = 1000


def _facility_mwh_per_hour(p_it_w: float, pue: float) -> float:
    """E_facility_MWh for one hour = (P_IT_W/1000)*1*hour*PUE/1000."""
    return (p_it_w / 1000.0) * 1.0 * (pue / 1000.0)


def _time_weighted_moer(
    points: list[dict[str, Any]],
    actual_start: datetime,
    actual_end: datetime,
) -> float:
    """Compute time-weighted average MOER (lb/MWh) over [actual_start, actual_end].

    Each data point covers the interval from its timestamp to the next point's
    timestamp. The first and last intervals are clipped to actual_start/actual_end.
    If only one point, its value is used directly.
    """
    if len(points) == 1:
        return float(points[0]["value"])

    total_seconds = (actual_end - actual_start).total_seconds()
    sorted_points = sorted(points, key=lambda p: p["point_time"])

    def _parse_utc(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    weighted_sum = 0.0
    for i, p in enumerate(sorted_points):
        t_i = _parse_utc(p["point_time"])
        t_next = (
            _parse_utc(sorted_points[i + 1]["point_time"])
            if i + 1 < len(sorted_points)
            else actual_end
        )
        interval_start = max(t_i, actual_start)
        interval_end = min(t_next, actual_end)
        overlap = max(0.0, (interval_end - interval_start).total_seconds())
        weighted_sum += float(p["value"]) * overlap

    return weighted_sum / total_seconds


class JobCarbonEstimator:
    """Forecast and actual CO₂ for a job using one WattTime client (token cache is per-client)."""

    __slots__ = ("_watt_time",)

    def __init__(self, watt_time: WattTimeClient) -> None:
        self._watt_time = watt_time

    def _weighted_forecast_moer_lb_per_mwh(
        self,
        watttime_regions: list[tuple[str, float]],
        client: Any = None,
    ) -> float:
        """Weighted average forecast MOER (lb CO₂/MWh) across WattTime mixture regions."""
        if not watttime_regions:
            return 0.0
        total = 0.0
        wt = self._watt_time
        for region, weight in watttime_regions:
            try:
                data = wt.get_forecast(region, horizon_hours=1, client=client)
                points = data.get("data", [])
                if points:
                    total += weight * float(points[0].get("value", 0))
                else:
                    total += weight * 400.0  # fallback default lb/MWh
            except WattTimeError:
                raise
            except Exception:
                total += weight * 400.0
        return total

    def estimate_region(
        self,
        job: JobSpec,
        cloud: str,
        cloud_region: str,
        watttime_regions: list[tuple[str, float]],
        mapping_confidence: float,
        *,
        moer_override: float | None = None,
        rng: random.Random | None = None,
    ) -> EstimateResult:
        """
        Monte Carlo carbon + cost estimate for one cloud region (forecast MOER, sampled power).

        Refuses to compute kg if WattTime units are not lbs_co2_per_mwh (unit gate).
        """
        rng = rng or random.Random()
        samples_kg: list[float] = []

        if moer_override is not None:
            moer_lb = moer_override
        else:
            try:
                moer_lb = self._weighted_forecast_moer_lb_per_mwh(watttime_regions, None)
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

    def compute_actual_run(
        self,
        job: JobSpec,
        cloud: str,
        cloud_region: str,
        watttime_regions: list[tuple[str, float]],
        actual_start: datetime,
        actual_end: datetime,
        estimated_co2_kg: float,
        *,
        historical_moer_override: list[dict[str, Any]] | None = None,
    ) -> ActualRunResult:
        """Integrate historical MOER over the run window; compare to the pre-run estimate mean."""
        if actual_end <= actual_start:
            raise ValueError(
                f"actual_end must be after actual_start: "
                f"{actual_start.isoformat()} >= {actual_end.isoformat()}"
            )

        actual_duration_hours = (actual_end - actual_start).total_seconds() / 3600
        wt = self._watt_time

        if historical_moer_override is not None:
            if not historical_moer_override:
                raise ValueError("historical_moer_override must not be empty")
            avg_moer_lb = _time_weighted_moer(historical_moer_override, actual_start, actual_end)
        else:
            avg_moer_lb = 0.0
            for region, weight in watttime_regions:
                points = wt.get_historical(region, actual_start, actual_end)
                if not points:
                    raise ValueError(
                        f"No historical MOER data returned for region {region!r} "
                        f"between {actual_start.isoformat()} and {actual_end.isoformat()}"
                    )
                avg_moer_lb += weight * _time_weighted_moer(points, actual_start, actual_end)

        rng = random.Random(0)
        samples_mwh: list[float] = []
        for _ in range(MONTE_CARLO_SAMPLES):
            params = sample_power_params(job, rng)
            facility_mwh = _facility_mwh_per_hour(params.p_it_w, params.pue) * actual_duration_hours
            samples_mwh.append(facility_mwh)
        mean_facility_mwh = sum(samples_mwh) / len(samples_mwh)

        actual_co2_kg = mean_facility_mwh * avg_moer_lb * LB_TO_KG
        actual_cost_usd = estimate_cost_usd(job.gpu_type, job.gpu_count, actual_duration_hours, cloud_region)

        co2_savings_pct = (
            (estimated_co2_kg - actual_co2_kg) / estimated_co2_kg * 100
            if estimated_co2_kg > 0
            else 0.0
        )

        return ActualRunResult(
            cloud=cloud,
            cloud_region=cloud_region,
            watttime_regions=watttime_regions,
            actual_start_utc=actual_start,
            actual_end_utc=actual_end,
            actual_duration_hours=actual_duration_hours,
            actual_co2_kg=actual_co2_kg,
            actual_cost_usd=actual_cost_usd,
            estimated_co2_kg=estimated_co2_kg,
            co2_savings_vs_estimate_pct=co2_savings_pct,
        )


def estimate_job_carbon_in_region(
    job: JobSpec,
    cloud: str,
    cloud_region: str,
    watttime_regions: list[tuple[str, float]],
    mapping_confidence: float,
    watt_time: WattTimeClient,
    *,
    moer_override: float | None = None,
    rng: random.Random | None = None,
) -> EstimateResult:
    """Backward-compatible wrapper around :class:`JobCarbonEstimator`."""
    return JobCarbonEstimator(watt_time).estimate_region(
        job,
        cloud,
        cloud_region,
        watttime_regions,
        mapping_confidence,
        moer_override=moer_override,
        rng=rng,
    )


def compute_actual_co2(
    job: JobSpec,
    cloud: str,
    cloud_region: str,
    watttime_regions: list[tuple[str, float]],
    actual_start: datetime,
    actual_end: datetime,
    wt: WattTimeClient,
    estimated_co2_kg: float,
    *,
    historical_moer_override: list[dict[str, Any]] | None = None,
) -> ActualRunResult:
    """Backward-compatible wrapper around :meth:`JobCarbonEstimator.compute_actual_run`."""
    return JobCarbonEstimator(wt).compute_actual_run(
        job,
        cloud,
        cloud_region,
        watttime_regions,
        actual_start,
        actual_end,
        estimated_co2_kg,
        historical_moer_override=historical_moer_override,
    )
