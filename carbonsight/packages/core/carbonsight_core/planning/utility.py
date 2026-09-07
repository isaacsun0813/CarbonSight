"""Utility and carbon estimation helpers for planning (SkyNomad Eq. 7 + 9 barebones)."""

from datetime import datetime

from carbonsight_core.estimator.carbon_model import LB_TO_KG, time_weighted_moer
from carbonsight_core.estimator.power_model import PUE_MEAN, power_it_w
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.types import JobMode


def facility_mwh_for_duration(job: JobSpec, duration_hours: float, pue: float = PUE_MEAN) -> float:
    """Facility MWh over duration using mean IT power and fixed PUE."""
    p_it_w = power_it_w(job)
    return (p_it_w / 1000.0) * duration_hours * (pue / 1000.0)


def estimate_window_carbon_kg(
    job: JobSpec,
    forecast_points: list[dict],
    start_utc: datetime,
    finish_utc: datetime,
    *,
    pue: float = PUE_MEAN,
) -> tuple[float, float]:
    """Return (carbon_kg, time_weighted_moer_lb_per_mwh) for [start, finish]."""
    duration_hours = (finish_utc - start_utc).total_seconds() / 3600.0
    if duration_hours <= 0:
        return 0.0, 0.0
    moer_lb = time_weighted_moer(forecast_points, start_utc, finish_utc)
    facility_mwh = facility_mwh_for_duration(job, duration_hours, pue=pue)
    carbon_kg = moer_lb * facility_mwh * LB_TO_KG
    return carbon_kg, moer_lb


def carbon_rate_kg_per_hr(job: JobSpec, moer_lb_per_mwh: float, pue: float = PUE_MEAN) -> float:
    """Instantaneous carbon emission rate (kg/hr) at given MOER."""
    facility_mwh_per_hr = facility_mwh_for_duration(job, 1.0, pue=pue)
    return moer_lb_per_mwh * facility_mwh_per_hr * LB_TO_KG


def progress_value(theta: float, theta_bar: float, c_od_min: float) -> float:
    """SkyNomad Eq. (7): monetary value of progress at current deadline pressure."""
    if c_od_min <= 0:
        return 0.0
    if theta_bar <= 0:
        return c_od_min * max(theta, 0.0)
    return c_od_min * (theta / theta_bar)


def deadline_pressure(remaining_work_hours: float, remaining_time_hours: float) -> float:
    """SkyNomad Eq. (5): required future progress rate to meet deadline."""
    if remaining_time_hours <= 0:
        return float("inf") if remaining_work_hours > 0 else 0.0
    return remaining_work_hours / remaining_time_hours


def calc_utility(
    progress_value_usd_per_hr: float,
    cost_rate_usd_per_hr: float,
    carbon_rate_kg_per_hr: float,
    mode: JobMode,
    *,
    expected_lifetime_hours: float,
    cold_start_hours: float = 0.5,
    migration_usd: float = 0.0,
    lambda_co2_usd_per_kg: float = 0.0,
) -> float:
    """SkyNomad Eq. (9) barebones: U = V*eta - cost - lambda*carbon - amortized migration."""
    if mode == JobMode.IDLE:
        return 0.0
    if expected_lifetime_hours <= 0:
        return 0.0
    if mode == JobMode.ON_DEMAND:
        eta = 1.0
        lifetime = expected_lifetime_hours
    else:
        lifetime = expected_lifetime_hours
        eta = max(0.0, lifetime - cold_start_hours) / lifetime
    amortized_migration = migration_usd / lifetime if lifetime > 0 else 0.0
    return (
        progress_value_usd_per_hr * eta
        - cost_rate_usd_per_hr
        - lambda_co2_usd_per_kg * carbon_rate_kg_per_hr
        - amortized_migration
    )
