"""Utility and carbon estimation helpers for planning (SkyNomad Eq. 7 + 9 barebones)."""

from datetime import UTC, datetime

from carbonsight_core.estimator.carbon_model import LB_TO_KG, time_weighted_moer
from carbonsight_core.estimator.power_model import expected_it_power_w
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.types import JobMode


def parse_forecast_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def it_mwh_for_duration(job: JobSpec, duration_hours: float) -> float:
    """IT MWh over duration; cooling and other facility overhead are excluded."""
    return expected_it_power_w(job) * duration_hours / 1_000_000.0


def estimate_window_carbon_kg(
    job: JobSpec,
    forecast_points: list[dict],
    start_utc: datetime,
    finish_utc: datetime,
) -> tuple[float, float]:
    """Return IT operational carbon and time-weighted MOER for [start, finish]."""
    duration_hours = (finish_utc - start_utc).total_seconds() / 3600.0
    if duration_hours <= 0:
        return 0.0, 0.0
    moer_lb = time_weighted_moer(forecast_points, start_utc, finish_utc)
    it_mwh = it_mwh_for_duration(job, duration_hours)
    carbon_kg = moer_lb * it_mwh * LB_TO_KG
    return carbon_kg, moer_lb


def carbon_rate_kg_per_hr(job: JobSpec, moer_lb_per_mwh: float) -> float:
    """Instantaneous IT operational carbon rate (kg/hr) at given MOER."""
    it_mwh_per_hr = it_mwh_for_duration(job, 1.0)
    return moer_lb_per_mwh * it_mwh_per_hr * LB_TO_KG


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
