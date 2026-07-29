"""Pick the lowest-carbon start time within a delay budget."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from carbonsight_core.estimator.carbon_model import _time_weighted_moer


@dataclass(frozen=True, slots=True)
class ScheduleChoice:
    """Result of choosing the best start time for a job."""

    start_utc: datetime
    delay_hours: float
    window_moer_lb_per_mwh: float
    now_window_moer_lb_per_mwh: float
    moer_reduction_pct: float


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def pick_lowest_carbon_start(
    forecast_points: list[dict],
    duration_hours: float,
    max_delay_hours: float,
) -> ScheduleChoice:
    """Find the start time within ``[t0, t0 + max_delay_hours]`` that minimises window MOER."""
    if not forecast_points:
        raise ValueError("forecast_points must not be empty")
    if max_delay_hours < 0:
        raise ValueError("max_delay_hours must be non-negative")
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    sorted_pts = sorted(forecast_points, key=lambda p: p["point_time"])
    t0 = _parse_utc(sorted_pts[0]["point_time"])
    deadline = t0 + timedelta(hours=max_delay_hours)
    window = timedelta(hours=duration_hours)

    candidates: list[tuple[datetime, float]] = []
    for p in sorted_pts:
        t = _parse_utc(p["point_time"])
        if t > deadline:
            break
        moer = _time_weighted_moer(sorted_pts, t, t + window)
        candidates.append((t, moer))

    if not candidates:
        candidates.append((t0, _time_weighted_moer(sorted_pts, t0, t0 + window)))

    best_start, best_moer = min(candidates, key=lambda c: (c[1], c[0]))
    now_moer = candidates[0][1]
    delay = (best_start - t0).total_seconds() / 3600.0
    reduction = (now_moer - best_moer) / now_moer * 100.0 if now_moer else 0.0

    return ScheduleChoice(
        start_utc=best_start,
        delay_hours=delay,
        window_moer_lb_per_mwh=best_moer,
        now_window_moer_lb_per_mwh=now_moer,
        moer_reduction_pct=reduction,
    )
