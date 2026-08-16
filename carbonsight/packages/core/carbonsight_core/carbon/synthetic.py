"""Explicit, non-physical carbon curves for demos and tests only."""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.watttime import SUPPORTED_MOER_UNIT

SYNTHETIC_WATTTIME_REGIONS: tuple[str, ...] = (
    "PJM_DC",
    "PJM_EASTERN_OH",
    "CAISO_NORTH",
    "PACW",
    "HQ",
    "IE",
    "UK",
    "FR",
    "DE",
    "SE",
    "IT",
    "JP_TK",
    "KOR",
    "SGP",
    "NEM_NSW",
    "IND",
    "BRA",
)

SYNTHETIC_BASE_MOER_LB_PER_MWH = 200.0
SYNTHETIC_MOER_SPREAD_LB_PER_MWH = 850
SYNTHETIC_DIURNAL_AMPLITUDE_LB_PER_MWH = 30.0
SYNTHETIC_MIN_MOER_LB_PER_MWH = 10.0


def synthetic_moer_for_region(
    region: str,
    *,
    base_lb_per_mwh: float = SYNTHETIC_BASE_MOER_LB_PER_MWH,
) -> float:
    """Return a deterministic, non-physical stand-in MOER for ``region``."""
    digest = hashlib.md5(region.encode("utf-8"), usedforsecurity=False).hexdigest()
    region_offset = int(digest[:8], 16) % SYNTHETIC_MOER_SPREAD_LB_PER_MWH
    return float(base_lb_per_mwh + region_offset)


def build_synthetic_forecast(
    region: str,
    *,
    horizon_hours: int = 24,
    step_minutes: int = 5,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build an explicitly synthetic MOER series for demos and tests."""
    series_start = now or datetime.now(UTC)
    if series_start.tzinfo is None:
        series_start = series_start.replace(tzinfo=UTC)
    base_lb_per_mwh = synthetic_moer_for_region(region)
    point_count = max(1, int(horizon_hours * 60 / step_minutes))
    points: list[dict[str, Any]] = []
    for index in range(point_count):
        wave_lb_per_mwh = SYNTHETIC_DIURNAL_AMPLITUDE_LB_PER_MWH * (
            (index % 12) / 12.0 - 0.5
        )
        point_time = series_start + timedelta(minutes=step_minutes * index)
        points.append(
            {
                "point_time": point_time.isoformat().replace("+00:00", "Z"),
                "value": max(
                    SYNTHETIC_MIN_MOER_LB_PER_MWH,
                    base_lb_per_mwh + wave_lb_per_mwh,
                ),
                "units": SUPPORTED_MOER_UNIT,
            }
        )
    return points
