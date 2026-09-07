"""Build RegionPlanInput rows from registry + WattTime (app-boundary HTTP)."""

from __future__ import annotations

import math
from datetime import datetime

from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.types import PlanConstraints, RegionPlanInput
from carbonsight_core.watttime import WattTimeClient, WattTimeError


def build_region_plan_inputs(
    registry: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    constraints: PlanConstraints,
    region_codes: list[str] | None = None,
) -> list[RegionPlanInput]:
    """
    Fetch MOER forecast per AWS region for dynamic/static planning.

    Uses primary wt_region only (mixture weighting TODO). Skips regions without
    mapping or on WattTime errors.
    """
    horizon_hours = math.ceil(
        (constraints.deadline_utc - constraints.now_utc).total_seconds() / 3600.0
        + job.duration_hours
    )
    horizon_hours = max(1, min(horizon_hours, 72))
    allowed = set(region_codes) if region_codes else None
    inputs: list[RegionPlanInput] = []

    for entry in registry.all_regions():
        if entry.provider.lower() != "aws" or not entry.wt_regions:
            continue
        if allowed is not None and entry.region_code not in allowed:
            continue
        wt_region = entry.wt_regions[0][0]
        try:
            forecast = watt_time.get_forecast(wt_region, horizon_hours=horizon_hours)
            points = forecast.get("data", [])
        except WattTimeError:
            continue
        if not points:
            continue
        inputs.append(
            RegionPlanInput(
                cloud_region=entry.region_code,
                watttime_regions=list(entry.wt_regions),
                forecast_points=points,
                mapping_confidence=entry.s_source,
            )
        )
    return inputs
