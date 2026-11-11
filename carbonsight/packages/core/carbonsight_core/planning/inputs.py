from __future__ import annotations

import math

from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.planning.types import PlanConstraints, RegionPlanInput
from carbonsight_core.watttime import WattTimeClient, WattTimeError


def build_region_plan_inputs(
    registry: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    constraints: PlanConstraints,
    region_codes: list[str] | frozenset[str] | None = None,
) -> list[RegionPlanInput]:
    """
    Fetch MOER forecast per AWS region for dynamic/static planning.

    Combines mixture mappings using the same weighted-MOER calculation as
    ``JobCarbonEstimator``. Skips regions without mapping or on WattTime errors.
    """
    horizon_hours = math.ceil(
        (constraints.deadline_utc - constraints.now_utc).total_seconds() / 3600.0
        + job.duration_hours
    )
    horizon_hours = max(1, min(horizon_hours, 72))
    allowed = frozenset(region_codes) if region_codes else None
    inputs: list[RegionPlanInput] = []

    for entry in registry.iter_plannable_aws_regions(allowed):
        try:
            forecasts = [
                (
                    weight,
                    watt_time.get_forecast(wt_region, horizon_hours=horizon_hours).get("data", []),
                )
                for wt_region, weight in entry.wt_regions
            ]
        except WattTimeError:
            continue
        point_times = {
            point["point_time"]
            for _, points in forecasts
            for point in points
        }
        if not point_times:
            continue
        values_by_region = [
            (weight, {point["point_time"]: float(point.get("value", 0)) for point in points})
            for weight, points in forecasts
        ]
        points = [
            {
                "point_time": point_time,
                "value": sum(
                    weight * values.get(point_time, 400.0)
                    for weight, values in values_by_region
                ),
            }
            for point_time in point_times
        ]
        inputs.append(
            RegionPlanInput(
                cloud_region=entry.region_code,
                watttime_regions=list(entry.wt_regions),
                forecast_points=sorted(points, key=lambda p: p["point_time"]),
                mapping_confidence=entry.s_source,
            )
        )
    return inputs
