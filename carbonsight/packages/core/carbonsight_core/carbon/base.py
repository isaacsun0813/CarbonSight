"""Carbon intensity: the provider contract and the shared mixture/job math.

Mirrors ``cloud/`` on the other axis. ``cloud/`` answers *what does this compute
cost and can I get it*; ``carbon/`` answers *how dirty is the power*. Keeping the
two apart means ``watttime/`` stays a plain API client with the source-selection
logic layered above it.

Every provider answers the same questions:

    get_forecast(region)                    raw MOER points for one grid region
    get_moer_lb_per_mwh(wt_regions, s, e)   mixture- and time-weighted MOER
    get_kg_per_hr(job, wt_regions, s, e)    kgCO2/hr for *this* job's power draw
    get_cost_per_hr(job, wt_regions, s, e)  the same, dollarised

``wt_regions`` is the ``[(wt_region, weight)]`` mixture from the mapping registry,
so a cloud region straddling two grids is priced correctly.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from carbonsight_core.estimator.power_model import sample_power_params
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime import LB_TO_KG, mixture_weighted_moer, time_weighted_moer

# Used when a mixture resolves to no usable series at all. Matches the estimator's
# long-standing fallback so the two agree.
FALLBACK_MOER_LB_PER_MWH = 400.0

# How many draws to average when estimating facility power. A single draw samples
# the utilisation and PUE distributions rather than their mean.
POWER_MODEL_SAMPLES = 64


@runtime_checkable
class CarbonIntensityProvider(Protocol):
    """Mixture- and job-aware MOER access."""

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """MOER points ``{point_time, value}`` (lb/MWh) for one grid region.

        ``anchor`` is the window start. Live providers ignore it; synthetic
        curves anchor on it so results do not drift with the calendar.
        """
        ...

    def get_moer_lb_per_mwh(
        self, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        """Mixture-weighted, time-weighted MOER over ``[start, end]``."""
        ...

    def get_kg_per_hr(
        self, job: JobSpec, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        """kgCO2 per wall-clock hour for this job in this grid mixture."""
        ...

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        """``get_kg_per_hr`` dollarised at the social cost of carbon."""
        ...

    def list_regions(self) -> list[str]:
        """Known grid region codes."""
        ...


def as_utc(moment: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than local time."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def facility_mwh_per_hr(
    job: JobSpec,
    *,
    samples: int = POWER_MODEL_SAMPLES,
    seed: int = 0,
) -> float:
    """Mean facility MWh drawn per wall-clock hour: ``P_IT * PUE / 1e6``.

    Averages ``samples`` draws from the same power model the estimator uses, on a
    fixed seed so rankings stay reproducible run to run.

    ``job.gpu_utilization`` (``--gpu-util`` / ``--nvidia-smi``) pins the GPU term
    instead of sampling it, in which case only PUE and CPU load still vary.
    """
    rng = random.Random(seed)
    total_watts = 0.0
    for _ in range(samples):
        params = sample_power_params(job, rng)
        total_watts += params.p_it_w * params.pue
    return total_watts / samples / 1_000_000.0


class ForecastBackedProvider:
    """Shared mixture/job math. Subclasses only supply ``get_forecast``."""

    def get_forecast(
        self,
        region: str,
        *,
        horizon_hours: int = 24,
        anchor: datetime | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_regions(self) -> list[str]:
        raise NotImplementedError

    def get_moer_lb_per_mwh(
        self, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        anchor = as_utc(start)
        series = [
            (points, weight)
            for points, weight in (
                (self.get_forecast(region, anchor=anchor), weight)
                for region, weight in wt_regions
            )
            if points
        ]
        if not series:
            return FALLBACK_MOER_LB_PER_MWH
        blended = mixture_weighted_moer(series, as_utc(start), as_utc(end))
        return blended or FALLBACK_MOER_LB_PER_MWH

    def get_kg_per_hr(
        self, job: JobSpec, wt_regions: list[tuple[str, float]], start: datetime, end: datetime
    ) -> float:
        moer_lb_per_mwh = self.get_moer_lb_per_mwh(wt_regions, start, end)
        return facility_mwh_per_hr(job) * moer_lb_per_mwh * LB_TO_KG

    def get_cost_per_hr(
        self,
        job: JobSpec,
        wt_regions: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> float:
        kg_per_hr = self.get_kg_per_hr(job, wt_regions, start, end)
        return kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight

    def get_moer(
        self,
        region: str,
        *,
        window_hours: float = 1.0,
        window_start: datetime | None = None,
    ) -> float:
        """Single-region convenience: time-weighted MOER over ``[t, t + window]``."""
        start = as_utc(window_start or datetime.now(UTC))
        end = start + timedelta(hours=max(window_hours, 1e-6))
        return time_weighted_moer(self.get_forecast(region, anchor=start), start, end)
