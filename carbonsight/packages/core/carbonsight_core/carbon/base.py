"""Carbon intensity: the provider contract and the shared mixture/job math.

Mirrors ``cloud/`` on the other axis. ``cloud/`` answers *what does this compute
cost and can I get it*; ``carbon/`` answers *how dirty is the power*. Keeping the
two apart means ``watttime/`` stays a plain API client with the source-selection
logic layered above it.

Every provider answers the same questions:

    get_forecast(region)                    raw MOER points for one grid region
    get_moer_lb_per_mwh(wt_regions, s, e)   mixture- and time-weighted MOER

``wt_regions`` is the ``[(wt_region, weight)]`` mixture from the mapping registry,
so a cloud region straddling two grids is priced correctly.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from carbonsight_core.watttime import mixture_weighted_moer


class CarbonProviderError(RuntimeError):
    """A configured carbon source could not be reached or returned nothing usable.

    Part of the provider contract, so it lives beside the Protocol rather than with
    any one implementation. Deliberately fatal: there used to be a
    ``FALLBACK_MOER_LB_PER_MWH = 400.0`` here that stood in whenever a mixture
    resolved to no usable series, which turned "the data source is down" into a
    confident-looking carbon number with nothing marking it invented.
    """


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

    def list_regions(self) -> list[str]:
        """Known grid region codes."""
        ...


def as_utc(moment: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than local time."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


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
        horizon_hours = max(1, math.ceil((as_utc(end) - anchor).total_seconds() / 3600))
        series = [
            (points, weight)
            for points, weight in (
                (
                    self.get_forecast(
                        region,
                        horizon_hours=horizon_hours,
                        anchor=anchor,
                    ),
                    weight,
                )
                for region, weight in wt_regions
            )
            if points
        ]
        named = ", ".join(region for region, _ in wt_regions) or "(none)"
        if sum(max(0.0, weight) for _, weight in wt_regions) <= 0.0:
            # A registry bug, not an outage: mixture_weighted_moer divides by
            # `total_weight or 1.0`, so all-zero weights blend to a flat 0 lb/MWh and
            # rank this region as the cleanest place on earth.
            raise ValueError(f"Mixture [{named}] has no positive weights.")
        if not series:
            # Every provider raises before handing back empty points, so reaching here
            # means the source is answering with nothing usable. There is no number to
            # report; the 400 lb/MWh that used to be returned was invented.
            raise CarbonProviderError(
                f"No forecast points for any grid region in the mixture [{named}]."
            )
        return mixture_weighted_moer(series, as_utc(start), as_utc(end))
