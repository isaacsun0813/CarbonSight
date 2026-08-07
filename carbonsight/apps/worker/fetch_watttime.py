"""Periodic MOER refresher: keeps the shared forecast cache (and Postgres) warm.

This is the piece that makes the central-credential design work. The worker holds
one WattTime login, refreshes every region on a timer, and writes into the shared
``ForecastCache`` plus ``grid_signal_cache``. CLI clients then read through the API
and never need credentials of their own.

**It never writes fabricated data.** The archive version fell back to a synthetic
curve whenever a fetch failed and wrote *that* into the cache and the database. That
is the worst place to fabricate: synthetic MOER is a hash of the region name, so a
single WattTime outage would persist physically meaningless numbers into the shared
store and serve them to every client as if they were real, indefinitely. A failed
region is now logged and skipped — the previous good value stays until it expires,
and a region we have never fetched simply has no entry. Stale-but-real beats
fresh-and-invented, and absent beats both.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import default_grid_regions
from carbonsight_core.watttime import WattTimeClient, get_global_forecast_cache

logger = logging.getLogger("carbonsight.worker.fetch_watttime")

DEFAULT_INTERVAL_SECONDS = 900  # 15 min, matching the forecast cache TTL
DEFAULT_RETENTION_HOURS = 168  # 7 days
DEFAULT_HORIZON_HOURS = 24
MOER_SIGNAL_TYPE = "co2_moer"
DEFAULT_MOER_UNITS = "lbs_co2_per_mwh"

_UPSERT_SQL = """
INSERT INTO grid_signal_cache (wt_region, signal_type, point_time, value, units)
VALUES (:wt_region, :signal_type, CAST(:point_time AS timestamptz), :value, :units)
ON CONFLICT (wt_region, signal_type, point_time)
DO UPDATE SET value = EXCLUDED.value, units = EXCLUDED.units
"""

_PRUNE_SQL = "DELETE FROM grid_signal_cache WHERE point_time < :cutoff"


@dataclass
class RefreshReport:
    """Outcome of one pass over every region."""

    regions_refreshed: int = 0
    regions_failed: int = 0
    points_written: int = 0
    rows_persisted: int = 0
    rows_pruned: int = 0
    failures: list[str] = field(default_factory=list)

    def as_log_line(self) -> str:
        line = (
            f"{self.regions_refreshed} refreshed, {self.regions_failed} failed, "
            f"{self.points_written} points, {self.rows_persisted} rows persisted, "
            f"{self.rows_pruned} pruned"
        )
        return line if not self.failures else f"{line} | failures: {'; '.join(self.failures)}"


def _interval_seconds() -> int:
    return int(os.environ.get("WORKER_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)))


def _retention_hours() -> int:
    return int(os.environ.get("GRID_SIGNAL_RETENTION_HOURS", str(DEFAULT_RETENTION_HOURS)))


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def _rows_for_region(region: str, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        point_time = point.get("point_time")
        if not point_time:
            continue
        try:
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "wt_region": region,
                "signal_type": MOER_SIGNAL_TYPE,
                "point_time": point_time,
                "value": value,
                "units": point.get("units") or DEFAULT_MOER_UNITS,
            }
        )
    return rows


def fetch_all_regions(
    config: Config | None = None,
    *,
    client: WattTimeClient | None = None,
    regions: list[str] | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> RefreshReport:
    """Refresh every region into the shared cache, then persist and prune.

    A region that fails is logged and skipped. Nothing synthetic is ever written.
    """
    cfg = config or Config.from_env()
    if not (cfg.watttime_username and cfg.watttime_password):
        raise RuntimeError(
            "The refresh worker needs WATTTIME_USERNAME and WATTTIME_PASSWORD. It exists "
            "to hold the one shared credential; without it there is nothing to refresh.",
        )

    cache = get_global_forecast_cache(cfg.forecast_cache_ttl_seconds)
    # allow_synthetic=False: a worker that invents data poisons the shared cache.
    watt_time = client or WattTimeClient(cfg, allow_synthetic=False)

    report = RefreshReport()
    rows: list[dict[str, Any]] = []

    for region in regions if regions is not None else default_grid_regions():
        try:
            payload = watt_time.get_forecast(region, horizon_hours=horizon_hours)
            points = WattTimeClient.normalize_forecast_payload(payload)
        except Exception as err:  # noqa: BLE001 - one bad region must not stop the pass
            report.regions_failed += 1
            report.failures.append(f"{region}: {err}")
            logger.warning("refresh failed for %s: %s", region, err)
            continue

        if not points:
            report.regions_failed += 1
            report.failures.append(f"{region}: empty forecast")
            logger.warning("refresh returned no points for %s; keeping previous value", region)
            continue

        cache.set(region, points)
        report.regions_refreshed += 1
        report.points_written += len(points)
        rows.extend(_rows_for_region(region, points))

    report.rows_persisted = persist_rows(rows)
    report.rows_pruned = prune_expired_rows()
    return report


def persist_rows(rows: list[dict[str, Any]], *, database_url: str | None = None) -> int:
    """Upsert forecast points into ``grid_signal_cache``. No DB configured => no-op."""
    dsn = database_url if database_url is not None else _database_url()
    if not dsn or not rows:
        return 0
    try:
        import sqlalchemy as sa
        from sqlalchemy.exc import SQLAlchemyError

        engine = sa.create_engine(dsn)
        with engine.begin() as connection:
            # executemany: the archive issued one round trip per point, which is
            # ~21 regions x 288 points = 6k statements per pass.
            connection.execute(sa.text(_UPSERT_SQL), rows)
        return len(rows)
    except (SQLAlchemyError, ImportError) as err:
        # ImportError: the Postgres driver (psycopg2) is an optional install, so a
        # deployment without it must degrade to cache-only, not crash the worker.
        logger.warning("persist skipped: %s", err)
        return 0


def prune_expired_rows(
    *, database_url: str | None = None, retention_hours: int | None = None
) -> int:
    """Drop rows older than the retention window. No DB configured => no-op."""
    dsn = database_url if database_url is not None else _database_url()
    if not dsn:
        return 0
    hours = retention_hours if retention_hours is not None else _retention_hours()
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    try:
        import sqlalchemy as sa
        from sqlalchemy.exc import SQLAlchemyError

        engine = sa.create_engine(dsn)
        with engine.begin() as connection:
            result = connection.execute(sa.text(_PRUNE_SQL), {"cutoff": cutoff.isoformat()})
        return int(result.rowcount or 0)
    except (SQLAlchemyError, ImportError) as err:
        logger.warning("prune skipped: %s", err)
        return 0


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    interval = _interval_seconds()
    logger.info("WattTime refresh worker starting (every %ss)", interval)
    while True:
        try:
            logger.info("refresh: %s", fetch_all_regions().as_log_line())
        except Exception as err:
            # One bad pass must not kill the loop; the next tick retries.
            logger.exception("refresh pass failed: %s", err)
        time.sleep(interval)


if __name__ == "__main__":
    main()
