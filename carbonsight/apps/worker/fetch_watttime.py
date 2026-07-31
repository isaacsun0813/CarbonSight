"""
Periodic job: fetch MOER for all known regions into ForecastCache and optional DB.

Table: grid_signal_cache (infra/schemas.sql). Retention via DELETE of old rows.
Docker-compose runs this worker alongside the API.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS
from carbonsight_core.watttime.cache import get_global_forecast_cache
from carbonsight_core.watttime.client import WattTimeClient, build_synthetic_forecast

logger = logging.getLogger("carbonsight.worker.fetch_watttime")

DEFAULT_INTERVAL_SECONDS = int(os.environ.get("WORKER_INTERVAL_SECONDS", "900"))  # 15 min
RETENTION_HOURS = int(os.environ.get("GRID_SIGNAL_RETENTION_HOURS", "168"))  # 7 days


def fetch_all_regions(config: Config | None = None) -> dict[str, int]:
    """Fetch forecasts for all default regions into the process cache (+ optional DB)."""
    cfg = config or Config.from_env()
    cache = get_global_forecast_cache(cfg.cache_ttl_seconds)
    client = WattTimeClient(cfg, allow_synthetic=True)
    counts = {"regions": 0, "points": 0, "synthetic": 0}

    rows_for_db: list[dict[str, Any]] = []
    for region in DEFAULT_CARBON_REGIONS:
        try:
            payload = client.get_forecast(region, horizon_hours=24)
            pts = WattTimeClient.normalize_forecast_payload(payload)
            if not pts:
                pts = build_synthetic_forecast(region)
                counts["synthetic"] += 1
            cache.set(region, pts)
            counts["regions"] += 1
            counts["points"] += len(pts)
            for p in pts:
                rows_for_db.append(
                    {
                        "wt_region": region,
                        "signal_type": "co2_moer",
                        "point_time": p.get("point_time"),
                        "value": float(p.get("value", 0)),
                        "units": p.get("units") or "lbs_co2_per_mwh",
                    }
                )
        except Exception as exc:
            logger.warning("fetch failed for %s: %s", region, exc)
            pts = build_synthetic_forecast(region)
            cache.set(region, pts)
            counts["regions"] += 1
            counts["synthetic"] += 1
            counts["points"] += len(pts)

    _maybe_write_db(rows_for_db)
    _maybe_prune_db()
    return counts


def _maybe_write_db(rows: list[dict[str, Any]]) -> None:
    """Best-effort insert into grid_signal_cache when DATABASE_URL is set."""
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn or not rows:
        return
    try:
        import sqlalchemy as sa

        engine = sa.create_engine(dsn)
        with engine.begin() as conn:
            for r in rows:
                conn.execute(
                    sa.text(
                        """
                        INSERT INTO grid_signal_cache
                            (wt_region, signal_type, point_time, value, units)
                        VALUES
                            (:wt_region, :signal_type, CAST(:point_time AS timestamptz), :value, :units)
                        ON CONFLICT (wt_region, signal_type, point_time)
                        DO UPDATE SET value = EXCLUDED.value, units = EXCLUDED.units
                        """
                    ),
                    r,
                )
    except Exception as exc:
        logger.warning("DB write skipped/failed: %s", exc)


def _maybe_prune_db() -> None:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        return
    cutoff = datetime.now(UTC) - timedelta(hours=RETENTION_HOURS)
    try:
        import sqlalchemy as sa

        engine = sa.create_engine(dsn)
        with engine.begin() as conn:
            conn.execute(
                sa.text("DELETE FROM grid_signal_cache WHERE point_time < :cutoff"),
                {"cutoff": cutoff.isoformat()},
            )
    except Exception as exc:
        logger.warning("DB prune skipped/failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    interval = DEFAULT_INTERVAL_SECONDS
    logger.info("WattTime fetch worker starting (interval=%ss)", interval)
    while True:
        stats = fetch_all_regions()
        logger.info("fetch complete: %s", stats)
        time.sleep(interval)


if __name__ == "__main__":
    main()
