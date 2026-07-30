"""
Fetcher worker: single writer that holds ONE WattTime credential and populates grid_signal_cache.

Usage:
  WATTTIME_USERNAME=... WATTTIME_PASSWORD=... python -m fetch_watttime --once
  WATTTIME_USERNAME=... python -m fetch_watttime --loop --interval-min 15

Design:
- Loads seed_registry.json, dedupes wt_regions
- For each wt_region, calls client.get_forecast(region, horizon=72)
- Upserts points into Postgres grid_signal_cache if DATABASE_URL set, else prints/log
- Rate limit safe: sequential calls, token cached 25min, concurrency 1
- Retention: DELETE older than 7 days (configurable via GRID_CACHE_RETENTION_DAYS)

For P0 validation without Postgres, runs in-memory and logs fetch counts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure core package on path when run as script
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "apps" / "api"))

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.watttime import WattTimeClient


def _registry_path() -> Path:
    # Default location
    return ROOT / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json"


def _load_registry(path: Path) -> set[str]:
    reg = Registry()
    reg.load_json(path)
    wt_regions = set()
    for entry in reg.all_regions():
        for wt_region, _weight in entry.wt_regions:
            wt_regions.add(wt_region)
    return wt_regions


def _parse_point_dt(point_time: str) -> datetime:
    return datetime.fromisoformat(point_time.replace("Z", "+00:00"))


def _upsert_to_postgres(
    wt_region: str,
    points: list[dict],
    db_url: str,
    signal_type: str = "co2_moer",
) -> int:
    """
    Upsert points into grid_signal_cache.
    Returns number of rows attempted.
    Tries psycopg (v3) -> psycopg2 -> sqlalchemy, else no-op with log.
    Handles optional fields: units, data_point_period_seconds, model_date.
    """
    if not points:
        return 0

    # Normalize points to tuples for insertion
    normalized = []
    for p in points:
        try:
            pt_str = p.get("point_time") or p.get("timestamp") or ""
            if not pt_str:
                continue
            dt = _parse_point_dt(pt_str)
            val = float(p["value"])
            units = p.get("units") or "lbs_co2_per_mwh"
            period = p.get("data_point_period_seconds") or p.get("freq") or None
            model_date = p.get("model_date")
            # model_date may be string YYYY-MM-DD
            if isinstance(model_date, str):
                try:
                    # keep as date string; DB expects DATE, driver will parse
                    from datetime import date

                    # try iso date
                    if "T" in model_date:
                        model_date_parsed = _parse_point_dt(model_date).date()
                    else:
                        model_date_parsed = date.fromisoformat(model_date[:10])
                except Exception:
                    model_date_parsed = None
            else:
                model_date_parsed = model_date
            normalized.append(
                {
                    "dt": dt,
                    "value": val,
                    "units": units,
                    "period": period,
                    "model_date": model_date_parsed,
                }
            )
        except Exception as e:
            print(f"  skip point {p} due to {e}")
            continue

    if not normalized:
        return 0

    # Attempt psycopg (v3)
    try:
        import psycopg  # type: ignore

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for n in normalized:
                    try:
                        cur.execute(
                            """
                            INSERT INTO grid_signal_cache
                              (wt_region, signal_type, point_time, value, units, data_point_period_seconds, model_date)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (wt_region, signal_type, point_time)
                            DO UPDATE SET value = EXCLUDED.value,
                                          units = EXCLUDED.units,
                                          data_point_period_seconds = COALESCE(EXCLUDED.data_point_period_seconds, grid_signal_cache.data_point_period_seconds),
                                          model_date = COALESCE(EXCLUDED.model_date, grid_signal_cache.model_date)
                            """,
                            (
                                wt_region,
                                signal_type,
                                n["dt"],
                                n["value"],
                                n["units"],
                                n["period"],
                                n["model_date"],
                            ),
                        )
                    except Exception as e:
                        print(f"  skip upsert {n['dt']} due to {e}")
                        continue
            conn.commit()
        return len(normalized)
    except ImportError:
        pass
    except Exception as e:
        # If it's not import error but connection error, still try other drivers
        print(f"psycopg upsert failed, trying psycopg2/sqlalchemy: {e}")

    # Attempt psycopg2
    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            for n in normalized:
                try:
                    cur.execute(
                        """
                        INSERT INTO grid_signal_cache
                          (wt_region, signal_type, point_time, value, units, data_point_period_seconds, model_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (wt_region, signal_type, point_time)
                        DO UPDATE SET value = EXCLUDED.value,
                                      units = EXCLUDED.units,
                                      data_point_period_seconds = COALESCE(EXCLUDED.data_point_period_seconds, grid_signal_cache.data_point_period_seconds),
                                      model_date = COALESCE(EXCLUDED.model_date, grid_signal_cache.model_date)
                        """,
                        (
                            wt_region,
                            signal_type,
                            n["dt"],
                            n["value"],
                            n["units"],
                            n["period"],
                            n["model_date"],
                        ),
                    )
                except Exception as e:
                    print(f"  skip upsert {n['dt']} due to {e}")
                    continue
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
        return len(normalized)
    except ImportError:
        pass
    except Exception as e:
        print(f"psycopg2 upsert failed, trying sqlalchemy: {e}")

    # Attempt sqlalchemy (sync engine)
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, future=True)
        with engine.begin() as conn:
            for n in normalized:
                try:
                    conn.execute(
                        text(
                            """
                            INSERT INTO grid_signal_cache
                              (wt_region, signal_type, point_time, value, units, data_point_period_seconds, model_date)
                            VALUES (:wt_region, :signal_type, :point_time, :value, :units, :period, :model_date)
                            ON CONFLICT (wt_region, signal_type, point_time)
                            DO UPDATE SET value = EXCLUDED.value,
                                          units = EXCLUDED.units,
                                          data_point_period_seconds = COALESCE(EXCLUDED.data_point_period_seconds, grid_signal_cache.data_point_period_seconds),
                                          model_date = COALESCE(EXCLUDED.model_date, grid_signal_cache.model_date)
                            """
                        ),
                        {
                            "wt_region": wt_region,
                            "signal_type": signal_type,
                            "point_time": n["dt"],
                            "value": n["value"],
                            "units": n["units"],
                            "period": n["period"],
                            "model_date": n["model_date"],
                        },
                    )
                except Exception as e:
                    print(f"  skip point {n['dt']} due to {e}")
                    continue
        return len(normalized)
    except ImportError:
        print("DB upsert skipped: no psycopg / psycopg2 / sqlalchemy installed")
        return 0
    except Exception as e:
        print(f"DB upsert failed (sqlalchemy path): {e}")
        return 0


def _cleanup_old_rows(db_url: str, retention_days: int = 7) -> int:
    """
    DELETE older than retention_days from grid_signal_cache.
    Returns number of rows deleted (if driver reports) or 0.
    Supports psycopg, psycopg2, sqlalchemy.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0

    # psycopg
    try:
        import psycopg  # type: ignore

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM grid_signal_cache WHERE point_time < %s",
                    (cutoff,),
                )
                deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
            conn.commit()
        print(f"Retention cleanup: deleted {deleted} rows older than {retention_days}d (psycopg)")
        return deleted
    except ImportError:
        pass
    except Exception as e:
        print(f"Retention cleanup psycopg failed: {e}, trying psycopg2/sqlalchemy")

    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM grid_signal_cache WHERE point_time < %s",
                (cutoff,),
            )
            deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
        print(f"Retention cleanup: deleted {deleted} rows older than {retention_days}d (psycopg2)")
        return deleted
    except ImportError:
        pass
    except Exception as e:
        print(f"Retention cleanup psycopg2 failed: {e}, trying sqlalchemy")

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, future=True)
        with engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM grid_signal_cache WHERE point_time < :cutoff"),
                {"cutoff": cutoff},
            )
            deleted = result.rowcount if hasattr(result, "rowcount") else 0
        print(f"Retention cleanup: deleted {deleted} rows older than {retention_days}d (sqlalchemy)")
        return deleted
    except ImportError:
        print("Retention cleanup skipped: no DB driver installed")
        return 0
    except Exception as e:
        print(f"Retention cleanup failed: {e}")
        return 0


def fetch_once(horizon_hours: int = 72, db_url: str | None = None) -> dict:
    cfg = Config.from_env()
    if not cfg.watttime_username or not cfg.watttime_password:
        print(
            "ERROR: WATTTIME_USERNAME/PASSWORD not set — cannot fetch. "
            "This worker must hold ONE credential server-side."
        )
        return {"error": "no creds"}

    reg_path = _registry_path()
    if not reg_path.exists():
        print(f"Registry not found at {reg_path}")
        return {"error": "no registry"}

    wt_regions = _load_registry(reg_path)
    print(f"Found {len(wt_regions)} distinct WT regions from registry: {sorted(wt_regions)[:10]}...")

    client = WattTimeClient(cfg)
    results = {}
    for wt_region in sorted(wt_regions):
        try:
            print(f"Fetching {wt_region} horizon={horizon_hours}h...")
            data = client.get_forecast(wt_region, horizon_hours=horizon_hours)
            points = data.get("data", []) or []
            print(f"  got {len(points)} points, first value={points[0]['value'] if points else 'none'}")
            results[wt_region] = len(points)
            if db_url:
                n = _upsert_to_postgres(wt_region, points, db_url)
                print(f"  upserted {n} rows to grid_signal_cache")
        except Exception as e:
            print(f"  FAILED {wt_region}: {e}")
            results[wt_region] = f"error: {e}"

    if db_url:
        retention_days = cfg.grid_cache_retention_days or 7
        try:
            _cleanup_old_rows(db_url, retention_days=retention_days)
        except Exception as e:
            print(f"Retention cleanup error: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="WattTime central fetcher worker")
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    parser.add_argument("--loop", action="store_true", help="Loop forever every interval")
    parser.add_argument("--interval-min", type=int, default=15, help="Interval minutes for loop mode")
    parser.add_argument("--horizon-hours", type=int, default=72, help="Forecast horizon")
    parser.add_argument("--retention-days", type=int, default=None, help="Override retention days (default from env GRID_CACHE_RETENTION_DAYS=7)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL") or os.environ.get("CARBONSIGHT_DB_URL")
    cfg = Config.from_env()
    if args.retention_days is not None:
        # allow CLI override via env for fetch_once
        os.environ["GRID_CACHE_RETENTION_DAYS"] = str(args.retention_days)

    if args.once or not args.loop:
        print(f"=== Fetch once horizon={args.horizon_hours}h db_url={'set' if db_url else 'not set'} ===")
        res = fetch_once(horizon_hours=args.horizon_hours, db_url=db_url)
        print(f"Done: {res}")
    else:
        print(f"=== Loop mode interval={args.interval_min}min horizon={args.horizon_hours}h ===")
        while True:
            try:
                res = fetch_once(horizon_hours=args.horizon_hours, db_url=db_url)
                print(f"[{datetime.now(UTC).isoformat()}] fetched {len(res)} regions")
            except Exception as e:
                print(f"Loop iteration failed: {e}")
            print(f"Sleeping {args.interval_min}min...")
            time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
