"""
GET /v1/carbon/forecast -> central WattTime cache so everyone can schedule without personal creds.

Design P0:
- Server holds ONE WattTime credential (WATTTIME_USERNAME/PASSWORD in API env)
- In-memory ForecastCache (TTL 15min) + optional Postgres grid_signal_cache for persistence
- Clients call this endpoint via ApiCarbonProvider (no WattTime creds needed)

This endpoint mirrors WattTime /v3/forecast but serves from central cache.
Postgres flow (production):
- If DATABASE_URL set, query grid_signal_cache where wt_region and point_time between start/end (or latest horizon)
- If DB miss and WattTime creds exist, fetch live and upsert (ON CONFLICT DO UPDATE)
- Retention handled by worker (DELETE older than 7 days)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query

from carbonsight_core.config import Config
from carbonsight_core.watttime import WattTimeClient
from carbonsight_core.watttime.cache import ForecastCache, _time_weighted_moer

router = APIRouter()

# Global in-memory cache per process — holds single credential's forecasts
_forecast_cache: ForecastCache | None = None


def _get_cache() -> ForecastCache | None:
    global _forecast_cache
    if _forecast_cache is not None:
        return _forecast_cache
    cfg = Config.from_env()
    if not cfg.watttime_username or not cfg.watttime_password:
        return None
    try:
        client = WattTimeClient(cfg)
        _forecast_cache = ForecastCache(client, ttl_seconds=cfg.cache_ttl_min * 60)
        return _forecast_cache
    except Exception:
        return None


def _get_database_url() -> str | None:
    cfg = Config.from_env()
    # Config already merges DATABASE_URL / CARBONSIGHT_DB_URL
    if cfg.database_url:
        return cfg.database_url
    return os.environ.get("DATABASE_URL") or os.environ.get("CARBONSIGHT_DB_URL") or None


def _parse_point_time_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _query_db_points(
    db_url: str,
    wt_region: str,
    signal_type: str = "co2_moer",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    horizon_hours: int = 72,
) -> list[dict[str, Any]]:
    """
    Query grid_signal_cache.
    - If start_dt and end_dt provided: WHERE point_time BETWEEN start and end
    - Else: latest horizon from now to now+horison_hours, with fallback to most recent rows
    Returns list of dicts {"point_time": iso, "value": float, "units": str}
    Handles both psycopg (v3), psycopg2, and sqlalchemy.
    """
    points: list[dict[str, Any]] = []

    # For horizon queries without explicit window:
    # - If start/end provided: BETWEEN start and end
    # - Else: BETWEEN now-1h and now+horizon (buffer to include points just inserted)
    now = datetime.now(UTC)
    if start_dt is not None and end_dt is not None:
        q_start = start_dt
        q_end = end_dt
    else:
        # Buffer 1h back to capture forecasts inserted moments ago
        q_start = start_dt or (now - timedelta(hours=1))
        q_end = end_dt or (now + timedelta(hours=horizon_hours))

    # Try psycopg v3
    try:
        import psycopg  # type: ignore

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT point_time, value, units
                    FROM grid_signal_cache
                    WHERE wt_region = %s AND signal_type = %s AND point_time BETWEEN %s AND %s
                    ORDER BY point_time ASC
                    """,
                    (wt_region, signal_type, q_start, q_end),
                )
                rows = cur.fetchall()
                if not rows and (start_dt is None or end_dt is None):
                    # Fallback: most recent rows for this region (may be slightly stale)
                    limit = max(50, horizon_hours * 12)
                    cur.execute(
                        """
                        SELECT point_time, value, units
                        FROM grid_signal_cache
                        WHERE wt_region = %s AND signal_type = %s
                        ORDER BY point_time DESC
                        LIMIT %s
                        """,
                        (wt_region, signal_type, limit),
                    )
                    rows = cur.fetchall()
                    rows = list(reversed(rows))

                for r in rows:
                    pt = r[0]
                    val = r[1]
                    units = r[2] if len(r) > 2 else "lbs_co2_per_mwh"
                    iso = pt.isoformat() if isinstance(pt, datetime) else str(pt)
                    points.append(
                        {
                            "point_time": iso,
                            "value": float(val),
                            "units": units or "lbs_co2_per_mwh",
                        }
                    )
        return points
    except ImportError:
        pass
    except Exception:
        pass

    # Try psycopg2
    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT point_time, value, units
                FROM grid_signal_cache
                WHERE wt_region = %s AND signal_type = %s AND point_time BETWEEN %s AND %s
                ORDER BY point_time ASC
                """,
                (wt_region, signal_type, q_start, q_end),
            )
            rows = cur.fetchall()
            if not rows and (start_dt is None or end_dt is None):
                limit = max(50, horizon_hours * 12)
                cur.execute(
                    """
                    SELECT point_time, value, units
                    FROM grid_signal_cache
                    WHERE wt_region = %s AND signal_type = %s
                    ORDER BY point_time DESC
                    LIMIT %s
                    """,
                    (wt_region, signal_type, limit),
                )
                rows = cur.fetchall()
                rows = list(reversed(rows))

            for r in rows:
                pt, val, units = r[0], r[1], r[2] if len(r) > 2 else "lbs_co2_per_mwh"
                iso = pt.isoformat() if isinstance(pt, datetime) else str(pt)
                points.append(
                    {
                        "point_time": iso,
                        "value": float(val),
                        "units": units or "lbs_co2_per_mwh",
                    }
                )
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
        return points
    except ImportError:
        pass
    except Exception:
        pass

    # Try sqlalchemy
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, future=True)
        with engine.connect() as conn:
            if start_dt is not None and end_dt is not None:
                result = conn.execute(
                    text(
                        """
                        SELECT point_time, value, units
                        FROM grid_signal_cache
                        WHERE wt_region = :wt_region AND signal_type = :signal_type
                          AND point_time BETWEEN :start AND :end
                        ORDER BY point_time ASC
                        """
                    ),
                    {
                        "wt_region": wt_region,
                        "signal_type": signal_type,
                        "start": q_start,
                        "end": q_end,
                    },
                )
                rows = result.fetchall()
            else:
                result = conn.execute(
                    text(
                        """
                        SELECT point_time, value, units
                        FROM grid_signal_cache
                        WHERE wt_region = :wt_region AND signal_type = :signal_type
                          AND point_time BETWEEN :start AND :end
                        ORDER BY point_time ASC
                        """
                    ),
                    {
                        "wt_region": wt_region,
                        "signal_type": signal_type,
                        "start": q_start,
                        "end": q_end,
                    },
                )
                rows = result.fetchall()
                if not rows:
                    limit = max(50, horizon_hours * 12)
                    result = conn.execute(
                        text(
                            """
                            SELECT point_time, value, units
                            FROM grid_signal_cache
                            WHERE wt_region = :wt_region AND signal_type = :signal_type
                            ORDER BY point_time DESC
                            LIMIT :limit
                            """
                        ),
                        {
                            "wt_region": wt_region,
                            "signal_type": signal_type,
                            "limit": limit,
                        },
                    )
                    rows = result.fetchall()
                    rows = list(reversed(rows))

            for r in rows:
                pt, val, units = r[0], r[1], r[2]
                iso = pt.isoformat() if isinstance(pt, datetime) else str(pt)
                points.append(
                    {
                        "point_time": iso,
                        "value": float(val),
                        "units": units or "lbs_co2_per_mwh",
                    }
                )
        return points
    except ImportError:
        return []
    except Exception:
        # Swallow errors -> fallback
        return []

    return points


def _upsert_points_to_db(
    db_url: str,
    wt_region: str,
    points: list[dict[str, Any]],
    signal_type: str = "co2_moer",
) -> int:
    """
    Upsert live-fetched points into grid_signal_cache for future DB hits.
    Supports psycopg, psycopg2, sqlalchemy.
    Returns number of points attempted.
    """
    if not points:
        return 0

    # Reuse logic from worker (simplified copy) to avoid circular import
    normalized = []
    for p in points:
        try:
            pt_str = p.get("point_time", "")
            if not pt_str:
                continue
            dt = _parse_point_time_to_dt(pt_str)
            val = float(p["value"])
            units = p.get("units") or "lbs_co2_per_mwh"
            period = p.get("data_point_period_seconds")
            model_date = p.get("model_date")
            if isinstance(model_date, str) and model_date:
                try:
                    from datetime import date

                    if "T" in model_date:
                        model_date = _parse_point_time_to_dt(model_date).date()
                    else:
                        model_date = date.fromisoformat(model_date[:10])
                except Exception:
                    model_date = None
            normalized.append(
                {
                    "dt": dt,
                    "value": val,
                    "units": units,
                    "period": period,
                    "model_date": model_date,
                }
            )
        except Exception:
            continue

    if not normalized:
        return 0

    # psycopg v3
    try:
        import psycopg  # type: ignore

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for n in normalized:
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
            conn.commit()
        return len(normalized)
    except ImportError:
        pass
    except Exception:
        pass

    # psycopg2
    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            for n in normalized:
                cur.execute(
                    """
                    INSERT INTO grid_signal_cache
                      (wt_region, signal_type, point_time, value, units, data_point_period_seconds, model_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (wt_region, signal_type, point_time)
                    DO UPDATE SET value = EXCLUDED.value,
                                  units = EXCLUDED.units
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
    except Exception:
        pass

    # sqlalchemy
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, future=True)
        with engine.begin() as conn:
            for n in normalized:
                conn.execute(
                    text(
                        """
                        INSERT INTO grid_signal_cache
                          (wt_region, signal_type, point_time, value, units, data_point_period_seconds, model_date)
                        VALUES (:wt_region, :signal_type, :point_time, :value, :units, :period, :model_date)
                        ON CONFLICT (wt_region, signal_type, point_time)
                        DO UPDATE SET value = EXCLUDED.value,
                                      units = EXCLUDED.units
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
        return len(normalized)
    except ImportError:
        return 0
    except Exception:
        return 0


@router.get("/carbon/forecast")
def get_carbon_forecast(
    wt_region: str = Query(..., description="WattTime region code, e.g. CAISO_NORTH"),
    horizon_hours: int = Query(72, ge=1, le=168, description="Forecast horizon hours, max 168"),
    signal_type: str = Query("co2_moer", description="Signal type"),
) -> dict[str, Any]:
    """
    Central proxy for WattTime forecast.

    Priority:
    1. Postgres grid_signal_cache if DATABASE_URL set (query by wt_region and point_time horizon)
    2. In-memory ForecastCache (TTL 15min)
    3. Live WattTime fetch if creds present (and upsert to DB if DATABASE_URL set)
    4. Empty fallback -> recommendations will use synthetic

    Response shape: {"wt_region":..., "horizon_hours":..., "points": [{"point_time":..., "value":...}], "source": "db"|"cache"|"live"|"empty"}
    """
    cfg = Config.from_env()
    db_url = _get_database_url()

    # 1) Try Postgres first if DATABASE_URL set
    if db_url:
        try:
            db_points = _query_db_points(
                db_url, wt_region, signal_type, start_dt=None, end_dt=None, horizon_hours=horizon_hours
            )
            if db_points:
                return {
                    "wt_region": wt_region,
                    "signal_type": signal_type,
                    "horizon_hours": horizon_hours,
                    "points": db_points,
                    "data": db_points,  # backward compat for ApiCarbonProvider expecting "data" or "points"
                    "source": "db",
                    "cached": True,
                }
        except Exception:
            # Fall through to cache/live
            pass

    cache = _get_cache()

    # 2) Try in-memory cache
    if cache is not None:
        try:
            points = cache.get_forecast_points(wt_region, horizon_hours=horizon_hours)
            if points:
                # If DB configured but was miss, upsert cache points for persistence
                if db_url:
                    try:
                        _upsert_points_to_db(db_url, wt_region, points, signal_type)
                    except Exception:
                        pass
                return {
                    "wt_region": wt_region,
                    "signal_type": signal_type,
                    "horizon_hours": horizon_hours,
                    "points": points,
                    "data": points,
                    "source": "cache" if cache.fetch_count > 0 else "live",
                    "cached": True,
                }
        except Exception:
            pass

    # 3) Live fetch if creds present, with DB upsert
    if cfg.watttime_username and cfg.watttime_password:
        try:
            client = WattTimeClient(cfg)
            data = client.get_forecast(wt_region, signal_type=signal_type, horizon_hours=horizon_hours)
            pts = data.get("data", []) or []
            if db_url and pts:
                try:
                    _upsert_points_to_db(db_url, wt_region, pts, signal_type)
                except Exception:
                    pass
            return {
                "wt_region": wt_region,
                "signal_type": signal_type,
                "horizon_hours": horizon_hours,
                "points": pts,
                "data": pts,
                "source": "live",
                "cached": False,
            }
        except Exception as e:
            return {
                "wt_region": wt_region,
                "signal_type": signal_type,
                "horizon_hours": horizon_hours,
                "points": [],
                "data": [],
                "source": "error",
                "error": str(e),
            }

    # 4) No creds, no cache, no DB — empty, client will use synthetic fallback
    return {
        "wt_region": wt_region,
        "signal_type": signal_type,
        "horizon_hours": horizon_hours,
        "points": [],
        "data": [],
        "source": "empty",
        "message": "No WattTime credentials on server and no cache. Use BYOK mode or set WATTTIME_USERNAME on server.",
    }


@router.get("/carbon/forecast/window")
def get_carbon_forecast_window(
    wt_region: str = Query(..., description="WattTime region"),
    start: str = Query(..., description="ISO start datetime UTC, e.g. 2026-01-01T00:00:00Z"),
    end: str = Query(..., description="ISO end datetime UTC"),
    horizon_hours: int = Query(72, ge=1, le=168),
    signal_type: str = Query("co2_moer", description="Signal type"),
) -> dict[str, Any]:
    """Time-weighted MOER for window [start,end] using central cache (DB -> memory -> synthetic)."""

    def _parse(ts: str) -> datetime:
        # Accept Z suffix
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    try:
        start_dt = _parse(start)
        end_dt = _parse(end)
    except Exception as e:
        return {"error": f"Invalid datetime: {e}", "points": [], "moer_lb_per_mwh": 400.0}

    cfg = Config.from_env()
    db_url = _get_database_url()

    # 1) Try DB range query
    if db_url:
        try:
            db_points = _query_db_points(
                db_url, wt_region, signal_type, start_dt=start_dt, end_dt=end_dt, horizon_hours=horizon_hours
            )
            if db_points:
                try:
                    moer = _time_weighted_moer(db_points, start_dt, end_dt)
                except Exception:
                    # fallback to first point if weighting fails
                    moer = float(db_points[0].get("value", 400.0)) if db_points else 400.0
                return {
                    "wt_region": wt_region,
                    "signal_type": signal_type,
                    "start": start,
                    "end": end,
                    "moer_lb_per_mwh": moer,
                    "source": "db",
                    "points": db_points,
                    "point_count": len(db_points),
                }
        except Exception:
            pass

    # 2) In-memory cache path
    cache = _get_cache()
    if cache is None:
        return {
            "wt_region": wt_region,
            "signal_type": signal_type,
            "start": start,
            "end": end,
            "moer_lb_per_mwh": 400.0,
            "source": "synthetic",
        }

    try:
        moer = cache.get_time_weighted_moer(
            [(wt_region, 1.0)], start_dt, end_dt, horizon_hours=horizon_hours
        )
    except Exception:
        moer = 400.0

    # If DB configured but was miss, optionally upsert cache points for persistence (best-effort)
    if db_url and cache is not None:
        try:
            pts = cache.get_forecast_points(wt_region, horizon_hours=horizon_hours)
            if pts:
                _upsert_points_to_db(db_url, wt_region, pts, signal_type)
        except Exception:
            pass

    return {
        "wt_region": wt_region,
        "signal_type": signal_type,
        "start": start,
        "end": end,
        "moer_lb_per_mwh": moer,
        "source": "cache",
    }
