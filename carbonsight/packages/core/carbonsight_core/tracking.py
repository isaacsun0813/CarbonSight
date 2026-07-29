"""Persistent run ledger backed by SQLite."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class RunRecord:
    """One tracked run."""

    run_id: str
    created_utc: datetime
    cloud_region: str
    gpu_type: str
    gpu_count: int
    duration_hours: float
    use_spot: bool
    baseline_region: str
    estimated_co2_kg: float
    baseline_co2_kg: float
    estimated_cost_usd: float
    baseline_cost_usd: float
    actual_co2_kg: float | None = None
    actual_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    """Aggregate stats across all runs."""

    n_runs: int
    total_estimated_co2_kg: float
    total_baseline_co2_kg: float
    total_co2_saved_kg: float
    co2_saved_pct: float
    total_estimated_cost_usd: float
    total_baseline_cost_usd: float
    total_cost_saved_usd: float
    cost_saved_pct: float


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    cloud_region TEXT NOT NULL,
    gpu_type TEXT NOT NULL,
    gpu_count INTEGER NOT NULL,
    duration_hours REAL NOT NULL,
    use_spot INTEGER NOT NULL,
    baseline_region TEXT NOT NULL,
    estimated_co2_kg REAL NOT NULL,
    baseline_co2_kg REAL NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    baseline_cost_usd REAL NOT NULL,
    actual_co2_kg REAL,
    actual_cost_usd REAL
)"""

_INSERT = """\
INSERT INTO runs (
    run_id, created_utc, cloud_region, gpu_type, gpu_count,
    duration_hours, use_spot, baseline_region,
    estimated_co2_kg, baseline_co2_kg,
    estimated_cost_usd, baseline_cost_usd,
    actual_co2_kg, actual_cost_usd
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_SELECT_ALL = "SELECT * FROM runs ORDER BY created_utc ASC"
_SELECT_ONE = "SELECT * FROM runs WHERE run_id = ?"


class RunLedger:
    """SQLite-backed run ledger."""

    __slots__ = ("_db_path", "_conn")

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def record(self, rec: RunRecord) -> None:
        """Insert a run record."""
        self._conn.execute(_INSERT, (
            rec.run_id,
            rec.created_utc.isoformat(),
            rec.cloud_region,
            rec.gpu_type,
            rec.gpu_count,
            rec.duration_hours,
            int(rec.use_spot),
            rec.baseline_region,
            rec.estimated_co2_kg,
            rec.baseline_co2_kg,
            rec.estimated_cost_usd,
            rec.baseline_cost_usd,
            rec.actual_co2_kg,
            rec.actual_cost_usd,
        ))
        self._conn.commit()

    def _row_to_record(self, row: tuple) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            created_utc=datetime.fromisoformat(row[1]),
            cloud_region=row[2],
            gpu_type=row[3],
            gpu_count=row[4],
            duration_hours=row[5],
            use_spot=bool(row[6]),
            baseline_region=row[7],
            estimated_co2_kg=row[8],
            baseline_co2_kg=row[9],
            estimated_cost_usd=row[10],
            baseline_cost_usd=row[11],
            actual_co2_kg=row[12],
            actual_cost_usd=row[13],
        )

    def get(self, run_id: str) -> RunRecord | None:
        """Fetch a single run by ID."""
        row = self._conn.execute(_SELECT_ONE, (run_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def all(self) -> list[RunRecord]:
        """All runs, oldest first."""
        return [self._row_to_record(r) for r in self._conn.execute(_SELECT_ALL).fetchall()]

    def summary(self) -> LedgerSummary:
        """Aggregate stats across all recorded runs."""
        runs = self.all()
        if not runs:
            return LedgerSummary(
                n_runs=0,
                total_estimated_co2_kg=0.0, total_baseline_co2_kg=0.0,
                total_co2_saved_kg=0.0, co2_saved_pct=0.0,
                total_estimated_cost_usd=0.0, total_baseline_cost_usd=0.0,
                total_cost_saved_usd=0.0, cost_saved_pct=0.0,
            )
        t_est_co2 = sum(r.estimated_co2_kg for r in runs)
        t_base_co2 = sum(r.baseline_co2_kg for r in runs)
        t_co2_saved = t_base_co2 - t_est_co2
        t_est_cost = sum(r.estimated_cost_usd for r in runs)
        t_base_cost = sum(r.baseline_cost_usd for r in runs)
        t_cost_saved = t_base_cost - t_est_cost
        return LedgerSummary(
            n_runs=len(runs),
            total_estimated_co2_kg=t_est_co2,
            total_baseline_co2_kg=t_base_co2,
            total_co2_saved_kg=t_co2_saved,
            co2_saved_pct=t_co2_saved / t_base_co2 * 100.0 if t_base_co2 else 0.0,
            total_estimated_cost_usd=t_est_cost,
            total_baseline_cost_usd=t_base_cost,
            total_cost_saved_usd=t_cost_saved,
            cost_saved_pct=t_cost_saved / t_base_cost * 100.0 if t_base_cost else 0.0,
        )
