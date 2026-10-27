"""Local SQLite history of spot availability observations.

Observations are things we actually saw: launch results, observed preemptions and (opt-in)
launch-then-terminate probes. They feed the lifetime prior in ``planning.survival``.
This is a local file like the run ledger (``runs.db``); there is no auth beyond the
``StoreAuthenticator`` hook, which is a stub to be replaced later.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from carbonsight_core.planning.survival import (
    SpotObservation,
    VirtualInstanceTracker,
    replay_observations,
)

# Observation sources. "sps" (Spot placement score) is reserved for future use.
SOURCE_PROBE = "probe"
SOURCE_LAUNCH = "launch"
SOURCE_PREEMPTION = "preemption"
SOURCE_MIGRATION = "migration"
SOURCE_SPS = "sps"


class StoreAuthError(Exception):
    """Raised when the store's authenticator rejects access."""


class StoreAuthenticator(Protocol):
    """Hook for future access control on the observation store."""

    def authenticate(self) -> bool: ...


class NoAuth:
    """Default authenticator: local file, always allowed."""

    def authenticate(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class StoredObservation:
    """One row of ``spot_observations``."""

    observed_utc: datetime
    region: str
    instance_type: str
    available: bool
    source: str = SOURCE_PROBE
    run_id: str | None = None
    detail: str = ""
    cloud: str = "aws"

    def to_observation(self) -> SpotObservation:
        return SpotObservation(self.observed_utc, self.available, self.source, self.run_id)


@dataclass(frozen=True, slots=True)
class AvailabilitySummary:
    """Per region + instance type roll-up of recorded observations."""

    region: str
    instance_type: str
    n_observations: int
    n_available: int
    first_seen: datetime
    last_seen: datetime

    @property
    def availability_rate(self) -> float:
        return self.n_available / self.n_observations if self.n_observations else 0.0


@dataclass(frozen=True, slots=True)
class LifetimeSamples:
    """Virtual-instance lifetimes (hours) replayed from stored observations."""

    complete: list[float]  # ended by an observed 1 -> 0 transition
    censored: list[float]  # still up when observation stopped (lower bound)


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS spot_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_utc TEXT NOT NULL,
    cloud TEXT NOT NULL,
    region TEXT NOT NULL,
    instance_type TEXT NOT NULL,
    available INTEGER NOT NULL,
    source TEXT NOT NULL,
    run_id TEXT,
    detail TEXT NOT NULL DEFAULT ''
)"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_spot_obs_region_type_time
ON spot_observations (region, instance_type, observed_utc)"""

_INSERT = """\
INSERT INTO spot_observations (
    observed_utc, cloud, region, instance_type, available, source, run_id, detail
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


def _to_iso(value: datetime) -> str:
    """Normalize to UTC with a fixed format so text ordering matches time ordering."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _from_iso(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class SpotObservationStore:
    """SQLite-backed spot observation history."""

    def __init__(
        self,
        db_path: str | Path,
        authenticator: StoreAuthenticator | None = None,
    ) -> None:
        # Authenticate before touching the filesystem so a rejected caller creates nothing.
        auth = authenticator if authenticator is not None else NoAuth()
        if not auth.authenticate():
            raise StoreAuthError("spot observation store: authentication failed")
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=5.0)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    # -- writes ---------------------------------------------------------------

    @staticmethod
    def _row(obs: StoredObservation) -> tuple:
        return (
            _to_iso(obs.observed_utc),
            obs.cloud,
            obs.region,
            obs.instance_type,
            int(obs.available),
            obs.source,
            obs.run_id,
            obs.detail,
        )

    def add(self, obs: StoredObservation) -> None:
        self._conn.execute(_INSERT, self._row(obs))
        self._conn.commit()

    def add_many(self, observations: Iterable[StoredObservation]) -> None:
        self._conn.executemany(_INSERT, [self._row(o) for o in observations])
        self._conn.commit()

    # -- reads ----------------------------------------------------------------

    def observations(
        self,
        region: str,
        instance_type: str,
        *,
        since: datetime | None = None,
    ) -> list[StoredObservation]:
        sql = (
            "SELECT observed_utc, cloud, region, instance_type, available, source, run_id, detail "
            "FROM spot_observations WHERE region = ? AND instance_type = ?"
        )
        params: list[object] = [region, instance_type]
        if since is not None:
            sql += " AND observed_utc >= ?"
            params.append(_to_iso(since))
        sql += " ORDER BY observed_utc, id"
        return [
            StoredObservation(
                observed_utc=_from_iso(r[0]),
                cloud=r[1],
                region=r[2],
                instance_type=r[3],
                available=bool(r[4]),
                source=r[5],
                run_id=r[6],
                detail=r[7],
            )
            for r in self._conn.execute(sql, params).fetchall()
        ]

    def history_for(
        self,
        regions: Sequence[str],
        instance_type: str,
        *,
        since: datetime | None = None,
    ) -> dict[str, list[SpotObservation]]:
        """Observations per region in the shape ``DynamicJob`` accepts."""
        return {
            region: [o.to_observation() for o in self.observations(region, instance_type, since=since)]
            for region in regions
        }

    def availability_summary(
        self,
        *,
        since: datetime | None = None,
        source: str | None = None,
    ) -> list[AvailabilitySummary]:
        sql = (
            "SELECT region, instance_type, COUNT(*), SUM(available), "
            "MIN(observed_utc), MAX(observed_utc) FROM spot_observations"
        )
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("observed_utc >= ?")
            params.append(_to_iso(since))
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY region, instance_type ORDER BY region, instance_type"
        return [
            AvailabilitySummary(
                region=r[0],
                instance_type=r[1],
                n_observations=r[2],
                n_available=int(r[3] or 0),
                first_seen=_from_iso(r[4]),
                last_seen=_from_iso(r[5]),
            )
            for r in self._conn.execute(sql, params).fetchall()
        ]

    def lifetime_samples(
        self,
        region: str,
        instance_type: str,
        *,
        max_gap_hours: float | None = None,
        since: datetime | None = None,
    ) -> LifetimeSamples:
        """Replay stored observations into virtual-instance lifetimes (hours)."""
        tracker = VirtualInstanceTracker(region=region)
        stored = self.observations(region, instance_type, since=since)
        replay_observations(
            tracker,
            [o.to_observation() for o in stored],
            max_gap_hours=max_gap_hours,
        )
        return LifetimeSamples(
            complete=list(tracker.model.complete_samples),
            censored=list(tracker.model.censored_samples),
        )

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SpotObservationStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
