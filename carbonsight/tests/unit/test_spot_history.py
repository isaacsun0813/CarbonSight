"""SQLite spot observation store: round trip, analytics, auth stub, lifetime replay."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from carbonsight_core.spot_history import (
    NoAuth,
    SpotObservationStore,
    StoreAuthError,
    StoredObservation,
)

UTC = UTC
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def _obs(
    hours: float,
    available: bool,
    *,
    region: str = "us-west-2",
    instance_type: str = "p4d.24xlarge",
    source: str = "probe",
    run_id: str | None = None,
) -> StoredObservation:
    return StoredObservation(
        observed_utc=T0 + timedelta(hours=hours),
        region=region,
        instance_type=instance_type,
        available=available,
        source=source,
        run_id=run_id,
    )


class TestStoreRoundTrip:
    def test_add_and_read_back_in_time_order(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "sub" / "runs.db") as store:
            store.add(_obs(2, False, source="preemption", run_id="r1"))
            store.add_many([_obs(0, True, source="launch", run_id="r1"), _obs(1, True)])
            rows = store.observations("us-west-2", "p4d.24xlarge")
        assert [(r.available, r.source) for r in rows] == [
            (True, "launch"),
            (True, "probe"),
            (False, "preemption"),
        ]
        assert rows[0].run_id == "r1" and rows[0].observed_utc == T0
        assert rows[0].observed_utc.tzinfo is not None

    def test_filters_by_region_type_and_since(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add_many(
                [
                    _obs(0, True),
                    _obs(5, True),
                    _obs(1, True, region="us-east-1"),
                    _obs(1, True, instance_type="p5.48xlarge"),
                ]
            )
            assert len(store.observations("us-west-2", "p4d.24xlarge")) == 2
            recent = store.observations("us-west-2", "p4d.24xlarge", since=T0 + timedelta(hours=1))
            assert [r.observed_utc for r in recent] == [T0 + timedelta(hours=5)]

    def test_persists_across_connections(self, tmp_path) -> None:
        path = tmp_path / "runs.db"
        with SpotObservationStore(path) as store:
            store.add(_obs(0, True))
        with SpotObservationStore(path) as store:
            assert len(store.observations("us-west-2", "p4d.24xlarge")) == 1

    def test_history_for_returns_planner_observations(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add(_obs(0, True, source="launch", run_id="r1"))
            history = store.history_for(["us-west-2", "eu-west-1"], "p4d.24xlarge")
        assert history["eu-west-1"] == []
        (only,) = history["us-west-2"]
        assert only.source == "launch" and only.run_id == "r1" and only.available


class TestAnalytics:
    def test_availability_summary(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add_many(
                [
                    _obs(0, True),
                    _obs(1, False),
                    _obs(2, True, source="launch"),
                    _obs(0, True, region="us-east-1"),
                ]
            )
            summary = store.availability_summary()
            probes_only = store.availability_summary(source="probe")
        west = next(s for s in summary if s.region == "us-west-2")
        assert (west.n_observations, west.n_available) == (3, 2)
        assert west.availability_rate == pytest.approx(2 / 3)
        assert west.first_seen == T0 and west.last_seen == T0 + timedelta(hours=2)
        assert [s.region for s in summary] == ["us-east-1", "us-west-2"]
        west_probe = next(s for s in probes_only if s.region == "us-west-2")
        assert west_probe.n_observations == 2

    def test_lifetime_samples_complete_and_censored(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add_many(
                [
                    _obs(0, True, run_id="a"),
                    _obs(3, False, run_id="a"),  # complete: 3h
                    _obs(10, True, run_id="b"),
                    _obs(12, True, run_id="b"),  # run b ends still up -> censored 2h
                ]
            )
            samples = store.lifetime_samples("us-west-2", "p4d.24xlarge")
        assert samples.complete == [3.0]
        assert samples.censored == [2.0]

    def test_lifetime_samples_gap_is_censored_not_chained(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add_many([_obs(0, True), _obs(2, True), _obs(100, True), _obs(101, False)])
            samples = store.lifetime_samples("us-west-2", "p4d.24xlarge", max_gap_hours=6.0)
        assert samples.censored == [2.0]
        assert samples.complete == [1.0]

    def test_empty_store(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            assert store.availability_summary() == []
            samples = store.lifetime_samples("us-west-2", "p4d.24xlarge")
        assert samples.complete == [] and samples.censored == []


class TestAuthStub:
    def test_no_auth_allows(self) -> None:
        assert NoAuth().authenticate() is True

    def test_rejected_authenticator_raises_and_creates_nothing(self, tmp_path) -> None:
        class Deny:
            def authenticate(self) -> bool:
                return False

        path = tmp_path / "nested" / "runs.db"
        with pytest.raises(StoreAuthError):
            SpotObservationStore(path, authenticator=Deny())
        assert not path.exists() and not path.parent.exists()

    def test_custom_authenticator_is_called(self, tmp_path) -> None:
        calls: list[int] = []

        class Allow:
            def authenticate(self) -> bool:
                calls.append(1)
                return True

        SpotObservationStore(tmp_path / "runs.db", authenticator=Allow()).close()
        assert calls == [1]


class TestAdvisorBandColumn:
    def test_band_round_trips(self, tmp_path) -> None:
        with SpotObservationStore(tmp_path / "runs.db") as store:
            store.add(
                StoredObservation(
                    observed_utc=T0,
                    region="us-west-2",
                    instance_type="p4d.24xlarge",
                    available=True,
                    advisor_band=3,
                )
            )
            store.add(_obs(1, False))
            rows = store.observations("us-west-2", "p4d.24xlarge")
        assert [r.advisor_band for r in rows] == [3, None]

    def test_legacy_db_is_migrated_in_place(self, tmp_path) -> None:
        import sqlite3

        path = tmp_path / "runs.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE spot_observations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "observed_utc TEXT NOT NULL, cloud TEXT NOT NULL, region TEXT NOT NULL, "
            "instance_type TEXT NOT NULL, available INTEGER NOT NULL, source TEXT NOT NULL, "
            "run_id TEXT, detail TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            "INSERT INTO spot_observations (observed_utc, cloud, region, instance_type, "
            "available, source) VALUES (?, 'aws', 'us-west-2', 'p4d.24xlarge', 1, 'probe')",
            (T0.isoformat(),),
        )
        conn.commit()
        conn.close()
        with SpotObservationStore(path) as store:
            store.add(_obs(1, False))
            rows = store.observations("us-west-2", "p4d.24xlarge")
        assert [r.advisor_band for r in rows] == [None, None]
        assert len(rows) == 2
