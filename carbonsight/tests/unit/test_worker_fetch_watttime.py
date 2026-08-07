"""The refresh worker: keeps the shared cache warm, and never poisons it.

The load-bearing class is `TestNeverPersistsFabricatedData`. This worker writes to a
*shared* cache and a *persistent* table read by every client, so fabricating here is
far worse than fabricating in a single CLI process: one outage would bake physically
meaningless numbers into the central store and serve them as real indefinitely.
"""

from __future__ import annotations

from typing import Any

import pytest
from carbonsight_core.config import Config
from carbonsight_core.watttime import (
    WattTimeError,
    build_synthetic_forecast,
    reset_global_forecast_cache,
)
from worker import fetch_watttime as worker

CREDENTIALLED = Config(watttime_username="user", watttime_password="secret")


class FakeWattTime:
    """Returns a canned payload per region; raises for regions in ``failing``."""

    def __init__(
        self,
        payloads: dict[str, Any] | None = None,
        failing: set[str] | None = None,
    ) -> None:
        self.payloads = payloads or {}
        self.failing = failing or set()
        self.calls: list[str] = []

    def get_forecast(self, region: str, horizon_hours: int = 24, **_: object) -> Any:
        self.calls.append(region)
        if region in self.failing:
            raise WattTimeError(f"upstream exploded for {region}")
        return self.payloads.get(region, {"data": [{"point_time": "2026-03-01T00:00:00Z", "value": 111.0}]})


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    reset_global_forecast_cache()
    yield
    reset_global_forecast_cache()


@pytest.fixture(autouse=True)
def _no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)


class TestRefreshPass:
    def test_refreshes_every_region_into_the_shared_cache(self) -> None:
        from carbonsight_core.watttime import get_global_forecast_cache

        client = FakeWattTime()
        report = worker.fetch_all_regions(CREDENTIALLED, client=client, regions=["IE", "SE"])
        assert report.regions_refreshed == 2
        assert report.regions_failed == 0
        assert get_global_forecast_cache().get("IE") is not None
        assert get_global_forecast_cache().get("SE") is not None

    def test_counts_points(self) -> None:
        payload = {"data": [{"point_time": f"2026-03-01T0{i}:00:00Z", "value": 100.0} for i in range(5)]}
        report = worker.fetch_all_regions(
            CREDENTIALLED, client=FakeWattTime({"IE": payload}), regions=["IE"]
        )
        assert report.points_written == 5

    def test_defaults_to_the_full_region_list(self) -> None:
        from carbonsight_core.watttime import SYNTHETIC_WATTTIME_REGIONS

        client = FakeWattTime()
        worker.fetch_all_regions(CREDENTIALLED, client=client)
        assert client.calls == list(SYNTHETIC_WATTTIME_REGIONS)

    def test_requires_credentials(self) -> None:
        """The worker exists to hold the one shared login; without it there's nothing to do."""
        with pytest.raises(RuntimeError, match="WATTTIME_USERNAME"):
            worker.fetch_all_regions(Config())


class TestPartialFailure:
    def test_one_bad_region_does_not_stop_the_others(self) -> None:
        client = FakeWattTime(failing={"SE"})
        report = worker.fetch_all_regions(
            CREDENTIALLED, client=client, regions=["IE", "SE", "FR"]
        )
        assert report.regions_refreshed == 2
        assert report.regions_failed == 1
        assert any("SE" in f for f in report.failures)

    def test_an_empty_forecast_counts_as_a_failure(self) -> None:
        report = worker.fetch_all_regions(
            CREDENTIALLED, client=FakeWattTime({"IE": {"data": []}}), regions=["IE"]
        )
        assert report.regions_failed == 1
        assert "empty forecast" in report.failures[0]


class TestNeverPersistsFabricatedData:
    def test_a_failed_region_is_left_absent_not_invented(self) -> None:
        from carbonsight_core.watttime import get_global_forecast_cache

        worker.fetch_all_regions(CREDENTIALLED, client=FakeWattTime(failing={"SE"}), regions=["SE"])
        assert get_global_forecast_cache().get("SE") is None, (
            "a failed fetch must leave the cache empty, not fill it with a synthetic curve"
        )

    def test_a_failed_region_keeps_its_previous_real_value(self) -> None:
        """Stale-but-real beats fresh-and-invented."""
        from carbonsight_core.watttime import get_global_forecast_cache

        real = [{"point_time": "2026-03-01T00:00:00Z", "value": 42.0}]
        get_global_forecast_cache().set("SE", real)
        worker.fetch_all_regions(CREDENTIALLED, client=FakeWattTime(failing={"SE"}), regions=["SE"])
        assert get_global_forecast_cache().get("SE") == real

    def test_nothing_synthetic_reaches_the_database(self) -> None:
        captured: list[dict[str, Any]] = []
        original = worker.persist_rows

        def spy(rows: list[dict[str, Any]], **kwargs: object) -> int:
            captured.extend(rows)
            return original(rows, **kwargs)  # type: ignore[arg-type]

        worker.persist_rows = spy  # type: ignore[assignment]
        try:
            worker.fetch_all_regions(
                CREDENTIALLED, client=FakeWattTime(failing={"SE", "IE"}), regions=["SE", "IE"]
            )
        finally:
            worker.persist_rows = original  # type: ignore[assignment]
        assert captured == [], "a failed region must contribute no rows"

    def test_the_client_is_built_with_synthetic_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt and braces: even the client can't hand the worker invented data."""
        seen: dict[str, Any] = {}

        class Recorder:
            def __init__(self, _cfg: Config, *, allow_synthetic: bool = True) -> None:
                seen["allow_synthetic"] = allow_synthetic

            def get_forecast(self, region: str, **_: object) -> Any:
                return {"data": [{"point_time": "2026-03-01T00:00:00Z", "value": 1.0}]}

        monkeypatch.setattr(worker, "WattTimeClient", Recorder)
        worker.fetch_all_regions(CREDENTIALLED, regions=["IE"])
        assert seen["allow_synthetic"] is False

    def test_synthetic_curves_are_arbitrary_which_is_why_this_matters(self) -> None:
        """Pinned as a fact: these values have no relationship to real grids."""
        sweden = sum(p["value"] for p in build_synthetic_forecast("SE")) / 288
        india = sum(p["value"] for p in build_synthetic_forecast("IND")) / 288
        assert sweden > india


class TestPersistence:
    def test_no_database_configured_is_a_no_op(self) -> None:
        assert worker.persist_rows([{"wt_region": "IE"}], database_url="") == 0
        assert worker.prune_expired_rows(database_url="") == 0

    def test_no_rows_is_a_no_op(self) -> None:
        assert worker.persist_rows([], database_url="postgresql://nope/nope") == 0

    def test_a_dead_database_does_not_crash_the_pass(self) -> None:
        """Persistence is best-effort; the cache refresh is the primary job."""
        assert worker.persist_rows(
            [{"wt_region": "IE", "signal_type": "co2_moer",
              "point_time": "2026-03-01T00:00:00Z", "value": 1.0, "units": "lbs_co2_per_mwh"}],
            database_url="postgresql://user:pw@127.0.0.1:1/none",
        ) == 0

    def test_rows_carry_the_full_key_and_units(self) -> None:
        rows = worker._rows_for_region(
            "IE", [{"point_time": "2026-03-01T00:00:00Z", "value": 250.5}]
        )
        assert rows == [
            {
                "wt_region": "IE",
                "signal_type": "co2_moer",
                "point_time": "2026-03-01T00:00:00Z",
                "value": 250.5,
                "units": "lbs_co2_per_mwh",
            }
        ]

    def test_malformed_points_are_dropped_not_written_as_zero(self) -> None:
        rows = worker._rows_for_region(
            "IE",
            [
                {"point_time": "2026-03-01T00:00:00Z", "value": 100.0},
                {"value": 200.0},                                   # no timestamp
                {"point_time": "2026-03-01T01:00:00Z", "value": None},  # unparseable
            ],
        )
        assert len(rows) == 1 and rows[0]["value"] == 100.0


class TestReportFormatting:
    def test_log_line_mentions_failures(self) -> None:
        report = worker.RefreshReport(regions_refreshed=1, regions_failed=1, failures=["SE: boom"])
        assert "SE: boom" in report.as_log_line()

    def test_clean_pass_has_no_failure_suffix(self) -> None:
        assert "failures" not in worker.RefreshReport(regions_refreshed=3).as_log_line()
