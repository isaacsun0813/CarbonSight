"""Tests for spot pricing, carbon-aware scheduling, and persistent run tracking."""

import random
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from carbonsight_core.estimator.carbon_model import estimate_job_carbon_in_region
from carbonsight_core.estimator.pricing import SPOT_PRICE_FRACTION, estimate_cost_usd
from carbonsight_core.models import JobSpec
from carbonsight_core.scheduler import ScheduleChoice, pick_lowest_carbon_start
from carbonsight_core.tracking import LedgerSummary, RunLedger, RunRecord


def _make_job(gpu_type: str = "A100", gpu_count: int = 1, duration_hours: float = 1.0) -> JobSpec:
    return JobSpec(gpu_type=gpu_type, gpu_count=gpu_count, duration_hours=duration_hours)


def _mock_wt() -> MagicMock:
    wt = MagicMock()
    wt.get_forecast.side_effect = AssertionError("HTTP must not be called with moer_override")
    return wt


# ---------------------------------------------------------------------------
# A. Spot pricing
# ---------------------------------------------------------------------------


class TestSpotPricing:
    def test_spot_fraction_value(self) -> None:
        assert SPOT_PRICE_FRACTION == 0.35

    def test_on_demand_unchanged_by_default(self) -> None:
        cost = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        assert cost == pytest.approx(4.10, rel=0.01)

    def test_spot_is_fraction_of_on_demand(self) -> None:
        on_demand = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        spot = estimate_cost_usd("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert spot == pytest.approx(on_demand * SPOT_PRICE_FRACTION, rel=0.001)

    def test_spot_false_equals_on_demand(self) -> None:
        a = estimate_cost_usd("A100", 1, 1.0, "eu-west-1")
        b = estimate_cost_usd("A100", 1, 1.0, "eu-west-1", use_spot=False)
        assert a == b

    def test_spot_threads_through_estimator(self) -> None:
        job = _make_job()
        on_demand = estimate_job_carbon_in_region(
            job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(),
            moer_override=400.0, rng=random.Random(42),
        )
        spot = estimate_job_carbon_in_region(
            job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(),
            moer_override=400.0, rng=random.Random(42), use_spot=True,
        )
        assert spot.expected_cost_usd == pytest.approx(
            on_demand.expected_cost_usd * SPOT_PRICE_FRACTION, rel=0.001,
        )
        assert spot.expected_co2_kg_mean == on_demand.expected_co2_kg_mean


# ---------------------------------------------------------------------------
# B. Carbon-aware scheduling
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _forecast(values: list[float], start: datetime = _T0, interval_min: int = 60) -> list[dict]:
    return [
        {"point_time": (start + timedelta(minutes=i * interval_min)).isoformat(), "value": v}
        for i, v in enumerate(values)
    ]


class TestScheduler:
    def test_empty_forecast_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pick_lowest_carbon_start([], 1.0, 6.0)

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pick_lowest_carbon_start(_forecast([400.0]), 1.0, -1.0)

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            pick_lowest_carbon_start(_forecast([400.0]), 0.0, 6.0)

    def test_single_point_immediate_start(self) -> None:
        sc = pick_lowest_carbon_start(_forecast([300.0]), 1.0, 6.0)
        assert sc.start_utc == _T0
        assert sc.delay_hours == pytest.approx(0.0)
        assert sc.moer_reduction_pct == pytest.approx(0.0)

    def test_picks_lowest_moer_window(self) -> None:
        pts = _forecast([500.0, 400.0, 200.0, 300.0])
        sc = pick_lowest_carbon_start(pts, 1.0, 3.0)
        assert sc.start_utc == _T0 + timedelta(hours=2)
        assert sc.delay_hours == pytest.approx(2.0)
        assert sc.window_moer_lb_per_mwh < sc.now_window_moer_lb_per_mwh

    def test_zero_delay_means_run_now(self) -> None:
        pts = _forecast([500.0, 200.0, 100.0])
        sc = pick_lowest_carbon_start(pts, 1.0, 0.0)
        assert sc.start_utc == _T0
        assert sc.delay_hours == pytest.approx(0.0)

    def test_ties_break_to_earliest(self) -> None:
        pts = _forecast([300.0, 300.0, 300.0])
        sc = pick_lowest_carbon_start(pts, 1.0, 2.0)
        assert sc.start_utc == _T0

    def test_reduction_percentage(self) -> None:
        pts = _forecast([400.0, 200.0, 200.0, 200.0])
        sc = pick_lowest_carbon_start(pts, 1.0, 3.0)
        assert sc.moer_reduction_pct > 0.0
        expected = (sc.now_window_moer_lb_per_mwh - sc.window_moer_lb_per_mwh) / sc.now_window_moer_lb_per_mwh * 100
        assert sc.moer_reduction_pct == pytest.approx(expected, rel=0.001)

    def test_respects_max_delay_boundary(self) -> None:
        # Best window is at hour 5, but max_delay=2 limits candidates to hours 0-2
        pts = _forecast([400.0, 350.0, 300.0, 250.0, 200.0, 100.0])
        sc = pick_lowest_carbon_start(pts, 1.0, 2.0)
        assert sc.delay_hours <= 2.0


# ---------------------------------------------------------------------------
# C. Persistent run tracking
# ---------------------------------------------------------------------------


def _sample_record(**overrides) -> RunRecord:
    defaults = dict(
        run_id="abc123",
        created_utc=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        cloud_region="eu-north-1",
        gpu_type="A100",
        gpu_count=8,
        duration_hours=2.0,
        use_spot=True,
        baseline_region="us-east-1",
        estimated_co2_kg=1.5,
        baseline_co2_kg=3.0,
        estimated_cost_usd=50.0,
        baseline_cost_usd=65.6,
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


class TestRunLedger:
    def test_empty_ledger_summary_all_zero(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        s = ledger.summary()
        assert s.n_runs == 0
        assert s.total_co2_saved_kg == 0.0
        assert s.co2_saved_pct == 0.0

    def test_record_and_get(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        rec = _sample_record()
        ledger.record(rec)
        got = ledger.get("abc123")
        assert got is not None
        assert got.run_id == "abc123"
        assert got.cloud_region == "eu-north-1"
        assert got.use_spot is True
        assert got.actual_co2_kg is None

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        assert ledger.get("nope") is None

    def test_all_returns_oldest_first(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        ledger.record(_sample_record(
            run_id="b", created_utc=datetime(2024, 6, 2, tzinfo=UTC),
        ))
        ledger.record(_sample_record(
            run_id="a", created_utc=datetime(2024, 6, 1, tzinfo=UTC),
        ))
        runs = ledger.all()
        assert [r.run_id for r in runs] == ["a", "b"]

    def test_summary_computes_savings(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        ledger.record(_sample_record(
            run_id="r1",
            estimated_co2_kg=1.0, baseline_co2_kg=2.0,
            estimated_cost_usd=50.0, baseline_cost_usd=100.0,
        ))
        ledger.record(_sample_record(
            run_id="r2",
            estimated_co2_kg=1.5, baseline_co2_kg=3.0,
            estimated_cost_usd=30.0, baseline_cost_usd=60.0,
        ))
        s = ledger.summary()
        assert s.n_runs == 2
        assert s.total_co2_saved_kg == pytest.approx(2.5)
        assert s.co2_saved_pct == pytest.approx(50.0)
        assert s.total_cost_saved_usd == pytest.approx(80.0)
        assert s.cost_saved_pct == pytest.approx(50.0)

    def test_summary_zero_baseline_no_division_error(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "test.db")
        ledger.record(_sample_record(
            run_id="r1",
            estimated_co2_kg=0.0, baseline_co2_kg=0.0,
            estimated_cost_usd=0.0, baseline_cost_usd=0.0,
        ))
        s = ledger.summary()
        assert s.co2_saved_pct == 0.0
        assert s.cost_saved_pct == 0.0

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        RunLedger(db).record(_sample_record())
        got = RunLedger(db).get("abc123")
        assert got is not None
        assert got.cloud_region == "eu-north-1"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db = tmp_path / "deep" / "nested" / "runs.db"
        ledger = RunLedger(db)
        ledger.record(_sample_record())
        assert db.exists()
