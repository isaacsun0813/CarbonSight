"""
Independent oracle for the Tier-1 task (spot pricing, carbon-aware time-shifting,
persistent run tracking). Grader-owned — do NOT add this file to the repo during a
model session; drop it into ``carbonsight/tests/unit/`` only when scoring.

All tests are deterministic and HTTP-free (``moer_override`` / overrides / tmp SQLite).
Run from ``carbonsight/``:  pytest tests/unit/test_tier1_oracle.py -v
"""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from carbonsight_core.estimator.carbon_model import estimate_job_carbon_in_region
from carbonsight_core.estimator.pricing import SPOT_PRICE_FRACTION, estimate_cost_usd
from carbonsight_core.models import JobSpec
from carbonsight_core.scheduler import ScheduleChoice, pick_lowest_carbon_start
from carbonsight_core.tracking import LedgerSummary, RunLedger, RunRecord

from carbonsight_cli.commands.run import patch_sky_yaml_with_cloud_region

UTC = timezone.utc


def _job(gpu_type: str = "A100", gpu_count: int = 1, duration_hours: float = 1.0) -> JobSpec:
    return JobSpec(gpu_type=gpu_type, gpu_count=gpu_count, duration_hours=duration_hours)


def _forecast(t0: datetime, values: list[float], step_min: int = 60) -> list[dict]:
    return [
        {"point_time": (t0 + timedelta(minutes=i * step_min)).isoformat(), "value": v}
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# A. Spot pricing
# ---------------------------------------------------------------------------

class TestSpotPricing:
    def test_spot_fraction_is_a_valid_discount(self) -> None:
        assert 0.0 < SPOT_PRICE_FRACTION < 1.0

    def test_on_demand_default_is_unchanged(self) -> None:
        assert estimate_cost_usd("A100", 1, 1.0, "us-east-1") == pytest.approx(4.10, rel=0.01)

    def test_spot_is_fraction_of_on_demand(self) -> None:
        on_demand = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        spot = estimate_cost_usd("A100", 1, 1.0, "us-east-1", use_spot=True)
        assert spot == pytest.approx(on_demand * SPOT_PRICE_FRACTION, rel=1e-6)

    def test_spot_cheaper_across_regions_and_gpus(self) -> None:
        for gpu in ("A100", "H100", "T4"):
            for region in ("us-east-1", "eu-west-1", "ap-south-1"):
                on_demand = estimate_cost_usd(gpu, 2, 3.0, region)
                spot = estimate_cost_usd(gpu, 2, 3.0, region, use_spot=True)
                assert spot < on_demand

    def test_estimate_region_cost_reflects_spot(self) -> None:
        """Spot must thread through the estimator, not just the raw pricing helper."""
        job = _job()
        wt = None  # moer_override => no client used
        on_demand = estimate_job_carbon_in_region(
            job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, wt, moer_override=400.0
        )
        spot = estimate_job_carbon_in_region(
            job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, wt, moer_override=400.0, use_spot=True
        )
        assert spot.expected_cost_usd == pytest.approx(
            on_demand.expected_cost_usd * SPOT_PRICE_FRACTION, rel=1e-6
        )
        # CO2 is independent of the billing model.
        assert spot.expected_co2_kg_mean == pytest.approx(on_demand.expected_co2_kg_mean, rel=0.15)

    def test_patch_sets_use_spot_true(self, tmp_path: Path) -> None:
        y = tmp_path / "j.yaml"
        y.write_text("name: t\nresources:\n  accelerators: A100:1\nduration: 1h\nrun: python x.py\n")
        patched = patch_sky_yaml_with_cloud_region(y, "aws", "eu-north-1", use_spot=True)
        assert "use_spot: true" in patched.lower()
        assert "eu-north-1" in patched

    def test_patch_omits_or_falsifies_spot_by_default(self, tmp_path: Path) -> None:
        y = tmp_path / "j.yaml"
        y.write_text("name: t\nresources:\n  accelerators: A100:1\nduration: 1h\nrun: python x.py\n")
        patched = patch_sky_yaml_with_cloud_region(y, "aws", "eu-north-1").lower()
        assert "use_spot: true" not in patched


# ---------------------------------------------------------------------------
# B. Carbon-aware time-shifting
# ---------------------------------------------------------------------------

class TestPickLowestCarbonStart:
    _T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_picks_greenest_window_and_reports_reduction(self) -> None:
        pts = _forecast(self._T0, [500, 500, 100, 100, 500])  # green valley at +2h/+3h
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=4.0)
        assert isinstance(choice, ScheduleChoice)
        assert choice.start_utc == self._T0 + timedelta(hours=2)  # earliest of the tied minima
        assert choice.delay_hours == pytest.approx(2.0)
        assert choice.window_moer_lb_per_mwh == pytest.approx(100.0)
        assert choice.now_window_moer_lb_per_mwh == pytest.approx(500.0)
        assert choice.moer_reduction_pct == pytest.approx(80.0)

    def test_zero_delay_forces_immediate_start(self) -> None:
        pts = _forecast(self._T0, [500, 100, 100])
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=0.0)
        assert choice.start_utc == self._T0
        assert choice.delay_hours == pytest.approx(0.0)
        assert choice.moer_reduction_pct == pytest.approx(0.0)

    def test_respects_max_delay_budget(self) -> None:
        # Greenest point is at +3h but the budget only reaches +1h.
        pts = _forecast(self._T0, [500, 400, 100, 50])
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=1.0)
        assert choice.start_utc == self._T0 + timedelta(hours=1)
        assert choice.window_moer_lb_per_mwh == pytest.approx(400.0)

    def test_time_weighted_window_spanning_points(self) -> None:
        # 30-min points; a 1h window at t0 spans [200 for 30m, 600 for 30m] -> 400.
        pts = _forecast(self._T0, [200, 600, 600, 600], step_min=30)
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=0.0)
        assert choice.now_window_moer_lb_per_mwh == pytest.approx(400.0)

    def test_ties_break_to_earliest(self) -> None:
        pts = _forecast(self._T0, [100, 100, 100])
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=2.0)
        assert choice.start_utc == self._T0
        assert choice.delay_hours == pytest.approx(0.0)

    def test_single_point_is_immediate_no_reduction(self) -> None:
        pts = _forecast(self._T0, [321.0])
        choice = pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=6.0)
        assert choice.delay_hours == pytest.approx(0.0)
        assert choice.window_moer_lb_per_mwh == pytest.approx(321.0)
        assert choice.moer_reduction_pct == pytest.approx(0.0)

    def test_empty_forecast_raises(self) -> None:
        with pytest.raises(ValueError):
            pick_lowest_carbon_start([], duration_hours=1.0, max_delay_hours=4.0)

    def test_negative_max_delay_raises(self) -> None:
        pts = _forecast(self._T0, [500, 100])
        with pytest.raises(ValueError):
            pick_lowest_carbon_start(pts, duration_hours=1.0, max_delay_hours=-1.0)

    def test_non_positive_duration_raises(self) -> None:
        pts = _forecast(self._T0, [500, 100])
        with pytest.raises(ValueError):
            pick_lowest_carbon_start(pts, duration_hours=0.0, max_delay_hours=4.0)


# ---------------------------------------------------------------------------
# C. Persistent run tracking
# ---------------------------------------------------------------------------

def _rec(run_id: str, base_co2: float, est_co2: float, base_cost: float, est_cost: float) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        created_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        cloud_region="eu-north-1",
        gpu_type="A100",
        gpu_count=1,
        duration_hours=1.0,
        use_spot=True,
        baseline_region="us-east-1",
        estimated_co2_kg=est_co2,
        baseline_co2_kg=base_co2,
        estimated_cost_usd=est_cost,
        baseline_cost_usd=base_cost,
    )


class TestRunLedger:
    def test_record_and_get_roundtrip(self, tmp_path: Path) -> None:
        led = RunLedger(tmp_path / "runs.db")
        rec = _rec("r1", 10.0, 4.0, 100.0, 35.0)
        led.record(rec)
        got = led.get("r1")
        assert got is not None
        assert got.run_id == "r1"
        assert got.cloud_region == "eu-north-1"
        assert got.estimated_co2_kg == pytest.approx(4.0)
        assert got.baseline_co2_kg == pytest.approx(10.0)
        assert got.use_spot is True

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        led = RunLedger(tmp_path / "runs.db")
        assert led.get("nope") is None

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """Proves real on-disk storage, not an in-memory dict."""
        db = tmp_path / "runs.db"
        RunLedger(db).record(_rec("r1", 10.0, 4.0, 100.0, 35.0))
        reopened = RunLedger(db)
        assert reopened.get("r1") is not None
        assert len(reopened.all()) == 1

    def test_all_returns_every_run(self, tmp_path: Path) -> None:
        led = RunLedger(tmp_path / "runs.db")
        led.record(_rec("r1", 10.0, 4.0, 100.0, 35.0))
        led.record(_rec("r2", 20.0, 5.0, 200.0, 70.0))
        assert {r.run_id for r in led.all()} == {"r1", "r2"}

    def test_summary_aggregates_savings(self, tmp_path: Path) -> None:
        led = RunLedger(tmp_path / "runs.db")
        led.record(_rec("r1", 10.0, 4.0, 100.0, 35.0))
        led.record(_rec("r2", 20.0, 5.0, 200.0, 70.0))
        s = led.summary()
        assert isinstance(s, LedgerSummary)
        assert s.n_runs == 2
        assert s.total_baseline_co2_kg == pytest.approx(30.0)
        assert s.total_estimated_co2_kg == pytest.approx(9.0)
        assert s.total_co2_saved_kg == pytest.approx(21.0)
        assert s.co2_saved_pct == pytest.approx(70.0)
        assert s.total_cost_saved_usd == pytest.approx(195.0)
        assert s.cost_saved_pct == pytest.approx(65.0)

    def test_summary_empty_is_zeroed_no_div_by_zero(self, tmp_path: Path) -> None:
        s = RunLedger(tmp_path / "runs.db").summary()
        assert s.n_runs == 0
        assert s.total_co2_saved_kg == pytest.approx(0.0)
        assert s.co2_saved_pct == pytest.approx(0.0)
        assert s.cost_saved_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI wiring smoke (HTTP-free: --help only)
# ---------------------------------------------------------------------------

def _cli_help(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **__import__("os").environ,
        "PYTHONPATH": f"{root / 'packages' / 'core'}:{root / 'apps' / 'cli'}:{root / 'apps' / 'api'}",
        # Pin a wide terminal so Rich help output doesn't wrap option names
        # (keeps these assertions deterministic in CI, where COLUMNS is unset -> 80).
        "COLUMNS": "200",
    }
    return subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", *args],
        capture_output=True, text=True, cwd=root, env=env, timeout=30,
    )


@pytest.fixture(scope="module")
def repo() -> Path:
    return Path(__file__).resolve().parents[2]


class TestCliWiring:
    def test_run_exposes_spot_and_max_delay(self, repo: Path) -> None:
        out = _cli_help(repo, "run", "--help")
        assert out.returncode == 0, out.stderr
        assert "--spot" in out.stdout
        assert "--max-delay" in out.stdout

    def test_history_and_report_commands_exist(self, repo: Path) -> None:
        top = _cli_help(repo, "--help")
        assert top.returncode == 0, top.stderr
        assert "history" in top.stdout
        assert "report" in top.stdout
        assert _cli_help(repo, "history", "--help").returncode == 0
        assert _cli_help(repo, "report", "--help").returncode == 0
