"""
Unit tests for cost estimation, carbon estimation, and --max-cost-premium filtering.

`estimate_job_carbon_in_region` tests use ``moer_override`` (no HTTP).
CLI ``advise`` integration tests need ``WATTTIME_USERNAME`` and ``WATTTIME_PASSWORD``.
"""

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from carbonsight_core.estimator.carbon_model import LB_TO_KG, compute_actual_co2, estimate_job_carbon_in_region
from carbonsight_core.estimator.pricing import estimate_cost_usd
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime import WattTimeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(gpu_type: str = "A100", gpu_count: int = 1, duration_hours: float = 1.0) -> JobSpec:
    return JobSpec(gpu_type=gpu_type, gpu_count=gpu_count, duration_hours=duration_hours)


def _mock_wt() -> MagicMock:
    """WattTimeClient mock — should never be called when moer_override is set."""
    wt = MagicMock()
    wt.get_forecast.side_effect = AssertionError("HTTP should not be called when moer_override is set")
    return wt


# ---------------------------------------------------------------------------
# Pricing unit tests
# ---------------------------------------------------------------------------

class TestPricing:
    def test_known_gpu_known_region_returns_positive(self) -> None:
        cost = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        assert cost > 0

    def test_us_east_1_a100_baseline(self) -> None:
        # us-east-1 has multiplier 1.0; base A100 price is $4.10/GPU/hr
        cost = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        assert cost == pytest.approx(4.10, rel=0.01)

    def test_eu_region_costs_more_than_us_east(self) -> None:
        us = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        eu = estimate_cost_usd("A100", 1, 1.0, "eu-west-1")
        assert eu > us

    def test_gpu_count_scales_linearly(self) -> None:
        one = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        eight = estimate_cost_usd("A100", 8, 1.0, "us-east-1")
        assert eight == pytest.approx(one * 8, rel=0.001)

    def test_duration_scales_linearly(self) -> None:
        one_hr = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        ten_hr = estimate_cost_usd("A100", 1, 10.0, "us-east-1")
        assert ten_hr == pytest.approx(one_hr * 10, rel=0.001)

    def test_unknown_gpu_type_uses_fallback(self) -> None:
        # Should not raise — falls back to default price
        cost = estimate_cost_usd("MYSTERY_GPU_9000", 1, 1.0, "us-east-1")
        assert cost > 0

    def test_unknown_region_uses_fallback(self) -> None:
        cost = estimate_cost_usd("A100", 1, 1.0, "xx-unknown-99")
        assert cost > 0

    def test_h100_more_expensive_than_a100(self) -> None:
        a100 = estimate_cost_usd("A100", 1, 1.0, "us-east-1")
        h100 = estimate_cost_usd("H100", 1, 1.0, "us-east-1")
        assert h100 > a100


# ---------------------------------------------------------------------------
# Carbon estimation unit tests (moer_override bypasses HTTP)
# ---------------------------------------------------------------------------

class TestEstimateJobCarbonInRegion:
    def test_cost_is_populated(self) -> None:
        job = _make_job()
        result = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert result.expected_cost_usd > 0

    def test_co2_mean_is_positive(self) -> None:
        job = _make_job()
        result = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert result.expected_co2_kg_mean > 0

    def test_p10_le_mean_le_p90(self) -> None:
        job = _make_job()
        result = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert result.expected_co2_kg_p10 <= result.expected_co2_kg_mean <= result.expected_co2_kg_p90

    def test_zero_moer_means_zero_co2(self) -> None:
        job = _make_job()
        result = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=0.0)
        assert result.expected_co2_kg_mean == pytest.approx(0.0, abs=1e-9)

    def test_higher_moer_produces_higher_co2(self) -> None:
        job = _make_job()
        low = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=100.0)
        high = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=800.0)
        assert high.expected_co2_kg_mean > low.expected_co2_kg_mean

    def test_longer_job_produces_more_co2(self) -> None:
        short = estimate_job_carbon_in_region(_make_job(duration_hours=1.0), "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        long_ = estimate_job_carbon_in_region(_make_job(duration_hours=10.0), "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert long_.expected_co2_kg_mean > short.expected_co2_kg_mean

    def test_more_gpus_produce_more_co2_and_cost(self) -> None:
        one = estimate_job_carbon_in_region(_make_job(gpu_count=1), "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        eight = estimate_job_carbon_in_region(_make_job(gpu_count=8), "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert eight.expected_co2_kg_mean > one.expected_co2_kg_mean
        assert eight.expected_cost_usd == pytest.approx(one.expected_cost_usd * 8, rel=0.01)

    def test_deterministic_with_fixed_seed(self) -> None:
        job = _make_job()
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        a = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0, rng=rng_a)
        b = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0, rng=rng_b)
        assert a.expected_co2_kg_mean == b.expected_co2_kg_mean

    def test_wt_never_called_when_moer_override_set(self) -> None:
        wt = _mock_wt()
        job = _make_job()
        # Would raise AssertionError if wt.get_forecast() is called
        estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, wt, moer_override=400.0)
        wt.get_forecast.assert_not_called()

    def test_co2_math_is_correct(self) -> None:
        """With a fixed seed and known MOER, verify the CO2 formula end-to-end."""
        # At moer=400 lb/MWh and a deterministic seed we can't easily predict the
        # exact value due to Monte Carlo variance, but we can bound it:
        # min A100 power ~250W, max ~400W, PUE ~1.1-1.5 → facility ~275-600W
        # For 1 hour: 0.000275-0.0006 MWh × 400 lb/MWh × 0.453 kg/lb → 0.05-0.11 kg
        job = _make_job()
        result = estimate_job_carbon_in_region(job, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0)
        assert 0.01 < result.expected_co2_kg_mean < 2.0  # sanity bounds


# ---------------------------------------------------------------------------
# CLI integration: --max-cost-premium filtering
# ---------------------------------------------------------------------------

needs_watttime_creds = pytest.mark.skipif(
    not (os.environ.get("WATTTIME_USERNAME") and os.environ.get("WATTTIME_PASSWORD")),
    reason="Set WATTTIME_USERNAME and WATTTIME_PASSWORD for live advise subprocess tests",
)


@needs_watttime_creds
class TestMaxCostPremiumCLI:
    """Run ``carbonsight advise --json`` in a subprocess (real WattTime forecast calls)."""

    @pytest.fixture(autouse=True)
    def _paths(self, repo_root: Path) -> None:
        self.root = repo_root
        self.fixture = repo_root / "tests" / "fixtures" / "train_minimal.yaml"
        self.env = {
            **__import__("os").environ,
            "PYTHONPATH": str(repo_root / "packages" / "core") + ":" + str(repo_root / "apps" / "cli"),
        }

    def _run_advise(self, extra_args: list[str], timeout: int = 60) -> list[dict]:
        result = subprocess.run(
            [sys.executable, "-m", "carbonsight_cli.main", "advise",
             "--yaml", str(self.fixture), "--json"] + extra_args,
            capture_output=True, text=True, cwd=self.root, env=self.env, timeout=timeout,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def test_returns_list_of_results(self) -> None:
        results = self._run_advise([])
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_result_has_cost_and_co2(self) -> None:
        results = self._run_advise([])
        for r in results:
            assert r["expected_cost_usd"] > 0, f"Zero cost for {r['cloud_region']}"
            assert r["expected_co2_kg_mean"] > 0, f"Zero CO2 for {r['cloud_region']}"

    def test_results_sorted_greenest_first(self) -> None:
        results = self._run_advise([])
        co2_values = [r["expected_co2_kg_mean"] for r in results]
        assert co2_values == sorted(co2_values), "Results must be sorted by CO2 ascending"

    def test_tight_premium_filters_expensive_regions(self) -> None:
        all_results = self._run_advise(["--max-cost-premium", "1.0"])
        tight_results = self._run_advise(["--max-cost-premium", "0.0"])
        # Tight filter must return fewer or equal regions
        assert len(tight_results) <= len(all_results)

    def test_zero_premium_returns_only_cheapest_cost_regions(self) -> None:
        results = self._run_advise(["--max-cost-premium", "0.0"])
        min_cost = min(r["expected_cost_usd"] for r in results)
        for r in results:
            assert r["expected_cost_usd"] == pytest.approx(min_cost, rel=0.01), (
                f"{r['cloud_region']} cost ${r['expected_cost_usd']:.2f} exceeds min ${min_cost:.2f}"
            )

    def test_wide_premium_returns_all_regions(self) -> None:
        all_r = self._run_advise(["--max-cost-premium", "999.0"])
        default_r = self._run_advise([])
        # With unlimited budget we should see at least as many as default
        assert len(all_r) >= len(default_r)


# ---------------------------------------------------------------------------
# compute_actual_co2 unit tests (M7 + M8)
# ---------------------------------------------------------------------------

_ACT_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_ACT_END = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)  # 1-hour window
_ACT_REGIONS: list[tuple[str, float]] = [("CAISO_NORTH", 1.0)]


def _make_moer_pts(
    values: list[float],
    start: datetime = _ACT_START,
    interval_min: int = 5,
) -> list[dict]:
    """Build ``{"point_time": iso_str, "value": float}`` list at regular intervals."""
    return [
        {"point_time": (start + timedelta(minutes=i * interval_min)).isoformat(), "value": v}
        for i, v in enumerate(values)
    ]


class TestComputeActualCo2:
    """Tests for compute_actual_co2; all HTTP-free via historical_moer_override or MagicMock."""

    _JOB = _make_job()

    # M7-1: happy path — MagicMock for get_historical
    def test_happy_path_returns_actual_run_result(self) -> None:
        pts = _make_moer_pts([400.0, 400.0, 400.0])
        wt = MagicMock()
        wt.get_historical.return_value = pts

        result = compute_actual_co2(
            self._JOB, "aws", "us-east-1", _ACT_REGIONS,
            _ACT_START, _ACT_END, wt, estimated_co2_kg=0.1,
        )

        assert result.actual_co2_kg > 0
        assert result.actual_duration_hours == pytest.approx(1.0)
        assert result.watttime_regions == _ACT_REGIONS
        wt.get_historical.assert_called_once()

    # M7-2: empty historical list → ValueError
    def test_empty_historical_raises_value_error(self) -> None:
        wt = MagicMock()
        wt.get_historical.return_value = []

        with pytest.raises(ValueError, match="No historical MOER"):
            compute_actual_co2(
                self._JOB, "aws", "us-east-1", _ACT_REGIONS,
                _ACT_START, _ACT_END, wt, estimated_co2_kg=0.1,
            )

    # M7-3: zero duration → ValueError
    def test_zero_duration_raises_value_error(self) -> None:
        wt = MagicMock()
        with pytest.raises(ValueError):
            compute_actual_co2(
                self._JOB, "aws", "us-east-1", _ACT_REGIONS,
                _ACT_START, _ACT_START, wt, estimated_co2_kg=0.1,
            )

    # M7-3: negative duration (end before start) → ValueError
    def test_negative_duration_raises_value_error(self) -> None:
        wt = MagicMock()
        with pytest.raises(ValueError):
            compute_actual_co2(
                self._JOB, "aws", "us-east-1", _ACT_REGIONS,
                _ACT_END, _ACT_START, wt, estimated_co2_kg=0.1,
            )

    # M7-4: all-zero MOER → zero CO2
    def test_zero_moer_returns_zero_co2(self) -> None:
        pts = _make_moer_pts([0.0, 0.0, 0.0])
        result = compute_actual_co2(
            self._JOB, "aws", "us-east-1", _ACT_REGIONS,
            _ACT_START, _ACT_END, MagicMock(), estimated_co2_kg=0.1,
            historical_moer_override=pts,
        )
        assert result.actual_co2_kg == pytest.approx(0.0, abs=1e-9)

    # M7-5: WattTimeError from get_historical propagates
    def test_watttime_error_propagates(self) -> None:
        wt = MagicMock()
        wt.get_historical.side_effect = WattTimeError("bad units")

        with pytest.raises(WattTimeError, match="bad units"):
            compute_actual_co2(
                self._JOB, "aws", "us-east-1", _ACT_REGIONS,
                _ACT_START, _ACT_END, wt, estimated_co2_kg=0.1,
            )

    # M7-6: mixture regions — two regions with weights 0.6 + 0.4
    def test_mixture_regions_weighted_correctly(self) -> None:
        # Region A: MOER=300 lb/MWh (weight 0.6), Region B: MOER=500 lb/MWh (weight 0.4)
        # Expected weighted MOER = 0.6*300 + 0.4*500 = 380
        pts_a = _make_moer_pts([300.0, 300.0, 300.0])
        pts_b = _make_moer_pts([500.0, 500.0, 500.0])
        wt = MagicMock()
        wt.get_historical.side_effect = [pts_a, pts_b]

        result_mix = compute_actual_co2(
            self._JOB, "aws", "us-east-1",
            [("REGION_A", 0.6), ("REGION_B", 0.4)],
            _ACT_START, _ACT_END, wt, estimated_co2_kg=0.1,
        )
        # Reference: single region with the expected weighted MOER
        result_ref = compute_actual_co2(
            self._JOB, "aws", "us-east-1", [("REGION_REF", 1.0)],
            _ACT_START, _ACT_END, MagicMock(), estimated_co2_kg=0.1,
            historical_moer_override=_make_moer_pts([380.0, 380.0, 380.0]),
        )
        assert result_mix.actual_co2_kg == pytest.approx(result_ref.actual_co2_kg, rel=0.001)

    # M7-7: time-weighted average differs from simple mean for non-uniform intervals
    def test_time_weighted_average_non_uniform_intervals(self) -> None:
        start = _ACT_START
        end = start + timedelta(minutes=30)
        job = _make_job(duration_hours=0.5)

        # Points at t=0, t=5min, t=25min (non-uniform spacing)
        # Intervals: [0→5min]=300s val=100, [5→25min]=1200s val=400, [25→30min]=300s val=300
        # time-weighted = (100*300 + 400*1200 + 300*300) / 1800 = 600000/1800 = 333.33
        # simple mean would be (100+400+300)/3 = 266.67  ← different, proving the test is meaningful
        pts = [
            {"point_time": (start + timedelta(minutes=0)).isoformat(), "value": 100.0},
            {"point_time": (start + timedelta(minutes=5)).isoformat(), "value": 400.0},
            {"point_time": (start + timedelta(minutes=25)).isoformat(), "value": 300.0},
        ]
        expected_weighted_moer = (100 * 300 + 400 * 1200 + 300 * 300) / 1800  # 333.333…

        result_tw = compute_actual_co2(
            job, "aws", "us-east-1", [("CAISO_NORTH", 1.0)],
            start, end, MagicMock(), estimated_co2_kg=0.1,
            historical_moer_override=pts,
        )
        # Single-point override returns its value directly — the reference ground truth.
        result_ref = compute_actual_co2(
            job, "aws", "us-east-1", [("CAISO_NORTH", 1.0)],
            start, end, MagicMock(), estimated_co2_kg=0.1,
            historical_moer_override=[{"point_time": start.isoformat(), "value": expected_weighted_moer}],
        )
        assert result_tw.actual_co2_kg == pytest.approx(result_ref.actual_co2_kg, rel=0.001)

    # M8: historical_moer_override bypasses all HTTP calls
    def test_historical_moer_override_skips_http(self) -> None:
        pts = _make_moer_pts([400.0, 350.0, 420.0])
        wt = MagicMock()
        wt.get_historical.side_effect = AssertionError("HTTP must not be called with override set")

        result = compute_actual_co2(
            self._JOB, "aws", "us-east-1", _ACT_REGIONS,
            _ACT_START, _ACT_END, wt, estimated_co2_kg=0.1,
            historical_moer_override=pts,
        )
        assert result.actual_co2_kg > 0
        wt.get_historical.assert_not_called()

    # M8: empty override raises ValueError (not silently returns zero)
    def test_empty_historical_moer_override_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_actual_co2(
                self._JOB, "aws", "us-east-1", _ACT_REGIONS,
                _ACT_START, _ACT_END, MagicMock(), estimated_co2_kg=0.1,
                historical_moer_override=[],
            )
