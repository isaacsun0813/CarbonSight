"""GPU utilization telemetry: nvidia-smi parsing and power model integration."""

import random

import pytest

from carbonsight_core.estimator.carbon_model import estimate_job_carbon_in_region
from carbonsight_core.estimator.power_model import _sample_u_gpu, sample_power_params
from carbonsight_core.models import JobSpec
from carbonsight_core.telemetry.nvidia_smi import parse_nvidia_smi_utilization_csv


def _mock_wt():
    from unittest.mock import MagicMock

    wt = MagicMock()
    wt.get_forecast.side_effect = AssertionError("HTTP should not be called when moer_override is set")
    return wt


class TestNvidiaSmiParse:
    def test_single_gpu(self) -> None:
        assert parse_nvidia_smi_utilization_csv("45\n") == [45.0]

    def test_multi_gpu_mean_ready(self) -> None:
        assert parse_nvidia_smi_utilization_csv("0\n100\n50\n") == [0.0, 100.0, 50.0]

    def test_ignores_garbage_lines(self) -> None:
        assert parse_nvidia_smi_utilization_csv("12\nnot_a_number\n3") == [12.0, 3.0]


class TestPowerModelGpuUtilization:
    def test_sample_u_gpu_fixed_when_spec_set(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0, gpu_utilization=0.42)
        rng = random.Random(999)
        assert _sample_u_gpu(job, rng) == 0.42

    def test_sample_u_gpu_sampled_when_unset(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        rng = random.Random(0)
        a = _sample_u_gpu(job, rng)
        rng = random.Random(0)
        b = _sample_u_gpu(job, rng)
        assert a == b
        assert 0.6 <= a <= 0.9

    def test_telemetry_narrows_mc_spread_vs_default(self) -> None:
        """Fixed GPU util removes u_gpu variance; p90-p10 band should shrink materially."""
        base = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        fixed = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0, gpu_utilization=0.75)
        rng = random.Random(42)
        base_spread = []
        fix_spread = []
        for _ in range(500):
            base_spread.append(sample_power_params(base, rng).p_it_w)
            fix_spread.append(sample_power_params(fixed, rng).p_it_w)
        w_base = max(base_spread) - min(base_spread)
        w_fix = max(fix_spread) - min(fix_spread)
        assert w_fix < w_base * 0.85


class TestEstimateWithGpuUtilization:
    def test_high_util_increases_co2_vs_low_util(self) -> None:
        low = JobSpec(
            gpu_type="A100", gpu_count=1, duration_hours=1.0, gpu_utilization=0.1
        )
        high = JobSpec(
            gpu_type="A100", gpu_count=1, duration_hours=1.0, gpu_utilization=0.99
        )
        rng = random.Random(0)
        r_low = estimate_job_carbon_in_region(
            low, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0, rng=rng
        )
        rng = random.Random(0)
        r_high = estimate_job_carbon_in_region(
            high, "aws", "us-east-1", [("PJM_DC", 1.0)], 0.9, _mock_wt(), moer_override=400.0, rng=rng
        )
        assert r_high.expected_co2_kg_mean > r_low.expected_co2_kg_mean
