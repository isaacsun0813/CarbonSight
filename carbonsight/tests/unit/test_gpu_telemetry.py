"""GPU utilization telemetry: nvidia-smi parsing and power model integration."""

import math
import random

import pytest
from carbonsight_core.estimator.carbon_model import estimate_job_carbon_in_region
from carbonsight_core.estimator.power_model import (
    GPU_POWER_CATALOG,
    _gpu_power,
    _sample_u_gpu,
    expected_it_power_w,
    gpu_power_limit_w,
    gpu_power_notes,
    power_it_w,
    sample_power_params,
)
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
    def test_expected_power_honors_observed_gpu_utilization(self) -> None:
        job = JobSpec(
            gpu_type="A100",
            gpu_count=1,
            duration_hours=1.0,
            gpu_utilization=0.42,
        )
        assert expected_it_power_w(job) == power_it_w(job, u_gpu=0.42, u_cpu=0.5)

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


@pytest.mark.parametrize("gpu,watts", [
    ("L4", 72), ("L40", 300), ("L40S", 350), ("A10", 150), ("A10G", 300),
    ("A100-PCIE-40GB", 250), ("A100-PCIE-80GB", 300), ("NVIDIA A100-SXM4-80GB", 400),
    ("H100-PCIE", 350), ("H100-SXM5", 700), ("H200-NVL", 600), ("B200", 1000),
])
def test_gpu_power_uses_sourced_hardware_spec(gpu: str, watts: float) -> None:
    job = JobSpec(gpu_type=gpu, gpu_count=2)
    assert gpu_power_limit_w(job) == watts
    assert _gpu_power(job, 1.0) == 2 * watts
    assert _gpu_power(job, 0.0) == pytest.approx(0.2 * watts)
    assert "source=https://" in gpu_power_notes(job)[0]


def test_unknown_gpu_requires_explicit_limit_and_can_be_modeled() -> None:
    with pytest.raises(ValueError, match="No sourced GPU power specification"):
        expected_it_power_w(JobSpec(gpu_type="future-gpu"))
    job = JobSpec(gpu_type="future-gpu", gpu_power_limit_w=120)
    assert _gpu_power(job, 1.0) == 120
    assert "user-provided" in gpu_power_notes(job)[0]
    assert expected_it_power_w(JobSpec(gpu_type="none", gpu_count=0)) > 0


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_explicit_power_limit_must_be_positive_and_finite(limit: float) -> None:
    with pytest.raises(ValueError):
        JobSpec(gpu_type="L4", gpu_power_limit_w=limit)


def test_gpu_catalog_has_unique_aliases_and_covers_aws_gpus() -> None:
    from carbonsight_core.cloud.aws.gpu_catalog import _AWS_GPU_CATALOG
    from carbonsight_core.estimator.power_model import _gpu_key

    aliases = []
    for spec in GPU_POWER_CATALOG["gpus"]:
        assert math.isfinite(spec["max_power_w"]) and spec["max_power_w"] > 0
        assert spec["source_url"].startswith("https://")
        aliases.extend(_gpu_key(name) for name in (spec["name"], *spec["aliases"]))
    assert len(aliases) == len(set(aliases))
    for gpu in _AWS_GPU_CATALOG:
        assert gpu_power_limit_w(JobSpec(gpu_type=gpu)) > 0


def test_default_deterministic_power_matches_sampling_mean() -> None:
    job = JobSpec(gpu_type="L4")
    rng = random.Random(42)
    sampled = [sample_power_params(job, rng).p_it_w for _ in range(1000)]
    assert sum(sampled) / len(sampled) == pytest.approx(expected_it_power_w(job), rel=0.01)


def test_api_requires_power_limit_for_unknown_gpu(monkeypatch) -> None:
    from carbonsight_api.main import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app.state, "registry_path", None, raising=False)
    client = TestClient(app)
    response = client.post("/v1/recommendations", json={"gpu_type": "unknown"})
    assert response.status_code == 422
    assert "gpu_power_limit_w" in response.json()["detail"]
    response = client.post(
        "/v1/recommendations", json={"gpu_type": "unknown", "gpu_power_limit_w": 72},
    )
    assert response.status_code == 200
