"""Characterization tests for ``run_launch`` (CLI wiring in commands/run.py).

These pin the current behavior of ``run_launch`` before its signature and
phase-split refactor: exit codes, dry-run/no-exec returns, the ledger write
that happens *before* those returns, and the static launch path. All external
services (registry, WattTime, ranking, ledger, SkyPilot) are faked.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from carbonsight_cli.commands import run as run_mod
from carbonsight_core.models import EstimateResult

JOB_YAML = {
    "name": "char-job",
    "resources": {"accelerators": "T4:1"},
    "duration": "1h",
    "run": "python train.py\n",
}


@dataclass
class FakeConfig:
    watttime_username: str = "user"
    watttime_password: str = "pass"
    live_aws_pricing: bool = False


class FakeLedger:
    records: list = []

    def __init__(self, db_path):
        self.db_path = db_path

    def record(self, rec) -> None:
        FakeLedger.records.append(rec)


def _estimate() -> EstimateResult:
    return EstimateResult(
        cloud="aws",
        cloud_region="eu-north-1",
        watttime_regions=[("SE", 1.0)],
        expected_cost_usd=10.0,
        expected_co2_kg_mean=1.0,
        expected_co2_kg_p10=0.8,
        expected_co2_kg_p90=1.2,
        mapping_confidence=1.0,
    )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch):
    """Fake every external seam of run_launch; return (yaml_path, calls dict)."""
    FakeLedger.records = []
    calls: dict = {"launches": []}

    yaml_path = tmp_path / "job.yaml"
    yaml_path.write_text(yaml.dump(JOB_YAML, sort_keys=False))
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")

    class FakeRegistry:
        def load_json(self, path) -> None:
            calls["registry_path"] = path

    class FakeEstimator:
        def __init__(self, watt_time):
            pass

        def compute_actual_run(self, **kwargs):
            return SimpleNamespace(
                actual_co2_kg=0.9,
                actual_duration_hours=1.0,
                actual_cost_usd=9.0,
                co2_savings_vs_estimate_pct=10.0,
            )

    def fake_launch(patched, **kwargs):
        calls["launches"].append((patched, kwargs))
        return 0

    monkeypatch.setattr(run_mod, "Registry", FakeRegistry)
    monkeypatch.setattr(
        run_mod, "resolve_registry_json_file", lambda *a, **k: registry_path
    )
    monkeypatch.setattr(
        run_mod.Config, "from_env", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(run_mod, "configure_pricing", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "WattTimeClient", lambda config: object())
    monkeypatch.setattr(
        run_mod.AwsRegionRankingService,
        "collect_estimates",
        lambda self, job, **kwargs: [_estimate()],
    )
    monkeypatch.setattr(run_mod, "RunLedger", FakeLedger)
    monkeypatch.setattr(run_mod, "launch_skypilot_with_patched_yaml", fake_launch)
    monkeypatch.setattr(run_mod, "JobCarbonEstimator", FakeEstimator)
    monkeypatch.setattr(run_mod, "_require_sky_cli", lambda: None)
    return yaml_path, calls


def _run(yaml_path: Path, **overrides):
    kwargs = {
        "checkpoint": False,
        "skip_preflight": True,
        "db_path": yaml_path.parent / "runs.db",
    }
    kwargs.update(overrides)
    return run_mod.run_launch(yaml_path, run_mod.LaunchOptions(**kwargs))


def test_dry_run_prints_patched_yaml_and_still_records_run(harness, capsys) -> None:
    yaml_path, calls = harness
    _run(yaml_path, dry_run=True)
    out = capsys.readouterr().out
    assert "Patched YAML for SkyPilot (dry run)" in out
    assert calls["launches"] == []
    # Characterization: the ledger write happens before the dry-run return.
    assert len(FakeLedger.records) == 1
    assert FakeLedger.records[0].cloud_region == "eu-north-1"


def test_no_exec_prints_patched_yaml_and_skips_launch(harness, capsys) -> None:
    yaml_path, calls = harness
    _run(yaml_path, no_exec=True)
    out = capsys.readouterr().out
    assert "Skipping launch (--no-exec)" in out
    assert calls["launches"] == []
    assert len(FakeLedger.records) == 1


def test_static_launch_happy_path(harness) -> None:
    yaml_path, calls = harness
    _run(yaml_path)
    assert len(calls["launches"]) == 1
    patched, kwargs = calls["launches"][0]
    assert yaml.safe_load(patched)["resources"]["infra"] == "aws/eu-north-1"
    assert len(FakeLedger.records) == 1


def test_validate_sky_raises_exit_zero_on_success(harness) -> None:
    yaml_path, calls = harness
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path, validate_sky=True)
    assert excinfo.value.exit_code == 0
    assert len(calls["launches"]) == 1
    assert calls["launches"][0][1]["dryrun"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"poll_seconds": 0},
        {"gpu_util": 1.5},
    ],
)
def test_validation_exits(harness, overrides) -> None:
    yaml_path, _calls = harness
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path, **overrides)
    assert excinfo.value.exit_code == 1


def test_missing_yaml_exits(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        run_mod.run_launch(
            tmp_path / "nope.yaml", run_mod.LaunchOptions(checkpoint=False)
        )
    assert excinfo.value.exit_code == 1


def test_empty_estimates_exits(harness, monkeypatch) -> None:
    yaml_path, _calls = harness
    monkeypatch.setattr(
        run_mod.AwsRegionRankingService,
        "collect_estimates",
        lambda self, job, **kwargs: [],
    )
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path)
    assert excinfo.value.exit_code == 1


def test_missing_credentials_exits(harness, monkeypatch) -> None:
    yaml_path, _calls = harness
    monkeypatch.setattr(
        run_mod.Config,
        "from_env",
        classmethod(lambda cls: FakeConfig(watttime_username="", watttime_password="")),
    )
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path)
    assert excinfo.value.exit_code == 1
