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
from carbonsight_cli.commands import job_setup
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

    monkeypatch.setattr(job_setup, "Registry", FakeRegistry)
    monkeypatch.setattr(
        job_setup, "resolve_registry_json_file", lambda *a, **k: registry_path
    )
    monkeypatch.setattr(
        job_setup.Config, "from_env", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(job_setup, "configure_pricing", lambda *a, **k: None)
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


@pytest.mark.parametrize("preview", [False, True])
def test_static_spot_orders_only_affordable_regions(harness, monkeypatch, capsys, preview) -> None:
    yaml_path, calls = harness
    resources = {"accelerators": "T4:1", "instance_type": "g4dn.xlarge", "disk_size": 50}
    yaml_path.write_text(yaml.safe_dump({**JOB_YAML, "resources": resources}))
    estimates = [
        _estimate().model_copy(update={
            "cloud_region": f"region-{index}",
            "expected_co2_kg_mean": float(max(1, index)),
            "expected_cost_usd": 100.0 if index == 2 else 10.0,
        })
        for index in range(8)
    ]
    monkeypatch.setattr(
        run_mod.AwsRegionRankingService, "collect_estimates", lambda *a, **k: estimates
    )
    score_calls = []

    def placement_scores(instance_type, nodes, regions):
        score_calls.append((instance_type, nodes, regions))
        return {"region-1": 9}

    monkeypatch.setattr(run_mod, "get_spot_placement_scores", placement_scores)
    _run(yaml_path, dry_run=preview)
    captured = capsys.readouterr()
    if preview:
        assert score_calls == []
        assert calls["launches"] == []
        patched = captured.out.split("--- Patched YAML for SkyPilot (dry run) ---\n")[1]
    else:
        assert score_calls[0][:2] == ("g4dn.xlarge", 1)
        assert len(calls["launches"]) == 1
        patched = calls["launches"][0][0]
        assert "SkyPilot launch region is unknown" in captured.out
        assert "Actual IT CO₂ estimate:" not in captured.out

    initial = ["region-0", "region-1"] if preview else ["region-1", "region-0"]
    assert FakeLedger.records[0].cloud_region == initial[0]
    assert yaml.safe_load(patched)["resources"] == {
        **resources,
        "use_spot": True,
        "ordered": [{"infra": f"aws/{region}"} for region in [*initial, "region-3", "region-4", "region-5"]],
    }


def test_static_no_spot_disables_spot_in_yaml(harness) -> None:
    yaml_path, calls = harness
    resources = {**JOB_YAML["resources"], "use_spot": True}
    yaml_path.write_text(yaml.safe_dump({**JOB_YAML, "resources": resources}))
    _run(yaml_path, use_spot=False)
    assert yaml.safe_load(calls["launches"][0][0])["resources"] == {
        **resources, "use_spot": False, "infra": "aws/eu-north-1",
    }


@pytest.mark.parametrize("location", [{"infra": "aws/us-east-1"}, {"region": "us-east-1"}])
def test_explicit_aws_region_pins_ordinary_launch(harness, monkeypatch, location) -> None:
    yaml_path, calls = harness
    resources = {**JOB_YAML["resources"], **location}
    yaml_path.write_text(yaml.safe_dump({**JOB_YAML, "resources": resources}))
    pinned = _estimate().model_copy(update={"cloud_region": "us-east-1", "expected_co2_kg_mean": 4.0})
    monkeypatch.setattr(
        run_mod.AwsRegionRankingService, "collect_estimates", lambda *a, **k: [_estimate(), pinned]
    )

    _run(yaml_path)

    assert yaml.safe_load(calls["launches"][0][0])["resources"] == {
        "accelerators": "T4:1", "infra": "aws/us-east-1", "use_spot": True,
    }
    assert FakeLedger.records[0].cloud_region == "us-east-1"


def test_explicit_aws_region_limits_deadline_plan(harness, monkeypatch) -> None:
    yaml_path, calls = harness
    resources = {**JOB_YAML["resources"], "infra": "aws/eu-north-1"}
    yaml_path.write_text(yaml.safe_dump({**JOB_YAML, "resources": resources}))
    other = _estimate().model_copy(update={"cloud_region": "us-east-1"})
    monkeypatch.setattr(
        run_mod.AwsRegionRankingService, "collect_estimates", lambda *a, **k: [other, _estimate()]
    )

    def check_plan_inputs(reg, watt_time, job, estimates, **kwargs):
        assert [row.cloud_region for row in estimates] == ["eu-north-1"]
        raise typer.Exit(9)

    monkeypatch.setattr(run_mod, "_build_dynamic_plan_inputs", check_plan_inputs)
    with pytest.raises(typer.Exit) as exc:
        _run(yaml_path, finish_by=run_mod.datetime(2030, 1, 1, tzinfo=run_mod.UTC))
    assert exc.value.exit_code == 9
    assert calls["launches"] == []


@pytest.mark.parametrize(
    ("location", "error"),
    [
        ({"infra": "aws/us-east-1"}, "No eligible estimate for the pinned AWS region"),
        ({"infra": "aws/us-east-1/us-east-1a"}, "availability-zone pin"),
        ({"region": "us-east-1", "zone": "us-east-1a"}, "availability-zone pin"),
        ({"ordered": [{"infra": "aws/us-east-1"}]}, "without ordered or any_of"),
        ({"any_of": [{"infra": "aws/us-east-1"}]}, "without ordered or any_of"),
    ],
)
def test_unsupported_or_unavailable_location_fails_closed(harness, capsys, location, error) -> None:
    yaml_path, calls = harness
    resources = {**JOB_YAML["resources"], **location}
    yaml_path.write_text(yaml.safe_dump({**JOB_YAML, "resources": resources}))

    with pytest.raises(typer.Exit) as exc:
        _run(yaml_path)
    assert exc.value.exit_code == 1
    assert error in capsys.readouterr().err
    assert calls["launches"] == []


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
        job_setup.Config,
        "from_env",
        classmethod(lambda cls: FakeConfig(watttime_username="", watttime_password="")),
    )
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path)
    assert excinfo.value.exit_code == 1


def test_unknown_runtime_launches_without_recording_job_totals(harness, capsys) -> None:
    yaml_path, calls = harness
    data = dict(JOB_YAML)
    del data["duration"]
    yaml_path.write_text(yaml.safe_dump(data))
    _run(yaml_path)
    assert len(calls["launches"]) == 1
    assert FakeLedger.records == []
    output = capsys.readouterr()
    assert "Runtime unknown" in output.err
    assert "vs estimate:" not in output.out


def test_deadline_without_runtime_fails_before_aws_calls(harness, monkeypatch) -> None:
    from datetime import UTC, datetime

    yaml_path, calls = harness
    data = dict(JOB_YAML)
    del data["duration"]
    yaml_path.write_text(yaml.safe_dump(data))

    def unexpected(*args, **kwargs):
        pytest.fail("Deadline must be validated before collecting estimates")

    monkeypatch.setattr(run_mod.AwsRegionRankingService, "collect_estimates", unexpected)
    with pytest.raises(typer.Exit) as excinfo:
        _run(yaml_path, finish_by=datetime(2030, 1, 1, tzinfo=UTC))
    assert excinfo.value.exit_code == 1
    assert calls["launches"] == []


@pytest.mark.parametrize("command", ["run", "train"])
@pytest.mark.parametrize("flags, expected", [([], False), (["--probe"], True)])
def test_probe_cli_requires_opt_in(tmp_path, monkeypatch, command, flags, expected) -> None:
    from carbonsight_cli.commands import train as train_mod
    from carbonsight_cli.main import app
    from typer.testing import CliRunner

    path = tmp_path / ("job.yaml" if command == "run" else "train.py")
    path.touch()
    options = []
    module = run_mod if command == "run" else train_mod
    monkeypatch.setattr(module, "run_launch", lambda path, opts: options.append(opts))
    args = [command, str(path), *flags]
    if command == "train":
        args.append("--launch")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert options[0].probe_before_plan is expected


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, []),
        ({"probe_before_plan": True}, ["eu-north-1"]),
        ({"probe_before_plan": True, "use_spot": False}, []),
        ({"probe_before_plan": True, "dry_run": True}, []),
    ],
)
def test_region_selection_only_probes_opted_in_spot_launches(
    monkeypatch, overrides, expected,
) -> None:
    from datetime import UTC, datetime

    region = "eu-north-1"
    context = SimpleNamespace(
        options=run_mod.LaunchOptions(**overrides),
        effective_finish_by=datetime(2030, 1, 1, tzinfo=UTC),
        effective_carbon_budget=None,
        effective_carbon_price=None,
        registry=None,
        watt_time=None,
        job=SimpleNamespace(gpu_type="T4"),
        estimates=[],
        db_path=None,
        config=None,
        spot_instance_type="g4dn.xlarge",
    )
    calls = []
    constraints = run_mod.PlanConstraints(
        deadline_utc=context.effective_finish_by,
        now_utc=datetime.now(UTC),
    )
    monkeypatch.setattr(
        run_mod, "_build_dynamic_plan_inputs",
        lambda *a, **k: (constraints, [SimpleNamespace(cloud_region=region)]),
    )
    monkeypatch.setattr(run_mod, "_load_spot_history", lambda *a: {})
    monkeypatch.setattr(run_mod, "_load_advisor_bands", lambda *a: {})
    monkeypatch.setattr(run_mod, "_make_prober", lambda *a: object())
    monkeypatch.setattr(
        run_mod, "_probe_before_plan", lambda prober, regions, *a, **k: calls.extend(regions),
    )
    monkeypatch.setattr(
        run_mod, "_pick_region_with_dynamic_plan", lambda *a, **k: (region, True, None, None),
    )
    run_mod._select_region(context)
    assert calls == expected
