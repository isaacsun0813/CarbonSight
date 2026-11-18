"""CLI coverage for the experimental static planner."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import yaml
from carbonsight_cli.commands import plan_static
from carbonsight_cli.main import app
from carbonsight_core.planning import RegionPlanInput
from typer.testing import CliRunner

runner = CliRunner()


def _write_task(tmp_path: Path, *, finish_by: str | None = None) -> Path:
    data = {
        "duration": "1h",
        "resources": {"accelerators": "A100:1"},
        "run": "python train.py",
    }
    if finish_by is not None:
        data["carbonsight"] = {"finish_by": finish_by}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _stub_external_inputs(monkeypatch, tmp_path: Path, *, with_regions: bool = True) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")

    class FakeRegistry:
        def load_json(self, path: Path) -> None:
            assert path == registry_path

    monkeypatch.setattr(plan_static, "Registry", FakeRegistry)
    monkeypatch.setattr(
        plan_static, "resolve_registry_json_file", lambda *args, **kwargs: registry_path,
    )
    monkeypatch.setattr(
        plan_static.Config,
        "from_env",
        lambda: SimpleNamespace(
            watttime_username="user",
            watttime_password="password",
            live_aws_pricing=False,
        ),
    )
    monkeypatch.setattr(plan_static, "configure_pricing", lambda *args, **kwargs: None)
    monkeypatch.setattr(plan_static, "WattTimeClient", lambda config: object())

    def fake_region_inputs(registry, watt_time, job, constraints):
        if not with_regions:
            return []
        t0 = constraints.now_utc
        return [
            RegionPlanInput(
                cloud_region="us-west-2",
                watttime_regions=[("PACW", 1.0)],
                forecast_points=[
                    {
                        "point_time": (t0 + timedelta(hours=hour)).isoformat(),
                        "value": 100.0,
                    }
                    for hour in range(4)
                ],
            )
        ]

    monkeypatch.setattr(plan_static, "build_region_plan_inputs", fake_region_inputs)


def test_plan_static_uses_yaml_deadline_and_emits_selected_candidate(
    monkeypatch, tmp_path: Path,
) -> None:
    task = _write_task(tmp_path, finish_by="2099-01-01T00:00:00Z")
    _stub_external_inputs(monkeypatch, tmp_path)

    result = runner.invoke(app, ["plan-static", "--yaml", str(task), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["experimental"] is True
    assert payload["advisory_only"] is True
    assert payload["best"]["cloud_region"] == "us-west-2"
    assert payload["best"]["mode"] in {"spot", "on_demand"}
    assert payload["feasible_count"] > 0


def test_plan_static_requires_deadline(tmp_path: Path) -> None:
    task = _write_task(tmp_path)

    result = runner.invoke(app, ["plan-static", "--yaml", str(task)])

    assert result.exit_code == 1
    assert "requires --finish-by or carbonsight.finish_by" in result.output


def test_plan_static_json_reports_no_feasible_candidate(
    monkeypatch, tmp_path: Path,
) -> None:
    task = _write_task(tmp_path, finish_by="2099-01-01T00:00:00Z")
    _stub_external_inputs(monkeypatch, tmp_path, with_regions=False)

    result = runner.invoke(app, ["plan-static", "--yaml", str(task), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["best"] is None
    assert payload["feasible_count"] == 0
