"""Resolve planning fields from CLI vs YAML carbonsight block."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from carbonsight_cli.commands.job_setup import (
    resolve_carbon_budget_kg,
    resolve_carbon_price,
    resolve_finish_by,
)


def _write_yaml(tmp_path: Path, carbonsight: dict) -> Path:
    path = tmp_path / "job.yaml"
    path.write_text(
        yaml.dump(
            {
                "name": "t",
                "carbonsight": carbonsight,
                "resources": {"accelerators": "T4:1"},
                "run": "python x.py\n",
            },
            default_flow_style=False,
            sort_keys=False,
        )
    )
    return path


def test_resolve_finish_by_from_yaml_when_cli_unset(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"finish_by": "2026-09-08T22:00:00Z"})
    resolved = resolve_finish_by(None, yaml_path)
    assert resolved == datetime(2026, 9, 8, 22, 0, tzinfo=UTC)


def test_resolve_finish_by_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"finish_by": "2026-09-08T22:00:00Z"})
    cli = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    resolved = resolve_finish_by(cli, yaml_path)
    assert resolved == cli


def test_resolve_carbon_budget_from_yaml_when_cli_unset(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"carbon_budget_kg": 12.5})
    assert resolve_carbon_budget_kg(None, yaml_path) == 12.5


def test_resolve_carbon_price_from_yaml_when_cli_unset(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"carbon_price": 3.5})
    assert resolve_carbon_price(None, yaml_path) == 3.5


def test_resolve_carbon_price_defaults_to_zero(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"finish_by": "2026-09-08T22:00:00Z"})
    assert resolve_carbon_price(None, yaml_path) == 0.0


def test_missing_duration_is_unknown_with_one_hour_comparison(tmp_path: Path) -> None:
    from carbonsight_cli.commands.job_setup import job_spec_from_sky_yaml

    job = job_spec_from_sky_yaml(_write_yaml(tmp_path, {}))
    assert job.duration_hours == 1.0
    assert not job.has_duration_estimate
    with pytest.raises(ValueError, match="explicit estimated duration"):
        job.require_duration_estimate()


def test_explicit_duration_is_preserved(tmp_path: Path) -> None:
    from carbonsight_cli.commands.job_setup import job_spec_from_sky_yaml

    path = _write_yaml(tmp_path, {})
    data = yaml.safe_load(path.read_text())
    data["duration"] = "30m"
    path.write_text(yaml.safe_dump(data))
    job = job_spec_from_sky_yaml(path)
    assert job.has_duration_estimate
    assert job.require_duration_estimate() == 0.5


@pytest.mark.parametrize("duration", ["0h", "-1h", "inf", "nan", "", "h"])
def test_invalid_duration_is_rejected(tmp_path: Path, duration: str) -> None:
    from carbonsight_cli.commands.job_setup import job_spec_from_sky_yaml

    path = _write_yaml(tmp_path, {})
    data = yaml.safe_load(path.read_text())
    data["duration"] = duration
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        job_spec_from_sky_yaml(path)


def test_train_yaml_omits_unknown_duration(tmp_path: Path) -> None:
    from carbonsight_cli.commands.train import build_train_yaml_dict

    data = build_train_yaml_dict(
        tmp_path / "train.py", name=None, accelerators="T4:1", duration=None,
        cpus=4, memory=16, cloud="aws",
    )
    assert "duration" not in data


@pytest.mark.parametrize("planner_name", ["plan_static_job", "plan_dynamic_job"])
def test_planners_reject_unknown_runtime(planner_name: str) -> None:
    from carbonsight_core import planning
    from carbonsight_core.models import JobSpec

    now = datetime(2030, 1, 1, tzinfo=UTC)
    constraints = planning.PlanConstraints(now_utc=now, deadline_utc=now)
    with pytest.raises(ValueError, match="explicit estimated duration"):
        getattr(planning, planner_name)(JobSpec(gpu_type="T4"), constraints, [])


def test_yaml_can_supply_power_limit_for_uncataloged_gpu(tmp_path: Path) -> None:
    from carbonsight_cli.commands.job_setup import load_job_spec
    from carbonsight_core.estimator.power_model import gpu_power_limit_w

    path = _write_yaml(tmp_path, {"gpu_power_limit_w": 72})
    data = yaml.safe_load(path.read_text())
    data["resources"]["accelerators"] = "future-gpu:1"
    path.write_text(yaml.safe_dump(data))
    job = load_job_spec(path, gpu_util=None, nvidia_smi=False)
    assert gpu_power_limit_w(job) == 72
