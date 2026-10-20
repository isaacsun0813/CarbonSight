"""Resolve planning fields from CLI vs YAML carbonsight block."""

from datetime import UTC, datetime
from pathlib import Path

import yaml

from carbonsight_cli.commands.advise import (
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
