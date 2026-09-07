"""SkyPilot YAML patching: strip CarbonSight-only keys."""

from pathlib import Path

import yaml

from carbonsight_cli.commands.advise import strip_carbonsight_only_yaml_keys
from carbonsight_cli.commands.run import patch_sky_yaml_with_cloud_region


def test_strip_carbonsight_only_yaml_keys() -> None:
    data = {
        "name": "t",
        "duration": "30m",
        "carbonsight": {"finish_by": "2026-09-08T22:00:00Z"},
        "resources": {"accelerators": "T4:1"},
        "run": "python train.py\n",
    }
    cleaned = strip_carbonsight_only_yaml_keys(data)
    assert "duration" not in cleaned
    assert "carbonsight" not in cleaned
    assert cleaned["name"] == "t"
    assert cleaned["run"] == "python train.py\n"


def test_patch_omits_carbonsight_keys(tmp_path: Path) -> None:
    y = tmp_path / "job.yaml"
    y.write_text(
        yaml.dump(
            {
                "name": "t",
                "duration": "1h",
                "carbonsight": {"finish_by": "2026-09-08T22:00:00Z"},
                "resources": {"accelerators": "A100:1"},
                "run": "python x.py\n",
            },
            default_flow_style=False,
            sort_keys=False,
        )
    )
    patched = patch_sky_yaml_with_cloud_region(y, "aws", "eu-north-1", use_spot=True)
    data = yaml.safe_load(patched)
    assert "duration" not in data
    assert "carbonsight" not in data
    assert data["resources"]["infra"] == "aws/eu-north-1"
    assert data["resources"]["use_spot"] is True
