"""
E2E tests for ``carbonsight mappings validate``.

- No credentials: subprocess smoke (CI default).
- Integration: live WattTime when creds present.
- VCR: in-process validate with replayed HTTP (cassette optional).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.mapping.validate import validate_registry_mappings, drift_report_to_dict
from carbonsight_core.watttime import WattTimeClient


def _cli_env(repo_root: Path, *, strip_watttime: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "packages" / "core") + ":" + str(repo_root / "apps" / "cli")
    if strip_watttime:
        env.pop("WATTTIME_USERNAME", None)
        env.pop("WATTTIME_PASSWORD", None)
    return env


def _run_validate(
    repo_root: Path,
    env: dict[str, str],
    *,
    json_out: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "carbonsight_cli.main", "mappings", "validate"]
    if json_out:
        cmd.append("--json")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=timeout,
    )


def test_validate_without_watttime_skips(repo_root: Path) -> None:
    """CLI exits 0 and reports skipped when WattTime credentials are unset."""
    env = _cli_env(repo_root, strip_watttime=True)
    result = _run_validate(repo_root, env, json_out=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip())
    assert data["status"] == "skipped"


@pytest.mark.integration
def test_validate_live_seed_registry(repo_root: Path) -> None:
    """Live validate against bundled seed registry when WattTime creds are set."""
    if not (os.environ.get("WATTTIME_USERNAME") and os.environ.get("WATTTIME_PASSWORD")):
        pytest.skip("Set WATTTIME_USERNAME and WATTTIME_PASSWORD for live E2E")

    env = _cli_env(repo_root, strip_watttime=False)
    result = _run_validate(repo_root, env, json_out=True, timeout=300)
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout.strip())
    assert data["status"] == "ok"
    assert data["regions_ok"] >= 1


@pytest.mark.e2e
def test_validate_in_process_mocked(tmp_path: Path) -> None:
    """In-process validate path with mocked WattTime (no HTTP)."""
    tiny_registry = tmp_path / "registry.json"
    tiny_registry.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "provider": "aws",
                        "region_code": "us-east-1",
                        "display_name": "US East",
                        "sites": [
                            {
                                "site_id": "s1",
                                "provider": "aws",
                                "region_code": "us-east-1",
                                "lat": 38.96,
                                "lon": -77.56,
                            },
                        ],
                        "wt_regions": [{"wt_region": "PJM_DC", "weight": 1.0}],
                    },
                ],
            }
        )
    )
    reg = Registry()
    reg.load_json(tiny_registry)
    from carbonsight_core.config import Config

    wt = WattTimeClient(Config.from_env())
    with patch.object(wt, "region_from_loc", return_value={"region": "PJM_DC"}):
        result = validate_registry_mappings(reg, wt)
    payload = drift_report_to_dict(result)
    assert payload["status"] == "ok"
    assert payload["regions_ok"] == 1
