"""
Full-path E2E for ``carbonsight advise``: real subprocess, real YAML, JSON contract.

- **No credentials:** must return valid JSON ``[]`` and exit 0 (matches CLI behavior).
- **With WattTime credentials:** must return a non-empty list of recommendations with
  expected fields (live registry + forecast). Marked ``@pytest.mark.integration``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REQUIRED_RESULT_KEYS = frozenset(
    {
        "cloud",
        "cloud_region",
        "watttime_regions",
        "expected_cost_usd",
        "expected_co2_kg_mean",
        "expected_co2_kg_p10",
        "expected_co2_kg_p90",
        "mapping_confidence",
        "notes",
    }
)


def _cli_env(repo_root: Path, *, strip_watttime: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "packages" / "core") + ":" + str(repo_root / "apps" / "cli")
    if strip_watttime:
        env.pop("WATTTIME_USERNAME", None)
        env.pop("WATTTIME_PASSWORD", None)
    return env


def _run_advise_json(repo_root: Path, fixture: Path, env: dict[str, str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", "advise", "--yaml", str(fixture), "--json"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=timeout,
    )


def test_advise_without_watttime_returns_empty_json_list(repo_root: Path) -> None:
    """CLI + YAML parse + code path through advise; no HTTP when credentials unset."""
    fixture = repo_root / "tests" / "fixtures" / "train_minimal.yaml"
    assert fixture.is_file()
    env = _cli_env(repo_root, strip_watttime=True)
    result = _run_advise_json(repo_root, fixture, env, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip())
    assert data == []


@pytest.mark.integration
def test_advise_live_watttime_full_pipeline(repo_root: Path) -> None:
    """End-to-end: registry + WattTime forecast + estimator → ranked JSON rows."""
    if not (os.environ.get("WATTTIME_USERNAME") and os.environ.get("WATTTIME_PASSWORD")):
        pytest.skip("Set WATTTIME_USERNAME and WATTTIME_PASSWORD for live E2E")

    fixture = repo_root / "tests" / "fixtures" / "train_minimal.yaml"
    env = _cli_env(repo_root, strip_watttime=False)
    result = _run_advise_json(repo_root, fixture, env, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout

    data: Any = json.loads(result.stdout.strip())
    assert isinstance(data, list), data
    assert len(data) >= 1, "expected at least one AWS region with mapping in seed registry"

    for row in data:
        assert set(row.keys()) >= _REQUIRED_RESULT_KEYS, row
        assert row["expected_co2_kg_p10"] <= row["expected_co2_kg_mean"] <= row["expected_co2_kg_p90"]
        assert row["expected_co2_kg_mean"] > 0
        assert row["mapping_confidence"] >= 0.0

    co2_values = [float(r["expected_co2_kg_mean"]) for r in data]
    assert co2_values == sorted(co2_values), "CLI must return regions sorted greenest-first"
