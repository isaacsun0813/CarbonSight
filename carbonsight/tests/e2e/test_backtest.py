"""E2E: backtest run produces metrics and is reproducible."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_backtest_run_small() -> None:
    """Run backtest with small n; assert exit 0 and metrics in output."""
    root = Path(__file__).resolve().parents[2]
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli")}
    result = subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", "backtest", "run", "--n", "20", "--days", "1", "--json"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stderr or result.stdout)
    data = json.loads(result.stdout)
    assert "carbon_savings_pct_mean" in data
    assert "regret_mean" in data
    assert "rank_top1_pct" in data
    assert data["n_workloads"] == 20


def test_backtest_reproducible() -> None:
    """Same seed -> same metrics."""
    root = Path(__file__).resolve().parents[2]
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli")}
    def run(seed: int) -> dict:
        r = subprocess.run(
            [sys.executable, "-m", "carbonsight_cli.main", "backtest", "run", "--n", "10", "--days", "1", "--seed", str(seed), "--json"],
            capture_output=True,
            text=True,
            cwd=root,
            env=env,
            timeout=20,
        )
        assert r.returncode == 0
        return json.loads(r.stdout)
    a = run(99)
    b = run(99)
    assert a["regret_mean"] == b["regret_mean"]
    assert a["rank_top1_pct"] == b["rank_top1_pct"]


def _spot_json(root: Path, *args: str) -> dict:
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli"),
    }
    r = subprocess.run(
        [
            sys.executable, "-m", "carbonsight_cli.main", "backtest", "run",
            "--spot", "--n", "4", "--days", "4", "--json", *args,
        ],
        capture_output=True, text=True, cwd=root, env=env, timeout=120,
    )
    assert r.returncode == 0, (r.stderr or r.stdout)
    return json.loads(r.stdout)


def test_spot_backtest_reports_named_baselines_and_spread() -> None:
    """The --spot path: real fields, a named baseline, and mean +/- sigma."""
    root = Path(__file__).resolve().parents[2]
    data = _spot_json(root, "--seeds", "1,2,3")

    assert data["seeds"] == [1, 2, 3]
    assert data["up_single_default_region"]  # the pin must be named, not averaged away

    metrics = data["metrics_mean_stdev"]
    for key in (
        "skynomad_cost_mean",
        "up_single_default_cost_mean",
        "up_single_best_cost_mean",
        "up_multi_cost_mean",
        "cost_savings_vs_up_single_default_pct",
        "cost_savings_vs_up_single_best_pct",
        "cost_savings_vs_up_multi_pct",
        "idle_hours_mean",
    ):
        mean, stdev = metrics[key]
        assert isinstance(mean, float) and isinstance(stdev, float)
        assert stdev >= 0.0

    assert any("constant per region" in c for c in data["caveats"])


def test_spot_backtest_savings_are_consistent_with_the_costs() -> None:
    root = Path(__file__).resolve().parents[2]
    m = _spot_json(root, "--seeds", "5")["metrics_mean_stdev"]
    sky = m["skynomad_cost_mean"][0]
    for cost_key, pct_key in (
        ("up_single_default_cost_mean", "cost_savings_vs_up_single_default_pct"),
        ("up_multi_cost_mean", "cost_savings_vs_up_multi_pct"),
    ):
        assert m[pct_key][0] == pytest.approx((1 - sky / m[cost_key][0]) * 100, rel=1e-6)


def test_spot_backtest_is_reproducible() -> None:
    root = Path(__file__).resolve().parents[2]
    a = _spot_json(root, "--seeds", "9")
    b = _spot_json(root, "--seeds", "9")
    assert a["metrics_mean_stdev"] == b["metrics_mean_stdev"]
