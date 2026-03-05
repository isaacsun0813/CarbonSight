"""E2E: backtest run produces metrics and is reproducible."""

import json
import subprocess
import sys
from pathlib import Path


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
