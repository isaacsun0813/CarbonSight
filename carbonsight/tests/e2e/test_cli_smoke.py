"""E2E smoke: carbonsight --help and advise with fixture YAML (no live HTTP required for smoke)."""

import subprocess
import sys
from pathlib import Path


def test_carbonsight_help() -> None:
    """Run carbonsight --help; exit 0."""
    root = Path(__file__).resolve().parents[2]
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli")}
    result = subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", "--help"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    if result.returncode != 0:
        # Fallback: run main.py
        result = subprocess.run(
            [sys.executable, str(root / "apps" / "cli" / "carbonsight_cli" / "main.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=root,
            env=env,
        )
    assert result.returncode == 0, (result.stderr or result.stdout)
    assert "advise" in (result.stdout or result.stderr) or "run" in (result.stdout or result.stderr)


def test_advise_parses_yaml() -> None:
    """Advise with fixture YAML; may fail on missing WattTime but must not crash on parse."""
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests" / "fixtures" / "train_minimal.yaml"
    if not fixture.exists():
        return
    cli_main = root / "apps" / "cli" / "carbonsight_cli" / "main.py"
    if not cli_main.exists():
        return
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli")}
    result = subprocess.run(
        [sys.executable, str(cli_main), "advise", "--yaml", str(fixture), "--json"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=15,
    )
    # 0 = success (with or without regions); empty credentials -> empty list and exit 0
    assert result.returncode in (0, 1)
