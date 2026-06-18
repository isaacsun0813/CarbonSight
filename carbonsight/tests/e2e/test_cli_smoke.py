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
    out = result.stdout or result.stderr
    assert "advise" in out and "train" in out and "run" in out


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
    assert result.returncode == 0


def test_train_script_without_watttime_returns_empty_json() -> None:
    """train script.py builds YAML and runs advise; no creds -> []."""
    root = Path(__file__).resolve().parents[2]
    stub = root.parent / "examples" / "skypilot" / "train_stub.py"
    if not stub.is_file():
        return
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "packages" / "core") + ":" + str(root / "apps" / "cli")}
    env.pop("WATTTIME_USERNAME", None)
    env.pop("WATTTIME_PASSWORD", None)
    result = subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", "train", str(stub), "--json"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
