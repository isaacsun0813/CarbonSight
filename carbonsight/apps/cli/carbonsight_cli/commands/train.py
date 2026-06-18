"""carbonsight train script.py — build a SkyPilot task from a Python file, then advise or launch."""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
import yaml

from carbonsight_cli.commands.advise import run_advise
from carbonsight_cli.commands.run import run_launch


def _run_line_for_script(script: Path) -> str:
    """``python <path>`` relative to cwd when possible (SkyPilot uploads the working directory)."""
    cwd = Path.cwd().resolve()
    resolved = script.resolve()
    try:
        rel = resolved.relative_to(cwd)
        arg = rel.as_posix()
    except ValueError:
        arg = resolved.as_posix()
    return f"python {arg}"


def build_train_yaml_dict(
    script: Path,
    *,
    name: str | None,
    accelerators: str,
    duration: str,
    cpus: int,
    memory: int,
    cloud: str,
) -> dict:
    """SkyPilot-style document: resources + duration + run."""
    task_name = name or f"carbonsight-{script.stem}"
    return {
        "name": task_name,
        "resources": {
            "cloud": cloud,
            "accelerators": accelerators,
            "cpus": cpus,
            "memory": memory,
        },
        "duration": duration,
        "run": _run_line_for_script(script) + "\n",
    }


def write_train_yaml(data: dict) -> Path:
    """Write YAML to a temp file; caller should unlink when done."""
    text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(text)
        return Path(f.name)


def train_cmd(
    script: Path = typer.Argument(..., help="Python training script to run in the cloud (e.g. train.py)"),
    launch: bool = typer.Option(
        False,
        "--launch",
        help="Pick greenest region and launch with SkyPilot (default: show advise table only).",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="SkyPilot task name (default: carbonsight-<script stem>).",
    ),
    accelerators: str = typer.Option(
        "A100:1",
        "--accelerators",
        "-a",
        help="SkyPilot accelerators string, e.g. A100:1 or T4:2.",
    ),
    duration: str = typer.Option(
        "1h",
        "--duration",
        "-d",
        help="Expected job length (e.g. 1h, 30m) for carbon/cost estimate.",
    ),
    cpus: int = typer.Option(8, "--cpus", help="CPU count in resources."),
    memory: int = typer.Option(32, "--memory", "-m", help="Memory (GiB) in resources."),
    cloud: str = typer.Option("aws", "--cloud", help="resources.cloud (provider hint for SkyPilot)."),
    explain: bool = typer.Option(False, "--explain", help="[advise] Extra context after the table"),
    json_out: bool = typer.Option(False, "--json", help="[advise] Machine-readable JSON"),
    dry_run: bool = typer.Option(False, "--dry-run", help="[launch] Print patched YAML only"),
    no_exec: bool = typer.Option(False, "--no-exec", help="[launch] Patch YAML but do not call sky"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="[launch] Skip AWS GPU quota check"),
    managed: bool = typer.Option(False, "--managed", help="[launch] Use sky jobs launch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="[launch] Pass --yes to SkyPilot"),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    max_cost_premium: float = typer.Option(
        0.20,
        "--max-cost-premium",
        help="Max extra cost vs cheapest region when choosing where to run.",
    ),
    gpu_util: float | None = typer.Option(
        None,
        "--gpu-util",
        help="GPU utilization in [0, 1] for the power model.",
    ),
    nvidia_smi: bool = typer.Option(False, "--nvidia-smi", help="Sample GPU utilization from nvidia-smi"),
) -> None:
    """Estimate carbon/cost for training ``script`` without writing a YAML by hand.

    Defaults: ``A100:1``, ``1h``, 8 CPUs, 32 GiB. Use ``--launch`` to pick a region and run SkyPilot.
    Run from your project root so the generated ``python path/to/script.py`` matches what Sky uploads.
    """
    if not script.is_file():
        typer.echo(f"Error: not a file: {script}", err=True)
        raise typer.Exit(1)

    data = build_train_yaml_dict(
        script,
        name=name,
        accelerators=accelerators,
        duration=duration,
        cpus=cpus,
        memory=memory,
        cloud=cloud,
    )
    tmp = write_train_yaml(data)
    try:
        if launch:
            run_launch(
                tmp,
                dry_run=dry_run,
                no_exec=no_exec,
                skip_preflight=skip_preflight,
                managed=managed,
                yes=yes,
                max_cost_premium=max_cost_premium,
                registry_path=registry_path,
                gpu_util=gpu_util,
                nvidia_smi=nvidia_smi,
            )
        else:
            run_advise(
                tmp,
                explain=explain,
                json_out=json_out,
                registry_path=registry_path,
                max_cost_premium=max_cost_premium,
                gpu_util=gpu_util,
                nvidia_smi=nvidia_smi,
            )
    finally:
        tmp.unlink(missing_ok=True)
