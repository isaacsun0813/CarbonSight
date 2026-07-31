"""
carbonsight schedule — joint region x mode ranking (cost + carbon + lifetime + deadline).

Ranks every mapped region in both spot and on-demand mode, plus idle, by

    U = V*eta - C_total - E/Lbar

with V(t) = C_od * theta/theta_tilde from the deadline-pressure model. Two rules
short-circuit the ranking: thrifty (p >= P) and safety net (T-t < P-p+2d).

``--progress-hours`` (p) and ``--elapsed-hours`` (t) drive V: at the default 0/0
the job is on track and V anchors at the cheapest on-demand total, so spot is
accepted and on-demand rejected. A job behind schedule raises V and buys
reliability. This command never launches anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.providers.carbon import get_carbon_provider
from carbonsight_core.providers.spot import get_spot_price_provider
from carbonsight_core.spot.scheduler_service import ranked_as_json, schedule_job
from rich.console import Console
from rich.table import Table

from carbonsight_cli.commands.advise import apply_gpu_telemetry_cli, job_spec_from_sky_yaml


def run_schedule(
    yaml_path: Path,
    *,
    deadline_hours: float | None = None,
    checkpoint_size_gb: float = 0.0,
    cold_start_minutes: float = 6.0,
    carbon_price: float = 50.0,
    carbon_weight: float = 1.0,
    egress_usd_per_gb: float = 0.02,
    current_region: str = "",
    progress_hours: float = 0.0,
    elapsed_hours: float = 0.0,
    registry_path: Path | None = None,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    json_out: bool = False,
    top_n: int | None = None,
) -> None:
    """Shared implementation so tests can drive the command without Typer."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    job = job_spec_from_sky_yaml(yaml_path)
    job, nvidia_failed = apply_gpu_telemetry_cli(job, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    if nvidia_failed:
        typer.echo(
            "Warning: --nvidia-smi could not read utilization. Using sampled GPU power.", err=True
        )

    # Paper default: 30h of compute against a 45h deadline, i.e. T = 1.5 P.
    deadline = deadline_hours if deadline_hours is not None else job.duration_hours * 1.5
    if deadline <= 0:
        typer.echo("Error: --deadline-hours must be positive", err=True)
        raise typer.Exit(1)
    progress_hours = min(progress_hours, job.duration_hours)

    job = job.model_copy(
        update={
            "deadline_hours": deadline,
            "checkpoint_size_gb": checkpoint_size_gb,
            "cold_start_minutes": cold_start_minutes,
            "carbon_price_usd_per_ton": carbon_price,
            "carbon_weight": carbon_weight,
            "progress_hours_done": progress_hours,
            "current_region": current_region,
        }
    )

    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    registry: Registry | None = None
    if rpath.exists():
        registry = Registry()
        registry.load_json(rpath)

    carbon = get_carbon_provider()
    spot = get_spot_price_provider()
    result = schedule_job(
        job,
        registry=registry,
        carbon=carbon,
        spot=spot,
        elapsed_hours=elapsed_hours,
        egress_usd_per_gb=egress_usd_per_gb,
        top_n=top_n,
    )
    progress = result.progress

    if result.action == "empty":
        typer.echo("[]" if json_out else "No mapped regions in the registry.")
        return

    if result.action == "thrifty":
        payload = {
            "action": "thrifty_idle",
            "reason": f"p={progress.p} >= P={progress.P}",
            "value_v": result.value_v,
        }
        typer.echo(
            json.dumps(payload, indent=2)
            if json_out
            else f"Thrifty: p={progress.p}h >= P={progress.P}h, job done — release the instance."
        )
        return

    if result.action == "safety_net":
        threshold = progress.remaining_work + 2 * result.cold_start_hr
        payload = {
            "action": "safety_net_on_demand",
            "reason": f"T-t={progress.remaining_time:.2f}h < P-p+2d={threshold:.2f}h",
            "chosen_region": result.safety_net_region,
            "total_cost_to_finish_usd": result.safety_net_total_cost,
            "value_v": result.value_v,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2))
        else:
            typer.echo(
                f"Safety net: remaining time {progress.remaining_time:.2f}h < "
                f"remaining work + 2d {threshold:.2f}h — switch to on-demand in "
                f"{result.safety_net_region} to finish "
                f"(${result.safety_net_total_cost:.2f}). V={result.value_v:.3f}"
            )
        return

    rows = ranked_as_json(result, job)
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return

    table = Table(
        title=(
            f"schedule | P={progress.P:.1f}h p={progress.p:.1f}h t={progress.t:.1f}h "
            f"T={progress.T:.1f}h | V={result.value_v:.3f} d={result.cold_start_hr:.2f}h "
            f"ckpt={checkpoint_size_gb:g}GB"
        )
    )
    table.add_column("#", justify="right")
    table.add_column("Region", style="cyan")
    table.add_column("Mode")
    table.add_column("Lbar h", justify="right")
    table.add_column("eta", justify="right")
    table.add_column("$/hr", justify="right")
    table.add_column("kgCO2/hr", justify="right")
    table.add_column("carbon $/hr", justify="right")
    table.add_column("C_total $/hr", justify="right")
    table.add_column("E/Lbar $/hr", justify="right")
    table.add_column("U", justify="right", style="bold")
    for i, r in enumerate(rows, 1):
        lbar = r["mean_lifetime_hr"]
        table.add_row(
            str(i),
            str(r["cloud_region"]),
            str(r["mode"]),
            f"{lbar:.2f}" if lbar is not None else "inf",
            f"{r['effectiveness_eta']:.3f}",
            f"{r['price_per_hr_usd']:.3f}",
            f"{r['carbon_kg_per_hr']:.4f}",
            f"{r['carbon_cost_per_hr_usd']:.4f}",
            f"{r['total_cost_per_hr_usd']:.3f}",
            f"{r['amortized_migration_per_hr']:.4f}",
            f"{r['utility_u']:.4f}",
        )
    Console().print(table)
    best = rows[0]
    typer.echo(
        f"  Carbon: {type(carbon).__name__}, spot: {type(spot).__name__} "
        f"(CARBONSIGHT_API_URL > WATTTIME_* > synthetic).\n"
        f"  Top action: {best['cloud_region']} {best['mode']} at U={best['utility_u']:.4f}; "
        f"idle scores 0, so a negative U means wait."
    )


def schedule(
    yaml_path: Path = typer.Option(..., "--yaml", "-y", help="SkyPilot-style job YAML"),
    deadline_hours: float | None = typer.Option(
        None, "--deadline-hours", help="Deadline T in hours from now (default 1.5x job duration)"
    ),
    checkpoint_size_gb: float = typer.Option(
        0.0, "--checkpoint-size-gb", help="Checkpoint size in GB, for migration cost E"
    ),
    cold_start_minutes: float = typer.Option(
        6.0, "--cold-start-minutes", help="Cold start d in minutes, before checkpoint restore"
    ),
    carbon_price: float = typer.Option(
        50.0, "--carbon-price", help="Social cost of carbon, USD per metric ton"
    ),
    carbon_weight: float = typer.Option(
        1.0, "--carbon-weight", help="Weight on the carbon lever relative to dollars"
    ),
    egress_usd_per_gb: float = typer.Option(
        0.02, "--egress-usd-per-gb", help="Inter-region egress price for migration cost"
    ),
    current_region: str = typer.Option(
        "", "--current-region", help="Region already holding the checkpoint (r0); no egress for it"
    ),
    progress_hours: float = typer.Option(
        0.0, "--progress-hours", help="Compute-hours already done (p); raises V when behind"
    ),
    elapsed_hours: float = typer.Option(
        0.0, "--elapsed-hours", help="Wall-clock hours since the job started (t)"
    ),
    registry_path: Path | None = typer.Option(None, "--registry", help="Mapping registry JSON"),
    gpu_util: float | None = typer.Option(None, "--gpu-util", help="GPU utilization in [0, 1]"),
    nvidia_smi: bool = typer.Option(
        False, "--nvidia-smi", help="Sample utilization from nvidia-smi"
    ),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON"),
    top_n: int | None = typer.Option(None, "--top", help="Show only the best N candidates"),
) -> None:
    """Rank region x mode by joint utility U (no instances are launched)."""
    run_schedule(
        yaml_path,
        deadline_hours=deadline_hours,
        checkpoint_size_gb=checkpoint_size_gb,
        cold_start_minutes=cold_start_minutes,
        carbon_price=carbon_price,
        carbon_weight=carbon_weight,
        egress_usd_per_gb=egress_usd_per_gb,
        current_region=current_region,
        progress_hours=progress_hours,
        elapsed_hours=elapsed_hours,
        registry_path=registry_path,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        json_out=json_out,
        top_n=top_n,
    )


if __name__ == "__main__":
    typer.run(schedule)
