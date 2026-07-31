"""carbonsight schedule — SkyNomad multi-lever ranking (cost + carbon + time + availability)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from carbonsight_cli.commands.advise import job_spec_from_sky_yaml
from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.spot.scheduler_service import ranked_as_json, schedule_job
from rich.console import Console
from rich.table import Table


def schedule(
    yaml_path: Path = typer.Option(..., "--yaml", "-y", help="SkyPilot-style job YAML"),
    deadline_hours: float = typer.Option(45.0, "--deadline-hours", help="Wall-clock deadline hours"),
    checkpoint_size_gb: float = typer.Option(100.0, "--checkpoint-size-gb", help="Checkpoint size GB"),
    carbon_price: float = typer.Option(
        50.0, "--carbon-price", help="Carbon price USD per metric ton for joint U"
    ),
    top_n: int = typer.Option(17, "--top", help="Show top N candidates"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON"),
    registry_path: Path | None = typer.Option(None, "--registry", help="Mapping registry JSON"),
) -> None:
    """Rank regions by joint utility U_s using carbon + spot + lifetime providers."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    job = job_spec_from_sky_yaml(yaml_path)
    job = job.model_copy(
        update={
            "deadline_hours": deadline_hours,
            "checkpoint_size_gb": checkpoint_size_gb,
            "carbon_price_usd_per_ton": carbon_price,
        }
    )

    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    reg: Registry | None = None
    if rpath.exists():
        reg = Registry()
        reg.load_json(rpath)

    result = schedule_job(job, registry=reg, config=Config.from_env(), top_n=top_n)
    rows = ranked_as_json(result)

    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return

    table = Table(title=f"SkyNomad schedule (deadline={deadline_hours}h, carbon=${carbon_price}/t)")
    table.add_column("#", justify="right")
    table.add_column("Region", style="cyan")
    table.add_column("U_s", justify="right")
    table.add_column("Spot $/GPU-h", justify="right")
    table.add_column("MOER", justify="right")
    table.add_column("CO2 kg", justify="right")
    table.add_column("Survival", justify="right")
    table.add_column("Lbar h", justify="right")
    for i, r in enumerate(rows, 1):
        table.add_row(
            str(i),
            str(r["region"]),
            f"{r['utility']:.3f}",
            f"{r['spot_price_usd_per_gpu_hr']:.3f}",
            f"{r['moer_lb_per_mwh']:.0f}",
            f"{r['carbon_kg']:.4f}",
            f"{r['survival']:.2f}",
            f"{r['lbar_hours']:.1f}",
        )
    Console().print(table)
    typer.echo(
        "  Providers: get_carbon_provider() "
        "(CARBONSIGHT_API_URL > WATTTIME_* > synthetic). "
        "Ranked by U_s = V*eta - C_total - E/Lbar."
    )


if __name__ == "__main__":
    typer.run(schedule)
