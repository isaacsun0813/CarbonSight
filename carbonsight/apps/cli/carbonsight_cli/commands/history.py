"""carbonsight history — list recorded runs."""

from pathlib import Path

import typer
from carbonsight_core.tracking import RunLedger
from rich.console import Console
from rich.table import Table

from carbonsight_cli.commands.run import _resolve_db_path


def history_cmd(
    db_path: Path | None = typer.Option(
        None, "--db", help="SQLite DB path (default: ~/.carbonsight/runs.db).",
    ),
) -> None:
    """Show all recorded CarbonSight runs."""
    ledger = RunLedger(_resolve_db_path(db_path))
    runs = ledger.all()
    if not runs:
        typer.echo("No runs recorded yet.")
        return

    table = Table(title=f"CarbonSight runs ({len(runs)})")
    table.add_column("ID", style="dim")
    table.add_column("Date")
    table.add_column("Region", style="cyan")
    table.add_column("GPU")
    table.add_column("Hours", justify="right")
    table.add_column("Spot")
    table.add_column("CO2 (kg)", justify="right")
    table.add_column("Cost ($)", justify="right")
    for r in runs:
        table.add_row(
            r.run_id,
            r.created_utc.strftime("%Y-%m-%d %H:%M"),
            r.cloud_region,
            f"{r.gpu_type}x{r.gpu_count}",
            f"{r.duration_hours:.1f}",
            "yes" if r.use_spot else "no",
            f"{r.estimated_co2_kg:.2f}",
            f"{r.estimated_cost_usd:.2f}",
        )
    Console().print(table)
