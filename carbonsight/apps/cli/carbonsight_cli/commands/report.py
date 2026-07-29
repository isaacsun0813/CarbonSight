"""carbonsight report — print cumulative savings vs baseline."""

from pathlib import Path

import typer
from carbonsight_core.tracking import RunLedger

from carbonsight_cli.commands.run import _resolve_db_path


def report_cmd(
    db_path: Path | None = typer.Option(
        None, "--db", help="SQLite DB path (default: ~/.carbonsight/runs.db).",
    ),
) -> None:
    """Print cumulative CO2 and cost savings vs baseline (us-east-1)."""
    ledger = RunLedger(_resolve_db_path(db_path))
    s = ledger.summary()
    if s.n_runs == 0:
        typer.echo("No runs recorded yet.")
        return

    typer.echo(f"Runs tracked: {s.n_runs}")
    typer.echo(
        f"CO2:  {s.total_estimated_co2_kg:.2f} kg chosen  vs  "
        f"{s.total_baseline_co2_kg:.2f} kg baseline  ->  "
        f"{s.total_co2_saved_kg:.2f} kg saved ({s.co2_saved_pct:.1f}%)"
    )
    typer.echo(
        f"Cost: ${s.total_estimated_cost_usd:.2f} chosen  vs  "
        f"${s.total_baseline_cost_usd:.2f} baseline  ->  "
        f"${s.total_cost_saved_usd:.2f} saved ({s.cost_saved_pct:.1f}%)"
    )
