"""carbonsight dashboard — Rich terminal visualization of savings and checkpoint impact."""

from __future__ import annotations

import math
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from carbonsight_core.tracking import RunLedger, RunRecord

from carbonsight_cli.commands.run import _resolve_db_path

MEAN_SPOT_LIFETIME_HOURS = 4.0


def _preemption_prob(duration_hours: float) -> float:
    return 1.0 - math.exp(-duration_hours / MEAN_SPOT_LIFETIME_HOURS)


def _bar(value: float, max_val: float, width: int = 30) -> str:
    if max_val <= 0:
        return ""
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _pct_text(saved: float, baseline: float, *, invert: bool = False) -> Text:
    if baseline <= 0:
        return Text("--", style="dim")
    pct = saved / baseline * 100.0
    style = "green" if pct > 0 else ("red" if pct < 0 else "dim")
    sign = "-" if not invert else "+"
    return Text(f"{sign}{abs(pct):.0f}%", style=style)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _summary_panel(runs: list[RunRecord], summary) -> Panel:
    grid = Table.grid(padding=(0, 6))
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")

    co2 = Text()
    co2.append(f"{summary.total_co2_saved_kg:.1f} kg\n", style="bold green")
    co2.append(f"{summary.co2_saved_pct:.0f}% vs baseline", style="green")

    cost = Text()
    cost.append(f"${summary.total_cost_saved_usd:.2f}\n", style="bold cyan")
    cost.append(f"{summary.cost_saved_pct:.0f}% vs baseline", style="cyan")

    spot_n = sum(1 for r in runs if r.use_spot)
    rtext = Text()
    rtext.append(f"{summary.n_runs} runs\n", style="bold yellow")
    rtext.append(f"{spot_n} on spot", style="yellow")

    header = Table.grid(padding=(0, 6))
    header.add_column(justify="center")
    header.add_column(justify="center")
    header.add_column(justify="center")
    header.add_row(
        Text("CARBON SAVED", style="bold dim"),
        Text("COST SAVED", style="bold dim"),
        Text("TRACKED", style="bold dim"),
    )
    header.add_row(co2, cost, rtext)
    return Panel(header, title="[bold]CarbonSight Dashboard[/bold]", border_style="green")


def _checkpoint_panel(runs: list[RunRecord]) -> Panel | None:
    spot_runs = [r for r in runs if r.use_spot]
    if not spot_runs:
        return None

    total_hours = sum(r.duration_hours for r in spot_runs)
    expected_preemptions = sum(_preemption_prob(r.duration_hours) for r in spot_runs)

    wasted_h_no_ckpt = sum(
        r.duration_hours * 0.5 * _preemption_prob(r.duration_hours) for r in spot_runs
    )
    wasted_co2_no_ckpt = sum(
        r.estimated_co2_kg * 0.5 * _preemption_prob(r.duration_hours) for r in spot_runs
    )
    wasted_cost_no_ckpt = sum(
        r.estimated_cost_usd * 0.5 * _preemption_prob(r.duration_hours) for r in spot_runs
    )

    ckpt_loss_per = 10.0 / 60.0
    wasted_h_with_ckpt = expected_preemptions * ckpt_loss_per

    saved_h = max(0, wasted_h_no_ckpt - wasted_h_with_ckpt)

    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim", min_width=28)
    tbl.add_column(style="bold", min_width=18)
    tbl.add_column(style="dim")

    tbl.add_row("Spot runs", f"{len(spot_runs)}", f"{total_hours:.1f}h total compute")
    tbl.add_row(
        "Expected preemptions",
        f"{expected_preemptions:.1f}",
        "~1 per 4h of spot runtime",
    )
    tbl.add_row("")
    tbl.add_row(
        "[red]Without checkpointing[/red]",
        f"[red]{wasted_h_no_ckpt:.1f}h wasted[/red]",
        f"[red]{wasted_co2_no_ckpt:.1f} kg CO2 / ${wasted_cost_no_ckpt:.0f}[/red]",
    )
    tbl.add_row(
        "[green]With checkpointing[/green]",
        f"[green]{wasted_h_with_ckpt:.1f}h wasted[/green]",
        "[green]~10 min lost per preemption[/green]",
    )
    tbl.add_row("")

    co2_preserved = wasted_co2_no_ckpt * (1 - wasted_h_with_ckpt / wasted_h_no_ckpt) if wasted_h_no_ckpt else 0
    tbl.add_row(
        "[bold]Compute preserved[/bold]",
        f"[bold green]{saved_h:.1f}h saved[/bold green]",
        f"[bold green]{co2_preserved:.1f} kg CO2 preserved[/bold green]",
    )

    return Panel(tbl, title="[bold]Checkpoint Resilience Impact[/bold] [dim](projected)[/dim]", border_style="magenta")


def _history_table(runs: list[RunRecord]) -> Table:
    tbl = Table(title="Run History", box=box.SIMPLE_HEAVY, row_styles=["", "dim"])
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Date", no_wrap=True)
    tbl.add_column("Region", style="cyan", no_wrap=True)
    tbl.add_column("GPU", no_wrap=True)
    tbl.add_column("Hrs", justify="right")
    tbl.add_column("Spot", justify="center")
    tbl.add_column("CO2", justify="right", no_wrap=True)
    tbl.add_column("Cost", justify="right", no_wrap=True)

    for r in runs:
        co2_pct = (r.baseline_co2_kg - r.estimated_co2_kg) / r.baseline_co2_kg * 100 if r.baseline_co2_kg else 0
        cost_pct = (r.baseline_cost_usd - r.estimated_cost_usd) / r.baseline_cost_usd * 100 if r.baseline_cost_usd else 0

        co2_cell = Text()
        co2_cell.append(f"{r.estimated_co2_kg:.2f}kg ")
        co2_cell.append(f"-{co2_pct:.0f}%", style="green" if co2_pct > 0 else "red")

        cost_cell = Text()
        cost_cell.append(f"${r.estimated_cost_usd:.0f} ")
        cost_cell.append(f"-{cost_pct:.0f}%", style="green" if cost_pct > 0 else "red")

        tbl.add_row(
            r.run_id[:8],
            r.created_utc.strftime("%m-%d %H:%M"),
            r.cloud_region,
            f"{r.gpu_type}x{r.gpu_count}",
            f"{r.duration_hours:.0f}",
            "[green]yes[/green]" if r.use_spot else "no",
            co2_cell,
            cost_cell,
        )
    return tbl


def _carbon_bars(runs: list[RunRecord], summary) -> Panel:
    lines: list[Text] = []

    max_val = max(summary.total_baseline_co2_kg, summary.total_estimated_co2_kg, 0.01)

    base_line = Text()
    base_line.append("  Baseline (us-east-1)  ", style="dim")
    base_line.append(_bar(summary.total_baseline_co2_kg, max_val), style="red")
    base_line.append(f"  {summary.total_baseline_co2_kg:.1f} kg CO2", style="bold")
    lines.append(base_line)

    chosen_line = Text()
    chosen_line.append("  Your runs (chosen)    ", style="dim")
    chosen_line.append(_bar(summary.total_estimated_co2_kg, max_val), style="green")
    chosen_line.append(f"  {summary.total_estimated_co2_kg:.1f} kg CO2", style="bold")
    saved_pct = f"  -{summary.co2_saved_pct:.0f}%" if summary.co2_saved_pct > 0 else ""
    chosen_line.append(saved_pct, style="bold green")
    lines.append(chosen_line)

    # Per-region breakdown if multiple regions used
    regions: dict[str, float] = {}
    for r in runs:
        regions[r.cloud_region] = regions.get(r.cloud_region, 0) + r.estimated_co2_kg
    if len(regions) > 1:
        lines.append(Text(""))
        for region, co2 in sorted(regions.items(), key=lambda x: x[1]):
            rline = Text()
            rline.append(f"  {region:<22}", style="cyan")
            rline.append(_bar(co2, max_val), style="cyan")
            rline.append(f"  {co2:.1f} kg", style="dim")
            lines.append(rline)

    body = Text("\n").join(lines)
    return Panel(body, title="[bold]Carbon Comparison[/bold]", border_style="blue")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def dashboard_cmd(
    db_path: Path | None = typer.Option(
        None, "--db", help="SQLite DB path (default: ~/.carbonsight/runs.db).",
    ),
) -> None:
    """Visual dashboard of carbon/cost savings and checkpoint resilience impact."""
    console = Console()
    ledger = RunLedger(_resolve_db_path(db_path))
    runs = ledger.all()
    summary = ledger.summary()

    if not runs:
        console.print(Panel(
            "[dim]No runs recorded yet. Use [bold]carbonsight run[/bold] or "
            "[bold]carbonsight train --launch[/bold] to get started.[/dim]",
            title="CarbonSight Dashboard",
            border_style="dim",
        ))
        return

    console.print()
    console.print(_summary_panel(runs, summary))
    console.print()

    ckpt = _checkpoint_panel(runs)
    if ckpt:
        console.print(ckpt)
        console.print()

    console.print(_carbon_bars(runs, summary))
    console.print()
    console.print(_history_table(runs))
    console.print()
