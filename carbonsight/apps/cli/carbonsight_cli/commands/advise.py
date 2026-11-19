"""carbonsight advise --yaml train.yaml [--explain] [--json]"""

import json
from pathlib import Path

import typer
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.watttime import WattTimeClient
from rich.console import Console
from rich.table import Table

from carbonsight_cli.commands.job_setup import (
    ensure_job_inputs,
    load_config,
    load_job_spec,
    load_registry,
    resolve_registry_file,
)


def run_advise(
    yaml_path: Path,
    *,
    explain: bool = False,
    json_out: bool = False,
    registry_path: Path | None = None,
    max_cost_premium: float = 0.20,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    live_pricing: bool = False,
    static_pricing: bool = False,
    use_spot: bool = False,
) -> None:
    """Shared implementation for ``advise`` and ``train``."""
    ensure_job_inputs(yaml_path, gpu_util)

    job = load_job_spec(yaml_path, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    reg = load_registry(resolve_registry_file(registry_path))

    config = load_config(live_pricing=live_pricing, static_pricing=static_pricing)
    if not config.watttime_username or not config.watttime_password:
        if json_out:
            typer.echo("[]")
        else:
            typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD to get recommendations.")
        return

    watt_time = WattTimeClient(config)
    ranking = AwsRegionRankingService(reg, watt_time)

    def _warn_estimate(region_code: str, err: BaseException) -> None:
        typer.echo(f"Warning: {region_code}: {err}", err=True)

    all_estimates = ranking.collect_estimates(job, use_spot=use_spot, on_estimate_error=_warn_estimate)

    if not all_estimates:
        if json_out:
            typer.echo("[]")
        else:
            typer.echo("No regions returned (check WattTime credentials and registry).")
        return

    visible, min_cost, cost_ceiling = AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(
        all_estimates, max_cost_premium
    )

    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in visible], indent=2))
        return

    premium_pct = int(max_cost_premium * 100)
    table = Table(title=f"Greenest regions within {premium_pct}% of cheapest (${min_cost:.2f})")
    table.add_column("Region", style="cyan")
    table.add_column("IT CO2 mean (kg)" if job.has_duration_estimate else "IT CO2 (kg/h)", justify="right")
    table.add_column("IT CO2 p10-p90" if job.has_duration_estimate else "IT CO2 p10-p90 (kg/h)", justify="right")
    table.add_column("Cost (USD)" if job.has_duration_estimate else "Cost (USD/h)", justify="right")
    table.add_column("vs cheapest", justify="right")
    table.add_column("Confidence", justify="right")
    for row in visible:
        premium = (row.expected_cost_usd - min_cost) / min_cost * 100 if min_cost else 0.0
        table.add_row(
            f"{row.cloud}/{row.cloud_region}",
            f"{row.expected_co2_kg_mean:.2f}",
            f"{row.expected_co2_kg_p10:.1f}-{row.expected_co2_kg_p90:.1f}",
            f"${row.expected_cost_usd:.2f}",
            f"+{premium:.0f}%",
            f"{row.mapping_confidence:.2f}",
        )
    Console().print(table)

    hidden = len(all_estimates) - len(visible)
    if hidden:
        typer.echo(
            f"  {hidden} region(s) hidden — too expensive vs cheapest. Use --max-cost-premium to widen."
        )
    if explain:
        typer.echo(
            f"  Green rank: IT operational CO2 (live WattTime data; facility overhead excluded). "
            f"Cost ceiling: ${cost_ceiling:.2f} (+{premium_pct}% above cheapest)."
        )


def advise(
    yaml_path: Path = typer.Option(..., "--yaml", "-y", help="Path to SkyPilot-style job YAML"),
    explain: bool = typer.Option(False, "--explain", help="Show top drivers"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    max_cost_premium: float = typer.Option(
        0.20, "--max-cost-premium",
        help=(
            "How much extra cost (as a fraction) you'll tolerate for a greener region. "
            "Default 0.20 = up to 20%% above the cheapest region. "
            "Use 0.0 to see only the cheapest option; use 1.0 to always pick the greenest regardless of price."
        ),
    ),
    gpu_util: float | None = typer.Option(
        None,
        "--gpu-util",
        help="Observed GPU utilization in [0, 1] for the power model.",
    ),
    nvidia_smi: bool = typer.Option(
        False,
        "--nvidia-smi",
        help="Sample GPU utilization from nvidia-smi on this machine (overrides YAML when successful).",
    ),
    live_pricing: bool = typer.Option(
        False,
        "--live-pricing",
        help="Use live AWS EC2 spot prices for cost estimates (when --spot).",
    ),
    static_pricing: bool = typer.Option(
        False,
        "--static-pricing",
        help="Force static cost tables instead of live AWS pricing.",
    ),
    spot: bool = typer.Option(
        False,
        "--spot/--no-spot",
        help="Estimate cost using spot pricing (static 35%% or live when --live-pricing).",
    ),
) -> None:
    """Print regions ranked greenest-first, filtered to those within your cost tolerance."""
    run_advise(
        yaml_path,
        explain=explain,
        json_out=json_out,
        registry_path=registry_path,
        max_cost_premium=max_cost_premium,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        live_pricing=live_pricing,
        static_pricing=static_pricing,
        use_spot=spot,
    )


if __name__ == "__main__":
    typer.run(advise)
