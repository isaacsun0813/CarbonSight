"""Experimental, advisory-only CLI for the one-shot static planner."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer
from carbonsight_core.planning import (
    PlanCandidate,
    PlanConstraints,
    build_region_plan_inputs,
    plan_static_job,
)
from carbonsight_core.watttime import WattTimeClient

from carbonsight_cli.commands.job_setup import (
    carbonsight_block_from_yaml,
    ensure_job_inputs,
    load_config,
    load_job_spec,
    load_registry,
    parse_finish_by,
    resolve_carbon_budget_kg,
    resolve_carbon_price,
    resolve_finish_by,
    resolve_registry_file,
)


def _candidate_dict(candidate: PlanCandidate) -> dict:
    data = asdict(candidate)
    data["mode"] = candidate.mode.value
    data["start_utc"] = candidate.start_utc.isoformat()
    data["finish_utc"] = candidate.finish_utc.isoformat()
    return data


def run_static_plan(
    yaml_path: Path,
    *,
    finish_by: datetime | None = None,
    carbon_budget_kg: float | None = None,
    carbon_price: float | None = None,
    max_cost_premium: float = 0.20,
    registry_path: Path | None = None,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    live_pricing: bool = False,
    static_pricing: bool = False,
    json_out: bool = False,
) -> None:
    """Run the experimental one-shot planner without launching or patching a task."""
    ensure_job_inputs(yaml_path, gpu_util)

    carbonsight_block = carbonsight_block_from_yaml(yaml_path)
    effective_finish_by = resolve_finish_by(
        finish_by, yaml_path, carbonsight_block=carbonsight_block,
    )
    if effective_finish_by is None:
        typer.echo(
            "Error: plan-static requires --finish-by or carbonsight.finish_by in YAML.",
            err=True,
        )
        raise typer.Exit(1)

    now_utc = datetime.now(UTC)
    if effective_finish_by <= now_utc:
        typer.echo("Error: finish-by must be in the future.", err=True)
        raise typer.Exit(1)

    job = load_job_spec(yaml_path, gpu_util=gpu_util, nvidia_smi=nvidia_smi)

    if not job.has_duration_estimate:
        typer.echo("Error: plan-static requires an explicit estimated duration.", err=True)
        raise typer.Exit(1)

    rpath = resolve_registry_file(registry_path)

    config = load_config(live_pricing=live_pricing, static_pricing=static_pricing)
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)

    registry = load_registry(rpath)
    constraints = PlanConstraints.from_finish_by(
        effective_finish_by,
        now_utc,
        carbon_budget_kg=resolve_carbon_budget_kg(
            carbon_budget_kg, yaml_path, carbonsight_block=carbonsight_block,
        ),
        max_cost_premium=max_cost_premium,
        lambda_co2_usd_per_kg=resolve_carbon_price(
            carbon_price, yaml_path, carbonsight_block=carbonsight_block,
        ),
    )
    regions = build_region_plan_inputs(
        registry, WattTimeClient(config), job, constraints,
    )
    result = plan_static_job(job, constraints, regions)
    payload = {
        "experimental": True,
        "advisory_only": True,
        "best": _candidate_dict(result.best) if result.best is not None else None,
        "feasible_count": len(result.feasible),
        "filtered_counts": {
            reason: len(candidates)
            for reason, candidates in result.filtered_out.items()
        },
    }

    if json_out:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo("Experimental static plan (advisory only; does not launch or wait):")
        if result.best is not None:
            best = result.best
            typer.echo(
                f"  Best: aws/{best.cloud_region}  mode={best.mode.value}  "
                f"start={best.start_utc.isoformat()}  finish={best.finish_utc.isoformat()}"
            )
            typer.echo(
                f"  Estimated IT carbon={best.carbon_kg:.3f}kg  "
                f"cost=${best.cost_usd:.2f}"
            )
        typer.echo(
            f"  Feasible candidates: {len(result.feasible)}  "
            + "  ".join(
                f"{reason} filtered: {len(candidates)}"
                for reason, candidates in result.filtered_out.items()
            )
        )

    if result.best is None:
        if not json_out:
            typer.echo("No feasible static plan found.", err=True)
        raise typer.Exit(1)


def plan_static_cmd(
    yaml_path: Path = typer.Option(..., "--yaml", "-y", help="Path to task YAML"),
    finish_by: str | None = typer.Option(
        None,
        "--finish-by",
        help="ISO-8601 deadline; otherwise uses YAML carbonsight.finish_by.",
    ),
    carbon_budget: float | None = typer.Option(
        None,
        "--carbon-budget",
        help="Hard carbon budget in kg CO2; otherwise uses YAML.",
    ),
    carbon_price: float | None = typer.Option(
        None,
        "--carbon-price",
        help="Shadow price in USD per kg CO2; otherwise uses YAML.",
    ),
    max_cost_premium: float = typer.Option(
        0.20,
        "--max-cost-premium",
        help="Maximum extra cost versus the cheapest feasible candidate.",
    ),
    registry_path: Path | None = typer.Option(
        None, "--registry", help="Override path to registry JSON",
    ),
    gpu_util: float | None = typer.Option(
        None, "--gpu-util", help="Observed GPU utilization in [0, 1].",
    ),
    nvidia_smi: bool = typer.Option(
        False, "--nvidia-smi", help="Sample GPU utilization from nvidia-smi.",
    ),
    live_pricing: bool = typer.Option(
        False, "--live-pricing", help="Use live AWS pricing where supported.",
    ),
    static_pricing: bool = typer.Option(
        False, "--static-pricing", help="Force static cost tables.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Output a machine-readable plan.",
    ),
) -> None:
    """EXPERIMENTAL: print a one-shot advisory plan; never launch or wait."""
    parsed_finish_by = parse_finish_by(finish_by) if finish_by else None
    run_static_plan(
        yaml_path,
        finish_by=parsed_finish_by,
        carbon_budget_kg=carbon_budget,
        carbon_price=carbon_price,
        max_cost_premium=max_cost_premium,
        registry_path=registry_path,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        live_pricing=live_pricing,
        static_pricing=static_pricing,
        json_out=json_out,
    )
