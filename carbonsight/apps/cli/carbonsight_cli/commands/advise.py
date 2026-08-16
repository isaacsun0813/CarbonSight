"""carbonsight advise --yaml train.yaml [--explain] [--json]"""

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from carbonsight_core.carbon import CarbonProviderError, get_carbon_provider
from carbonsight_core.config import Config
from carbonsight_core.estimator.pricing import configure_pricing
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.telemetry.nvidia_smi import sample_mean_gpu_utilization_from_nvidia_smi
from carbonsight_core.watttime import WattTimeClient
from rich.console import Console
from rich.table import Table


def parse_sky_accelerators_string(acc: str) -> tuple[str, int]:
    """SkyPilot accelerators → (gpu_type, gpu_count). Supports 'A100:1' and '1x A100'."""
    gpu_count, gpu_type = 1, "A100"
    if not acc:
        return gpu_type, gpu_count
    acc = acc.strip()
    if "x" in acc or "X" in acc:
        parts = acc.replace("x", " ").replace("X", " ").split()
        if len(parts) >= 2:
            try:
                gpu_count = int(parts[0])
                gpu_type = parts[1].split(":")[0] if ":" in parts[1] else parts[1]
            except ValueError:
                gpu_type = parts[0]
    elif ":" in acc:
        left, right = acc.split(":", 1)
        gpu_type = left.strip()
        try:
            gpu_count = int(right.strip().split(":")[-1])
        except ValueError:
            pass
    return gpu_type, gpu_count


def parse_duration_hours(raw: Any) -> float:
    """YAML duration '1h' / '30m' → hours. Default 1.0."""
    if raw is None:
        return 1.0
    s = str(raw).strip().lower()
    if not s:
        return 1.0
    num = float(s.replace("h", "").replace("m", "").strip() or "1")
    if "m" in s:
        num /= 60.0
    return num if num > 0 else 1.0


def job_spec_from_sky_yaml(path: Path) -> JobSpec:
    """SkyPilot-style YAML → JobSpec.

    Optional ``carbonsight: { gpu_utilization: 0.75 }`` fixes GPU utilization for the power model
    (same effect as ``--gpu-util`` / ``--nvidia-smi`` on the CLI).
    """
    data: dict = yaml.safe_load(path.read_text()) or {}
    resources = data.get("resources") or {}
    acc = resources.get("accelerators") or ""
    gpu_type, gpu_count = parse_sky_accelerators_string(acc if isinstance(acc, str) else "")
    duration = parse_duration_hours(data.get("duration"))
    gpu_utilization: float | None = None
    cs = data.get("carbonsight")
    if isinstance(cs, dict) and cs.get("gpu_utilization") is not None:
        try:
            gpu_utilization = float(cs["gpu_utilization"])
        except (TypeError, ValueError):
            gpu_utilization = None
    return JobSpec(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        duration_hours=duration,
        cpu_count=resources.get("cpus"),
        mem_gib=resources.get("memory"),
        gpu_utilization=gpu_utilization,
    )


def apply_gpu_telemetry_cli(
    job: JobSpec,
    *,
    gpu_util: float | None,
    nvidia_smi: bool,
) -> tuple[JobSpec, bool]:
    """Apply CLI GPU telemetry. Returns ``(job, nvidia_smi_failed)``.

    Precedence: ``--gpu-util`` overrides everything; else ``--nvidia-smi`` samples this host;
    else YAML ``carbonsight.gpu_utilization`` on ``job`` is kept.
    """
    if gpu_util is not None:
        return job.model_copy(update={"gpu_utilization": gpu_util}), False
    if nvidia_smi:
        u = sample_mean_gpu_utilization_from_nvidia_smi()
        if u is None:
            return job, True
        return job.model_copy(update={"gpu_utilization": u}), False
    return job, False


def resolve_live_aws_pricing(
    config: Config,
    *,
    live_pricing: bool,
    static_pricing: bool,
) -> bool:
    """CLI/env precedence: --static-pricing > --live-pricing > CARBONSIGHT_LIVE_AWS_PRICING."""
    if static_pricing and live_pricing:
        typer.echo("Error: use only one of --live-pricing and --static-pricing.", err=True)
        raise typer.Exit(1)
    if static_pricing:
        return False
    if live_pricing:
        return True
    return config.live_aws_pricing


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
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    if gpu_util is not None and not (0.0 <= gpu_util <= 1.0):
        typer.echo("Error: --gpu-util must be between 0 and 1.", err=True)
        raise typer.Exit(1)

    job = job_spec_from_sky_yaml(yaml_path)
    job, nvidia_failed = apply_gpu_telemetry_cli(job, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    if nvidia_failed:
        typer.echo(
            "Warning: --nvidia-smi could not read utilization (no GPU driver here?). "
            "Using default sampled GPU power.",
            err=True,
        )
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(registry_path, carbonsight_package_root=package_root, cwd=Path.cwd())
    if not rpath.exists():
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)

    reg = Registry()
    reg.load_json(rpath)

    config = Config.from_env()
    configure_pricing(
        config,
        live_aws_pricing=resolve_live_aws_pricing(
            config, live_pricing=live_pricing, static_pricing=static_pricing,
        ),
    )
    try:
        carbon_provider = get_carbon_provider(config)
    except (CarbonProviderError, ValueError) as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err

    watt_time = WattTimeClient(config)
    ranking = AwsRegionRankingService(reg, watt_time, carbon_provider=carbon_provider)

    def _warn_estimate(region_code: str, err: BaseException) -> None:
        typer.echo(f"Warning: {region_code}: {err}", err=True)

    try:
        all_estimates = ranking.collect_estimates(
            job, use_spot=use_spot, on_estimate_error=_warn_estimate,
        )
    except CarbonProviderError as err:
        typer.echo(
            "Error: CarbonSight's carbon data source is unavailable, so no ranking "
            "can be produced.\n"
            f"  {err}\n"
            "  Nothing is estimated from defaults -- an invented carbon number would "
            "be indistinguishable from a real one.",
            err=True,
        )
        raise typer.Exit(1) from err

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
    table.add_column("CO2 (kg)", justify="right")
    table.add_column("CO2 p10-p90", justify="right")
    table.add_column("Cost (USD)", justify="right")
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
            f"  Green rank: CO2 emissions (live WattTime data). "
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
        help="Observed GPU utilization in [0, 1] (overrides YAML carbonsight.gpu_utilization).",
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
