"""carbonsight advise --yaml train.yaml [--explain] [--json]"""

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import estimate_option
from carbonsight_core.mapping.registry import Registry, _confidence
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.watttime import WattTimeClient, WattTimeError

# Default registry: from this file up 4 levels to carbonsight repo root
_REGISTRY_REL_PATH = ("packages", "core", "carbonsight_core", "mapping", "seed_registry.json")


def _parse_accelerators(acc: str) -> tuple[str, int]:
    """Parse SkyPilot accelerators string to (gpu_type, gpu_count). Supports 'A100:1' and '1x A100'."""
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


def _parse_duration_hours(raw: Any) -> float:
    """Parse duration from YAML (e.g. '1h', '30m') to hours. Default 1.0."""
    if raw is None:
        return 1.0
    s = str(raw).strip().lower()
    if not s:
        return 1.0
    num = float(s.replace("h", "").replace("m", "").strip() or "1")
    if "m" in s:
        num /= 60.0
    return num if num > 0 else 1.0


def _job_spec_from_yaml(path: Path) -> JobSpec:
    """Parse SkyPilot-style YAML into JobSpec (gpu_type, gpu_count, duration, etc.)."""
    data: dict = yaml.safe_load(path.read_text()) or {}
    resources = data.get("resources") or {}
    acc = resources.get("accelerators") or ""
    gpu_type, gpu_count = _parse_accelerators(acc if isinstance(acc, str) else "")
    duration = _parse_duration_hours(data.get("duration"))
    return JobSpec(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        duration_hours=duration,
        cpu_count=resources.get("cpus"),
        mem_gib=resources.get("memory"),
    )


def _default_registry_path() -> Path:
    """Repo root is 4 levels up from this file (commands -> carbonsight_cli -> cli -> apps -> carbonsight)."""
    root = Path(__file__).resolve().parents[4]
    return root.joinpath(*_REGISTRY_REL_PATH)


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
) -> None:
    """Print regions ranked greenest-first, filtered to those within your cost tolerance."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    job = _job_spec_from_yaml(yaml_path)
    reg = Registry()
    rpath = registry_path or _default_registry_path()
    if not rpath.exists():
        cwd = Path.cwd()
        fallbacks = (
            cwd.joinpath("carbonsight", *_REGISTRY_REL_PATH),
            cwd.joinpath(*_REGISTRY_REL_PATH),
        )
        rpath = next((p for p in fallbacks if p.exists()), rpath)
    if rpath.exists():
        reg.load_json(rpath)
    else:
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        if json_out:
            typer.echo("[]")
        else:
            typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD to get recommendations.")
        return

    wt = WattTimeClient(config)
    results: list[EstimateResult] = []

    for entry in reg.all_regions():
        if entry.wt_regions and entry.provider.lower() == "aws":
            try:
                conf = _confidence(
                    entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency
                )
                res = estimate_option(
                    job,
                    entry.provider,
                    entry.region_code,
                    entry.wt_regions,
                    conf,
                    wt,
                )
                results.append(res)
            except WattTimeError as e:
                typer.echo(f"Warning: {entry.region_code}: {e}", err=True)
            except Exception as e:
                typer.echo(f"Warning: {entry.region_code}: {e}", err=True)

    if not results:
        if json_out:
            typer.echo("[]")
        else:
            typer.echo("No regions returned (check WattTime credentials and registry).")
        return

    # Sort all results by CO2 — green is always the primary objective.
    results.sort(key=lambda r: r.expected_co2_kg_mean)

    # Cost-premium filter: hide regions that cost more than (1 + max_cost_premium) × cheapest.
    # This keeps carbon as the ranking criterion while making cost a hard constraint.
    min_cost = min(r.expected_cost_usd for r in results)
    cost_ceiling = min_cost * (1 + max_cost_premium)
    affordable = [r for r in results if r.expected_cost_usd <= cost_ceiling]

    # Always show at least the single greenest region even if it blows the budget.
    visible = affordable if affordable else results[:1]

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
    for r in visible:
        premium = (r.expected_cost_usd - min_cost) / min_cost * 100
        table.add_row(
            f"{r.cloud}/{r.cloud_region}",
            f"{r.expected_co2_kg_mean:.2f}",
            f"{r.expected_co2_kg_p10:.1f}-{r.expected_co2_kg_p90:.1f}",
            f"${r.expected_cost_usd:.2f}",
            f"+{premium:.0f}%",
            f"{r.mapping_confidence:.2f}",
        )
    Console().print(table)

    hidden = len(results) - len(visible)
    if hidden:
        typer.echo(f"  {hidden} region(s) hidden — too expensive vs cheapest. Use --max-cost-premium to widen.")
    if explain:
        typer.echo(f"  Green rank: CO2 emissions (live WattTime data). Cost ceiling: ${cost_ceiling:.2f} (+{premium_pct}% above cheapest).")


if __name__ == "__main__":
    typer.run(advise)
