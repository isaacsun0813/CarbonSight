"""carbonsight run train.yaml [--dry-run] [--no-exec] [--managed] [--max-cost-premium N]"""

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from carbonsight_cli.commands.advise import (
    _REGISTRY_REL_PATH,
    _default_registry_path,
    _job_spec_from_yaml,
)
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import compute_actual_co2, estimate_option
from carbonsight_core.mapping.registry import Registry, _confidence
from carbonsight_core.models import EstimateResult
from carbonsight_core.preflight.quota import QuotaChecker
from carbonsight_core.watttime import WattTimeClient, WattTimeError


def _patch_yaml_with_region(yaml_path: Path, cloud: str, region: str) -> str:
    """Return patched YAML string with resources.cloud and resources.region set."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    data["resources"]["cloud"] = cloud
    data["resources"]["region"] = region
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _launch_skypilot(patched_yaml: str, *, managed: bool, yes: bool) -> int:
    """
    Write patched YAML to a temp file and call SkyPilot.
    Returns the subprocess exit code.

    sky launch        → on-demand / reserved instances
    sky jobs launch   → managed spot jobs (preemptible, auto-recovery)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(patched_yaml)
        tmp_path = Path(f.name)

    try:
        cmd = ["sky", "jobs", "launch"] if managed else ["sky", "launch"]
        if yes:
            cmd.append("--yes")
        cmd.append(str(tmp_path))

        typer.echo(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def run_cmd(
    yaml_path: Path = typer.Argument(..., help="Path to SkyPilot-style job YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print patched YAML only, do not launch"),
    no_exec: bool = typer.Option(False, "--no-exec", help="Pick region and patch YAML but skip launch"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Skip AWS GPU quota check"),
    managed: bool = typer.Option(False, "--managed", help="Use sky jobs launch (managed spot) instead of sky launch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Pass --yes to SkyPilot (skip confirmation prompt)"),
    max_cost_premium: float = typer.Option(
        0.20, "--max-cost-premium",
        help="Max extra cost vs cheapest region (0.20 = up to 20%% more). Greener region wins within budget.",
    ),
    registry_path: Path | None = typer.Option(None, "--registry", help="Override path to registry JSON"),
) -> None:
    """Pick the greenest affordable region, patch YAML, and launch via SkyPilot."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    # --- Load job spec and registry ---
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
    if not rpath.exists():
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)
    reg.load_json(rpath)

    # --- WattTime client ---
    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)
    wt = WattTimeClient(config)

    # --- Preflight: AWS quota check ---
    quota_checker = QuotaChecker() if not skip_preflight else None

    # --- Estimate all regions ---
    results: list[EstimateResult] = []
    for entry in reg.all_regions():
        if not entry.wt_regions or entry.provider.lower() != "aws":
            continue
        if quota_checker:
            qr = quota_checker.check_gpu_quota(
                entry.region_code, job.gpu_count, gpu_type=job.gpu_type
            )
            if not qr.allowed:
                typer.echo(f"  Skipping {entry.region_code}: {qr.reason}", err=True)
                continue
        try:
            conf = _confidence(entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency)
            res = estimate_option(job, entry.provider, entry.region_code, entry.wt_regions, conf, wt)
            results.append(res)
        except WattTimeError as e:
            typer.echo(f"  Warning {entry.region_code}: {e}", err=True)
        except Exception as e:
            typer.echo(f"  Warning {entry.region_code}: {e}", err=True)

    if not results:
        typer.echo("No regions available after quota and WattTime checks.", err=True)
        raise typer.Exit(1)

    # --- Apply cost-premium filter, pick greenest within budget ---
    results.sort(key=lambda r: r.expected_co2_kg_mean)
    min_cost = min(r.expected_cost_usd for r in results)
    cost_ceiling = min_cost * (1 + max_cost_premium)
    affordable = [r for r in results if r.expected_cost_usd <= cost_ceiling] or [results[0]]
    best = affordable[0]  # greenest within budget (already sorted by CO2)

    cloud, region = best.cloud, best.cloud_region
    typer.echo(
        f"\nChosen: {cloud}/{region}  "
        f"CO2={best.expected_co2_kg_mean:.2f}kg  "
        f"Cost=${best.expected_cost_usd:.2f}  "
        f"Confidence={best.mapping_confidence:.2f}"
    )

    patched = _patch_yaml_with_region(yaml_path, cloud, region)

    if dry_run:
        typer.echo("\n--- Patched YAML (dry run) ---")
        typer.echo(patched)
        return

    if no_exec:
        typer.echo("\n--- Patched YAML (--no-exec) ---")
        typer.echo(patched)
        typer.echo("Skipping launch (--no-exec).")
        return

    # --- Check SkyPilot is installed ---
    sky_check = subprocess.run(["sky", "--version"], capture_output=True)
    if sky_check.returncode != 0:
        typer.echo(
            "SkyPilot not found. Install it with: pip install skypilot[aws]\n"
            "Then re-run without --no-exec.",
            err=True,
        )
        raise typer.Exit(1)

    start_time = datetime.now(timezone.utc)
    exit_code = _launch_skypilot(patched, managed=managed, yes=yes)
    if exit_code != 0:
        raise typer.Exit(exit_code)

    end_time = datetime.now(timezone.utc)
    try:
        actual = compute_actual_co2(
            job=job,
            cloud=cloud,
            cloud_region=region,
            watttime_regions=best.watttime_regions,
            actual_start=start_time,
            actual_end=end_time,
            wt=wt,
            estimated_co2_kg=best.expected_co2_kg_mean,
        )
        pct = actual.co2_savings_vs_estimate_pct
        sign = "+" if pct >= 0 else ""
        typer.echo(
            f"\nActual CO\u2082: {actual.actual_co2_kg:.3f}kg | "
            f"Duration: {actual.actual_duration_hours:.1f}h | "
            f"Cost: ${actual.actual_cost_usd:.2f} | "
            f"vs estimate: {sign}{pct:.1f}%"
        )
    except (ValueError, WattTimeError) as e:
        typer.echo(f"\nWarning: could not compute actual CO\u2082: {e}", err=True)
