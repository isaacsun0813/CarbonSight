"""carbonsight run train.yaml [--dry-run] [--no-exec]"""

from pathlib import Path

import typer
import yaml

from carbonsight_cli.commands.advise import (
    _default_registry_path,
    _job_spec_from_yaml,
)
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import estimate_option
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.preflight.quota import QuotaChecker
from carbonsight_core.watttime import WattTimeClient, WattTimeError


def _patch_yaml_with_region(yaml_path: Path, cloud: str, region: str) -> str:
    """Patch SkyPilot YAML: set resources.region (and cloud). Return patched YAML string."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    data["resources"]["cloud"] = cloud
    data["resources"]["region"] = region
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def run_cmd(
    yaml_path: Path = typer.Argument(..., help="Path to SkyPilot-style job YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only output patched YAML"),
    no_exec: bool = typer.Option(False, "--no-exec", help="Do not launch SkyPilot"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Do not check AWS quota"),
) -> None:
    """Run advise + preflight + patch YAML + optionally launch via SkyPilot."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    job = _job_spec_from_yaml(yaml_path)
    reg = Registry()
    rpath = _default_registry_path()
    if not rpath.exists():
        for candidate in [
            Path.cwd() / "carbonsight" / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json",
            Path.cwd() / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json",
        ]:
            if candidate.exists():
                rpath = candidate
                break
    if not rpath.exists():
        typer.echo("No registry found.", err=True)
        raise typer.Exit(1)
    reg.load_json(rpath)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)
    wt = WattTimeClient(config)
    quota_checker = QuotaChecker() if not skip_preflight else None
    best: tuple[str, str] | None = None
    best_kg = float("inf")

    for entry in reg.all_regions():
        if entry.wt_regions and entry.provider.lower() == "aws":
            if quota_checker:
                qr = quota_checker.check_gpu_quota(entry.region_code, job.gpu_count)
                if not qr.allowed:
                    typer.echo(f"Skipping {entry.region_code}: {qr.reason}", err=True)
                    continue
            try:
                res = estimate_option(
                    job,
                    entry.provider,
                    entry.region_code,
                    entry.wt_regions,
                    0.35 * entry.s_source + 0.25 * entry.s_geo + 0.25 * entry.s_wt_stability + 0.15 * entry.s_recency,
                    wt,
                )
                if res.expected_co2_kg_mean < best_kg:
                    best_kg = res.expected_co2_kg_mean
                    best = (entry.provider, entry.region_code)
            except (WattTimeError, Exception):
                pass

    if not best:
        typer.echo("No region available (check WattTime and registry).", err=True)
        raise typer.Exit(1)

    cloud, region = best
    patched = _patch_yaml_with_region(yaml_path, cloud, region)
    typer.echo(f"Chosen region: {cloud}/{region}")
    typer.echo("--- Patched YAML ---")
    typer.echo(patched)

    if not dry_run and not no_exec:
        # SkyPilot launch would go here: subprocess.run(["sky", "jobs", "launch", ...])
        typer.echo("Launch not implemented. Use --dry-run or --no-exec to only patch.")
