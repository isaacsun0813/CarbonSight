"""carbonsight run train.yaml [--dry-run] [--no-exec] [--managed] [--max-cost-premium N]"""

import math
import os
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
from carbonsight_core.estimator.pricing import estimate_cost_usd
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.preflight.quota import QuotaChecker
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.scheduler import pick_lowest_carbon_start
from carbonsight_core.tracking import RunLedger, RunRecord
from carbonsight_core.checkpoint import (
    Framework,
    apply_checkpoint_patch_to_yaml,
    build_checkpoint_config,
    build_checkpoint_yaml_patch,
    detect_framework,
    extract_script_path_from_run_command,
    shim_local_path,
)
from carbonsight_core.watttime import WattTimeClient, WattTimeError

from carbonsight_cli.commands.advise import (
    job_spec_from_sky_yaml,
    apply_gpu_telemetry_cli,
    parse_duration_hours,
)

BASELINE_REGION = "us-east-1"


def patch_sky_yaml_with_cloud_region(
    yaml_path: Path, cloud: str, region: str, *, use_spot: bool = False,
) -> str:
    """Inject resources.cloud and resources.region into SkyPilot YAML; return full document text."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    data["resources"]["cloud"] = cloud
    data["resources"]["region"] = region
    if use_spot:
        data["resources"]["use_spot"] = True
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def launch_skypilot_with_patched_yaml(patched_yaml: str, *, managed: bool, yes: bool) -> int:
    """Write YAML to temp file, run sky launch or sky jobs launch. Returns process exit code."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(patched_yaml)
        tmp_path = Path(f.name)

    try:
        cmd = ["sky", "jobs", "launch"] if managed else ["sky", "launch"]
        if yes:
            cmd.append("--yes")
        cmd.append(str(tmp_path))

        typer.echo(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_db_path(db_opt: Path | None) -> Path:
    if db_opt is not None:
        return db_opt
    env = os.environ.get("CARBONSIGHT_DB")
    if env:
        return Path(env)
    return Path.home() / ".carbonsight" / "runs.db"


def run_launch(
    yaml_path: Path,
    *,
    dry_run: bool = False,
    no_exec: bool = False,
    skip_preflight: bool = False,
    managed: bool = False,
    yes: bool = False,
    max_cost_premium: float = 0.20,
    registry_path: Path | None = None,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    use_spot: bool = True,
    max_delay: str = "0h",
    db_path: Path | None = None,
    checkpoint: bool = True,
    checkpoint_bucket: str | None = None,
    checkpoint_interval: int = 500,
    script_path: Path | None = None,
) -> None:
    """Shared implementation for ``run`` and ``train``."""
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
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)

    watt_time = WattTimeClient(config)
    quota_checker = QuotaChecker() if not skip_preflight else None
    ranking = AwsRegionRankingService(reg, watt_time, quota_checker=quota_checker)

    def _quota_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _estimate_warn(region_code: str, err: BaseException) -> None:
        typer.echo(f"  Warning {region_code}: {err}", err=True)

    estimates = ranking.collect_estimates(
        job,
        use_spot=use_spot,
        on_quota_skip=_quota_skip,
        on_estimate_error=_estimate_warn,
    )

    if not estimates:
        typer.echo("No regions available after quota and WattTime checks.", err=True)
        raise typer.Exit(1)

    best = AwsRegionRankingService.pick_best_region_for_launch(estimates, max_cost_premium)
    if best is None:
        typer.echo("No region selected.", err=True)
        raise typer.Exit(1)

    cloud, region = best.cloud, best.cloud_region
    typer.echo(
        f"\nChosen: {cloud}/{region}  "
        f"CO2={best.expected_co2_kg_mean:.2f}kg  "
        f"Cost=${best.expected_cost_usd:.2f}  "
        f"Confidence={best.mapping_confidence:.2f}"
    )

    max_delay_hours = parse_duration_hours(max_delay)
    if max_delay_hours > 0:
        try:
            chosen_entry = next(
                e for e in reg.all_regions()
                if e.region_code == region and e.provider.lower() == "aws"
            )
            horizon = math.ceil(max_delay_hours + job.duration_hours)
            forecast = watt_time.get_forecast(
                chosen_entry.wt_regions[0][0],
                horizon_hours=horizon,
            )
            pts = forecast.get("data", [])
            if pts:
                sched = pick_lowest_carbon_start(pts, job.duration_hours, max_delay_hours)
                if sched.delay_hours < 0.01:
                    typer.echo("Schedule: run now (already the lowest-carbon window).")
                else:
                    typer.echo(
                        f"Schedule: wait {sched.delay_hours:.1f}h "
                        f"(start {sched.start_utc:%H:%M} UTC) -> "
                        f"-{sched.moer_reduction_pct:.1f}% carbon vs running now."
                    )
        except Exception as e:
            typer.echo(f"Warning: could not compute schedule: {e}", err=True)

    # Baseline estimate for tracking
    baseline_cost = estimate_cost_usd(
        job.gpu_type, job.gpu_count, job.duration_hours, BASELINE_REGION, use_spot=use_spot,
    )
    baseline_est = next(
        (e for e in estimates if e.cloud_region == BASELINE_REGION),
        None,
    )
    baseline_co2 = baseline_est.expected_co2_kg_mean if baseline_est else best.expected_co2_kg_mean

    rec = RunRecord(
        run_id=uuid.uuid4().hex[:12],
        created_utc=datetime.now(UTC),
        cloud_region=region,
        gpu_type=job.gpu_type,
        gpu_count=job.gpu_count,
        duration_hours=job.duration_hours,
        use_spot=use_spot,
        baseline_region=BASELINE_REGION,
        estimated_co2_kg=best.expected_co2_kg_mean,
        baseline_co2_kg=baseline_co2,
        estimated_cost_usd=best.expected_cost_usd,
        baseline_cost_usd=baseline_cost,
    )
    ledger = RunLedger(_resolve_db_path(db_path))
    ledger.record(rec)

    # Checkpoint resilience
    ckpt_patch = None
    if checkpoint:
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        script_source = ""
        if script_path and script_path.is_file():
            script_source = script_path.read_text()
        else:
            extracted = extract_script_path_from_run_command(raw.get("run", ""))
            if extracted:
                candidate = yaml_path.parent / extracted
                if not candidate.is_file():
                    candidate = Path.cwd() / extracted
                if candidate.is_file():
                    script_source = candidate.read_text()

        framework = detect_framework(script_source) if script_source else Framework.UNKNOWN
        task_name = raw.get("name", "carbonsight-job")
        original_run = raw.get("run", "").strip()

        ckpt_cfg = build_checkpoint_config(
            task_name, framework,
            save_interval_steps=checkpoint_interval,
            bucket_uri=checkpoint_bucket,
        )
        ckpt_patch = build_checkpoint_yaml_patch(ckpt_cfg, original_run, shim_local_path())

        if use_spot and not managed:
            managed = True
            typer.echo("Auto-enabling --managed for spot + checkpoint resilience.")

        typer.echo(f"Checkpoint: framework={framework.value}, storage={ckpt_cfg.storage_name or checkpoint_bucket}")

    if ckpt_patch:
        data = yaml.safe_load(yaml_path.read_text()) or {}
        if "resources" not in data:
            data["resources"] = {}
        data["resources"]["cloud"] = cloud
        data["resources"]["region"] = region
        if use_spot:
            data["resources"]["use_spot"] = True
        data = apply_checkpoint_patch_to_yaml(data, ckpt_patch)
        patched = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        patched = patch_sky_yaml_with_cloud_region(yaml_path, cloud, region, use_spot=use_spot)

    if dry_run:
        typer.echo("\n--- Patched YAML (dry run) ---")
        typer.echo(patched)
        return

    if no_exec:
        typer.echo("\n--- Patched YAML (--no-exec) ---")
        typer.echo(patched)
        typer.echo("Skipping launch (--no-exec).")
        return

    sky_check = subprocess.run(["sky", "--version"], capture_output=True)
    if sky_check.returncode != 0:
        typer.echo(
            "SkyPilot not found. Install it with: pip install skypilot[aws]\n"
            "Then re-run without --no-exec.",
            err=True,
        )
        raise typer.Exit(1)

    start_time = datetime.now(UTC)
    exit_code = launch_skypilot_with_patched_yaml(patched, managed=managed, yes=yes)
    if exit_code != 0:
        raise typer.Exit(exit_code)

    end_time = datetime.now(UTC)
    try:
        actual = JobCarbonEstimator(watt_time).compute_actual_run(
            job=job,
            cloud=cloud,
            cloud_region=region,
            watttime_regions=best.watttime_regions,
            actual_start=start_time,
            actual_end=end_time,
            estimated_co2_kg=best.expected_co2_kg_mean,
        )
        pct = actual.co2_savings_vs_estimate_pct
        sign = "+" if pct >= 0 else ""
        typer.echo(
            f"\nActual CO₂: {actual.actual_co2_kg:.3f}kg | "
            f"Duration: {actual.actual_duration_hours:.1f}h | "
            f"Cost: ${actual.actual_cost_usd:.2f} | "
            f"vs estimate: {sign}{pct:.1f}%"
        )
    except (ValueError, WattTimeError) as e:
        typer.echo(f"\nWarning: could not compute actual CO₂: {e}", err=True)


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
    spot: bool = typer.Option(
        True,
        "--spot/--no-spot",
        help="Use spot instances for cost estimates and YAML patch (default: on).",
    ),
    max_delay: str = typer.Option(
        "0h",
        "--max-delay",
        help="Max hours to delay start for greener carbon (e.g. 6h). 0h = run immediately.",
    ),
    db_path: Path | None = typer.Option(
        None, "--db", help="SQLite DB path for run tracking (default: ~/.carbonsight/runs.db).",
    ),
    checkpoint: bool = typer.Option(
        True,
        "--checkpoint/--no-checkpoint",
        help="Enable checkpoint resilience for spot training (default: on).",
    ),
    checkpoint_bucket: str | None = typer.Option(
        None,
        "--checkpoint-bucket",
        help="S3 bucket URI for checkpoint storage (default: auto-managed SkyPilot Storage).",
    ),
    checkpoint_interval: int = typer.Option(
        500,
        "--checkpoint-interval",
        help="Save checkpoint every N training steps.",
    ),
) -> None:
    """Pick the greenest affordable region, patch YAML, and launch via SkyPilot."""
    run_launch(
        yaml_path,
        dry_run=dry_run,
        no_exec=no_exec,
        skip_preflight=skip_preflight,
        managed=managed,
        yes=yes,
        max_cost_premium=max_cost_premium,
        registry_path=registry_path,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        use_spot=spot,
        max_delay=max_delay,
        db_path=db_path,
        checkpoint=checkpoint,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_interval=checkpoint_interval,
    )
