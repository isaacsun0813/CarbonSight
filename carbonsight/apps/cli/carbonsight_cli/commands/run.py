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
from carbonsight_core.checkpoint import (
    Framework,
    apply_checkpoint_patch_to_yaml,
    build_checkpoint_config,
    build_checkpoint_yaml_patch,
    detect_framework,
    extract_script_path_from_run_command,
    shim_local_path,
)
from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.cloud.aws.quota import QuotaChecker
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
from carbonsight_core.estimator.pricing import configure_pricing, estimate_cost_usd
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.planning import (
    JobMode,
    PlanConstraints,
    DynamicPlanResult,
    build_region_plan_inputs,
    plan_dynamic_job,
)
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.scheduler import pick_lowest_carbon_start
from carbonsight_core.tracking import RunLedger, RunRecord
from carbonsight_core.watttime import WattTimeClient, WattTimeError

from carbonsight_cli.commands.advise import (
    apply_gpu_telemetry_cli,
    carbonsight_block_from_yaml,
    job_spec_from_sky_yaml,
    parse_duration_hours,
    parse_finish_by,
    resolve_carbon_budget_kg,
    resolve_carbon_price,
    resolve_finish_by,
    resolve_live_aws_pricing,
    strip_carbonsight_only_yaml_keys,
)

BASELINE_REGION = "us-east-1"


def _echo_dynamic_plan(dyn) -> None:
    """Print concise dynamic plan trajectory."""
    typer.echo("\nDynamic plan (advisory simulation):")
    typer.echo(
        f"  Deadline met: {dyn.deadline_met}  "
        f"Cost: ${dyn.total_cost_usd:.2f}  Carbon: {dyn.total_carbon_kg:.2f}kg"
    )
    if dyn.initial_launch:
        typer.echo(
            f"  Initial launch: {dyn.initial_launch.region} "
            f"({dyn.initial_launch.mode.value})"
        )
    for ev in dyn.events[:12]:
        typer.echo(
            f"  {ev.time_utc:%Y-%m-%d %H:%M}Z {ev.event_type.value} "
            f"{ev.region or '-'} progress={ev.progress_hours:.2f}h"
        )
    if len(dyn.events) > 12:
        typer.echo(f"  ... {len(dyn.events) - 12} more events")


def _apply_cloud_region_to_resources(
    resources: dict, cloud: str, region: str, *, use_spot: bool = False,
) -> None:
    """Set SkyPilot ``resources.infra`` and drop legacy cloud/region/zone keys."""
    resources["infra"] = f"{cloud}/{region}"
    for key in ("cloud", "region", "zone"):
        resources.pop(key, None)
    if use_spot:
        resources["use_spot"] = True


def patch_sky_yaml_with_cloud_region(
    yaml_path: Path, cloud: str, region: str, *, use_spot: bool = False,
) -> str:
    """Inject resources.infra (cloud/region) into SkyPilot YAML; return SkyPilot-ready text."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    _apply_cloud_region_to_resources(data["resources"], cloud, region, use_spot=use_spot)
    return yaml.dump(
        strip_carbonsight_only_yaml_keys(data),
        default_flow_style=False,
        sort_keys=False,
    )


def _require_sky_cli() -> None:
    sky_check = subprocess.run(["sky", "--version"], capture_output=True)
    if sky_check.returncode != 0:
        typer.echo(
            "SkyPilot not found. Install it with: pip install skypilot[aws]",
            err=True,
        )
        raise typer.Exit(1)


def launch_skypilot_with_patched_yaml(
    patched_yaml: str,
    *,
    managed: bool,
    yes: bool,
    dryrun: bool = False,
    down: bool = False,
    cwd: Path | None = None,
) -> int:
    """Write YAML to temp file, run sky launch or sky jobs launch. Returns process exit code."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(patched_yaml)
        tmp_path = Path(f.name)

    try:
        cmd = ["sky", "jobs", "launch"] if managed else ["sky", "launch"]
        if yes:
            cmd.append("--yes")
        if dryrun:
            cmd.append("--dryrun")
        if down:
            cmd.append("--down")
        cmd.append(str(tmp_path))

        typer.echo(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=cwd).returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_db_path(db_opt: Path | None) -> Path:
    if db_opt is not None:
        return db_opt
    env = os.environ.get("CARBONSIGHT_DB")
    if env:
        return Path(env)
    return Path.home() / ".carbonsight" / "runs.db"


def _echo_finish_by_source(effective_finish_by: datetime, cli_finish_by: datetime | None) -> None:
    source = "CLI" if cli_finish_by is not None else "YAML"
    typer.echo(f"Finish-by ({source}): {effective_finish_by.isoformat()}")


def _pick_region_with_dynamic_plan(
    reg: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    estimates: list[EstimateResult],
    *,
    effective_finish_by: datetime,
    effective_carbon_budget: float | None,
    effective_carbon_price: float,
    max_cost_premium: float,
) -> tuple[str, bool, EstimateResult, DynamicPlanResult]:
    now_utc = datetime.now(UTC)
    if effective_finish_by <= now_utc:
        typer.echo("Error: finish-by must be in the future.", err=True)
        raise typer.Exit(1)

    visible, _, _ = AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(
        estimates, max_cost_premium,
    )
    allowed = frozenset(e.cloud_region for e in visible)
    constraints = PlanConstraints.from_finish_by(
        effective_finish_by,
        now_utc,
        carbon_budget_kg=effective_carbon_budget,
        max_cost_premium=max_cost_premium,
        lambda_co2_usd_per_kg=effective_carbon_price,
    )
    try:
        region_inputs = build_region_plan_inputs(
            reg, watt_time, job, constraints, region_codes=allowed,
        )
    except WattTimeError as err:
        typer.echo(f"Error fetching forecasts for dynamic plan: {err}", err=True)
        raise typer.Exit(1)
    if not region_inputs:
        typer.echo("No regions with WattTime forecasts for dynamic planning.", err=True)
        raise typer.Exit(1)

    dyn = plan_dynamic_job(job, constraints, region_inputs)
    _echo_dynamic_plan(dyn)
    if not dyn.deadline_met or dyn.initial_launch is None:
        typer.echo("Dynamic plan: no feasible launch before finish-by.", err=True)
        raise typer.Exit(1)

    region = dyn.initial_launch.region
    use_spot = dyn.initial_launch.mode == JobMode.SPOT
    best = next((e for e in estimates if e.cloud_region == region), None)
    if best is None:
        typer.echo(f"No estimate row for planned region {region}.", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"\nChosen (dynamic): aws/{region}  "
        f"mode={dyn.initial_launch.mode.value}  "
        f"finish-by={effective_finish_by.isoformat()}"
    )
    return region, use_spot, best, dyn


def _pick_static_region(
    estimates: list[EstimateResult],
    max_cost_premium: float,
) -> tuple[str, str, EstimateResult]:
    best = AwsRegionRankingService.pick_best_region_for_launch(estimates, max_cost_premium)
    if best is None:
        typer.echo("No region selected.", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"\nChosen: {best.cloud}/{best.cloud_region}  "
        f"CO2={best.expected_co2_kg_mean:.2f}kg  "
        f"Cost=${best.expected_cost_usd:.2f}  "
        f"Confidence={best.mapping_confidence:.2f}"
    )
    return best.cloud, best.cloud_region, best


def _maybe_echo_carbon_schedule(
    reg: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    region: str,
    max_delay: str,
) -> None:
    max_delay_hours = parse_duration_hours(max_delay)
    if max_delay_hours <= 0:
        return
    try:
        chosen_entry = reg.get_entry("aws", region)
        if chosen_entry is None or not chosen_entry.wt_regions:
            return
        horizon = math.ceil(max_delay_hours + job.duration_hours)
        forecast = watt_time.get_forecast(
            chosen_entry.wt_regions[0][0],
            horizon_hours=horizon,
        )
        pts = forecast.get("data", [])
        if not pts:
            return
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
    live_pricing: bool = False,
    static_pricing: bool = False,
    finish_by: datetime | None = None,
    carbon_budget_kg: float | None = None,
    carbon_price: float | None = None,
    validate_sky: bool = False,
    sky_smoke: bool = False,
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
    configure_pricing(
        config,
        live_aws_pricing=resolve_live_aws_pricing(
            config, live_pricing=live_pricing, static_pricing=static_pricing,
        ),
    )
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)

    watt_time = WattTimeClient(config)
    quota_checker = QuotaChecker() if not skip_preflight else None
    availability_checker = InstanceAvailabilityChecker(config) if not skip_preflight else None
    enabled_regions_provider = EnabledRegionsProvider(config) if not skip_preflight else None
    ranking = AwsRegionRankingService(
        reg,
        watt_time,
        quota_checker=quota_checker,
        availability_checker=availability_checker,
        enabled_regions_provider=enabled_regions_provider,
    )

    def _enabled_region_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _quota_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _availability_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _estimate_warn(region_code: str, err: BaseException) -> None:
        typer.echo(f"  Warning {region_code}: {err}", err=True)

    estimates = ranking.collect_estimates(
        job,
        use_spot=use_spot,
        on_enabled_region_skip=_enabled_region_skip,
        on_quota_skip=_quota_skip,
        on_availability_skip=_availability_skip,
        on_estimate_error=_estimate_warn,
    )

    if not estimates:
        typer.echo(
            "No regions available after enabled-region, quota, availability, and WattTime checks.",
            err=True,
        )
        raise typer.Exit(1)

    carbonsight_block = carbonsight_block_from_yaml(yaml_path)
    effective_finish_by = resolve_finish_by(
        finish_by, yaml_path, carbonsight_block=carbonsight_block,
    )
    effective_carbon_budget = resolve_carbon_budget_kg(
        carbon_budget_kg, yaml_path, carbonsight_block=carbonsight_block,
    )
    effective_carbon_price = resolve_carbon_price(
        carbon_price, yaml_path, carbonsight_block=carbonsight_block,
    )

    cloud = "aws"
    region: str
    best: EstimateResult

    if effective_finish_by is not None:
        _echo_finish_by_source(effective_finish_by, finish_by)
        region, use_spot, best, _dyn = _pick_region_with_dynamic_plan(
            reg,
            watt_time,
            job,
            estimates,
            effective_finish_by=effective_finish_by,
            effective_carbon_budget=effective_carbon_budget,
            effective_carbon_price=effective_carbon_price,
            max_cost_premium=max_cost_premium,
        )
    else:
        cloud, region, best = _pick_static_region(estimates, max_cost_premium)
        _maybe_echo_carbon_schedule(reg, watt_time, job, region, max_delay)

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
        _apply_cloud_region_to_resources(data["resources"], cloud, region, use_spot=use_spot)
        data = apply_checkpoint_patch_to_yaml(data, ckpt_patch)
        patched = yaml.dump(
            strip_carbonsight_only_yaml_keys(data),
            default_flow_style=False,
            sort_keys=False,
        )
    else:
        patched = patch_sky_yaml_with_cloud_region(yaml_path, cloud, region, use_spot=use_spot)

    sky_cwd = yaml_path.parent.resolve()

    if dry_run:
        typer.echo("\n--- Patched YAML for SkyPilot (dry run) ---")
        typer.echo(patched)
        return

    if no_exec:
        typer.echo("\n--- Patched YAML for SkyPilot (--no-exec) ---")
        typer.echo(patched)
        typer.echo("Skipping launch (--no-exec).")
        return

    if validate_sky:
        _require_sky_cli()
        typer.echo("\nValidating patched YAML with SkyPilot (`sky launch --dryrun`)...")
        exit_code = launch_skypilot_with_patched_yaml(
            patched, managed=managed, yes=True, dryrun=True, cwd=sky_cwd,
        )
        if exit_code == 0:
            typer.echo("SkyPilot accepted the patched YAML.")
        raise typer.Exit(exit_code)

    if sky_smoke:
        _require_sky_cli()
        smoke_data = yaml.safe_load(patched) or {}
        smoke_data["run"] = "echo 'carbonsight sky-smoke ok'\n"
        smoke_data.pop("file_mounts", None)
        smoke_data.pop("envs", None)
        setup = smoke_data.get("setup")
        if isinstance(setup, str) and "/ckpt" in setup:
            smoke_data["setup"] = "echo 'carbonsight sky-smoke setup'\n"
        smoke_yaml = yaml.dump(smoke_data, default_flow_style=False, sort_keys=False)
        typer.echo("\nSky smoke test: minimal job with `sky launch --down` (auto teardown)...")
        exit_code = launch_skypilot_with_patched_yaml(
            smoke_yaml, managed=False, yes=True, down=True, cwd=sky_cwd,
        )
        raise typer.Exit(exit_code)

    _require_sky_cli()

    start_time = datetime.now(UTC)
    exit_code = launch_skypilot_with_patched_yaml(
        patched, managed=managed, yes=yes, cwd=sky_cwd,
    )
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
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip AWS preflight (enabled regions, GPU quota, instance offerings)",
    ),
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
        help="Observed GPU utilization in [0, 1] for the power model.",
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
    finish_by: str | None = typer.Option(
        None,
        "--finish-by",
        help="ISO-8601 finish-by deadline (enables dynamic planner). When omitted, uses YAML carbonsight.finish_by.",
    ),
    carbon_budget: float | None = typer.Option(
        None,
        "--carbon-budget",
        help="Hard carbon budget in kg CO2 for dynamic planning. When omitted, uses YAML carbonsight.carbon_budget_kg.",
    ),
    carbon_price: float | None = typer.Option(
        None,
        "--carbon-price",
        help="Shadow price in USD per kg CO2 for dynamic utility scoring. When omitted, uses YAML carbonsight.carbon_price.",
    ),
    validate_sky: bool = typer.Option(
        False,
        "--validate-sky",
        help="After patching, run `sky launch --dryrun` to verify SkyPilot accepts the YAML.",
    ),
    sky_smoke: bool = typer.Option(
        False,
        "--sky-smoke",
        help="Launch a minimal echo job via SkyPilot with `--down` (provisions briefly, then tears down).",
    ),
) -> None:
    """Pick the greenest affordable region, patch YAML, and launch via SkyPilot."""
    parsed_finish_by = parse_finish_by(finish_by) if finish_by else None
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
        live_pricing=live_pricing,
        static_pricing=static_pricing,
        finish_by=parsed_finish_by,
        carbon_budget_kg=carbon_budget,
        carbon_price=carbon_price,
        validate_sky=validate_sky,
        sky_smoke=sky_smoke,
    )
