"""
carbonsight schedule --yaml train.yaml --deadline-hours 45 --checkpoint-size-gb 100

Joint 2D rank (region x mode x time) with SkyNomad-inspired utility + carbon lever.

Implements per paper Sec 4.3-4.7 extended:

- Spot availability: synthetic L̄ per region (real tracker from ledger in future)
- Lifetime prediction: synthetic, or via LifetimeStats if ledger populated
- Future progress value V(t)=C_od_total_min * θ/θ̃
- Unified model: U = V*η - C_total - E/L̄, C_total = spot $ + carbon $ (kg/1000*price*weight)
- Thrifty (p>=P → idle) and Safety Net (T-t < P-p+2d → cheapest OD)
- Central WattTime cache: uses CarbonIntensityProvider factory (API_URL > WattTime creds > synthetic) so everyone can schedule without personal creds

This command does NOT launch real instances — it prints ranked candidates with U, L̄, η, $/hr, kg/hr, carbon $, migration.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.providers.carbon import get_carbon_provider
from carbonsight_core.spot.availability import AvailabilityTracker
from carbonsight_core.spot.progress import ProgressState, select_cheapest_od_region, ODCandidate
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    StaticSpotPriceProvider,
    rank_candidates,
    total_cost_per_hr,
    effectiveness,
)
from carbonsight_cli.commands.advise import (
    job_spec_from_sky_yaml,
    apply_gpu_telemetry_cli,
)

# Synthetic L̄ per region for MVP when no real tracker data
# Generally available regions (paper) get higher L̄, others lower.
_GENERALLY_AVAILABLE = {
    "us-east-1": 12.0,
    "us-west-2": 10.0,
    "eu-west-1": 11.0,
    "eu-north-1": 9.0,
    "ap-southeast-1": 8.0,
}
_DEFAULT_SPOT_LIFETIME_HR = 3.5  # volatile regions per paper Fig 2


def _synthetic_lifetime_for_region(region_code: str) -> float:
    return _GENERALLY_AVAILABLE.get(region_code, _DEFAULT_SPOT_LIFETIME_HR)


def run_schedule(
    yaml_path: Path,
    *,
    deadline_hours: float | None = None,
    checkpoint_size_gb: float = 0.0,
    cold_start_minutes: float = 6.0,
    carbon_price_usd_per_ton: float = 50.0,
    carbon_weight: float = 1.0,
    egress_usd_per_gb: float = 0.02,
    current_region: str = "us-east-1",
    registry_path: Path | None = None,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    json_out: bool = False,
    max_candidates: int = 30,
) -> None:
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    job = job_spec_from_sky_yaml(yaml_path)
    job, nvidia_failed = apply_gpu_telemetry_cli(job, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    if nvidia_failed:
        typer.echo(
            "Warning: --nvidia-smi could not read utilization. Using default sampled GPU power.",
            err=True,
        )

    # Deadline handling: if None, use P*1.5 like paper 30h compute /45h deadline
    P = job.duration_hours
    T = deadline_hours if deadline_hours is not None else P * 1.5
    if T <= 0:
        typer.echo("Error: deadline must be positive", err=True)
        raise typer.Exit(1)

    # Progress state at t=0, p=0
    progress = ProgressState(p=0.0, P=P, t=0.0, T=T)
    cold_start_hr = cold_start_minutes / 60.0
    # Add checkpoint restore time ~ 0.002 hr per GB (~7 sec/GB @10Gbps)
    if checkpoint_size_gb > 0:
        cold_start_hr += checkpoint_size_gb * 0.002

    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(registry_path, carbonsight_package_root=package_root, cwd=Path.cwd())
    if not rpath.exists():
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)

    reg = Registry()
    reg.load_json(rpath)

    config = Config.from_env()
    # Central cache: factory picks API_URL > WattTime creds > synthetic
    carbon_provider = get_carbon_provider(api_url=config.carbonsight_api_url or None)
    spot_provider = StaticSpotPriceProvider(default_gpu_type=job.gpu_type)
    migration_estimator = MigrationCostEstimator(
        egress_usd_per_gb=egress_usd_per_gb,
        carbon_price_usd_per_ton=carbon_price_usd_per_ton,
        carbon_weight=carbon_weight,
    )

    # Current time window for carbon queries: now -> now+L̄
    now_utc = datetime.now(UTC)

    candidates: list[CandidateState] = []
    od_candidates_for_safety: list[ODCandidate] = []

    # For future: AvailabilityTracker from ledger if exists
    tracker = AvailabilityTracker()

    for entry in reg.all_regions():
        if not entry.wt_regions or entry.provider.lower() != "aws":
            continue
        region_code = entry.region_code

        # Synthetic lifetime for MVP
        mean_lifetime_hr = _synthetic_lifetime_for_region(region_code)

        # Spot and OD $/hr total for gpu_count
        spot_price_per_hr = spot_provider.get_spot_price_usd_per_hr_total(
            region_code, job.gpu_type, job.gpu_count
        )
        od_price_per_hr = spot_provider.get_od_price_usd_per_hr(
            region_code, job.gpu_type, job.gpu_count
        )

        # Carbon kg/hr using provider with window [now, now+L̄] for spot, [now, now+remaining] for OD
        # Use same job for power model
        try:
            # Spot window: now -> now+L̄
            end_spot = now_utc + timedelta(hours=mean_lifetime_hr)
            carbon_kg_spot = carbon_provider.get_kg_per_hr(
                job, entry.wt_regions, now_utc, end_spot
            )
        except Exception:
            carbon_kg_spot = 0.15  # fallback synthetic

        try:
            end_od = now_utc + timedelta(hours=progress.remaining_work)
            carbon_kg_od = carbon_provider.get_kg_per_hr(
                job, entry.wt_regions, now_utc, end_od
            )
        except Exception:
            carbon_kg_od = carbon_kg_spot

        migration_cost = 0.0
        if region_code != current_region and checkpoint_size_gb > 0:
            migration_cost = migration_estimator.estimate(checkpoint_size_gb)

        # Spot candidate
        candidates.append(
            CandidateState(
                region=region_code,
                mode="spot",
                mean_lifetime_hr=mean_lifetime_hr,
                price_per_hr=spot_price_per_hr,
                carbon_kg_per_hr=carbon_kg_spot,
                migration_cost=migration_cost,
            )
        )
        # OD candidate (infinite lifetime)
        candidates.append(
            CandidateState(
                region=region_code,
                mode="on_demand",
                mean_lifetime_hr=math.inf,
                price_per_hr=od_price_per_hr,
                carbon_kg_per_hr=carbon_kg_od,
                migration_cost=migration_cost,
            )
        )
        od_candidates_for_safety.append(
            ODCandidate(
                region=region_code,
                od_price_per_hr=od_price_per_hr,
                migration_cost=migration_cost,
                carbon_kg_per_hr=carbon_kg_od,
            )
        )

    # Idle candidate
    candidates.append(
        CandidateState(
            region="idle",
            mode="idle",
            mean_lifetime_hr=0.0,
            price_per_hr=0.0,
            carbon_kg_per_hr=0.0,
            migration_cost=0.0,
        )
    )

    # Compute C_od_total_min for V anchoring
    if od_candidates_for_safety:
        c_total_od_min = min(
            total_cost_per_hr(
                c.od_price_per_hr,
                c.carbon_kg_per_hr,
                carbon_price_usd_per_ton,
                carbon_weight,
            )
            for c in od_candidates_for_safety
        )
    else:
        c_total_od_min = 4.0

    # Future progress value
    V = progress.future_progress_value(c_total_od_min)

    # Thrifty check
    if progress.is_thrifty():
        if json_out:
            typer.echo(json.dumps([{"action": "idle", "reason": "p>=P thrifty", "V": V}], indent=2))
        else:
            typer.echo(f"Thrifty: progress done p={progress.p} >= P={progress.P}, safe to idle. V={V:.2f}")
        return

    # Safety net check
    if progress.is_safety_net(cold_start_hr):
        best = select_cheapest_od_region(
            od_candidates_for_safety,
            progress.remaining_work,
            cold_start_hr,
            carbon_price_usd_per_ton,
            carbon_weight,
        )
        if best is not None:
            best_cand, best_cost = best
            if json_out:
                typer.echo(
                    json.dumps(
                        {
                            "action": "safety_net_on_demand",
                            "reason": f"T-t={progress.remaining_time:.1f}h < P-p+2d={progress.remaining_work+2*cold_start_hr:.1f}h",
                            "chosen_region": best_cand.region,
                            "total_cost_to_finish": best_cost,
                            "V": V,
                        },
                        indent=2,
                    )
                )
            else:
                typer.echo(
                    f"Safety Net triggered: remaining_time {progress.remaining_time:.1f}h < "
                    f"remaining_work+2d {progress.remaining_work+2*cold_start_hr:.1f}h — "
                    f"switch to cheapest OD {best_cand.region} cost ${best_cost:.2f} to finish, V={V:.2f}"
                )
            return

    # Rank by utility
    ranked = rank_candidates(candidates, V, cold_start_hr, carbon_price_usd_per_ton, carbon_weight)

    # Trim to max_candidates
    visible = ranked[:max_candidates]

    if json_out:
        out = []
        for cand, u in visible:
            eta = effectiveness(cand.mean_lifetime_hr, cold_start_hr)
            c_total = total_cost_per_hr(
                cand.price_per_hr, cand.carbon_kg_per_hr, carbon_price_usd_per_ton, carbon_weight
            )
            out.append(
                {
                    "cloud_region": cand.region,
                    "mode": cand.mode,
                    "mean_lifetime_hr": cand.mean_lifetime_hr if not math.isinf(cand.mean_lifetime_hr) else 9999,
                    "effectiveness_eta": eta,
                    "price_per_hr_usd": cand.price_per_hr,
                    "carbon_kg_per_hr": cand.carbon_kg_per_hr,
                    "carbon_cost_per_hr_usd": cand.carbon_kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight,
                    "total_cost_per_hr_usd": c_total,
                    "migration_cost_usd": cand.migration_cost,
                    "amortized_migration_per_hr": cand.migration_cost / cand.mean_lifetime_hr if cand.mean_lifetime_hr and not math.isinf(cand.mean_lifetime_hr) else 0.0,
                    "utility_U": u,
                    "V": V,
                }
            )
        typer.echo(json.dumps(out, indent=2))
        return

    # Rich table
    table = Table(
        title=f"SkyNomad+Carbon schedule | P={P:.1f}h T={T:.1f}h V={V:.2f} | cold_start={cold_start_hr:.2f}h ckpt={checkpoint_size_gb}GB"
    )
    table.add_column("Rank", justify="right")
    table.add_column("Region")
    table.add_column("Mode")
    table.add_column("L̄ (h)", justify="right")
    table.add_column("η", justify="right")
    table.add_column("$/hr", justify="right")
    table.add_column("kgCO2/hr", justify="right")
    table.add_column("carbon $/hr", justify="right")
    table.add_column("C_total $/hr", justify="right")
    table.add_column("E/L̄ $/hr", justify="right")
    table.add_column("U", justify="right", style="bold")

    for idx, (cand, u) in enumerate(visible, start=1):
        eta = effectiveness(cand.mean_lifetime_hr, cold_start_hr)
        c_total = total_cost_per_hr(
            cand.price_per_hr, cand.carbon_kg_per_hr, carbon_price_usd_per_ton, carbon_weight
        )
        carbon_dollar = cand.carbon_kg_per_hr / 1000.0 * carbon_price_usd_per_ton * carbon_weight
        amort = cand.migration_cost / cand.mean_lifetime_hr if cand.mean_lifetime_hr and not math.isinf(cand.mean_lifetime_hr) else 0.0
        lh_str = f"{cand.mean_lifetime_hr:.1f}" if not math.isinf(cand.mean_lifetime_hr) else "∞"
        table.add_row(
            str(idx),
            cand.region,
            cand.mode,
            lh_str,
            f"{eta:.2f}",
            f"${cand.price_per_hr:.2f}",
            f"{cand.carbon_kg_per_hr:.3f}",
            f"${carbon_dollar:.3f}",
            f"${c_total:.2f}",
            f"${amort:.3f}",
            f"{u:.2f}",
        )

    Console().print(table)
    typer.echo(
        f"Carbon provider: {type(carbon_provider).__name__} (CARBONSIGHT_API_URL set → central cache, no WattTime creds needed)\n"
        f"Spot provider: {type(spot_provider).__name__} (Static fallback, your boto3 wrapper will implement same protocol)\n"
        f"Top action: launch {visible[0][0].region} {visible[0][0].mode} if U>0, else idle. Δ margin 0.05 to avoid flapping."
    )


def schedule_cmd(
    yaml_path: Path = typer.Option(..., "--yaml", "-y", help="Path to SkyPilot-style job YAML"),
    deadline_hours: float | None = typer.Option(None, "--deadline-hours", help="Deadline T hours from now (default P*1.5)"),
    checkpoint_size_gb: float = typer.Option(0.0, "--checkpoint-size-gb", help="Checkpoint size GB for migration cost E"),
    cold_start_minutes: float = typer.Option(6.0, "--cold-start-minutes", help="Cold start d minutes (default 6)"),
    carbon_price_usd_per_ton: float = typer.Option(50.0, "--carbon-price", help="Social cost of carbon $/ton"),
    carbon_weight: float = typer.Option(1.0, "--carbon-weight", help="Lambda weight for carbon vs $"),
    egress_usd_per_gb: float = typer.Option(0.02, "--egress-usd-per-gb", help="Egress $/GB for migration"),
    current_region: str = typer.Option("us-east-1", "--current-region", help="Current region r0 for migration cost"),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    gpu_util: float | None = typer.Option(None, "--gpu-util", help="GPU utilization [0,1]"),
    nvidia_smi: bool = typer.Option(False, "--nvidia-smi", help="Sample nvidia-smi"),
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
    max_candidates: int = typer.Option(30, "--max-candidates", help="Max candidates to show"),
) -> None:
    """Joint region×mode rank with SkyNomad utility + carbon lever (no real launch)."""
    run_schedule(
        yaml_path,
        deadline_hours=deadline_hours,
        checkpoint_size_gb=checkpoint_size_gb,
        cold_start_minutes=cold_start_minutes,
        carbon_price_usd_per_ton=carbon_price_usd_per_ton,
        carbon_weight=carbon_weight,
        egress_usd_per_gb=egress_usd_per_gb,
        current_region=current_region,
        registry_path=registry_path,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        json_out=json_out,
        max_candidates=max_candidates,
    )
