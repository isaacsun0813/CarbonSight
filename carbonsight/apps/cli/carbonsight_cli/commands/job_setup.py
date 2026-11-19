"""Shared YAML/resolve/setup helpers for CLI job commands.

Command-agnostic helpers used by ``advise``, ``plan-static``, and ``run``:
parsing SkyPilot YAML into a ``JobSpec``, resolving ``carbonsight`` block
fields (CLI wins over YAML), and loading the job, registry, and config.
No Typer commands are defined here, and no helper encodes a command's
credential policy or output format — callers keep those.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from carbonsight_core.config import Config
from carbonsight_core.estimator.power_model import gpu_power_limit_w
from carbonsight_core.estimator.pricing import configure_pricing
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.telemetry.nvidia_smi import sample_mean_gpu_utilization_from_nvidia_smi


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
    num = float(s.replace("h", "").replace("m", "").strip())
    if "m" in s:
        num /= 60.0
    return num


def parse_finish_by(raw: Any) -> datetime:
    """Parse ISO-8601 finish-by timestamp; ensure timezone-aware UTC."""
    if isinstance(raw, datetime):
        dt = raw
    else:
        dt = datetime.fromisoformat(str(raw).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def finish_by_from_yaml(
    path: Path,
    *,
    carbonsight_block: dict | None = None,
) -> datetime | None:
    """Read optional ``carbonsight.finish_by`` from a SkyPilot YAML file."""
    cs = carbonsight_block if carbonsight_block is not None else carbonsight_block_from_yaml(path)
    if cs.get("finish_by") is None:
        return None
    return parse_finish_by(cs["finish_by"])


def carbonsight_block_from_yaml(path: Path) -> dict:
    """Return the ``carbonsight`` mapping from a task YAML, or ``{}``."""
    data: dict = yaml.safe_load(path.read_text()) or {}
    cs = data.get("carbonsight")
    return cs if isinstance(cs, dict) else {}


def resolve_finish_by(
    cli_finish_by: datetime | None,
    yaml_path: Path,
    *,
    carbonsight_block: dict | None = None,
) -> datetime | None:
    """Deadline for dynamic planning: explicit CLI wins; else YAML ``carbonsight.finish_by``."""
    if cli_finish_by is not None:
        return cli_finish_by
    return finish_by_from_yaml(yaml_path, carbonsight_block=carbonsight_block)


def resolve_carbon_budget_kg(
    cli_carbon_budget_kg: float | None,
    yaml_path: Path,
    *,
    carbonsight_block: dict | None = None,
) -> float | None:
    """Carbon budget: explicit CLI wins; else optional YAML ``carbonsight.carbon_budget_kg``."""
    if cli_carbon_budget_kg is not None:
        return cli_carbon_budget_kg
    cs = (
        carbonsight_block
        if carbonsight_block is not None
        else carbonsight_block_from_yaml(yaml_path)
    )
    raw = cs.get("carbon_budget_kg")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def resolve_carbon_price(
    cli_carbon_price: float | None,
    yaml_path: Path,
    *,
    carbonsight_block: dict | None = None,
) -> float:
    """Shadow carbon price: explicit CLI wins; else YAML ``carbonsight.carbon_price``; else 0."""
    if cli_carbon_price is not None:
        return cli_carbon_price
    cs = (
        carbonsight_block
        if carbonsight_block is not None
        else carbonsight_block_from_yaml(yaml_path)
    )
    raw = cs.get("carbon_price")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


CARBONSIGHT_ONLY_YAML_TOP_LEVEL_KEYS = ("carbonsight", "duration")


def strip_carbonsight_only_yaml_keys(data: dict) -> dict:
    """Remove CarbonSight planning keys before handing YAML to SkyPilot."""
    cleaned = dict(data)
    for key in CARBONSIGHT_ONLY_YAML_TOP_LEVEL_KEYS:
        cleaned.pop(key, None)
    return cleaned


def job_spec_from_sky_yaml(path: Path) -> JobSpec:
    """SkyPilot-style YAML → JobSpec.

    GPU utilization for the power model is set via CLI ``--gpu-util`` or ``--nvidia-smi``.
    """
    data: dict = yaml.safe_load(path.read_text()) or {}
    resources = data.get("resources") or {}
    acc = resources.get("accelerators") or ""
    gpu_type, gpu_count = parse_sky_accelerators_string(acc if isinstance(acc, str) else "")
    duration = data.get("duration")
    return JobSpec(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        **({"duration_hours": parse_duration_hours(duration)} if duration is not None else {}),
        gpu_power_limit_w=carbonsight_block_from_yaml(path).get("gpu_power_limit_w"),
        cpu_count=resources.get("cpus"),
        mem_gib=resources.get("memory"),
    )


def apply_gpu_telemetry_cli(
    job: JobSpec,
    *,
    gpu_util: float | None,
    nvidia_smi: bool,
) -> tuple[JobSpec, bool]:
    """Apply CLI GPU telemetry. Returns ``(job, nvidia_smi_failed)``.

    Precedence: ``--gpu-util`` overrides ``--nvidia-smi``; otherwise power model samples 0.6–0.9.
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
    return bool(config.live_aws_pricing)


def ensure_job_inputs(yaml_path: Path, gpu_util: float | None) -> None:
    """Check the job YAML exists and ``gpu_util`` is in range; exit 1 otherwise."""
    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)
    if gpu_util is not None and not (0.0 <= gpu_util <= 1.0):
        typer.echo("Error: --gpu-util must be between 0 and 1.", err=True)
        raise typer.Exit(1)


def load_job_spec(yaml_path: Path, *, gpu_util: float | None, nvidia_smi: bool) -> JobSpec:
    """Load the ``JobSpec`` and apply CLI GPU telemetry, warning if sampling failed.

    Assumes ``ensure_job_inputs`` has already run.
    """
    job = job_spec_from_sky_yaml(yaml_path)
    try:
        gpu_power_limit_w(job)
    except ValueError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    job, nvidia_failed = apply_gpu_telemetry_cli(job, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    if nvidia_failed:
        typer.echo(
            "Warning: --nvidia-smi could not read utilization (no GPU driver here?). "
            "Using default sampled GPU power.",
            err=True,
        )
    if not job.has_duration_estimate:
        typer.echo(
            "Runtime unknown: carbon and cost estimates compare one hour of execution, "
            "not the whole job. Completion is determined by SkyPilot job status.",
            err=True,
        )
    return job


def resolve_registry_file(registry_path: Path | None) -> Path:
    """Resolve the registry JSON path; exit 1 if it does not exist."""
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath: Path = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    if not rpath.exists():
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)
    return rpath


def load_registry(registry_file: Path) -> Registry:
    """Load the mapping registry from ``registry_file``."""
    registry = Registry()
    registry.load_json(registry_file)
    return registry


def load_config(*, live_pricing: bool, static_pricing: bool) -> Config:
    """Load env config and apply the live/static pricing flags. No credential check."""
    config = Config.from_env()
    configure_pricing(
        config,
        live_aws_pricing=resolve_live_aws_pricing(
            config, live_pricing=live_pricing, static_pricing=static_pricing
        ),
    )
    return config
