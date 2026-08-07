"""carbonsight mappings validate / refresh"""

import json
from pathlib import Path

import typer
from carbonsight_core.config import Config
from carbonsight_core.mapping.refresh import RefreshReport, refresh_registry_mappings
from carbonsight_core.mapping.registry import Registry, write_registry_json
from carbonsight_core.mapping.validate import (
    DriftReport,
    drift_report_to_dict,
    skipped_drift_report_to_dict,
    validate_registry_mappings,
)
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    registry_json_path,
    resolve_registry_json_file,
)
from carbonsight_core.watttime import WattTimeClient

mappings_group = typer.Typer(help="Mapping registry and drift checks")


def _format_wt_regions(wt_regions: list[tuple[str, float]]) -> str:
    if not wt_regions:
        return "[]"
    parts = [f"{wt}:{weight:g}" for wt, weight in wt_regions]
    return "[" + ", ".join(parts) + "]"


def _print_drift_report(drift_report: DriftReport) -> None:
    for region in drift_report.regions:
        label = f"{region.provider}/{region.region_code}"
        wt_fmt = _format_wt_regions(region.live_wt_regions)
        conf = f"confidence={region.mapping_confidence:.2f} ({region.confidence_label})"
        if region.drift:
            stored_fmt = _format_wt_regions(region.stored_wt_regions)
            typer.echo(f"DRIFT {label}: stored={stored_fmt} live={wt_fmt} {conf}")
        else:
            typer.echo(f"OK {label}: {wt_fmt} {conf}")
    for warning in drift_report.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if drift_report.has_drift:
        typer.echo(
            f"Summary: {drift_report.regions_drift} drift(s), {drift_report.regions_ok} ok, "
            f"{len(drift_report.warnings)} warning(s)",
        )
    else:
        typer.echo(
            f"Summary: no drift ({drift_report.regions_ok} ok), {len(drift_report.warnings)} warning(s)",
        )


def _print_refresh_report(refresh_report: RefreshReport) -> None:
    for change in refresh_report.changes:
        label = f"{change.provider}/{change.region_code}"
        old_fmt = _format_wt_regions(change.old_wt_regions)
        new_fmt = _format_wt_regions(change.new_wt_regions)
        if change.changed:
            typer.echo(f"UPDATE {label}: {old_fmt} -> {new_fmt}")
        else:
            typer.echo(f"OK {label}: {new_fmt} (no change)")
    for warning in refresh_report.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(
        f"Summary: {refresh_report.regions_updated} updated, "
        f"{refresh_report.regions_unchanged} unchanged, {len(refresh_report.warnings)} warning(s)",
    )


@mappings_group.command("validate")
def validate(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Run drift checks (region-from-loc vs stored); print diff and confidence."""
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    registry_path = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    if not registry_path.exists():
        typer.echo("No registry found. Use --registry.", err=True)
        raise typer.Exit(1)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        skip_msg = (
            "Set WATTTIME_USERNAME and WATTTIME_PASSWORD to validate against live API."
        )
        if json_out:
            typer.echo(json.dumps(skipped_drift_report_to_dict(skip_msg)))
        else:
            typer.echo(skip_msg, err=True)
            typer.echo("OK (no drift check without credentials).")
        raise typer.Exit(0)

    registry = Registry()
    registry.load_json(registry_path)
    watt_time = WattTimeClient(config)
    drift_report = validate_registry_mappings(registry, watt_time)

    if json_out:
        typer.echo(json.dumps(drift_report_to_dict(drift_report)))
    else:
        _print_drift_report(drift_report)

    if drift_report.has_drift:
        raise typer.Exit(1)


@mappings_group.command("refresh")
def refresh(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    write: bool = typer.Option(False, "--write", help="Write refreshed registry JSON (default: dry-run preview)"),
    out: Path | None = typer.Option(None, "--out", help="Output path when using --write"),
) -> None:
    """Re-resolve WattTime regions from site coordinates; dry-run by default."""
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    registry_path = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd(),
    )
    if not registry_path.exists():
        typer.echo("No registry found. Use --registry.", err=True)
        raise typer.Exit(1)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD to refresh mappings.", err=True)
        raise typer.Exit(1)

    registry = Registry()
    registry.load_json(registry_path)
    watt_time = WattTimeClient(config)
    refresh_report = refresh_registry_mappings(registry, watt_time)
    _print_refresh_report(refresh_report)

    if not write:
        typer.echo("Dry-run: no file written (use --write to persist).")
        return

    bundled_seed = registry_json_path(package_root).resolve()
    write_path = out.resolve() if out is not None else registry_path.resolve()
    if write_path == bundled_seed and out is None:
        typer.echo(
            "Refusing to overwrite bundled seed_registry.json in place. "
            "Use --out to write a copy elsewhere.",
            err=True,
        )
        raise typer.Exit(1)

    write_registry_json(registry, write_path)
    typer.echo(f"Wrote refreshed registry to {write_path}")


if __name__ == "__main__":
    typer.run(mappings_group)
