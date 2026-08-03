"""carbonsight mappings validate / refresh"""

import json
from pathlib import Path

import typer
from carbonsight_core.config import Config
from carbonsight_core.mapping.refresh import refresh_registry_mappings
from carbonsight_core.mapping.registry import Registry, write_registry_json
from carbonsight_core.mapping.validate import (
    skipped_validate_result_to_dict,
    validate_registry_mappings,
    validate_result_to_dict,
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


def _print_validate_result(result) -> None:
    for region in result.regions:
        label = f"{region.provider}/{region.region_code}"
        wt_fmt = _format_wt_regions(region.live_wt_regions)
        conf = f"confidence={region.mapping_confidence:.2f} ({region.confidence_label})"
        if region.drift:
            stored_fmt = _format_wt_regions(region.stored_wt_regions)
            typer.echo(f"DRIFT {label}: stored={stored_fmt} live={wt_fmt} {conf}")
        else:
            typer.echo(f"OK {label}: {wt_fmt} {conf}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if result.has_drift:
        typer.echo(
            f"Summary: {result.regions_drift} drift(s), {result.regions_ok} ok, "
            f"{len(result.warnings)} warning(s)",
        )
    else:
        typer.echo(
            f"Summary: no drift ({result.regions_ok} ok), {len(result.warnings)} warning(s)",
        )


def _print_refresh_result(result) -> None:
    for change in result.changes:
        label = f"{change.provider}/{change.region_code}"
        old_fmt = _format_wt_regions(change.old_wt_regions)
        new_fmt = _format_wt_regions(change.new_wt_regions)
        if change.changed:
            typer.echo(f"UPDATE {label}: {old_fmt} -> {new_fmt}")
        else:
            typer.echo(f"OK {label}: {new_fmt} (no change)")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(
        f"Summary: {result.regions_updated} updated, "
        f"{result.regions_unchanged} unchanged, {len(result.warnings)} warning(s)",
    )


@mappings_group.command("validate")
def validate(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Run drift checks (region-from-loc vs stored); print diff and confidence."""
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    if not rpath.exists():
        typer.echo("No registry found. Use --registry.", err=True)
        raise typer.Exit(1)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        skip_msg = (
            "Set WATTTIME_USERNAME and WATTTIME_PASSWORD to validate against live API."
        )
        if json_out:
            typer.echo(json.dumps(skipped_validate_result_to_dict(skip_msg)))
        else:
            typer.echo(skip_msg, err=True)
            typer.echo("OK (no drift check without credentials).")
        raise typer.Exit(0)

    reg = Registry()
    reg.load_json(rpath)
    wt = WattTimeClient(config)
    result = validate_registry_mappings(reg, wt)

    if json_out:
        typer.echo(json.dumps(validate_result_to_dict(result)))
    else:
        _print_validate_result(result)

    if result.has_drift:
        raise typer.Exit(1)


@mappings_group.command("refresh")
def refresh(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
    write: bool = typer.Option(False, "--write", help="Write refreshed registry JSON (default: dry-run preview)"),
    out: Path | None = typer.Option(None, "--out", help="Output path when using --write"),
) -> None:
    """Re-resolve WattTime regions from site coordinates; dry-run by default."""
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd(),
    )
    if not rpath.exists():
        typer.echo("No registry found. Use --registry.", err=True)
        raise typer.Exit(1)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD to refresh mappings.", err=True)
        raise typer.Exit(1)

    reg = Registry()
    reg.load_json(rpath)
    wt = WattTimeClient(config)
    result = refresh_registry_mappings(reg, wt)
    _print_refresh_result(result)

    if not write:
        typer.echo("Dry-run: no file written (use --write to persist).")
        return

    bundled_seed = registry_json_path(package_root).resolve()
    write_path = out.resolve() if out is not None else rpath.resolve()
    if write_path == bundled_seed and out is None:
        typer.echo(
            "Refusing to overwrite bundled seed_registry.json in place. "
            "Use --out to write a copy elsewhere.",
            err=True,
        )
        raise typer.Exit(1)

    write_registry_json(reg, write_path)
    typer.echo(f"Wrote refreshed registry to {write_path}")


if __name__ == "__main__":
    typer.run(mappings_group)
