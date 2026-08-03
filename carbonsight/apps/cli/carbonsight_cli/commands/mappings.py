"""carbonsight mappings validate / refresh"""

from pathlib import Path

import typer
from carbonsight_core.config import Config
from carbonsight_core.mapping.refresh import refresh_registry_mappings
from carbonsight_core.mapping.registry import Registry, write_registry_json
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    registry_json_path,
    resolve_registry_json_file,
)
from carbonsight_core.watttime import WattTimeClient, WattTimeError

mappings_group = typer.Typer(help="Mapping registry and drift checks")


def _format_wt_regions(wt_regions: list[tuple[str, float]]) -> str:
    if not wt_regions:
        return "[]"
    parts = [f"{wt}:{weight:g}" for wt, weight in wt_regions]
    return "[" + ", ".join(parts) + "]"


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
) -> None:
    """Run drift checks (region-from-loc vs stored); print diff and confidence."""
    reg = Registry()
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(
        registry_path, carbonsight_package_root=package_root, cwd=Path.cwd()
    )
    if not rpath.exists():
        typer.echo("No registry found. Use --registry.", err=True)
        raise typer.Exit(1)
    reg.load_json(rpath)

    config = Config.from_env()
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD to validate against live API.", err=True)
        typer.echo("OK (no drift check without credentials).")
        raise typer.Exit(0)

    wt = WattTimeClient(config)
    drift_count = 0
    for entry in reg.all_regions():
        for site in entry.sites:
            try:
                data = wt.region_from_loc(site.lat, site.lon, signal_type="co2_moer")
                current_wt = data.get("region") or data.get("abbrev") or str(data)
                stored = next((r for r, _ in entry.wt_regions), None) if entry.wt_regions else None
                if stored and current_wt != stored:
                    typer.echo(
                        f"DRIFT: {entry.provider}/{entry.region_code} site {site.site_id}: "
                        f"stored={stored} live={current_wt}",
                    )
                    drift_count += 1
            except WattTimeError as e:
                typer.echo(f"Warning: {entry.region_code} {site.site_id}: {e}", err=True)
            except Exception as e:
                typer.echo(f"Warning: {entry.region_code} {site.site_id}: {e}", err=True)

    if drift_count == 0:
        typer.echo("OK: no drift detected (all region-from-loc match stored).")
    else:
        typer.echo(f"Found {drift_count} drift(s). Update registry or revalidate.")


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
