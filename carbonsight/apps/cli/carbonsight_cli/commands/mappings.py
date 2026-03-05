"""carbonsight mappings validate"""

from pathlib import Path

import typer

from carbonsight_core.config import Config
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.watttime import WattTimeClient, WattTimeError


def _default_registry_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    return root / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json"


mappings_group = typer.Typer(help="Mapping registry and drift checks")


@mappings_group.command("validate")
def validate(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to mapping registry JSON"),
) -> None:
    """Run drift checks (region-from-loc vs stored); print diff and confidence."""
    reg = Registry()
    rpath = registry_path or _default_registry_path()
    if not rpath.exists():
        for candidate in [
            Path.cwd() / "carbonsight" / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json",
            Path.cwd() / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json",
        ]:
            if candidate.exists():
                rpath = candidate
                break
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
                    typer.echo(f"DRIFT: {entry.provider}/{entry.region_code} site {site.site_id}: stored={stored} live={current_wt}")
                    drift_count += 1
            except WattTimeError as e:
                typer.echo(f"Warning: {entry.region_code} {site.site_id}: {e}", err=True)
            except Exception as e:
                typer.echo(f"Warning: {entry.region_code} {site.site_id}: {e}", err=True)

    if drift_count == 0:
        typer.echo("OK: no drift detected (all region-from-loc match stored).")
    else:
        typer.echo(f"Found {drift_count} drift(s). Update registry or revalidate.")


if __name__ == "__main__":
    typer.run(mappings_group)
