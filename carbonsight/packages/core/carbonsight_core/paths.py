"""Locate packaged files (registry JSON) from CLI cwd and package root."""

from pathlib import Path

# Under `carbonsight/` repo (sibling to `apps/`, `packages/`)
REGISTRY_JSON_SEGMENTS = ("packages", "core", "carbonsight_core", "mapping", "seed_registry.json")


def registry_json_path(carbonsight_package_root: Path) -> Path:
    """Absolute path to `seed_registry.json` inside the checked-out package tree."""
    return carbonsight_package_root.joinpath(*REGISTRY_JSON_SEGMENTS)


def resolve_registry_json_file(
    explicit_path: Path | None,
    *,
    carbonsight_package_root: Path,
    cwd: Path,
) -> Path:
    """
    Prefer `explicit_path` if it exists; else try cwd fallbacks; else default under package root.

    Returns a path even when missing — callers use `.exists()` before load.
    """
    default = registry_json_path(carbonsight_package_root)
    candidate = explicit_path or default
    if candidate.exists():
        return candidate
    for fallback in (
        cwd.joinpath("carbonsight", *REGISTRY_JSON_SEGMENTS),
        cwd.joinpath(*REGISTRY_JSON_SEGMENTS),
    ):
        if fallback.exists():
            return fallback
    return candidate


def carbonsight_package_root_from_cli_command_file(command_file: Path, *, parents_up: int = 4) -> Path:
    """
    From `.../carbonsight_cli/commands/foo.py`, walk up to the `carbonsight/` dir that contains `apps/` and `packages/`.
    """
    return command_file.resolve().parents[parents_up]
