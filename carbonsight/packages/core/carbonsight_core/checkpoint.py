"""Framework detection, checkpoint config, and SkyPilot YAML patching for spot resilience."""

import ast
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Framework(Enum):
    HUGGINGFACE = "huggingface"
    LIGHTNING = "lightning"
    PYTORCH = "pytorch"
    UNKNOWN = "unknown"


_HF_MODULES = {"transformers"}
_LIGHTNING_MODULES = {"pytorch_lightning", "lightning"}
_TORCH_MODULES = {"torch"}

SHIM_REMOTE_PATH = "/carbonsight/shim.py"
DEFAULT_CKPT_MOUNT = "/ckpt"
DEFAULT_SAVE_INTERVAL = 500


def detect_framework(script_source: str) -> Framework:
    """Detect ML framework from Python source via import analysis."""
    try:
        tree = ast.parse(script_source)
    except SyntaxError:
        return Framework.UNKNOWN

    top_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_modules.add(node.module.split(".")[0])

    if top_modules & _HF_MODULES:
        return Framework.HUGGINGFACE
    if top_modules & _LIGHTNING_MODULES:
        return Framework.LIGHTNING
    if top_modules & _TORCH_MODULES:
        return Framework.PYTORCH
    return Framework.UNKNOWN


def extract_script_path_from_run_command(run_cmd: str) -> str | None:
    """Extract the ``.py`` script path from a run command like ``python train.py --flag``."""
    for line in run_cmd.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        for part in parts[1:]:
            if part.endswith(".py") and not part.startswith("-"):
                return part
    return None


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """All settings needed to configure checkpoint resilience."""

    framework: Framework
    task_name: str
    ckpt_mount_path: str = DEFAULT_CKPT_MOUNT
    save_interval_steps: int = DEFAULT_SAVE_INTERVAL
    bucket_uri: str | None = None
    storage_name: str | None = None


def build_checkpoint_config(
    task_name: str,
    framework: Framework,
    *,
    save_interval_steps: int = DEFAULT_SAVE_INTERVAL,
    bucket_uri: str | None = None,
) -> CheckpointConfig:
    """Build a checkpoint config, auto-generating a SkyPilot Storage name when no bucket is given."""
    storage_name = None if bucket_uri else f"carbonsight-ckpt-{task_name}"
    return CheckpointConfig(
        framework=framework,
        task_name=task_name,
        save_interval_steps=save_interval_steps,
        bucket_uri=bucket_uri,
        storage_name=storage_name,
    )


@dataclass(frozen=True, slots=True)
class CheckpointYamlPatch:
    """YAML-level changes to add checkpoint support."""

    file_mounts: dict = field(default_factory=dict)
    envs: dict = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)
    run_command: str = ""


def build_checkpoint_yaml_patch(
    config: CheckpointConfig,
    original_run_cmd: str,
    shim_local_path: str,
) -> CheckpointYamlPatch:
    """Generate all SkyPilot YAML modifications for checkpoint support."""
    mounts: dict = {}
    if config.bucket_uri:
        mounts[config.ckpt_mount_path] = config.bucket_uri
    else:
        mounts[config.ckpt_mount_path] = {
            "name": config.storage_name,
            "mode": "MOUNT",
        }
    mounts[SHIM_REMOTE_PATH] = shim_local_path

    envs = {
        "CARBONSIGHT_CKPT_DIR": config.ckpt_mount_path,
        "CARBONSIGHT_FRAMEWORK": config.framework.value,
        "CARBONSIGHT_SAVE_STEPS": str(config.save_interval_steps),
        "CARBONSIGHT_RESUME": "1",
    }

    run_cmd = original_run_cmd.strip()
    wrapped = f"python {SHIM_REMOTE_PATH} {run_cmd}\n"

    return CheckpointYamlPatch(
        file_mounts=mounts,
        envs=envs,
        setup_commands=[f"mkdir -p {config.ckpt_mount_path}"],
        run_command=wrapped,
    )


def apply_checkpoint_patch_to_yaml(yaml_data: dict, patch: CheckpointYamlPatch) -> dict:
    """Merge a checkpoint patch into a SkyPilot YAML dict. Returns a new dict."""
    out = dict(yaml_data)

    existing_mounts = dict(out.get("file_mounts") or {})
    existing_mounts.update(patch.file_mounts)
    out["file_mounts"] = existing_mounts

    existing_envs = dict(out.get("envs") or {})
    existing_envs.update(patch.envs)
    out["envs"] = existing_envs

    existing_setup = (out.get("setup") or "").rstrip()
    new_setup_block = "\n".join(patch.setup_commands)
    if existing_setup:
        out["setup"] = existing_setup + "\n" + new_setup_block + "\n"
    else:
        out["setup"] = new_setup_block + "\n"

    out["run"] = patch.run_command
    return out


def shim_local_path() -> str:
    """Absolute path to ``checkpoint_shim.py`` within this package (for file_mounts upload)."""
    return str(Path(__file__).parent / "checkpoint_shim.py")
