"""Tests for checkpoint resilience: framework detection, config building, YAML patching, shim logic."""

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from carbonsight_core.checkpoint import (
    DEFAULT_CKPT_MOUNT,
    DEFAULT_SAVE_INTERVAL,
    SHIM_REMOTE_PATH,
    CheckpointConfig,
    CheckpointYamlPatch,
    Framework,
    apply_checkpoint_patch_to_yaml,
    build_checkpoint_config,
    build_checkpoint_yaml_patch,
    detect_framework,
    extract_script_path_from_run_command,
    shim_local_path,
)
from carbonsight_core.checkpoint_shim import (
    find_latest_checkpoint,
    inject_initial_args,
    inject_resume_args,
)


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


class TestDetectFramework:
    def test_huggingface_import(self) -> None:
        src = "from transformers import Trainer\ntrainer = Trainer()\n"
        assert detect_framework(src) == Framework.HUGGINGFACE

    def test_huggingface_submodule(self) -> None:
        src = "import transformers.models\n"
        assert detect_framework(src) == Framework.HUGGINGFACE

    def test_lightning_import(self) -> None:
        src = "import pytorch_lightning as pl\ntrainer = pl.Trainer()\n"
        assert detect_framework(src) == Framework.LIGHTNING

    def test_lightning_new_namespace(self) -> None:
        src = "from lightning import Trainer\n"
        assert detect_framework(src) == Framework.LIGHTNING

    def test_pytorch_raw(self) -> None:
        src = "import torch\nimport torch.nn as nn\n"
        assert detect_framework(src) == Framework.PYTORCH

    def test_hf_takes_precedence_over_torch(self) -> None:
        src = "import torch\nfrom transformers import AutoModel\n"
        assert detect_framework(src) == Framework.HUGGINGFACE

    def test_lightning_takes_precedence_over_torch(self) -> None:
        src = "import torch\nimport pytorch_lightning as pl\n"
        assert detect_framework(src) == Framework.LIGHTNING

    def test_no_ml_imports(self) -> None:
        src = "import os\nimport json\nprint('hello')\n"
        assert detect_framework(src) == Framework.UNKNOWN

    def test_syntax_error(self) -> None:
        assert detect_framework("def broken(") == Framework.UNKNOWN

    def test_empty_source(self) -> None:
        assert detect_framework("") == Framework.UNKNOWN


# ---------------------------------------------------------------------------
# Script path extraction
# ---------------------------------------------------------------------------


class TestExtractScriptPath:
    def test_simple_python_call(self) -> None:
        assert extract_script_path_from_run_command("python train.py") == "train.py"

    def test_python3_call(self) -> None:
        assert extract_script_path_from_run_command("python3 train.py --lr 1e-4") == "train.py"

    def test_torchrun(self) -> None:
        assert extract_script_path_from_run_command("torchrun --nproc_per_node=4 train.py") == "train.py"

    def test_multiline_with_comments(self) -> None:
        cmd = "# setup\npip install -r requirements.txt\npython train.py\n"
        assert extract_script_path_from_run_command(cmd) == "train.py"

    def test_nested_path(self) -> None:
        assert extract_script_path_from_run_command("python src/train.py") == "src/train.py"

    def test_no_python_script(self) -> None:
        assert extract_script_path_from_run_command("echo hello") is None

    def test_module_run(self) -> None:
        assert extract_script_path_from_run_command("python -m torch.distributed.launch") is None


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


class TestBuildCheckpointConfig:
    def test_auto_storage_name(self) -> None:
        cfg = build_checkpoint_config("my-job", Framework.HUGGINGFACE)
        assert cfg.storage_name == "carbonsight-ckpt-my-job"
        assert cfg.bucket_uri is None
        assert cfg.save_interval_steps == DEFAULT_SAVE_INTERVAL

    def test_explicit_bucket(self) -> None:
        cfg = build_checkpoint_config("my-job", Framework.PYTORCH, bucket_uri="s3://my-bucket/ckpt")
        assert cfg.bucket_uri == "s3://my-bucket/ckpt"
        assert cfg.storage_name is None

    def test_custom_interval(self) -> None:
        cfg = build_checkpoint_config("j", Framework.LIGHTNING, save_interval_steps=100)
        assert cfg.save_interval_steps == 100

    def test_frozen(self) -> None:
        cfg = build_checkpoint_config("j", Framework.PYTORCH)
        with pytest.raises(AttributeError):
            cfg.framework = Framework.UNKNOWN  # type: ignore[misc]


# ---------------------------------------------------------------------------
# YAML patch generation
# ---------------------------------------------------------------------------


class TestBuildCheckpointYamlPatch:
    def _make_patch(self, framework: Framework = Framework.HUGGINGFACE) -> CheckpointYamlPatch:
        cfg = build_checkpoint_config("test-task", framework)
        return build_checkpoint_yaml_patch(cfg, "python train.py --epochs 10", "/local/shim.py")

    def test_file_mounts_contain_ckpt_and_shim(self) -> None:
        patch = self._make_patch()
        assert DEFAULT_CKPT_MOUNT in patch.file_mounts
        assert SHIM_REMOTE_PATH in patch.file_mounts
        assert patch.file_mounts[SHIM_REMOTE_PATH] == "/local/shim.py"

    def test_auto_storage_mount_structure(self) -> None:
        patch = self._make_patch()
        mount = patch.file_mounts[DEFAULT_CKPT_MOUNT]
        assert isinstance(mount, dict)
        assert mount["name"] == "carbonsight-ckpt-test-task"
        assert mount["mode"] == "MOUNT"

    def test_explicit_bucket_mount(self) -> None:
        cfg = build_checkpoint_config("t", Framework.PYTORCH, bucket_uri="s3://my/ckpt")
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        assert patch.file_mounts[DEFAULT_CKPT_MOUNT] == "s3://my/ckpt"

    def test_envs_set(self) -> None:
        patch = self._make_patch()
        assert patch.envs["CARBONSIGHT_CKPT_DIR"] == DEFAULT_CKPT_MOUNT
        assert patch.envs["CARBONSIGHT_FRAMEWORK"] == "huggingface"
        assert patch.envs["CARBONSIGHT_SAVE_STEPS"] == str(DEFAULT_SAVE_INTERVAL)
        assert patch.envs["CARBONSIGHT_RESUME"] == "1"

    def test_run_command_wraps_with_shim(self) -> None:
        patch = self._make_patch()
        assert patch.run_command.startswith(f"python {SHIM_REMOTE_PATH}")
        assert "python train.py --epochs 10" in patch.run_command

    def test_setup_creates_ckpt_dir(self) -> None:
        patch = self._make_patch()
        assert any(f"mkdir -p {DEFAULT_CKPT_MOUNT}" in c for c in patch.setup_commands)


# ---------------------------------------------------------------------------
# YAML patch application
# ---------------------------------------------------------------------------


class TestApplyCheckpointPatchToYaml:
    def test_preserves_existing_fields(self) -> None:
        original = {
            "name": "my-task",
            "resources": {"accelerators": "A100:1"},
            "duration": "2h",
            "run": "python train.py\n",
        }
        cfg = build_checkpoint_config("my-task", Framework.HUGGINGFACE)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        result = apply_checkpoint_patch_to_yaml(original, patch)

        assert result["name"] == "my-task"
        assert result["resources"]["accelerators"] == "A100:1"
        assert result["duration"] == "2h"

    def test_merges_file_mounts(self) -> None:
        original = {"file_mounts": {"/data": "s3://my-data"}, "run": "python train.py\n"}
        cfg = build_checkpoint_config("t", Framework.PYTORCH)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        result = apply_checkpoint_patch_to_yaml(original, patch)

        assert "/data" in result["file_mounts"]
        assert DEFAULT_CKPT_MOUNT in result["file_mounts"]
        assert SHIM_REMOTE_PATH in result["file_mounts"]

    def test_merges_envs(self) -> None:
        original = {"envs": {"MY_VAR": "hello"}, "run": "python train.py\n"}
        cfg = build_checkpoint_config("t", Framework.PYTORCH)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        result = apply_checkpoint_patch_to_yaml(original, patch)

        assert result["envs"]["MY_VAR"] == "hello"
        assert result["envs"]["CARBONSIGHT_FRAMEWORK"] == "pytorch"

    def test_appends_setup(self) -> None:
        original = {"setup": "pip install torch\n", "run": "python train.py\n"}
        cfg = build_checkpoint_config("t", Framework.PYTORCH)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        result = apply_checkpoint_patch_to_yaml(original, patch)

        assert "pip install torch" in result["setup"]
        assert "mkdir -p" in result["setup"]

    def test_does_not_mutate_original(self) -> None:
        original = {"name": "t", "run": "python train.py\n"}
        cfg = build_checkpoint_config("t", Framework.PYTORCH)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        apply_checkpoint_patch_to_yaml(original, patch)

        assert "file_mounts" not in original
        assert "envs" not in original

    def test_roundtrip_through_yaml_dump_load(self) -> None:
        original = {
            "name": "t",
            "resources": {"accelerators": "A100:1", "cloud": "aws", "region": "us-east-1"},
            "run": "python train.py\n",
        }
        cfg = build_checkpoint_config("t", Framework.HUGGINGFACE)
        patch = build_checkpoint_yaml_patch(cfg, "python train.py", "/shim.py")
        result = apply_checkpoint_patch_to_yaml(original, patch)

        text = yaml.dump(result, default_flow_style=False)
        reloaded = yaml.safe_load(text)

        assert reloaded["file_mounts"][DEFAULT_CKPT_MOUNT]["name"] == "carbonsight-ckpt-t"
        assert reloaded["envs"]["CARBONSIGHT_FRAMEWORK"] == "huggingface"
        assert SHIM_REMOTE_PATH in reloaded["run"]


# ---------------------------------------------------------------------------
# Shim: find_latest_checkpoint
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint:
    def test_finds_huggingface_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-500").mkdir()
        (tmp_path / "checkpoint-300").mkdir()
        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith("checkpoint-500")

    def test_finds_lightning_ckpt(self, tmp_path: Path) -> None:
        f = tmp_path / "epoch=2.ckpt"
        f.write_text("dummy")
        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith(".ckpt")

    def test_finds_pytorch_pt(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_text("dummy")
        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith(".pt")

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_hf_preferred_over_pt(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "model.pt").write_text("dummy")
        result = find_latest_checkpoint(str(tmp_path))
        assert "checkpoint-100" in result


# ---------------------------------------------------------------------------
# Shim: arg injection
# ---------------------------------------------------------------------------


class TestArgInjection:
    def test_initial_args_huggingface(self) -> None:
        os.environ["CARBONSIGHT_CKPT_DIR"] = "/ckpt"
        os.environ["CARBONSIGHT_SAVE_STEPS"] = "500"
        result = inject_initial_args(["python", "train.py"], "huggingface")
        assert "--output_dir" in result
        assert "--save_strategy" in result
        assert "--save_steps" in result
        assert "/ckpt" in result

    def test_initial_args_lightning(self) -> None:
        result = inject_initial_args(["python", "train.py"], "lightning")
        assert "--trainer.default_root_dir" in result

    def test_initial_args_unknown_passthrough(self) -> None:
        cmd = ["python", "train.py", "--flag"]
        result = inject_initial_args(cmd, "unknown")
        assert result == cmd

    def test_resume_args_huggingface(self) -> None:
        os.environ["CARBONSIGHT_CKPT_DIR"] = "/ckpt"
        os.environ["CARBONSIGHT_SAVE_STEPS"] = "500"
        result = inject_resume_args(["python", "train.py"], "huggingface", "/ckpt/checkpoint-500")
        assert "--resume_from_checkpoint" in result
        assert "/ckpt/checkpoint-500" in result
        assert "--output_dir" in result

    def test_resume_args_lightning(self) -> None:
        result = inject_resume_args(["python", "train.py"], "lightning", "/ckpt/last.ckpt")
        assert "--ckpt_path" in result
        assert "/ckpt/last.ckpt" in result

    def test_resume_args_unknown_passthrough(self) -> None:
        cmd = ["python", "train.py"]
        result = inject_resume_args(cmd, "unknown", "/ckpt/model.pt")
        assert result == cmd


# ---------------------------------------------------------------------------
# Shim local path
# ---------------------------------------------------------------------------


class TestShimLocalPath:
    def test_points_to_existing_file(self) -> None:
        p = shim_local_path()
        assert Path(p).exists()
        assert p.endswith("checkpoint_shim.py")


# ---------------------------------------------------------------------------
# CLI wiring smoke (--help only, no HTTP)
# ---------------------------------------------------------------------------

import subprocess
import sys


def _cli_help(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": f"{root / 'packages' / 'core'}:{root / 'apps' / 'cli'}:{root / 'apps' / 'api'}",
    }
    return subprocess.run(
        [sys.executable, "-m", "carbonsight_cli.main", *args],
        capture_output=True, text=True, cwd=root, env=env, timeout=30,
    )


@pytest.fixture(scope="module")
def repo() -> Path:
    return Path(__file__).resolve().parents[2]


class TestCheckpointCliWiring:
    def test_run_exposes_checkpoint_options(self, repo: Path) -> None:
        out = _cli_help(repo, "run", "--help")
        assert out.returncode == 0, out.stderr
        assert "--checkpoint" in out.stdout
        assert "--checkpoint-bucket" in out.stdout
        assert "--checkpoint-interval" in out.stdout

    def test_train_exposes_checkpoint_options(self, repo: Path) -> None:
        out = _cli_help(repo, "train", "--help")
        assert out.returncode == 0, out.stderr
        assert "--checkpoint" in out.stdout
        assert "--checkpoint-bucket" in out.stdout
        assert "--checkpoint-interval" in out.stdout
