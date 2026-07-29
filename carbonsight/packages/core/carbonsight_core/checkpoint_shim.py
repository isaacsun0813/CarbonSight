#!/usr/bin/env python3
"""CarbonSight checkpoint shim — uploaded to remote instance, zero external deps.

Wraps a training command with SIGTERM handling and framework-aware checkpoint
injection. Reads configuration from environment variables set by CarbonSight.
"""

import glob
import os
import re
import signal
import subprocess
import sys


CKPT_DIR = os.environ.get("CARBONSIGHT_CKPT_DIR", "/ckpt")
FRAMEWORK = os.environ.get("CARBONSIGHT_FRAMEWORK", "unknown")
SAVE_STEPS = os.environ.get("CARBONSIGHT_SAVE_STEPS", "500")
RESUME = os.environ.get("CARBONSIGHT_RESUME", "1") == "1"


def _log(msg):
    # type: (str) -> None
    print("[carbonsight] " + msg, file=sys.stderr, flush=True)


def find_latest_checkpoint(ckpt_dir):
    # type: (str) -> str | None
    """Find the most recent checkpoint in *ckpt_dir* by naming convention."""
    # HuggingFace: checkpoint-NNNN directories
    hf_dirs = sorted(
        glob.glob(os.path.join(ckpt_dir, "checkpoint-*")),
        key=lambda p: _extract_step_number(p),
    )
    if hf_dirs:
        return hf_dirs[-1]

    # Lightning: *.ckpt files
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True),
        key=os.path.getmtime,
    )
    if ckpt_files:
        return ckpt_files[-1]

    # Raw PyTorch: *.pt files
    pt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "**", "*.pt"), recursive=True),
        key=os.path.getmtime,
    )
    if pt_files:
        return pt_files[-1]

    return None


def _extract_step_number(path):
    # type: (str) -> int
    m = re.search(r"(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def inject_initial_args(cmd, framework):
    # type: (list, str) -> list
    """Append framework-specific checkpoint args for a fresh run."""
    if framework == "huggingface":
        return cmd + [
            "--output_dir", CKPT_DIR,
            "--save_strategy", "steps",
            "--save_steps", SAVE_STEPS,
        ]
    if framework == "lightning":
        return cmd + ["--trainer.default_root_dir", CKPT_DIR]
    return cmd


def inject_resume_args(cmd, framework, ckpt_path):
    # type: (list, str, str) -> list
    """Append framework-specific resume args on top of initial args."""
    cmd = inject_initial_args(cmd, framework)
    if framework == "huggingface":
        return cmd + ["--resume_from_checkpoint", ckpt_path]
    if framework == "lightning":
        return cmd + ["--ckpt_path", ckpt_path]
    return cmd


def main():
    # type: () -> None
    cmd = sys.argv[1:]
    if not cmd:
        _log("Usage: checkpoint_shim.py <command> [args...]")
        sys.exit(1)

    latest = find_latest_checkpoint(CKPT_DIR) if RESUME else None
    if latest:
        _log("Resuming from checkpoint: " + latest)
        cmd = inject_resume_args(cmd, FRAMEWORK, latest)
    else:
        _log("No checkpoint found, starting fresh.")
        cmd = inject_initial_args(cmd, FRAMEWORK)

    _log("Running: " + " ".join(cmd))
    proc = subprocess.Popen(cmd)

    def _sigterm_handler(signum, frame):
        _log("SIGTERM received (spot preemption). Sending SIGINT for graceful save...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=110)
        except subprocess.TimeoutExpired:
            _log("Timeout waiting for graceful exit. Terminating.")
            proc.terminate()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
