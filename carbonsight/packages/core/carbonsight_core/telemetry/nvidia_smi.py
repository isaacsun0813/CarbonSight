"""
Read GPU utilization from nvidia-smi for use as JobSpec.gpu_utilization.

Uses ``utilization.gpu`` (compute), which is a rough duty-cycle signal, not power.
Still useful to anchor the power model when you have a live machine.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Final

_NVIDIA_SMI: Final[str] = "nvidia-smi"
_TIMEOUT_SEC: Final[float] = 15.0


def parse_nvidia_smi_utilization_csv(text: str) -> list[float]:
    """
    Parse lines from ``nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits``.

    Returns per-GPU utilization in [0, 100] as reported by the driver (percent).
    """
    values: list[float] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def sample_mean_gpu_utilization_from_nvidia_smi() -> float | None:
    """
    Run nvidia-smi and return mean GPU utilization in [0, 1], or None if unavailable.

    Averages ``utilization.gpu`` across all visible GPUs.
    """
    if not shutil.which(_NVIDIA_SMI):
        return None
    try:
        proc = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    percents = parse_nvidia_smi_utilization_csv(proc.stdout)
    if not percents:
        return None
    mean_pct = sum(percents) / len(percents)
    return max(0.0, min(1.0, mean_pct / 100.0))
