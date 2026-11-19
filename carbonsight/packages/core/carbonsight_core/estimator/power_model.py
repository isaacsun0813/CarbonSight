"""
Power model per design doc: P_IT = P_base + P_gpu + P_cpu + P_mem + P_net.

This module models IT equipment only. Cooling and other facility overhead are
excluded because CarbonSight does not have facility-level measurements.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from carbonsight_core.models import JobSpec

ASSUMED_GPU_IDLE_FRACTION = 0.1
GPU_POWER_CATALOG = json.loads(Path(__file__).with_name("gpu_power_specs.json").read_text())


def _gpu_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key.removeprefix("nvidia").removeprefix("tesla")


_GPU_POWER_SPECS = {
    _gpu_key(alias): spec
    for spec in GPU_POWER_CATALOG["gpus"]
    for alias in (spec["name"], *spec["aliases"])
}


P_BASE_W = 200.0
W_PER_GIB = 0.4
CPU_TDP_W = 150.0
CPU_IDLE_W = 30.0
NET_OVERHEAD_W = 20.0
DEFAULT_GPU_UTIL = 0.75
DEFAULT_CPU_UTIL = 0.5


@dataclass
class PowerParams:
    """Sampled IT-power parameters."""

    p_it_w: float


def gpu_power_limit_w(job: JobSpec) -> float:
    if job.gpu_count == 0:
        return 0.0
    if job.gpu_power_limit_w is not None:
        return job.gpu_power_limit_w
    spec = _GPU_POWER_SPECS.get(_gpu_key(job.gpu_type))
    if spec is None:
        raise ValueError(
            f"No sourced GPU power specification for {job.gpu_type!r}. "
            "Set carbonsight.gpu_power_limit_w in YAML or gpu_power_limit_w in the API request."
        )
    return float(spec["max_power_w"])


def gpu_power_notes(job: JobSpec) -> list[str]:
    limit = gpu_power_limit_w(job)
    if job.gpu_count == 0:
        return []
    source = (
        "user-provided" if job.gpu_power_limit_w is not None
        else _GPU_POWER_SPECS[_gpu_key(job.gpu_type)]["source_url"]
    )
    return [
        f"power:gpu_max={limit:g}W_per_gpu; source={source}",
        "power:modeled; assumed idle=10% of maximum; linear utilization curve; not measured",
    ]


def _gpu_power(job: JobSpec, u_gpu: float) -> float:
    cap = gpu_power_limit_w(job)
    idle = ASSUMED_GPU_IDLE_FRACTION * cap
    return job.gpu_count * (idle + u_gpu * (cap - idle))


def _cpu_power(job: JobSpec, u_cpu: float) -> float:
    sockets = (job.cpu_count or 8) // 8 or 1
    return sockets * (CPU_IDLE_W + u_cpu * (CPU_TDP_W - CPU_IDLE_W))


def _mem_power(job: JobSpec) -> float:
    gib = job.mem_gib or (job.gpu_count * 40)
    return gib * W_PER_GIB


def power_it_w(job: JobSpec, u_gpu: float = DEFAULT_GPU_UTIL, u_cpu: float = DEFAULT_CPU_UTIL) -> float:
    """P_IT = P_base + P_gpu + P_cpu + P_mem + P_net."""
    return (
        P_BASE_W
        + _gpu_power(job, u_gpu)
        + _cpu_power(job, u_cpu)
        + _mem_power(job)
        + NET_OVERHEAD_W
    )


def expected_it_power_w(job: JobSpec) -> float:
    """Expected IT watts, honoring observed GPU utilization when supplied."""
    u_gpu = job.gpu_utilization if job.gpu_utilization is not None else DEFAULT_GPU_UTIL
    return power_it_w(job, u_gpu=u_gpu, u_cpu=DEFAULT_CPU_UTIL)


def _sample_u_gpu(job: JobSpec, rng: random.Random) -> float:
    """GPU utilization factor for the power curve; fixed when telemetry is present."""
    if job.gpu_utilization is not None:
        return job.gpu_utilization
    return rng.uniform(0.6, 0.9)


def sample_power_params(job: JobSpec, rng: random.Random | None = None) -> PowerParams:
    """Sample GPU/CPU utilization and return IT power; facility overhead is excluded."""
    rng = rng or random
    u_gpu = _sample_u_gpu(job, rng)
    u_cpu = rng.uniform(0.4, 0.6)
    p_it = power_it_w(job, u_gpu, u_cpu)
    return PowerParams(p_it_w=p_it)
