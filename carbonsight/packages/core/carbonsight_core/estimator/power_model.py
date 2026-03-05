"""
Power model per design doc: P_IT = P_base + P_gpu + P_cpu + P_mem + P_net.
GPU: P_gpu = gpu_count * (P_idle + u * (P_cap - P_idle)); T4 70W, A100 250-400W, H100 700W.
PUE default 1.20; Normal(1.20, 0.05) truncated [1.05, 1.60] for uncertainty.
"""

import random
from dataclasses import dataclass

from carbonsight_core.models import JobSpec

# GPU defaults (W): (idle, cap) per design
GPU_DEFAULTS: dict[str, tuple[float, float]] = {
    "t4": (25.0, 70.0),
    "a100": (80.0, 400.0),
    "a100-40": (80.0, 400.0),
    "a100-80": (80.0, 400.0),
    "h100": (150.0, 700.0),
    "h100-80": (150.0, 700.0),
}
DEFAULT_GPU = (80.0, 350.0)
P_BASE_W = 200.0
W_PER_GIB = 0.4
CPU_TDP_W = 150.0
CPU_IDLE_W = 30.0
NET_OVERHEAD_W = 20.0
DEFAULT_GPU_UTIL = 0.7
DEFAULT_CPU_UTIL = 0.5
PUE_MEAN = 1.20
PUE_SIGMA = 0.05
PUE_MIN, PUE_MAX = 1.05, 1.60


@dataclass
class PowerParams:
    """Sampled or fixed power parameters."""

    p_it_w: float
    pue: float


def _gpu_power(job: JobSpec, u_gpu: float) -> float:
    key = job.gpu_type.lower().replace(" ", "-")
    idle, cap = GPU_DEFAULTS.get(key, DEFAULT_GPU)
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


def sample_pue(rng: random.Random | None = None) -> float:
    """PUE ~ Normal(1.20, 0.05) truncated [1.05, 1.60]."""
    rng = rng or random
    while True:
        u = rng.gauss(PUE_MEAN, PUE_SIGMA)
        if PUE_MIN <= u <= PUE_MAX:
            return u


def sample_power_params(job: JobSpec, rng: random.Random | None = None) -> PowerParams:
    """Sample u_gpu, PUE and return P_IT and PUE."""
    rng = rng or random
    u_gpu = rng.uniform(0.6, 0.9)
    u_cpu = rng.uniform(0.4, 0.6)
    p_it = power_it_w(job, u_gpu, u_cpu)
    pue = sample_pue(rng)
    return PowerParams(p_it_w=p_it, pue=pue)
