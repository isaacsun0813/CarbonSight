"""
GPU type → EC2 instance mapping for AWS pricing and availability APIs.

AWS APIs use instance types (e.g. p4d.24xlarge), not SkyPilot accelerator strings (e.g. A100).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuInstanceSpec:
    """Representative EC2 instance for a JobSpec.gpu_type."""

    instance_type: str
    gpus_per_instance: int


_GPU_CATALOG: dict[str, GpuInstanceSpec] = {
    "A100": GpuInstanceSpec(instance_type="p4d.24xlarge", gpus_per_instance=8),
    "H100": GpuInstanceSpec(instance_type="p5.48xlarge", gpus_per_instance=8),
    "V100": GpuInstanceSpec(instance_type="p3.16xlarge", gpus_per_instance=8),
    "T4": GpuInstanceSpec(instance_type="g4dn.xlarge", gpus_per_instance=1),
    "A10G": GpuInstanceSpec(instance_type="g5.xlarge", gpus_per_instance=1),
}


def _normalize_gpu_type(gpu_type: str) -> str:
    return gpu_type.strip().upper().split(":")[0].split("-")[0]


def instance_type_for_gpu(gpu_type: str) -> str | None:
    """Return EC2 instance type for a GPU label, or None if unknown."""
    spec = _GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
    return spec.instance_type if spec else None


def gpus_per_instance(gpu_type: str) -> int:
    """Return GPU count on the representative instance; 1 if unknown."""
    spec = _GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
    return spec.gpus_per_instance if spec else 1


def gpu_instance_spec(gpu_type: str) -> GpuInstanceSpec | None:
    """Return full spec for a GPU label, or None if unknown."""
    return _GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
