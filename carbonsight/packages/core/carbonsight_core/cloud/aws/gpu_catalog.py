"""SkyPilot GPU label → representative EC2 instance type for AWS APIs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AwsGpuInstanceSpec:
    """Representative EC2 instance for a JobSpec.gpu_type."""

    instance_type: str
    gpus_per_instance: int


_AWS_GPU_CATALOG: dict[str, AwsGpuInstanceSpec] = {
    "A100": AwsGpuInstanceSpec(instance_type="p4d.24xlarge", gpus_per_instance=8),
    "H100": AwsGpuInstanceSpec(instance_type="p5.48xlarge", gpus_per_instance=8),
    "V100": AwsGpuInstanceSpec(instance_type="p3.16xlarge", gpus_per_instance=8),
    "T4": AwsGpuInstanceSpec(instance_type="g4dn.xlarge", gpus_per_instance=1),
    "A10G": AwsGpuInstanceSpec(instance_type="g5.xlarge", gpus_per_instance=1),
}


def _normalize_gpu_type(gpu_type: str) -> str:
    return gpu_type.strip().upper().split(":")[0].split("-")[0]


def aws_instance_type_for_gpu(gpu_type: str) -> str | None:
    """Return EC2 instance type for a GPU label, or None if unknown."""
    spec = _AWS_GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
    return spec.instance_type if spec else None


def aws_gpus_per_instance(gpu_type: str) -> int:
    """Return GPU count on the representative instance; 1 if unknown."""
    spec = _AWS_GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
    return spec.gpus_per_instance if spec else 1


def aws_gpu_instance_spec(gpu_type: str) -> AwsGpuInstanceSpec | None:
    """Return full EC2 spec for a GPU label, or None if unknown."""
    return _AWS_GPU_CATALOG.get(_normalize_gpu_type(gpu_type))
