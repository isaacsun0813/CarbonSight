"""SkyPilot GPU label → representative EC2 instance type for AWS APIs."""

from dataclasses import dataclass
from typing import Any


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


def spot_placement_score_request(
    gpu_type: str,
    regions: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Static ``ec2:GetSpotPlacementScores`` request kwargs for a GPU label (future use).

    NOT called anywhere yet. Caveats from the AWS docs, so nobody over-reads the result:
    the score (1-10) is a point-in-time hint about whether a request is *likely* to succeed;
    it is not a guarantee and not an interruption or lifetime measurement. Scores vary with the
    exact request configuration and change over time, so re-query for a fresh one; AWS limits
    how many distinct configurations may be queried per 24h. GPU families (P/G/Trn/Inf) are
    exempt from the 3-instance-type minimum, so a single type is a valid request.

    Returns None for GPU labels not in the catalog.
    """
    instance_type = aws_instance_type_for_gpu(gpu_type)
    if instance_type is None:
        return None
    request: dict[str, Any] = {
        "InstanceTypes": [instance_type],
        "TargetCapacity": 1,
        "TargetCapacityUnitType": "units",
    }
    if regions:
        request["RegionNames"] = list(regions)
    return request
