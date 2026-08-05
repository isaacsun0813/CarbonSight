"""AWS cloud provider infrastructure."""

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.aws.gpu_catalog import (
    AwsGpuInstanceSpec,
    aws_gpu_instance_spec,
    aws_gpus_per_instance,
    aws_instance_type_for_gpu,
)

__all__ = [
    "AwsGpuInstanceSpec",
    "BaseAWSProvider",
    "HAS_BOTO",
    "aws_gpu_instance_spec",
    "aws_gpus_per_instance",
    "aws_instance_type_for_gpu",
]
