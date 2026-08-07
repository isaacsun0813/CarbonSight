"""AWS cloud provider: session, catalog, pricing, and preflight checks."""

from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker
from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.cloud.aws.gpu_catalog import (
    AwsGpuInstanceSpec,
    aws_gpu_instance_spec,
    aws_gpus_per_instance,
    aws_instance_type_for_gpu,
)
from carbonsight_core.cloud.aws.ondemand_pricing import OnDemandPriceProvider
from carbonsight_core.cloud.aws.pricing_locations import pricing_location_for_region
from carbonsight_core.cloud.aws.quota import QuotaChecker, QuotaResult
from carbonsight_core.cloud.aws.spot_pricing import SpotPriceProvider

__all__ = [
    "AwsGpuInstanceSpec",
    "BaseAWSProvider",
    "EnabledRegionsProvider",
    "HAS_BOTO",
    "InstanceAvailabilityChecker",
    "OnDemandPriceProvider",
    "QuotaChecker",
    "QuotaResult",
    "SpotPriceProvider",
    "aws_gpu_instance_spec",
    "aws_gpus_per_instance",
    "aws_instance_type_for_gpu",
    "pricing_location_for_region",
]
