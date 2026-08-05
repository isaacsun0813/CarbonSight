"""Live EC2 spot pricing via describe_spot_price_history."""

from botocore.exceptions import ClientError

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.aws.gpu_catalog import aws_gpu_instance_spec

SPOT_HISTORY_MAX_RESULTS = 50


class SpotPriceProvider(BaseAWSProvider):
    """Fetch regional spot $/GPU/hr from EC2 spot price history; cache per (region, instance_type)."""

    def spot_price_per_gpu_hour(self, gpu_type: str, region: str) -> float | None:
        """
        Return minimum recent spot $/GPU/hr in the region, or None if unavailable.

        Uses min SpotPrice across AZs (conservative for cost ranking).
        """
        if not HAS_BOTO:
            return None

        spec = aws_gpu_instance_spec(gpu_type)
        if spec is None:
            return None

        instance_type = spec.instance_type
        cache_key = (region, instance_type)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ec2 = self._client("ec2", region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=SPOT_HISTORY_MAX_RESULTS,
            )
            history = resp.get("SpotPriceHistory", [])
            if not history:
                return None

            min_instance_hr = min(float(entry["SpotPrice"]) for entry in history)
            price_per_gpu = min_instance_hr / spec.gpus_per_instance
            return self._set_cached(cache_key, price_per_gpu)
        except ClientError:
            return None
        except Exception:
            return None
