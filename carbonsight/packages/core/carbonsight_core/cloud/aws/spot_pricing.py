"""Live EC2 spot pricing via describe_spot_price_history."""

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider, ClientError
from carbonsight_core.cloud.aws.gpu_catalog import aws_gpu_instance_spec

SPOT_HISTORY_MAX_RESULTS = 50


class SpotPriceProvider(BaseAWSProvider):
    """Fetch regional spot $/GPU/hr from EC2 spot price history; cache per (region, instance_type)."""

    def spot_price_per_gpu_hour(self, gpu_type: str, region: str) -> float | None:
        """
        Return the cheapest recent spot $/GPU/hr in the region, or None if unavailable.

        Takes ``min`` over the returned history, i.e. the lowest price seen in any AZ
        at any point in the window. This is **optimistic**, not conservative: the
        quoted figure is a best case you may not get, since you cannot choose which
        AZ you land in. No ``StartTime`` is passed, so the window is however far back
        ``MaxResults`` points happen to reach.

        Fine for ranking regions against each other (the bias is broadly similar
        across regions); read the absolute number with that caveat in mind. A median,
        or the most recent price per AZ, would be a defensible alternative.
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
