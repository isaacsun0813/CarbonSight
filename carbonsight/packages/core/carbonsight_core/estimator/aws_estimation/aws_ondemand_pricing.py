"""Live EC2 on-demand pricing via AWS Price List API (get_products)."""

import json

from botocore.exceptions import ClientError

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider
from carbonsight_core.cloud.aws.gpu_catalog import aws_gpu_instance_spec
from carbonsight_core.estimator.aws_estimation.aws_pricing_locations import (
    pricing_location_for_region,
)

PRICING_API_REGION = "us-east-1"
EC2_SERVICE_CODE = "AmazonEC2"


def parse_ondemand_instance_price_usd(price_list_entry: str) -> float | None:
    """Parse hourly on-demand instance USD from one get_products PriceList JSON string."""
    try:
        product = json.loads(price_list_entry)
        on_demand = product.get("terms", {}).get("OnDemand", {})
        if not on_demand:
            return None
        first_term = next(iter(on_demand.values()))
        dimensions = first_term.get("priceDimensions", {})
        if not dimensions:
            return None
        first_dim = next(iter(dimensions.values()))
        usd = first_dim.get("pricePerUnit", {}).get("USD")
        if usd is None:
            return None
        return float(usd)
    except (json.JSONDecodeError, StopIteration, TypeError, ValueError):
        return None


def _build_get_products_filters(instance_type: str, pricing_location: str) -> list[dict[str, str]]:
    return [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "location", "Value": pricing_location},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
    ]


class OnDemandPriceProvider(BaseAWSProvider):
    """Fetch on-demand $/GPU/hr from pricing.get_products; cache per (instance_type, location)."""

    def ondemand_price_per_gpu_hour(self, gpu_type: str, region_code: str) -> float | None:
        """Return on-demand $/GPU/hr for the region, or None if unavailable."""
        if not HAS_BOTO:
            return None

        spec = aws_gpu_instance_spec(gpu_type)
        if spec is None:
            return None

        pricing_location = pricing_location_for_region(region_code)
        if pricing_location is None:
            return None

        instance_type = spec.instance_type
        cache_key = (instance_type, pricing_location)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            client = self._client("pricing", PRICING_API_REGION)
            resp = client.get_products(
                ServiceCode=EC2_SERVICE_CODE,
                Filters=_build_get_products_filters(instance_type, pricing_location),
                MaxResults=1,
            )
            price_list = resp.get("PriceList", [])
            if not price_list:
                return None

            instance_hr = parse_ondemand_instance_price_usd(price_list[0])
            if instance_hr is None:
                return None

            price_per_gpu = instance_hr / spec.gpus_per_instance
            return self._set_cached(cache_key, price_per_gpu)
        except ClientError:
            return None
        except Exception:
            return None
