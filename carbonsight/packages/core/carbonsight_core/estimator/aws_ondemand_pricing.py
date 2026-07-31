"""
Live EC2 on-demand pricing via AWS Price List API (get_products).

Fails open: returns None on missing boto3, unknown GPU/region, or API errors.
"""

import json
import time
from typing import Any

from carbonsight_core.config import Config
from carbonsight_core.estimator.aws_pricing_locations import pricing_location_for_region
from carbonsight_core.estimator.gpu_catalog import gpu_instance_spec

try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

PRICING_API_REGION = "us-east-1"
EC2_SERVICE_CODE = "AmazonEC2"


def parse_ondemand_instance_price_usd(price_list_entry: str) -> float | None:
    """
    Parse hourly on-demand instance USD from one get_products PriceList JSON string.

    Walks terms.OnDemand only (not Reserved / SavingsPlan).
    """
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


class OnDemandPriceProvider:
    """Fetch on-demand $/GPU/hr from pricing.get_products; cache per (instance_type, location)."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._cache_ttl = cache_ttl_seconds or self._config.pricing_cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._client: Any = None
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            if not _HAS_BOTO:
                raise RuntimeError("boto3 not installed")
            profile = self._config.aws_profile
            self._session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return self._session

    def _pricing_client(self) -> Any:
        if self._client is None:
            self._client = self._get_session().client("pricing", region_name=PRICING_API_REGION)
        return self._client

    def _get_cached(self, instance_type: str, pricing_location: str) -> float | None:
        key = (instance_type, pricing_location)
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, instance_type: str, pricing_location: str, price_per_gpu_hr: float) -> None:
        self._cache[(instance_type, pricing_location)] = (time.time() + self._cache_ttl, price_per_gpu_hr)

    def ondemand_price_per_gpu_hour(self, gpu_type: str, region_code: str) -> float | None:
        """Return on-demand $/GPU/hr for the region, or None if unavailable."""
        if not _HAS_BOTO:
            return None

        spec = gpu_instance_spec(gpu_type)
        if spec is None:
            return None

        pricing_location = pricing_location_for_region(region_code)
        if pricing_location is None:
            return None

        instance_type = spec.instance_type
        cached = self._get_cached(instance_type, pricing_location)
        if cached is not None:
            return cached

        try:
            client = self._pricing_client()
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
            self._set_cached(instance_type, pricing_location, price_per_gpu)
            return price_per_gpu
        except ClientError:
            return None
        except Exception:
            return None
