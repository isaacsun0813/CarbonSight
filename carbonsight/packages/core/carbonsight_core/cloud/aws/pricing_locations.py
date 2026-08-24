"""
AWS region codes → Pricing API ``location`` display names for get_products filters.

The Pricing API uses human-readable locations (e.g. ``US East (N. Virginia)``),
not ``us-east-1``. Sources: AWS Price List / EC2 on-demand pricing pages.
"""

REGION_TO_PRICING_LOCATION: dict[str, str] = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "ca-central-1": "Canada (Central)",
    "eu-west-1": "Europe (Ireland)",
    "eu-west-2": "Europe (London)",
    "eu-west-3": "Europe (Paris)",
    "eu-central-1": "Europe (Frankfurt)",
    "eu-north-1": "Europe (Stockholm)",
    "eu-south-1": "Europe (Milan)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "sa-east-1": "South America (Sao Paulo)",
    "me-south-1": "Middle East (Bahrain)",
    "af-south-1": "Africa (Cape Town)",
}


def pricing_location_for_region(region_code: str) -> str | None:
    """Return Pricing API location string for an AWS region code, or None if unknown."""
    return REGION_TO_PRICING_LOCATION.get(region_code.strip().lower())
