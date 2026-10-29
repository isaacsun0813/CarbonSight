"""AWS Spot Instance Advisor bands (trailing-30-day interruption frequency).

The advisor page documents its bands: "Frequency of interruption represents the rate at which
Spot has reclaimed capacity during the trailing month" in ranges <5%, 5-10%, 10-15%, 15-20%,
>20%. The JSON behind that page is NOT a documented AWS API and may change or disappear, so
callers must treat this as optional and fail open.

The bands are coarse, regional, and have ill-defined units (the top band covers anything up to
100%), so they are retained as metadata and do not determine predicted hours of life.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

ADVISOR_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"


class SpotAdvisorError(Exception):
    """The advisor data could not be fetched or parsed."""


def parse_advisor_bands(
    data: dict[str, Any],
    instance_type: str,
    regions: Sequence[str],
    *,
    os_name: str = "Linux",
) -> dict[str, int]:
    """
    Band index per region (0 = <5% ... 4 = >20%) for ``instance_type``.

    Regions where the type is not listed are omitted. Band labels from the file's ``ranges``
    are checked so a format change is not silently misread.
    """
    try:
        labels = [r["label"] for r in data["ranges"]]
        advisor = data["spot_advisor"]
    except (KeyError, TypeError) as err:
        raise SpotAdvisorError(f"unexpected advisor data format: {err}") from err
    if labels != ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]:
        raise SpotAdvisorError(f"unexpected advisor bands: {labels}")

    bands: dict[str, int] = {}
    for region in regions:
        entry = advisor.get(region, {}).get(os_name, {}).get(instance_type)
        if not entry:
            continue
        index = entry.get("r")
        if isinstance(index, int) and 0 <= index < len(labels):
            bands[region] = index
    return bands


def fetch_advisor_bands(
    instance_type: str,
    regions: Sequence[str],
    *,
    timeout_seconds: float = 20.0,
    client: httpx.Client | None = None,
) -> dict[str, int]:
    """Download the advisor JSON and return band indices; raises ``SpotAdvisorError``."""
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
    try:
        response = http.get(ADVISOR_URL)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as err:
        raise SpotAdvisorError(f"could not fetch spot advisor data: {err}") from err
    finally:
        if owns_client:
            http.close()
    return parse_advisor_bands(data, instance_type, regions)
