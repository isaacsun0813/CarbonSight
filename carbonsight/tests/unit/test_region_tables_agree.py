"""The AWS region universe is defined in three places; they must not drift apart.

``seed_registry.json`` is the source of truth: ``collect_estimates`` loops over it.
Two static tables key off the same region codes:

- ``REGION_TO_PRICING_LOCATION`` -- region -> Pricing API display name. A miss here
  silently disables *live* on-demand pricing for that region (falls back to static).
- ``_REGION_MULTIPLIER`` -- region -> static price multiplier. A miss falls back to
  ``_DEFAULT_MULTIPLIER``.

Both fall back quietly, so drift produces worse numbers with no error. These tests
turn that silence into a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from carbonsight_core.cloud.aws.pricing_locations import REGION_TO_PRICING_LOCATION
from carbonsight_core.estimator.pricing import _REGION_MULTIPLIER

SEED_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "core"
    / "carbonsight_core"
    / "mapping"
    / "seed_registry.json"
)


@pytest.fixture(scope="module")
def registry_region_codes() -> set[str]:
    payload = json.loads(SEED_REGISTRY.read_text())
    regions = payload["regions"] if isinstance(payload, dict) else payload
    return {entry["region_code"] for entry in regions if entry["provider"].lower() == "aws"}


def test_registry_is_non_empty(registry_region_codes: set[str]) -> None:
    assert registry_region_codes, "seed_registry.json has no AWS regions"


def test_every_registry_region_has_a_pricing_location(registry_region_codes: set[str]) -> None:
    """Without this, live on-demand pricing silently degrades to the static table."""
    missing = registry_region_codes - set(REGION_TO_PRICING_LOCATION)
    assert not missing, (
        f"registry regions with no Pricing API location: {sorted(missing)}. "
        "Add them to REGION_TO_PRICING_LOCATION or live on-demand pricing will "
        "silently fall back to static estimates for these regions."
    )


def test_every_registry_region_has_a_price_multiplier(registry_region_codes: set[str]) -> None:
    """Without this, static pricing uses _DEFAULT_MULTIPLIER with no warning."""
    missing = registry_region_codes - set(_REGION_MULTIPLIER)
    assert not missing, (
        f"registry regions with no price multiplier: {sorted(missing)}. "
        "Add them to _REGION_MULTIPLIER or static cost estimates will quietly "
        "use the default fallback multiplier."
    )


def test_no_orphan_entries_in_the_pricing_tables(registry_region_codes: set[str]) -> None:
    """Entries for regions the registry doesn't know about are dead weight."""
    orphan_locations = set(REGION_TO_PRICING_LOCATION) - registry_region_codes
    orphan_multipliers = set(_REGION_MULTIPLIER) - registry_region_codes
    assert not orphan_locations and not orphan_multipliers, (
        f"pricing-location orphans: {sorted(orphan_locations)}; "
        f"multiplier orphans: {sorted(orphan_multipliers)}. "
        "Either add these regions to the registry or drop the stale entries."
    )
