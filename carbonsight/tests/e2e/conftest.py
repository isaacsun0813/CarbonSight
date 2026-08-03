"""E2E VCR configuration for replayed WattTime HTTP."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        "filter_hosts": ["api.watttime.org"],
        "record_mode": "none",
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    return str(Path(__file__).resolve().parent / "cassettes")
