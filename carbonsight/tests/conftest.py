"""Pytest config; load dotenv for E2E if present."""
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Add e2e marker."""
    config.addinivalue_line("markers", "e2e: end-to-end tests (may use vcr replay)")
    config.addinivalue_line(
        "markers",
        "integration: needs live credentials or external services (skipped in default CI)",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repo root (carbonsight/)."""
    return Path(__file__).resolve().parents[1]
