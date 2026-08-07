"""boto3 is an optional extra; the package must still import and fail open without it.

CI installs ``.[dev]``, which pulls in boto3, so the ``HAS_BOTO = False`` branches in
every AWS module are never exercised by the normal suite. A plain
``pip install carbonsight`` has no boto3 — and previously the whole ``cloud.aws``
package raised ImportError there, because four modules did an unguarded
``from botocore.exceptions import ClientError`` at module top (botocore ships with
boto3). Every fallback path was unreachable code.

These tests run in a subprocess with boto3 and botocore blocked at the import hook,
which is the only reliable way to simulate absence once a module is cached.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "packages" / "core"

_BLOCKER = """
import sys
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("boto3", "botocore"):
            raise ImportError("No module named %r" % name)
sys.meta_path.insert(0, _Blocker())
"""


def _run_without_boto3(body: str) -> subprocess.CompletedProcess[str]:
    """Execute ``body`` in a subprocess where boto3/botocore cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(CORE), "PATH": "/usr/bin:/bin"},
        timeout=60,
    )


def test_boto3_is_genuinely_blocked_in_the_harness() -> None:
    """Guard the guard: if the blocker stopped working these tests would pass vacuously."""
    proc = _run_without_boto3("""
        try:
            import boto3
            print("NOT_BLOCKED")
        except ImportError:
            print("BLOCKED")
    """)
    assert "BLOCKED" in proc.stdout, proc.stderr


def test_cloud_aws_package_imports_without_boto3() -> None:
    """The regression: this raised ImportError before ClientError was centralised."""
    proc = _run_without_boto3("""
        from carbonsight_core.cloud import aws
        print("HAS_BOTO", aws.HAS_BOTO)
    """)
    assert proc.returncode == 0, f"cloud.aws failed to import without boto3:\n{proc.stderr}"
    assert "HAS_BOTO False" in proc.stdout


def test_every_aws_module_imports_without_boto3() -> None:
    proc = _run_without_boto3("""
        import importlib
        for name in [
            "base", "gpu_catalog", "availability", "enabled_regions",
            "quota", "spot_pricing", "ondemand_pricing", "pricing_locations",
        ]:
            importlib.import_module("carbonsight_core.cloud.aws." + name)
        print("ALL_IMPORTED")
    """)
    assert proc.returncode == 0, proc.stderr
    assert "ALL_IMPORTED" in proc.stdout


def test_providers_fail_open_without_boto3() -> None:
    """No boto3 must mean 'skip the check', never 'block the region'."""
    proc = _run_without_boto3("""
        from carbonsight_core.cloud.aws import (
            EnabledRegionsProvider, InstanceAvailabilityChecker,
            OnDemandPriceProvider, QuotaChecker, SpotPriceProvider,
        )
        assert QuotaChecker().check_gpu_quota("us-east-1", 8).allowed is True
        assert InstanceAvailabilityChecker().check_instance_offering(
            "us-east-1", "A100").available is True
        assert EnabledRegionsProvider().enabled_region_codes() is None
        assert SpotPriceProvider().spot_price_per_gpu_hour("A100", "us-east-1") is None
        assert OnDemandPriceProvider().ondemand_price_per_gpu_hour("A100", "us-east-1") is None
        print("FAILS_OPEN")
    """)
    assert proc.returncode == 0, proc.stderr
    assert "FAILS_OPEN" in proc.stdout


def test_cost_estimates_fall_back_to_static_tables_without_boto3() -> None:
    """Live pricing off => the old static behaviour, not a crash."""
    proc = _run_without_boto3("""
        from carbonsight_core.estimator.pricing import configure_pricing, estimate_job_cost
        configure_pricing(live_aws_pricing=True)   # asked for live, but boto3 is absent
        cost = estimate_job_cost("A100", 1, 1.0, "us-east-1", use_spot=False)
        assert cost.usd > 0, cost
        assert "cost:static_ondemand_fallback" in cost.notes, cost.notes
        print("STATIC_FALLBACK")
    """)
    assert proc.returncode == 0, proc.stderr
    assert "STATIC_FALLBACK" in proc.stdout
