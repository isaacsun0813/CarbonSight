"""Quota preflight picks the spot or on-demand quota family that matches the launch."""

import pytest

from carbonsight_core.cloud.aws.quota import QuotaChecker
from carbonsight_core.config import Config

G_VT_ON_DEMAND = "L-DB2E81BA"
G_VT_SPOT = "L-3819A6DF"
P_ON_DEMAND = "L-417A185B"
P_SPOT = "L-7212CCBC"


class _FakeServiceQuotas:
    def __init__(self, values: dict[str, float]) -> None:
        self._values = values
        self.requested_codes: list[str] = []

    def get_service_quota(self, ServiceCode: str, QuotaCode: str) -> dict:
        self.requested_codes.append(QuotaCode)
        return {"Quota": {"Value": self._values[QuotaCode]}}


def _checker_with(fake: _FakeServiceQuotas) -> QuotaChecker:
    checker = QuotaChecker(Config())
    checker._client = lambda service_name, region_name: fake  # type: ignore[method-assign]
    return checker


@pytest.mark.parametrize(
    ("gpu", "use_spot", "expected"),
    [
        ("T4", False, G_VT_ON_DEMAND),
        ("T4", True, G_VT_SPOT),
        ("A10G", False, G_VT_ON_DEMAND),
        ("A10G", True, G_VT_SPOT),
        ("A100", False, P_ON_DEMAND),
        ("A100", True, P_SPOT),
        ("H100", False, P_ON_DEMAND),
        ("V100", True, P_SPOT),
    ],
)
def test_quota_code_for_gpu(gpu: str, use_spot: bool, expected: str) -> None:
    assert QuotaChecker.quota_code_for_gpu(gpu, use_spot=use_spot) == expected


def test_a100_is_not_mapped_to_g_and_vt_quota() -> None:
    assert QuotaChecker.quota_code_for_gpu("A100") not in {G_VT_ON_DEMAND, G_VT_SPOT}
    assert QuotaChecker.quota_code_for_gpu("A100", use_spot=True) not in {G_VT_ON_DEMAND, G_VT_SPOT}


def test_default_is_on_demand() -> None:
    assert QuotaChecker.quota_code_for_gpu("T4") == QuotaChecker.quota_code_for_gpu("T4", use_spot=False)


def test_spot_flag_reaches_service_quotas_api() -> None:
    fake = _FakeServiceQuotas({G_VT_ON_DEMAND: 4.0, G_VT_SPOT: 0.0})
    checker = _checker_with(fake)

    spot = checker.check_gpu_quota("ap-southeast-1", 1, gpu_type="T4", use_spot=True)
    assert fake.requested_codes == [G_VT_SPOT]
    assert spot.allowed is False
    assert spot.quota_code == G_VT_SPOT
    assert G_VT_SPOT in spot.reason

    on_demand = checker.check_gpu_quota("ap-southeast-1", 1, gpu_type="T4")
    assert fake.requested_codes == [G_VT_SPOT, G_VT_ON_DEMAND]
    assert on_demand.allowed is True
    assert on_demand.quota_code == G_VT_ON_DEMAND


def test_spot_and_on_demand_cache_do_not_collide() -> None:
    fake = _FakeServiceQuotas({G_VT_ON_DEMAND: 4.0, G_VT_SPOT: 0.0})
    checker = _checker_with(fake)

    assert checker.check_gpu_quota("us-east-1", 1, gpu_type="T4").allowed is True
    assert checker.check_gpu_quota("us-east-1", 1, gpu_type="T4", use_spot=True).allowed is False
    # Second round is served from cache with the same answers.
    assert checker.check_gpu_quota("us-east-1", 1, gpu_type="T4").allowed is True
    assert checker.check_gpu_quota("us-east-1", 1, gpu_type="T4", use_spot=True).allowed is False
    assert len(fake.requested_codes) == 2
