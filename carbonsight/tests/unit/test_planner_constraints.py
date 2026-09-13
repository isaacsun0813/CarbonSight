"""Unit tests for planning constraint filters."""

from datetime import datetime, timezone

from carbonsight_core.planning.constraints import filter_by_priority
from carbonsight_core.planning.types import JobMode, PlanCandidate, PlanConstraints

UTC = timezone.utc


def _candidate(
    *,
    cost: float,
    carbon: float = 1.0,
    start: datetime | None = None,
    finish: datetime | None = None,
) -> PlanCandidate:
    t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    return PlanCandidate(
        cloud_region="us-east-1",
        mode=JobMode.SPOT,
        start_utc=start or t0,
        finish_utc=finish or datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        cost_usd=cost,
        carbon_kg=carbon,
        utility=0.0,
        moer_lb_per_mwh=400.0,
    )


def test_cost_premium_single_pass_partitions_over_ceiling() -> None:
    t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    constraints = PlanConstraints(
        deadline_utc=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        now_utc=t0,
        max_cost_premium=0.0,
    )
    cheap = _candidate(cost=10.0)
    expensive = _candidate(cost=20.0)
    survivors, filtered = filter_by_priority([cheap, expensive], constraints)
    assert survivors == [cheap]
    assert filtered["cost_premium"] == [expensive]


def test_cost_premium_negative_premium_keeps_cheapest_when_all_over_ceiling() -> None:
    t0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    constraints = PlanConstraints(
        deadline_utc=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        now_utc=t0,
        max_cost_premium=-0.25,
    )
    a = _candidate(cost=30.0, carbon=2.0)
    b = _candidate(cost=20.0, carbon=3.0)
    survivors, filtered = filter_by_priority([a, b], constraints)
    assert survivors == [b]
    assert filtered["cost_premium"] == []
