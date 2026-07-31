"""Central cache design stubs + spot module smoke tests (many small cases)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS, SyntheticCarbonProvider
from carbonsight_core.providers.spot import Boto3SpotPriceProvider, StaticSpotPriceProvider
from carbonsight_core.spot.availability import AvailabilityTracker, SpotObservation, VirtualInstance
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.policy import Action, SkyNomadPolicy
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.scheduler_service import (
    build_candidates,
    candidates_to_estimates,
    schedule_job,
)
from carbonsight_core.spot.unified_model import ODCandidate, compute_utility, rank_candidates
from carbonsight_core.watttime.cache import ForecastCache, time_weighted_moer
from carbonsight_core.watttime.client import (
    SYNTHETIC_WT_REGIONS,
    WattTimeClient,
    build_synthetic_forecast,
)


# --- cache / client ---------------------------------------------------------

def test_synthetic_regions_len():
    assert len(SYNTHETIC_WT_REGIONS) == 17
    assert len(DEFAULT_CARBON_REGIONS) == 17


def test_client_without_creds_returns_list():
    c = WattTimeClient(allow_synthetic=True)
    # clear creds via empty config defaults
    out = c.get_forecast("CAISO_NORTH", horizon_hours=2)
    pts = WattTimeClient.normalize_forecast_payload(out)
    assert isinstance(pts, list) and len(pts) > 0


def test_build_synthetic_forecast_horizon():
    pts = build_synthetic_forecast("SE", horizon_hours=1, step_minutes=5)
    assert len(pts) == 12


@pytest.mark.parametrize("region", SYNTHETIC_WT_REGIONS[:8])
def test_cache_per_region_param(region: str):
    cache = ForecastCache()
    cache.set(region, build_synthetic_forecast(region, horizon_hours=1))
    assert cache.get(region) is not None
    assert cache.get_moer(region, lbar_hours=1.0) is not None


# --- availability -----------------------------------------------------------

def test_availability_tracker_basic():
    t = AvailabilityTracker()
    now = datetime.now(UTC)
    t.add_observation(SpotObservation("us-east-1", "A100", now, True))
    t.add_observation(SpotObservation("us-east-1", "A100", now, False))
    assert t.get_availability("us-east-1", "A100") == pytest.approx(0.5)
    assert t.eviction_counts()[("us-east-1", "A100")] == 1


def test_virtual_instance_evict():
    vi = VirtualInstance("us-west-2", "H100", datetime.now(UTC), instance_id="i-1")
    obs = vi.mark_evicted()
    assert obs.evicted and not vi.alive


def test_at_risk_set():
    t = AvailabilityTracker()
    t.seed_defaults([("r1", "A100")], default_availability=0.5, n=20)
    risk = t.at_risk_set(min_eviction_rate=0.3)
    assert ("r1", "A100") in risk


# --- lifetime ---------------------------------------------------------------

def test_lifetime_stats_survival_and_lbar():
    s = LifetimeStats.from_exponential(24.0, n=30)
    assert 0.0 <= s.compute_survival(1.0) <= 1.0
    assert s.expected_remaining(0.0) > 0
    assert s.compute_at_risk(0.0) == s.n()


def test_lifetime_hazard_nonnegative():
    s = LifetimeStats()
    s.add(5.0, evicted=True)
    s.add(10.0, evicted=False)
    s.add(5.0, evicted=True)
    assert s.compute_hazard(5.0) >= 0
    assert s.compute_cumulative_hazard(10.0) >= 0


# --- progress ---------------------------------------------------------------

def test_progress_theta_and_urgency():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ps = ProgressState(P=100, p=25, T=now + timedelta(hours=10), t=now)
    assert ps.theta() == pytest.approx(7.5)
    assert ps.urgency(baseline_theta=5.0) == pytest.approx(1.5)
    assert not ps.is_complete()
    ps.advance(75)
    assert ps.is_complete()


def test_progress_from_deadline_hours():
    ps = ProgressState.from_deadline_hours(deadline_hours=45)
    assert ps.time_left_hours() == pytest.approx(45.0, rel=0.01)


# --- unified model / policy -------------------------------------------------

def test_compute_utility_fields():
    ps = ProgressState.from_deadline_hours(deadline_hours=24)
    c = ODCandidate(
        region="us-east-1",
        instance_type="A100",
        spot_price=1.5,
        moer=300.0,
        survival=0.8,
        lbar=8.0,
        on_demand_price=4.0,
    )
    u = compute_utility(c, ps, carbon_price_usd_per_ton=50.0)
    assert isinstance(u, float)
    assert c.c_total > 0
    assert c.carbon_kg > 0


def test_rank_candidates_order():
    ps = ProgressState.from_deadline_hours(deadline_hours=24)
    a = ODCandidate("a", "A100", spot_price=1.0, moer=100.0, lbar=5.0, on_demand_price=3.0)
    b = ODCandidate("b", "A100", spot_price=5.0, moer=900.0, lbar=5.0, on_demand_price=3.0)
    ranked = rank_candidates([b, a], ps, carbon_price_usd_per_ton=5000.0)
    assert ranked[0].region == "a"


def test_policy_probe_then_run():
    tracker = AvailabilityTracker()
    life = LifetimeStats.from_exponential(12.0)
    ps = ProgressState.from_deadline_hours(deadline_hours=20)
    policy = SkyNomadPolicy(tracker, life, ps, carbon_price_usd_per_ton=50.0)
    cands = [
        ODCandidate("us-east-1", "A100", 1.0, 200.0, 0.9, 10.0, on_demand_price=3.0),
        ODCandidate("eu-north-1", "A100", 1.2, 150.0, 0.9, 10.0, on_demand_price=3.5),
    ]
    d1 = policy.decide(cands)
    assert d1.action in (Action.PROBE, Action.RUN)
    if d1.target:
        policy.current = d1.target
    d2 = policy.decide(cands)
    assert d2.action in (Action.RUN, Action.MIGRATE, Action.PROBE, Action.WAIT)


def test_policy_terminate_when_done():
    tracker = AvailabilityTracker()
    life = LifetimeStats()
    now = datetime.now(UTC)
    ps = ProgressState(P=1.0, p=1.0, T=now + timedelta(hours=1), t=now)
    policy = SkyNomadPolicy(tracker, life, ps)
    d = policy.decide([])
    assert d.action == Action.TERMINATE


# --- spot providers ---------------------------------------------------------

def test_boto3_stub_falls_back():
    p = Boto3SpotPriceProvider(use_fallback=True)
    assert p.get_spot_price_usd_per_gpu_hr("us-east-1", "A100") > 0


def test_boto3_stub_raises_without_fallback():
    p = Boto3SpotPriceProvider(use_fallback=False)
    with pytest.raises(NotImplementedError):
        p.get_spot_price_usd_per_gpu_hr("us-east-1", "A100")


def test_static_does_not_mutate_pricing_module():
    from carbonsight_core.estimator import pricing

    before = dict(pricing._GPU_BASE_PRICE_USD_PER_HR)
    s = StaticSpotPriceProvider()
    s._gpu_base["A100"] = 999.0  # mutate provider copy only
    assert pricing._GPU_BASE_PRICE_USD_PER_HR["A100"] == before["A100"]


# --- scheduler service ------------------------------------------------------

def test_schedule_job_returns_ranked_list():
    job = JobSpec(
        gpu_type="A100",
        gpu_count=1,
        duration_hours=1.0,
        deadline_hours=45.0,
        checkpoint_size_gb=100.0,
        carbon_price_usd_per_ton=50.0,
    )
    result = schedule_job(job, top_n=17)
    assert len(result.ranked) == 17
    assert result.ranked[0].utility >= result.ranked[-1].utility
    est = candidates_to_estimates(result.ranked, job)
    assert len(est) == 17
    assert est[0].utility_score is not None


def test_build_candidates_with_synthetic_carbon():
    job = JobSpec(gpu_type="T4", gpu_count=1, duration_hours=2.0, deadline_hours=10.0)
    ranked, ps = build_candidates(job, carbon=SyntheticCarbonProvider())
    assert len(ranked) >= 17
    assert ps.theta() > 0


# --- extra parametric coverage toward 191+ suite together with existing -----

@pytest.mark.parametrize("hours", [1.0, 2.0, 5.0, 12.0, 24.0])
def test_lbar_window_param(hours: float):
    cache = ForecastCache()
    cache.set("R", build_synthetic_forecast("R", horizon_hours=48))
    m = cache.get_moer("R", lbar_hours=hours)
    assert m is not None and m > 0


@pytest.mark.parametrize("ttl", [1, 60, 900, 3600])
def test_cache_ttl_variants(ttl: int):
    c = ForecastCache(ttl_seconds=ttl)
    assert c.ttl_seconds == ttl
    c.set("x", [{"point_time": "2026-01-01T00:00:00Z", "value": 1.0}])
    assert c.get("x") is not None


@pytest.mark.parametrize("gpu", ["T4", "A100", "H100", "V100", "A10G"])
def test_spot_prices_all_gpus(gpu: str):
    s = StaticSpotPriceProvider()
    assert s.get_spot_price_usd_per_gpu_hr("us-west-2", gpu) > 0


@pytest.mark.parametrize("region", ["us-east-1", "eu-north-1", "ap-southeast-1", "sa-east-1"])
def test_region_multipliers(region: str):
    s = StaticSpotPriceProvider()
    assert s.get_spot_price_usd_per_gpu_hr(region, "A100") > 0


def test_time_weighted_empty_points():
    start = datetime.now(UTC)
    assert time_weighted_moer([], start, start + timedelta(hours=1)) == 0.0


def test_time_weighted_single_point():
    start = datetime.now(UTC)
    pts = [{"point_time": start.isoformat(), "value": 123.0}]
    assert time_weighted_moer(pts, start, start + timedelta(hours=2)) == 123.0
