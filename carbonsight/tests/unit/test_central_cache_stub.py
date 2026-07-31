"""Central cache, spot providers, and the shared scheduler_service orchestration."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import DEFAULT_CARBON_REGIONS, SyntheticCarbonProvider
from carbonsight_core.providers.spot import Boto3SpotPriceProvider, StaticSpotPriceProvider
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.scheduler_service import (
    build_candidates,
    candidates_to_estimates,
    cold_start_hours,
    mean_lifetime_hr,
    progress_from_job,
    ranked_as_json,
    registry_pairs,
    schedule_job,
)
from carbonsight_core.watttime.cache import ForecastCache, time_weighted_moer
from carbonsight_core.watttime.client import (
    SYNTHETIC_WT_REGIONS,
    WattTimeClient,
    build_synthetic_forecast,
)

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

# The 21 real AWS regions the seed registry maps, including the four A truncated away.
EXPECTED_REGION_COUNT = 21
TRUNCATED_BY_A = {"ap-east-1", "sa-east-1", "me-south-1", "af-south-1"}


def _job(**kwargs) -> JobSpec:
    base = dict(gpu_type="A100", gpu_count=1, duration_hours=1.0, deadline_hours=45.0)
    base.update(kwargs)
    return JobSpec(**base)


# --- cache / client ---------------------------------------------------------


def test_synthetic_regions_len():
    assert len(SYNTHETIC_WT_REGIONS) == 17
    assert len(DEFAULT_CARBON_REGIONS) == 17


def test_client_without_creds_returns_list():
    pts = WattTimeClient.normalize_forecast_payload(
        WattTimeClient(allow_synthetic=True).get_forecast("CAISO_NORTH", horizon_hours=2)
    )
    assert isinstance(pts, list) and len(pts) > 0


def test_build_synthetic_forecast_horizon():
    assert len(build_synthetic_forecast("SE", horizon_hours=1, step_minutes=5)) == 12


@pytest.mark.parametrize("region", SYNTHETIC_WT_REGIONS[:8])
def test_cache_per_region_param(region: str):
    cache = ForecastCache()
    cache.set(region, build_synthetic_forecast(region, horizon_hours=1))
    assert cache.get(region) is not None
    assert cache.get_moer(region, lbar_hours=1.0) is not None


@pytest.mark.parametrize("hours", [1.0, 2.0, 5.0, 12.0, 24.0])
def test_lbar_window_param(hours: float):
    cache = ForecastCache()
    cache.set("R", build_synthetic_forecast("R", horizon_hours=48))
    moer = cache.get_moer("R", lbar_hours=hours)
    assert moer is not None and moer > 0


@pytest.mark.parametrize("ttl", [1, 60, 900, 3600])
def test_cache_ttl_variants(ttl: int):
    cache = ForecastCache(ttl_seconds=ttl)
    assert cache.ttl_seconds == ttl
    cache.set("x", [{"point_time": "2026-01-01T00:00:00Z", "value": 1.0}])
    assert cache.get("x") is not None


def test_time_weighted_empty_points():
    assert time_weighted_moer([], NOW, NOW + timedelta(hours=1)) == 0.0


def test_time_weighted_single_point():
    pts = [{"point_time": NOW.isoformat(), "value": 123.0}]
    assert time_weighted_moer(pts, NOW, NOW + timedelta(hours=2)) == 123.0


# --- spot providers ---------------------------------------------------------


def test_boto3_stub_falls_back():
    provider = Boto3SpotPriceProvider(use_fallback=True)
    assert provider.get_spot_price_usd_per_gpu_hr("us-east-1", "A100") > 0


def test_boto3_stub_raises_without_fallback():
    provider = Boto3SpotPriceProvider(use_fallback=False)
    with pytest.raises(NotImplementedError):
        provider.get_spot_price_usd_per_gpu_hr("us-east-1", "A100")


def test_static_does_not_mutate_pricing_module():
    from carbonsight_core.estimator import pricing

    before = dict(pricing._GPU_BASE_PRICE_USD_PER_HR)
    provider = StaticSpotPriceProvider()
    provider._gpu_base["A100"] = 999.0  # mutate the provider's copy only
    assert pricing._GPU_BASE_PRICE_USD_PER_HR["A100"] == before["A100"]


@pytest.mark.parametrize("gpu", ["T4", "A100", "H100", "V100", "A10G"])
def test_spot_prices_all_gpus(gpu: str):
    assert StaticSpotPriceProvider().get_spot_price_usd_per_gpu_hr("us-west-2", gpu) > 0


@pytest.mark.parametrize("region", ["us-east-1", "eu-north-1", "ap-southeast-1", "sa-east-1"])
def test_region_multipliers(region: str):
    assert StaticSpotPriceProvider().get_spot_price_usd_per_gpu_hr(region, "A100") > 0


# --- registry coverage: no fabrication, no truncation -----------------------


def test_registry_pairs_returns_every_real_region():
    codes = {region for _cloud, region, _wt, _conf in registry_pairs(None)}
    assert len(codes) == EXPECTED_REGION_COUNT
    assert TRUNCATED_BY_A <= codes, "A's [:17] cap dropped four real AWS regions"


def test_no_fabricated_region_codes():
    codes = {region for _cloud, region, _wt, _conf in registry_pairs(None)}
    assert not any(c.startswith("synthetic-") for c in codes)


# --- scheduler service ------------------------------------------------------


def test_progress_from_job_is_hours():
    job = _job(duration_hours=30.0, deadline_hours=45.0, progress_hours_done=5.0)
    progress = progress_from_job(job, elapsed_hours=10.0)
    assert (progress.p, progress.P, progress.t, progress.T) == (5.0, 30.0, 10.0, 45.0)


def test_cold_start_includes_checkpoint_restore():
    bare = cold_start_hours(_job(cold_start_minutes=6.0, checkpoint_size_gb=0.0))
    with_ckpt = cold_start_hours(_job(cold_start_minutes=6.0, checkpoint_size_gb=100.0))
    assert bare == pytest.approx(0.1)
    assert with_ckpt > bare


def test_mean_lifetime_varies_across_regions():
    values = {round(mean_lifetime_hr(r), 4) for r in ("us-east-1", "sa-east-1", "eu-north-1")}
    assert len(values) == 3
    assert all(v > 0 for v in values)


def test_build_candidates_covers_region_times_mode_plus_idle():
    candidates, od_options, inputs, _progress = build_candidates(_job(), now=NOW)
    assert len(inputs) == EXPECTED_REGION_COUNT
    assert len(od_options) == EXPECTED_REGION_COUNT
    assert len(candidates) == EXPECTED_REGION_COUNT * 2 + 1
    assert sum(1 for c in candidates if c.is_idle) == 1
    assert all(math.isinf(c.mean_lifetime_hr) for c in candidates if c.is_od)


def test_current_region_pays_no_migration():
    _c, _od, inputs, _p = build_candidates(
        _job(checkpoint_size_gb=100.0, current_region="us-east-1"), now=NOW
    )
    assert inputs["us-east-1"].migration_cost == 0.0
    assert inputs["eu-west-1"].migration_cost > 0.0


def test_schedule_job_ranks_all_regions_and_modes():
    result = schedule_job(_job(checkpoint_size_gb=100.0), now=NOW)
    assert result.action == "rank"
    assert len(result.ranked) == EXPECTED_REGION_COUNT * 2 + 1
    utilities = [u for _c, u in result.ranked]
    assert utilities == sorted(utilities, reverse=True)


def test_schedule_rows_vary_on_every_lever():
    """The gate A failed: L-bar, eta, carbon and E/L-bar must all differ across rows."""
    job = _job(checkpoint_size_gb=100.0)
    rows = [r for r in ranked_as_json(schedule_job(job, now=NOW), job) if r["mode"] == "spot"]
    for column in (
        "mean_lifetime_hr",
        "effectiveness_eta",
        "carbon_kg_per_hr",
        "amortized_migration_per_hr",
    ):
        assert len({round(r[column], 6) for r in rows}) > 1, f"{column} is constant across rows"


def test_estimates_are_per_region_and_greenest_first():
    estimates = schedule_job(_job(), now=NOW).estimates
    assert len(estimates) == EXPECTED_REGION_COUNT
    assert {e.cloud_region for e in estimates} >= TRUNCATED_BY_A
    co2 = [e.expected_co2_kg_mean for e in estimates]
    assert co2 == sorted(co2)


def test_estimate_cost_is_over_job_duration_not_spot_lifetime():
    """A billed max(Lbar, duration) — 26.8h of spot for a 1h job, ~27x over."""
    job = _job(duration_hours=1.0)
    result = schedule_job(job, now=NOW)
    for estimate in result.estimates:
        region = result.inputs[estimate.cloud_region]
        assert estimate.expected_cost_usd == pytest.approx(
            region.spot_price_per_hr * job.duration_hours
        )
        assert estimate.expected_co2_kg_mean == pytest.approx(
            region.carbon_kg_per_hr * job.duration_hours
        )


def test_estimate_cost_scales_linearly_with_duration():
    one = schedule_job(_job(duration_hours=1.0), now=NOW).estimates
    two = schedule_job(_job(duration_hours=2.0), now=NOW).estimates
    by_region = {e.cloud_region: e for e in one}
    for estimate in two:
        assert estimate.expected_cost_usd == pytest.approx(
            by_region[estimate.cloud_region].expected_cost_usd * 2
        )


def test_thrifty_short_circuits():
    job = _job(duration_hours=10.0, progress_hours_done=10.0, deadline_hours=45.0)
    assert schedule_job(job, now=NOW).action == "thrifty"


def test_safety_net_short_circuits_and_picks_a_real_region():
    job = _job(duration_hours=1.0, deadline_hours=1.0, checkpoint_size_gb=100.0)
    result = schedule_job(job, now=NOW)
    assert result.action == "safety_net"
    assert result.safety_net_region in result.inputs
    assert result.safety_net_total_cost > 0


def test_deadline_pressure_moves_v_and_the_ranking():
    """A's V was pinned at 2*C_od regardless of deadline; this must not regress."""
    job = _job(duration_hours=1.0, deadline_hours=45.0, checkpoint_size_gb=100.0)
    on_track = schedule_job(job, now=NOW)

    behind_job = _job(
        duration_hours=1.0, deadline_hours=45.0, checkpoint_size_gb=100.0, progress_hours_done=0.1
    )
    behind = schedule_job(behind_job, elapsed_hours=20.0, now=NOW)

    assert behind.value_v > on_track.value_v * 2
    order_before = [(c.region, c.mode) for c, _u in on_track.ranked]
    order_after = [(c.region, c.mode) for c, _u in behind.ranked]
    assert order_before != order_after


def test_v_anchors_at_cheapest_od_when_on_track():
    """theta == theta_tilde == P/T at t=0, so V == C_od_min exactly."""
    job = _job(duration_hours=30.0, deadline_hours=45.0)
    result = schedule_job(job, now=NOW)
    cheapest_od_total = min(
        c.price_per_hr + c.carbon_kg_per_hr / 1000.0 * job.carbon_price_usd_per_ton
        for c, _u in result.ranked
        if c.is_od
    )
    assert result.value_v == pytest.approx(cheapest_od_total, rel=1e-9)


def test_top_n_truncates():
    assert len(schedule_job(_job(), now=NOW, top_n=5).ranked) == 5


def test_candidates_to_estimates_handles_empty_ranking():
    job = _job()
    result = schedule_job(job, now=NOW)
    result.ranked = []
    rows = candidates_to_estimates(result, job)
    assert len(rows) == EXPECTED_REGION_COUNT
    assert all(r.utility_score is None for r in rows)


def test_synthetic_carbon_provider_is_injectable():
    candidates, _od, inputs, _p = build_candidates(
        _job(), carbon=SyntheticCarbonProvider(), now=NOW
    )
    assert candidates and all(i.carbon_kg_per_hr > 0 for i in inputs.values())


# --- progress model ---------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "total", "t", "deadline", "theta"),
    [(0, 10, 0, 5, 2.0), (5, 10, 0, 5, 1.0), (9, 10, 0, 2, 0.5)],
)
def test_theta_cases(p, total, t, deadline, theta):
    state = ProgressState(p=p, P=total, t=t, T=deadline)
    assert state.deadline_pressure == pytest.approx(theta)
