"""Carbon provider factory + green-vs-dirty lever tests."""

from __future__ import annotations

import os

import pytest

from carbonsight_core.config import Config
from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import (
    DEFAULT_CARBON_REGIONS,
    ApiCarbonProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    get_carbon_provider,
)
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import ODCandidate, rank_candidates
from carbonsight_core.watttime.client import synthetic_moer_for_region


def test_default_regions_count_is_17():
    assert len(DEFAULT_CARBON_REGIONS) == 17


def test_synthetic_provider_forecast_and_moer():
    p = SyntheticCarbonProvider()
    pts = p.get_forecast("CAISO_NORTH")
    assert len(pts) >= 1
    assert "value" in pts[0]
    m = p.get_moer("CAISO_NORTH", lbar_hours=3.0)
    assert m > 0


def test_factory_prefers_api_url(monkeypatch):
    monkeypatch.setenv("CARBONSIGHT_API_URL", "http://localhost:8001")
    monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
    monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
    p = get_carbon_provider(Config.from_env())
    assert isinstance(p, ApiCarbonProvider)


def test_factory_prefers_watttime_when_no_api(monkeypatch):
    monkeypatch.delenv("CARBONSIGHT_API_URL", raising=False)
    monkeypatch.setenv("WATTTIME_USERNAME", "user")
    monkeypatch.setenv("WATTTIME_PASSWORD", "pass")
    p = get_carbon_provider(Config.from_env())
    assert isinstance(p, WattTimeCarbonProvider)


def test_factory_synthetic_fallback(monkeypatch):
    monkeypatch.delenv("CARBONSIGHT_API_URL", raising=False)
    monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
    monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
    p = get_carbon_provider(Config.from_env())
    assert isinstance(p, SyntheticCarbonProvider)


def test_api_provider_falls_back_without_server():
    p = ApiCarbonProvider(base_url="http://127.0.0.1:1")  # nothing listening
    pts = p.get_forecast("CAISO_NORTH")
    assert len(pts) >= 1
    assert len(p.list_regions()) == 17


def test_watttime_provider_green_vs_dirty_lever():
    """High carbon price must flip ranking toward greener (lower MOER) region."""
    green = "GREEN_GRID_AAA"
    dirty = "DIRTY_GRID_ZZZ"
    # Force known spread via synthetic hash — pick regions with clear MOER gap
    # Use SyntheticCarbonProvider point_moer; if too close, override MOER directly
    g_moer = synthetic_moer_for_region(green)
    d_moer = synthetic_moer_for_region(dirty)
    if abs(g_moer - d_moer) < 50:
        g_moer, d_moer = 100.0, 800.0
    if g_moer > d_moer:
        g_moer, d_moer = d_moer, g_moer
        green, dirty = dirty, green

    progress = ProgressState.from_deadline_hours(deadline_hours=45.0)
    # Identical economics except MOER
    base = dict(
        instance_type="A100",
        spot_price=1.0,
        survival=0.9,
        lbar=10.0,
        on_demand_price=3.0,
        gpu_count=1,
        efficiency=1.0,
    )
    green_c = ODCandidate(region=green, moer=g_moer, **base)
    dirty_c = ODCandidate(region=dirty, moer=d_moer, **base)

    # Low carbon price: costs dominate — either order ok; just compute
    low = rank_candidates(
        [green_c, dirty_c], progress, carbon_price_usd_per_ton=1.0
    )
    assert len(low) == 2

    # High carbon price: green must win
    high = rank_candidates(
        [
            ODCandidate(region=green, moer=g_moer, **base),
            ODCandidate(region=dirty, moer=d_moer, **base),
        ],
        progress,
        carbon_price_usd_per_ton=20_000.0,
    )
    assert high[0].moer < high[1].moer
    assert high[0].region == green


def test_static_spot_price_positive():
    s = StaticSpotPriceProvider()
    px = s.get_spot_price_usd_per_gpu_hr("us-east-1", "A100")
    assert px > 0
    od = s.get_on_demand_usd_per_gpu_hr("us-east-1", "A100")
    assert od > px


def test_job_spec_extensions():
    j = JobSpec(
        gpu_type="A100",
        gpu_count=1,
        duration_hours=1.0,
        deadline_hours=45.0,
        checkpoint_size_gb=100.0,
        cold_start_minutes=5.0,
        carbon_price_usd_per_ton=50.0,
    )
    assert j.deadline_hours == 45.0
    assert j.checkpoint_size_gb == 100.0
