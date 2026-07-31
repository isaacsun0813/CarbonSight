"""Carbon provider factory, mixture/job-aware surface, and the green-vs-dirty lever."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.config import Config
from carbonsight_core.models import JobSpec
from carbonsight_core.providers.carbon import (
    DEFAULT_CARBON_REGIONS,
    ApiCarbonProvider,
    CarbonIntensityProvider,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    facility_mwh_per_hr,
    get_carbon_provider,
)
from carbonsight_core.providers.spot import StaticSpotPriceProvider
from carbonsight_core.spot.unified_model import CandidateState, rank_candidates
from carbonsight_core.watttime.client import synthetic_moer_for_region

START = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
END = START + timedelta(hours=3)


def _job(gpu_type: str = "A100", gpu_count: int = 1, **kwargs) -> JobSpec:
    return JobSpec(gpu_type=gpu_type, gpu_count=gpu_count, duration_hours=1.0, **kwargs)


# --- factory ----------------------------------------------------------------


def test_default_regions_count_is_17():
    assert len(DEFAULT_CARBON_REGIONS) == 17


def test_factory_prefers_api_url(monkeypatch):
    monkeypatch.setenv("CARBONSIGHT_API_URL", "http://localhost:8001")
    monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
    monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
    assert isinstance(get_carbon_provider(Config.from_env()), ApiCarbonProvider)


def test_factory_prefers_watttime_when_no_api(monkeypatch):
    monkeypatch.delenv("CARBONSIGHT_API_URL", raising=False)
    monkeypatch.setenv("WATTTIME_USERNAME", "user")
    monkeypatch.setenv("WATTTIME_PASSWORD", "pass")
    assert isinstance(get_carbon_provider(Config.from_env()), WattTimeCarbonProvider)


def test_factory_synthetic_fallback(monkeypatch):
    monkeypatch.delenv("CARBONSIGHT_API_URL", raising=False)
    monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
    monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
    assert isinstance(get_carbon_provider(Config.from_env()), SyntheticCarbonProvider)


@pytest.mark.parametrize(
    "provider",
    [SyntheticCarbonProvider(), ApiCarbonProvider(base_url="http://127.0.0.1:1")],
)
def test_providers_satisfy_one_protocol(provider):
    """Every provider answers the same mixture- and job-aware surface."""
    assert isinstance(provider, CarbonIntensityProvider)


def test_api_provider_falls_back_without_server():
    p = ApiCarbonProvider(base_url="http://127.0.0.1:1")  # nothing listening
    assert len(p.get_forecast("CAISO_NORTH")) >= 1
    assert len(p.list_regions()) == 17
    assert p.get_moer_lb_per_mwh([("CAISO_NORTH", 1.0)], START, END) > 0


# --- synthetic MOER must vary per region ------------------------------------


def test_synthetic_provider_forecast_and_moer():
    p = SyntheticCarbonProvider()
    pts = p.get_forecast("CAISO_NORTH")
    assert len(pts) >= 1 and "value" in pts[0]
    assert p.get_moer("CAISO_NORTH", lbar_hours=3.0) > 0


def test_synthetic_moer_is_not_flat_across_regions():
    """A flat 400 lb/MWh everywhere would make the carbon lever invisible.

    The curves are hash-derived over a 850 lb/MWh span, so a couple of the 17
    regions can collide; what matters is the spread, not perfect uniqueness.
    """
    p = SyntheticCarbonProvider()
    values = {
        round(p.get_moer_lb_per_mwh([(r, 1.0)], START, END), 3) for r in DEFAULT_CARBON_REGIONS
    }
    assert len(values) >= len(DEFAULT_CARBON_REGIONS) - 2
    assert max(values) - min(values) > 300.0


def test_mixture_weighting_blends_two_grids():
    p = SyntheticCarbonProvider()
    a = p.get_moer_lb_per_mwh([("SE", 1.0)], START, END)
    b = p.get_moer_lb_per_mwh([("IND", 1.0)], START, END)
    mixed = p.get_moer_lb_per_mwh([("SE", 0.5), ("IND", 0.5)], START, END)
    assert mixed == pytest.approx((a + b) / 2, rel=1e-6)


def test_empty_mixture_falls_back():
    assert SyntheticCarbonProvider().get_moer_lb_per_mwh([], START, END) == 400.0


# --- job-aware power model --------------------------------------------------


def test_kg_per_hr_scales_with_gpu_count():
    p = SyntheticCarbonProvider()
    one = p.get_kg_per_hr(_job(gpu_count=1), [("SE", 1.0)], START, END)
    four = p.get_kg_per_hr(_job(gpu_count=4), [("SE", 1.0)], START, END)
    assert four > one


def test_h100_emits_more_than_a100():
    """The power model must see GPU type; a fixed kW constant would not."""
    p = SyntheticCarbonProvider()
    a100 = p.get_kg_per_hr(_job("A100"), [("SE", 1.0)], START, END)
    h100 = p.get_kg_per_hr(_job("H100"), [("SE", 1.0)], START, END)
    assert h100 > a100


def test_gpu_utilization_moves_carbon():
    p = SyntheticCarbonProvider()
    low = p.get_kg_per_hr(_job(gpu_utilization=0.1), [("SE", 1.0)], START, END)
    high = p.get_kg_per_hr(_job(gpu_utilization=1.0), [("SE", 1.0)], START, END)
    assert high > low


def test_facility_mwh_is_deterministic():
    job = _job()
    assert facility_mwh_per_hr(job) == facility_mwh_per_hr(job)


def test_cost_per_hr_tracks_carbon_price():
    p = SyntheticCarbonProvider()
    args = (_job(), [("SE", 1.0)], START, END)
    cheap = p.get_cost_per_hr(*args, 10.0)
    dear = p.get_cost_per_hr(*args, 1000.0)
    assert dear == pytest.approx(cheap * 100, rel=1e-9)


def test_zero_carbon_price_zeroes_the_lever():
    assert SyntheticCarbonProvider().get_cost_per_hr(_job(), [("SE", 1.0)], START, END, 0.0) == 0.0


# --- carbon as a ranking lever ----------------------------------------------


def _candidate(region: str, carbon_kg: float, price: float = 1.0) -> CandidateState:
    return CandidateState(region, "spot", 10.0, price, carbon_kg)


def test_high_carbon_price_flips_ranking_toward_green():
    green = _candidate("green", carbon_kg=0.05, price=1.10)
    dirty = _candidate("dirty", carbon_kg=0.90, price=1.00)

    cheap_first = rank_candidates(
        [green, dirty], v=5.0, cold_start_hr=0.1, carbon_price_usd_per_ton=0.0
    )
    assert cheap_first[0][0].region == "dirty"

    green_first = rank_candidates(
        [green, dirty], v=5.0, cold_start_hr=0.1, carbon_price_usd_per_ton=20_000.0
    )
    assert green_first[0][0].region == "green"


def test_synthetic_moer_is_deterministic_per_region():
    assert synthetic_moer_for_region("CAISO_NORTH") == synthetic_moer_for_region("CAISO_NORTH")


def test_static_spot_price_positive():
    s = StaticSpotPriceProvider()
    spot = s.get_spot_price_usd_per_gpu_hr("us-east-1", "A100")
    assert spot > 0
    assert s.get_on_demand_usd_per_gpu_hr("us-east-1", "A100") > spot


def test_job_spec_extensions():
    j = JobSpec(
        gpu_type="A100",
        gpu_count=1,
        duration_hours=1.0,
        deadline_hours=45.0,
        checkpoint_size_gb=100.0,
        cold_start_minutes=5.0,
        carbon_price_usd_per_ton=50.0,
        carbon_weight=2.0,
        progress_hours_done=0.5,
        current_region="us-east-1",
    )
    assert j.deadline_hours == 45.0
    assert j.checkpoint_size_gb == 100.0
    assert j.carbon_weight == 2.0
    assert j.progress_hours_done == 0.5
    assert j.current_region == "us-east-1"
