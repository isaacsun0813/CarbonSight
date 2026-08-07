"""Carbon provider contract, factory selection, and the no-fabrication rule.

The load-bearing test here is the last class. Synthetic MOER is a hash of the region
name, so it is arbitrary against real grids — Sweden scores dirtier than India. That
is fine for an offline demo and unacceptable as a fallback, because a user cannot
tell a fabricated ranking from a real one. A configured-but-broken source must fail.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from carbonsight_core.carbon import (
    ApiCarbonProvider,
    CarbonIntensityProvider,
    CarbonProviderError,
    SyntheticCarbonProvider,
    WattTimeCarbonProvider,
    facility_mwh_per_hr,
    get_carbon_provider,
)
from carbonsight_core.config import Config
from carbonsight_core.models import JobSpec
from carbonsight_core.watttime import LB_TO_KG, WattTimeError

START = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
END = START + timedelta(hours=4)


@pytest.fixture
def job() -> JobSpec:
    return JobSpec(gpu_type="A100", gpu_count=8, duration_hours=4.0)


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def dead(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("network unreachable")

    monkeypatch.setattr(httpx.Client, "request", dead)
    monkeypatch.setattr(httpx.Client, "get", dead)
    monkeypatch.setattr(httpx.Client, "post", dead)


class TestProtocolConformance:
    def test_all_three_satisfy_the_protocol(self) -> None:
        assert isinstance(SyntheticCarbonProvider(), CarbonIntensityProvider)
        assert isinstance(
            WattTimeCarbonProvider(config=Config(watttime_username="u", watttime_password="p")),
            CarbonIntensityProvider,
        )
        assert isinstance(
            ApiCarbonProvider(base_url="http://localhost:8001"), CarbonIntensityProvider
        )


class TestSyntheticProvider:
    def test_regions_are_not_flat(self) -> None:
        """A flat curve would make the carbon lever invisible in the demo."""
        provider = SyntheticCarbonProvider()
        values = {
            region: provider.get_moer_lb_per_mwh([(region, 1.0)], START, END)
            for region in provider.list_regions()
        }
        assert len(set(round(v, 3) for v in values.values())) > 10
        assert max(values.values()) - min(values.values()) > 300

    def test_same_window_is_reproducible(self) -> None:
        a = SyntheticCarbonProvider().get_moer_lb_per_mwh([("IE", 1.0)], START, END)
        b = SyntheticCarbonProvider().get_moer_lb_per_mwh([("IE", 1.0)], START, END)
        assert a == b

    def test_anchoring_makes_it_calendar_independent(self) -> None:
        far_future = START + timedelta(days=400)
        provider = SyntheticCarbonProvider()
        near = provider.get_moer_lb_per_mwh([("IE", 1.0)], START, END)
        far = provider.get_moer_lb_per_mwh([("IE", 1.0)], far_future, far_future + timedelta(hours=4))
        assert near == pytest.approx(far, rel=1e-9)

    def test_mixture_lands_between_its_legs(self) -> None:
        provider = SyntheticCarbonProvider()
        pure_ie = provider.get_moer_lb_per_mwh([("IE", 1.0)], START, END)
        pure_se = provider.get_moer_lb_per_mwh([("SE", 1.0)], START, END)
        blend = provider.get_moer_lb_per_mwh([("IE", 0.5), ("SE", 0.5)], START, END)
        assert min(pure_ie, pure_se) < blend < max(pure_ie, pure_se)
        assert blend == pytest.approx((pure_ie + pure_se) / 2, rel=1e-6)


class TestJobAwareness:
    def test_kg_per_hr_scales_with_gpu_count(self, job: JobSpec) -> None:
        provider = SyntheticCarbonProvider()
        one = provider.get_kg_per_hr(job.model_copy(update={"gpu_count": 1}), [("IE", 1.0)], START, END)
        eight = provider.get_kg_per_hr(job.model_copy(update={"gpu_count": 8}), [("IE", 1.0)], START, END)
        assert eight > one * 4

    def test_kg_per_hr_is_moer_times_power_times_conversion(self, job: JobSpec) -> None:
        provider = SyntheticCarbonProvider()
        moer = provider.get_moer_lb_per_mwh([("IE", 1.0)], START, END)
        expected = facility_mwh_per_hr(job) * moer * LB_TO_KG
        assert provider.get_kg_per_hr(job, [("IE", 1.0)], START, END) == pytest.approx(expected)

    def test_gpu_utilization_pins_the_draw(self, job: JobSpec) -> None:
        idle = facility_mwh_per_hr(job.model_copy(update={"gpu_utilization": 0.05}))
        busy = facility_mwh_per_hr(job.model_copy(update={"gpu_utilization": 0.95}))
        assert busy > idle

    def test_cost_per_hr_is_kg_dollarised(self, job: JobSpec) -> None:
        provider = SyntheticCarbonProvider()
        kg = provider.get_kg_per_hr(job, [("IE", 1.0)], START, END)
        cost = provider.get_cost_per_hr(job, [("IE", 1.0)], START, END, 50.0, 1.0)
        assert cost == pytest.approx(kg / 1000.0 * 50.0)

    def test_carbon_weight_scales_linearly(self, job: JobSpec) -> None:
        provider = SyntheticCarbonProvider()
        single = provider.get_cost_per_hr(job, [("IE", 1.0)], START, END, 50.0, 1.0)
        double = provider.get_cost_per_hr(job, [("IE", 1.0)], START, END, 50.0, 2.0)
        assert double == pytest.approx(single * 2)

    def test_power_averaging_beats_a_single_draw(self, job: JobSpec) -> None:
        """64 draws, fixed seed: reproducible, and not an outlier of the distribution."""
        averaged = facility_mwh_per_hr(job)
        one_shot = facility_mwh_per_hr(job, samples=1)
        assert averaged == facility_mwh_per_hr(job)
        assert abs(averaged - one_shot) / averaged < 0.20


class TestFactorySelection:
    def test_api_url_wins(self) -> None:
        cfg = Config(
            carbonsight_api_url="http://localhost:8001",
            watttime_username="u",
            watttime_password="p",
        )
        assert isinstance(get_carbon_provider(cfg), ApiCarbonProvider)

    def test_credentials_are_second(self) -> None:
        cfg = Config(watttime_username="u", watttime_password="p")
        assert isinstance(get_carbon_provider(cfg), WattTimeCarbonProvider)

    def test_nothing_configured_falls_to_synthetic(self) -> None:
        assert isinstance(get_carbon_provider(Config()), SyntheticCarbonProvider)

    def test_half_a_credential_is_not_a_credential(self) -> None:
        assert isinstance(
            get_carbon_provider(Config(watttime_username="u")), SyntheticCarbonProvider
        )


class TestNeverFabricateARanking:
    """A configured source that fails must raise, not invent numbers."""

    def test_synthetic_values_really_are_physically_arbitrary(self) -> None:
        """The reason fabrication is unacceptable, pinned as a fact."""
        provider = SyntheticCarbonProvider()
        sweden = provider.get_moer_lb_per_mwh([("SE", 1.0)], START, END)
        india = provider.get_moer_lb_per_mwh([("IND", 1.0)], START, END)
        assert sweden > india, (
            "synthetic curves are hashes, not physics: Sweden scores dirtier than "
            "India. Fine for a demo, never acceptable as a fallback."
        )

    def test_watttime_provider_raises_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = WattTimeCarbonProvider(
            config=Config(watttime_username="u", watttime_password="p")
        )
        _no_network(monkeypatch)
        with pytest.raises((httpx.HTTPError, WattTimeError)):
            provider.get_moer_lb_per_mwh([("SE", 1.0)], START, END)

    def test_api_provider_raises_when_server_is_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = ApiCarbonProvider(base_url="http://localhost:8001")
        _no_network(monkeypatch)
        with pytest.raises(CarbonProviderError, match="unreachable"):
            provider.get_forecast("SE")

    def test_api_provider_raises_on_an_empty_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def empty(*_a: object, **_k: object) -> httpx.Response:
            return httpx.Response(200, json={"data": []}, request=httpx.Request("GET", "http://x"))

        monkeypatch.setattr(httpx.Client, "get", empty)
        with pytest.raises(CarbonProviderError, match="no forecast points"):
            ApiCarbonProvider(base_url="http://localhost:8001").get_forecast("SE")

    def test_api_provider_needs_a_url(self) -> None:
        with pytest.raises(ValueError, match="base URL"):
            ApiCarbonProvider(base_url="", config=Config())

    def test_api_provider_records_server_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        points = [{"point_time": START.isoformat().replace("+00:00", "Z"), "value": 123.0}]

        def ok(*_a: object, **_k: object) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": points, "source": "db"},
                request=httpx.Request("GET", "http://x"),
            )

        monkeypatch.setattr(httpx.Client, "get", ok)
        provider = ApiCarbonProvider(base_url="http://localhost:8001")
        provider.get_forecast("SE")
        assert provider.last_source("SE") == "db"
