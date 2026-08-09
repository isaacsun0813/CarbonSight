"""When the carbon data source is down, CarbonSight says so. It does not guess.

The estimator used to substitute a flat 400 lb/MWh for any region whose forecast
failed. That produced a complete-looking ranking out of nothing: every number was
invented, nothing in the output said so, and the caller could not tell it from a
real answer. For a tool whose entire job is reporting carbon, that is the worst
possible failure mode -- worse than crashing, because it is silent.

The rule these tests hold: a *configured* source that cannot answer is fatal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from carbonsight_core.carbon import (
    CarbonProviderError,
    ForecastBackedProvider,
    SyntheticCarbonProvider,
)
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import (
    CarbonDataUnavailableError,
    JobCarbonEstimator,
)
from carbonsight_core.mapping.registry import CloudRegionEntry, Registry
from carbonsight_core.models import JobSpec
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.watttime import WattTimeClient, WattTimeError

JOB = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
CREDS = Config(watttime_username="user", watttime_password="secret")


class DeadWattTime:
    def get_forecast(self, *_a: object, **_k: object) -> Any:
        raise httpx.ConnectError("connection refused")


class EmptyWattTime:
    def get_forecast(self, *_a: object, **_k: object) -> Any:
        return {"data": [], "units": "lbs_co2_per_mwh"}


class DeadProvider:
    def get_moer_lb_per_mwh(self, *_a: object, **_k: object) -> float:
        raise httpx.ConnectError("connection refused")


def _registry() -> Registry:
    registry = Registry()
    for code, grid in [("eu-west-1", "IE"), ("eu-north-1", "SE")]:
        entry = CloudRegionEntry(
            provider="aws", region_code=code, display_name=code, wt_regions=[(grid, 1.0)]
        )
        registry._regions[("aws", code)] = entry  # noqa: SLF001
    return registry


class TestTheEstimatorRefusesToGuess:
    def test_a_dead_source_raises_instead_of_returning_400(self) -> None:
        estimator = JobCarbonEstimator(DeadWattTime())  # type: ignore[arg-type]
        with pytest.raises(CarbonDataUnavailableError):
            estimator.estimate_region(JOB, "aws", "eu-west-1", [("IE", 1.0)], 0.9)

    def test_an_empty_forecast_raises_instead_of_returning_400(self) -> None:
        estimator = JobCarbonEstimator(EmptyWattTime())  # type: ignore[arg-type]
        with pytest.raises(CarbonDataUnavailableError, match="no forecast points"):
            estimator.estimate_region(JOB, "aws", "eu-west-1", [("IE", 1.0)], 0.9)

    def test_a_dead_provider_raises_too(self) -> None:
        """The provider path must fail the same way as the direct WattTime path."""
        estimator = JobCarbonEstimator(
            DeadWattTime(),  # type: ignore[arg-type]
            carbon_provider=DeadProvider(),  # type: ignore[arg-type]
        )
        with pytest.raises(CarbonDataUnavailableError):
            estimator.estimate_region(JOB, "aws", "eu-west-1", [("IE", 1.0)], 0.9)

    def test_no_estimate_ever_comes_back_as_the_old_400_default(self) -> None:
        """Belt and braces: the specific number that used to be invented."""
        estimator = JobCarbonEstimator(DeadWattTime())  # type: ignore[arg-type]
        try:
            result = estimator.estimate_region(JOB, "aws", "eu-west-1", [("IE", 1.0)], 0.9)
        except CarbonDataUnavailableError:
            return
        pytest.fail(f"expected a refusal, got a fabricated estimate: {result.expected_co2_kg_mean} kg")

    def test_the_unit_gate_is_still_fail_closed(self) -> None:
        """Wrong units were already fatal; that must not have regressed."""

        class WrongUnits:
            def get_forecast(self, *_a: object, **_k: object) -> Any:
                raise WattTimeError("Unsupported unit 'g_co2_per_kwh'; refusing to compute")

        estimator = JobCarbonEstimator(WrongUnits())  # type: ignore[arg-type]
        with pytest.raises(WattTimeError):
            estimator.estimate_region(JOB, "aws", "eu-west-1", [("IE", 1.0)], 0.9)


class TestTheRankingFailsWholeNotPerRegion:
    def test_a_dead_source_aborts_the_ranking(self) -> None:
        """Every region fails identically, so 21 warnings and an empty table is
        a worse answer than one sentence saying the service is down."""
        ranking = AwsRegionRankingService(
            _registry(),
            WattTimeClient(CREDS, allow_synthetic=False),
            carbon_provider=DeadProvider(),
        )
        with pytest.raises(CarbonDataUnavailableError):
            ranking.collect_estimates(JOB)

    def test_it_does_not_degrade_to_a_truncated_ranking(self) -> None:
        """A partial list would look like a real answer with fewer options."""
        warnings: list[str] = []
        ranking = AwsRegionRankingService(
            _registry(),
            WattTimeClient(CREDS, allow_synthetic=False),
            carbon_provider=DeadProvider(),
        )
        with pytest.raises(CarbonDataUnavailableError):
            ranking.collect_estimates(JOB, on_estimate_error=lambda r, e: warnings.append(r))
        assert warnings == [], "a dead source must not be reported as per-region warnings"


class TestTheUserSeesAnHonestMessage:
    def test_cli_exits_nonzero_with_one_readable_sentence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from carbonsight_cli.main import app
        from typer.testing import CliRunner

        monkeypatch.setenv("WATTTIME_USERNAME", "real")
        monkeypatch.setenv("WATTTIME_PASSWORD", "real")
        monkeypatch.delenv("CARBONSIGHT_API_URL", raising=False)

        def dead(*_a: object, **_k: object) -> None:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx.Client, "request", dead)
        monkeypatch.setattr(httpx.Client, "post", dead)

        fixture = "tests/fixtures/train_minimal.yaml"
        result = CliRunner().invoke(app, ["advise", "--yaml", fixture])

        assert result.exit_code == 1, "a dead carbon source must not exit 0"
        combined = (result.output or "") + (result.stderr or "")
        assert "carbon data source is unavailable" in combined
        # and it must not blame the user's (valid) credentials
        assert "check WattTime credentials" not in combined


class TestSyntheticIsStillAvailableForTheDemo:
    def test_no_credentials_at_all_still_gets_synthetic(self) -> None:
        """Removing the fallback must not break the offline demo path."""
        payload = WattTimeClient(Config(), allow_synthetic=True).get_forecast("SE")
        assert payload["data"], "the no-credentials demo should still produce a curve"
        assert payload.get("meta", {}).get("source") == "synthetic"

    def test_and_that_curve_is_labelled_as_synthetic(self) -> None:
        payload = WattTimeClient(Config(), allow_synthetic=True).get_forecast("IE")
        assert payload["meta"]["source"] == "synthetic"


class TestTheMixtureMathRefusesToGuess:
    """`ForecastBackedProvider.get_moer_lb_per_mwh` had its own copy of the 400."""

    def test_a_mixture_with_no_usable_series_raises(self) -> None:
        class NoPoints(ForecastBackedProvider):
            def get_forecast(self, region: str, **_k: object) -> list[dict[str, Any]]:
                return []

        with pytest.raises(CarbonProviderError, match="No forecast points"):
            NoPoints().get_moer_lb_per_mwh(
                [("IE", 0.5), ("SE", 0.5)],
                datetime(2026, 3, 1, tzinfo=UTC),
                datetime(2026, 3, 1, 1, tzinfo=UTC),
            )

    def test_a_zero_weight_mixture_is_an_error_not_zero_emissions(self) -> None:
        """mixture_weighted_moer would blend these to a flat 0 lb/MWh -- cleanest on earth."""

        class Real(ForecastBackedProvider):
            def get_forecast(self, region: str, **_k: object) -> list[dict[str, Any]]:
                return [{"point_time": "2026-03-01T00:00:00Z", "value": 500.0}]

        with pytest.raises(ValueError, match="no positive weights"):
            Real().get_moer_lb_per_mwh(
                [("IE", 0.0), ("SE", 0.0)],
                datetime(2026, 3, 1, tzinfo=UTC),
                datetime(2026, 3, 1, 1, tzinfo=UTC),
            )

    def test_a_bad_mixture_only_costs_that_region_not_the_whole_ranking(self) -> None:
        """A registry bug is per-region. Only a dead *source* is allowed to be fatal."""
        registry = Registry()
        registry._regions[("aws", "eu-west-1")] = CloudRegionEntry(  # noqa: SLF001
            provider="aws", region_code="eu-west-1", display_name="eu-west-1",
            wt_regions=[("IE", 0.0)],
        )
        registry._regions[("aws", "eu-north-1")] = CloudRegionEntry(  # noqa: SLF001
            provider="aws", region_code="eu-north-1", display_name="eu-north-1",
            wt_regions=[("SE", 1.0)],
        )
        warned: list[str] = []
        ranking = AwsRegionRankingService(
            registry,
            WattTimeClient(CREDS, allow_synthetic=False),
            carbon_provider=SyntheticCarbonProvider(),
        )
        estimates = ranking.collect_estimates(
            JOB, on_estimate_error=lambda r, e: warned.append(r)
        )
        assert warned == ["eu-west-1"]
        assert [e.cloud_region for e in estimates] == ["eu-north-1"]

    def test_a_real_reading_still_comes_through_untouched(self) -> None:
        """The refusals must not have swallowed the happy path."""
        start = datetime(2026, 3, 1, tzinfo=UTC)

        class Real(ForecastBackedProvider):
            def get_forecast(self, region: str, **_k: object) -> list[dict[str, Any]]:
                return [
                    {"point_time": "2026-03-01T00:00:00Z", "value": 300.0},
                    {"point_time": "2026-03-01T00:30:00Z", "value": 300.0},
                    {"point_time": "2026-03-01T01:00:00Z", "value": 300.0},
                ]

        got = Real().get_moer_lb_per_mwh([("IE", 1.0)], start, start + timedelta(hours=1))
        assert got == pytest.approx(300.0)

    def test_there_is_only_one_carbon_provider_error_class(self) -> None:
        """It moved to base.py so base and providers can both raise the same one."""
        from carbonsight_core.carbon import providers

        assert providers.CarbonProviderError is CarbonProviderError


class TestActualRunAlsoRefusesToGuess:
    def test_historical_gaps_do_not_become_invented_emissions(self) -> None:
        """compute_actual_run reports what a finished job DID emit."""
        estimator = JobCarbonEstimator(DeadWattTime())  # type: ignore[arg-type]
        start = datetime(2026, 3, 1, tzinfo=UTC)
        with pytest.raises(Exception) as exc:
            estimator.compute_actual_run(
                JOB, "aws", "eu-west-1", [("IE", 1.0)], start, start + timedelta(hours=1), 1.0
            )
        assert not isinstance(exc.value, AssertionError)
