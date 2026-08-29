"""
Single orchestration path for ranking AWS regions by estimated carbon.

CLI and API use :class:`AwsRegionRankingService` so region loops, enabled-region,
quota, availability, and confidence rules stay in sync.
Sequential WattTime calls: the client token cache is not thread-safe for parallel forecasts.
"""

from collections.abc import Callable

from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.cloud.aws.quota import QuotaChecker
from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
from carbonsight_core.mapping.registry import Registry, mapping_confidence
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.watttime import WattTimeClient, WattTimeError

_ENABLED_REGION_SKIP_REASON = "region not enabled for this AWS account"


class AwsRegionRankingService:
    """Load registry + WattTime once; enumerate AWS regions and build :class:`EstimateResult` rows."""

    __slots__ = (
        "_registry",
        "_quota_checker",
        "_availability_checker",
        "_enabled_regions_provider",
        "_estimator",
    )

    def __init__(
        self,
        registry: Registry,
        watt_time: WattTimeClient,
        *,
        quota_checker: QuotaChecker | None = None,
        availability_checker: InstanceAvailabilityChecker | None = None,
        enabled_regions_provider: EnabledRegionsProvider | None = None,
    ) -> None:
        self._registry = registry
        self._quota_checker = quota_checker
        self._availability_checker = availability_checker
        self._enabled_regions_provider = enabled_regions_provider
        self._estimator = JobCarbonEstimator(watt_time)

    def collect_estimates(
        self,
        job: JobSpec,
        *,
        use_spot: bool = False,
        on_enabled_region_skip: Callable[[str, str], None] | None = None,
        on_quota_skip: Callable[[str, str], None] | None = None,
        on_availability_skip: Callable[[str, str], None] | None = None,
        on_estimate_error: Callable[[str, BaseException], None] | None = None,
    ) -> list[EstimateResult]:
        """
        One pass over registry: optional enabled-region, quota, availability filters, then estimate.

        Callbacks are optional I/O hooks (e.g. typer.echo); core stays free of CLI dependencies.
        """
        enabled_regions: frozenset[str] | None = None
        if self._enabled_regions_provider is not None:
            enabled_regions = self._enabled_regions_provider.enabled_region_codes()

        estimates: list[EstimateResult] = []
        for entry in self._registry.all_regions():
            if not entry.wt_regions or entry.provider.lower() != "aws":
                continue
            if enabled_regions is not None and entry.region_code not in enabled_regions:
                if on_enabled_region_skip is not None:
                    on_enabled_region_skip(entry.region_code, _ENABLED_REGION_SKIP_REASON)
                continue
            if self._quota_checker is not None:
                quota_result = self._quota_checker.check_gpu_quota(
                    entry.region_code, job.gpu_count, gpu_type=job.gpu_type
                )
                if not quota_result.allowed:
                    if on_quota_skip is not None:
                        on_quota_skip(entry.region_code, quota_result.reason)
                    continue
            if self._availability_checker is not None:
                avail = self._availability_checker.check_instance_offering(
                    entry.region_code, job.gpu_type,
                )
                if not avail.available:
                    if on_availability_skip is not None:
                        on_availability_skip(entry.region_code, avail.reason)
                    continue
            conf = mapping_confidence(
                entry.s_source, entry.s_geo, entry.s_wt_stability, entry.s_recency
            )
            try:
                estimates.append(
                    self._estimator.estimate_region(
                        job,
                        entry.provider,
                        entry.region_code,
                        entry.wt_regions,
                        conf,
                        use_spot=use_spot,
                    )
                )
            except WattTimeError as err:
                if on_estimate_error is not None:
                    on_estimate_error(entry.region_code, err)
            except Exception as err:
                if on_estimate_error is not None:
                    on_estimate_error(entry.region_code, err)
        return estimates

    @staticmethod
    def rank_greenest_first_then_cost_ceiling(
        estimates: list[EstimateResult],
        max_cost_premium: float,
    ) -> tuple[list[EstimateResult], float, float]:
        """
        Sort by lowest mean CO₂; keep regions with cost <= (1 + premium) * cheapest.

        If none qualify, return the single greenest row so output is never empty when `estimates` is non-empty.

        Returns ``(visible_rows, min_cost_usd, cost_ceiling_usd)``.
        """
        if not estimates:
            return [], 0.0, 0.0
        sorted_by_co2 = sorted(estimates, key=lambda r: r.expected_co2_kg_mean)
        min_cost = min(r.expected_cost_usd for r in sorted_by_co2)
        cost_ceiling = min_cost * (1 + max_cost_premium)
        affordable = [r for r in sorted_by_co2 if r.expected_cost_usd <= cost_ceiling]
        visible = affordable if affordable else sorted_by_co2[:1]
        return visible, min_cost, cost_ceiling

    @staticmethod
    def pick_best_region_for_launch(
        estimates: list[EstimateResult],
        max_cost_premium: float,
    ) -> EstimateResult | None:
        """Greenest region within cost premium; None if ``estimates`` is empty."""
        visible, _, _ = AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(
            estimates, max_cost_premium
        )
        return visible[0] if visible else None


def collect_aws_region_estimates(
    job: JobSpec,
    registry: Registry,
    watt_time: WattTimeClient,
    *,
    quota_checker: QuotaChecker | None = None,
    availability_checker: InstanceAvailabilityChecker | None = None,
    enabled_regions_provider: EnabledRegionsProvider | None = None,
    on_enabled_region_skip: Callable[[str, str], None] | None = None,
    on_quota_skip: Callable[[str, str], None] | None = None,
    on_availability_skip: Callable[[str, str], None] | None = None,
    on_estimate_error: Callable[[str, BaseException], None] | None = None,
) -> list[EstimateResult]:
    """Backward-compatible wrapper around :meth:`AwsRegionRankingService.collect_estimates`."""
    return AwsRegionRankingService(
        registry,
        watt_time,
        quota_checker=quota_checker,
        availability_checker=availability_checker,
        enabled_regions_provider=enabled_regions_provider,
    ).collect_estimates(
        job,
        on_enabled_region_skip=on_enabled_region_skip,
        on_quota_skip=on_quota_skip,
        on_availability_skip=on_availability_skip,
        on_estimate_error=on_estimate_error,
    )


def rank_greenest_first_then_cost_ceiling(
    estimates: list[EstimateResult],
    max_cost_premium: float,
) -> tuple[list[EstimateResult], float, float]:
    """Delegate to :meth:`AwsRegionRankingService.rank_greenest_first_then_cost_ceiling`."""
    return AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(estimates, max_cost_premium)


def pick_best_region_for_launch(
    estimates: list[EstimateResult],
    max_cost_premium: float,
) -> EstimateResult | None:
    """Delegate to :meth:`AwsRegionRankingService.pick_best_region_for_launch`."""
    return AwsRegionRankingService.pick_best_region_for_launch(estimates, max_cost_premium)
