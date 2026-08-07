"""
AWS on-demand GPU instance pricing (per GPU per hour, USD).

Static tables are the default. When live AWS pricing is enabled:
- on-demand: Pricing API get_products
- spot: EC2 describe_spot_price_history

Sources (static):
  - https://aws.amazon.com/ec2/pricing/on-demand/ (sampled Feb 2026)
  - Base prices are for us-east-1; regional multipliers scale them.
"""

from dataclasses import dataclass, field

from carbonsight_core.cloud.aws.ondemand_pricing import OnDemandPriceProvider
from carbonsight_core.cloud.aws.spot_pricing import SpotPriceProvider
from carbonsight_core.config import Config

# Base price per GPU per hour (us-east-1 on-demand)
# Instance reference:
#   A100  → p4d.24xlarge  ($32.77/hr ÷ 8 GPUs)
#   H100  → p5.48xlarge   ($98.32/hr ÷ 8 GPUs)
#   V100  → p3.16xlarge   ($24.48/hr ÷ 8 GPUs)
#   T4    → g4dn.xlarge   ($0.526/hr ÷ 1 GPU)
#   A10G  → g5.xlarge     ($1.006/hr ÷ 1 GPU)
_GPU_BASE_PRICE_USD_PER_HR: dict[str, float] = {
    "A100": 4.10,
    "H100": 12.29,
    "V100": 3.06,
    "T4":   0.53,
    "A10G": 1.01,
}
_DEFAULT_GPU_PRICE = 4.10  # fallback if gpu_type not in table

# Regional price multiplier relative to us-east-1.
_REGION_MULTIPLIER: dict[str, float] = {
    "us-east-1":      1.00,
    "us-east-2":      1.00,
    "us-west-1":      1.10,
    "us-west-2":      1.00,
    "ca-central-1":   1.10,
    "eu-west-1":      1.15,
    "eu-west-2":      1.20,
    "eu-west-3":      1.20,
    "eu-central-1":   1.15,
    "eu-north-1":     1.15,
    "eu-south-1":     1.20,
    "ap-northeast-1": 1.20,
    "ap-northeast-2": 1.15,
    "ap-northeast-3": 1.20,
    "ap-southeast-1": 1.20,
    "ap-southeast-2": 1.20,
    "ap-south-1":     1.10,
    "ap-east-1":      1.30,
    "sa-east-1":      1.40,
    "me-south-1":     1.30,
    "af-south-1":     1.30,
}
_DEFAULT_MULTIPLIER = 1.20  # conservative fallback for unknown regions

SPOT_PRICE_FRACTION = 0.35

_live_aws_pricing: bool = False
_spot_provider: SpotPriceProvider | None = None
_ondemand_provider: OnDemandPriceProvider | None = None


@dataclass
class CostEstimate:
    """Job cost in USD with optional provenance notes."""

    usd: float
    notes: list[str] = field(default_factory=list)


def configure_pricing(
    config: Config | None = None,
    *,
    live_aws_pricing: bool | None = None,
) -> None:
    """Set module-level live pricing flag and AWS providers (call from CLI before ranking)."""
    global _live_aws_pricing, _spot_provider, _ondemand_provider
    cfg = config or Config.from_env()
    if live_aws_pricing is not None:
        _live_aws_pricing = live_aws_pricing
    else:
        _live_aws_pricing = cfg.live_aws_pricing
    if _live_aws_pricing:
        _spot_provider = SpotPriceProvider(cfg)
        _ondemand_provider = OnDemandPriceProvider(cfg)
    else:
        _spot_provider = None
        _ondemand_provider = None


def reset_pricing_state() -> None:
    """Reset module state (for tests)."""
    global _live_aws_pricing, _spot_provider, _ondemand_provider
    _live_aws_pricing = False
    _spot_provider = None
    _ondemand_provider = None


def _static_on_demand_per_gpu_hour(gpu_type: str, cloud_region: str) -> float:
    base = _GPU_BASE_PRICE_USD_PER_HR.get(gpu_type.upper(), _DEFAULT_GPU_PRICE)
    multiplier = _REGION_MULTIPLIER.get(cloud_region, _DEFAULT_MULTIPLIER)
    return base * multiplier


def _on_demand_per_gpu_hour(gpu_type: str, cloud_region: str) -> tuple[float, list[str]]:
    """Resolve on-demand $/GPU/hr with live API when enabled, else static tables."""
    if _live_aws_pricing and _ondemand_provider is not None:
        live = _ondemand_provider.ondemand_price_per_gpu_hour(gpu_type, cloud_region)
        if live is not None:
            return live, ["cost:live_ondemand"]
        return _static_on_demand_per_gpu_hour(gpu_type, cloud_region), ["cost:static_ondemand_fallback"]
    return _static_on_demand_per_gpu_hour(gpu_type, cloud_region), []


def estimate_job_cost(
    gpu_type: str,
    gpu_count: int,
    duration_hours: float,
    cloud_region: str,
    *,
    use_spot: bool = False,
) -> CostEstimate:
    """Return estimated job cost with provenance notes."""
    if not use_spot:
        per_gpu_hr, notes = _on_demand_per_gpu_hour(gpu_type, cloud_region)
        return CostEstimate(usd=per_gpu_hr * gpu_count * duration_hours, notes=list(notes))

    if _live_aws_pricing and _spot_provider is not None:
        spot_per_gpu_hr = _spot_provider.spot_price_per_gpu_hour(gpu_type, cloud_region)
        if spot_per_gpu_hr is not None:
            return CostEstimate(
                usd=spot_per_gpu_hr * gpu_count * duration_hours,
                notes=["cost:live_spot"],
            )
        return CostEstimate(
            usd=_static_spot_job_cost(gpu_type, gpu_count, duration_hours, cloud_region),
            notes=["cost:static_spot_fallback"],
        )

    return CostEstimate(
        usd=_static_spot_job_cost(gpu_type, gpu_count, duration_hours, cloud_region),
    )


def _static_spot_job_cost(
    gpu_type: str,
    gpu_count: int,
    duration_hours: float,
    cloud_region: str,
) -> float:
    per_gpu_hr, _ = _on_demand_per_gpu_hour(gpu_type, cloud_region)
    on_demand = per_gpu_hr * gpu_count * duration_hours
    return on_demand * SPOT_PRICE_FRACTION


def estimate_cost_usd(
    gpu_type: str,
    gpu_count: int,
    duration_hours: float,
    cloud_region: str,
    *,
    use_spot: bool = False,
) -> float:
    """Return estimated cost in USD for the job in the given region."""
    return estimate_job_cost(
        gpu_type, gpu_count, duration_hours, cloud_region, use_spot=use_spot,
    ).usd
