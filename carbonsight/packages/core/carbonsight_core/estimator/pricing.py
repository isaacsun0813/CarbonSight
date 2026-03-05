"""
AWS on-demand GPU instance pricing (per GPU per hour, USD).

Sources:
  - https://aws.amazon.com/ec2/pricing/on-demand/ (sampled Feb 2026)
  - Base prices are for us-east-1; regional multipliers scale them.

Design: keep this as a plain lookup dict so it's easy to update without
touching estimation logic. gpu_type keys match JobSpec.gpu_type values
(e.g. "A100", "H100", "V100", "T4", "A10G").
"""

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
# Europe and APAC carry a premium; emerging market regions (af, me, sa) are higher still.
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


def estimate_cost_usd(
    gpu_type: str,
    gpu_count: int,
    duration_hours: float,
    cloud_region: str,
) -> float:
    """Return estimated on-demand cost in USD for the job in the given region."""
    base = _GPU_BASE_PRICE_USD_PER_HR.get(gpu_type.upper(), _DEFAULT_GPU_PRICE)
    multiplier = _REGION_MULTIPLIER.get(cloud_region, _DEFAULT_MULTIPLIER)
    return base * multiplier * gpu_count * duration_hours
