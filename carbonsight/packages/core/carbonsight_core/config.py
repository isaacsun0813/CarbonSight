"""Load config from environment; no secrets in config files."""

import os
from dataclasses import dataclass

# TTL for the in-process WattTime forecast cache (15 minutes).
DEFAULT_FORECAST_CACHE_TTL_SECONDS = 900


@dataclass
class Config:
    """Runtime config from env."""

    watttime_username: str = ""
    watttime_password: str = ""
    carbon_provider: str = "watttime"
    use_demo_account: bool = False
    log_level: str = "INFO"
    cache_tokens: bool = True
    request_timeout: int = 15
    max_retries: int = 3
    live_aws_pricing: bool = False
    pricing_cache_ttl_seconds: int = 3600
    forecast_cache_ttl_seconds: int = DEFAULT_FORECAST_CACHE_TTL_SECONDS
    aws_profile: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        def str_to_bool(s: str) -> bool:
            return s.strip().lower() in ("1", "true", "yes")

        return cls(
            watttime_username=os.environ.get("WATTTIME_USERNAME", ""),
            watttime_password=os.environ.get("WATTTIME_PASSWORD", ""),
            carbon_provider=os.environ.get("CARBON_PROVIDER", "watttime"),
            use_demo_account=str_to_bool(os.environ.get("USE_DEMO_ACCOUNT", "false")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            cache_tokens=str_to_bool(os.environ.get("CACHE_TOKENS", "true")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "15")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            live_aws_pricing=str_to_bool(os.environ.get("CARBONSIGHT_LIVE_AWS_PRICING", "false")),
            pricing_cache_ttl_seconds=int(os.environ.get("CARBONSIGHT_PRICING_CACHE_TTL", "3600")),
            forecast_cache_ttl_seconds=int(
                os.environ.get(
                    "CARBONSIGHT_FORECAST_CACHE_TTL", str(DEFAULT_FORECAST_CACHE_TTL_SECONDS)
                )
            ),
            aws_profile=os.environ.get("AWS_PROFILE") or None,
        )
