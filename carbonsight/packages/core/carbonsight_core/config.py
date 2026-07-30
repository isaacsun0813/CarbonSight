"""Load config from environment; no secrets in config files."""

import os
from dataclasses import dataclass


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
    carbonsight_api_url: str = ""
    cache_ttl_min: int = 15
    carbon_price_usd_per_ton: float = 50.0
    carbon_weight: float = 1.0
    database_url: str = ""
    grid_cache_retention_days: int = 7

    @classmethod
    def from_env(cls) -> "Config":
        def str_to_bool(s: str) -> bool:
            return s.strip().lower() in ("1", "true", "yes")

        # DATABASE_URL handling: support DATABASE_URL and CARBONSIGHT_DB_URL
        db_url = (
            os.environ.get("DATABASE_URL", "")
            or os.environ.get("CARBONSIGHT_DB_URL", "")
            or ""
        )

        return cls(
            watttime_username=os.environ.get("WATTTIME_USERNAME", ""),
            watttime_password=os.environ.get("WATTTIME_PASSWORD", ""),
            carbon_provider=os.environ.get("CARBON_PROVIDER", "watttime"),
            use_demo_account=str_to_bool(os.environ.get("USE_DEMO_ACCOUNT", "false")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            cache_tokens=str_to_bool(os.environ.get("CACHE_TOKENS", "true")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "15")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            carbonsight_api_url=os.environ.get("CARBONSIGHT_API_URL", ""),
            cache_ttl_min=int(os.environ.get("CARBONSIGHT_CACHE_TTL_MIN", "15")),
            carbon_price_usd_per_ton=float(os.environ.get("CARBONSIGHT_CARBON_PRICE_USD_PER_TON", "50.0")),
            carbon_weight=float(os.environ.get("CARBONSIGHT_CARBON_WEIGHT", "1.0")),
            database_url=db_url,
            grid_cache_retention_days=int(os.environ.get("GRID_CACHE_RETENTION_DAYS", "7")),
        )
