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
    # Central API proxy (CLI clients set this; no personal WattTime creds needed)
    carbonsight_api_url: str = ""
    # Forecast cache TTL (seconds); default 15 minutes
    cache_ttl_seconds: int = 900
    # Social cost of carbon used to dollarize emissions in joint ranking
    carbon_price_usd_per_ton: float = 50.0
    # Postgres DSN for the persistent grid_signal_cache (API + worker)
    database_url: str = ""

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
            carbonsight_api_url=os.environ.get("CARBONSIGHT_API_URL", "").rstrip("/"),
            cache_ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "900")),
            carbon_price_usd_per_ton=float(os.environ.get("CARBON_PRICE_USD_PER_TON", "50")),
            database_url=(
                os.environ.get("DATABASE_URL") or os.environ.get("CARBONSIGHT_DB_URL") or ""
            ),
        )


# Module-level defaults for importers that prefer constants
CACHE_TTL_SECONDS = 900
CARBON_PRICE_USD_PER_TON = 50.0
