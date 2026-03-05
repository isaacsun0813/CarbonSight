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
        )
