"""Shared boto3 session, client pool, and TTL cache for AWS-backed providers."""

from __future__ import annotations

import time
from collections.abc import Hashable
from typing import Any

from carbonsight_core.config import Config

try:
    import boto3

    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

_HAS_BOTO = HAS_BOTO


class BaseAWSProvider:
    """Boto3 session/client management and generic TTL cache."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._cache_ttl = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else self._config.pricing_cache_ttl_seconds
        )
        self._cache: dict[Hashable, tuple[float, Any]] = {}
        self._session: Any = None
        self._clients: dict[str, Any] = {}

    def _get_session(self) -> Any:
        if self._session is None:
            if not HAS_BOTO:
                raise RuntimeError("boto3 not installed")
            profile = self._config.aws_profile
            self._session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return self._session

    def _client(self, service_name: str, region_name: str) -> Any:
        key = f"{service_name}:{region_name}"
        if key not in self._clients:
            self._clients[key] = self._get_session().client(service_name, region_name=region_name)
        return self._clients[key]

    def _get_cached(self, key: Hashable) -> Any | None:
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, key: Hashable, value: Any) -> Any:
        self._cache[key] = (time.time() + self._cache_ttl, value)
        return value
