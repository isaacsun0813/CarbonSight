"""
SkyNomad Sec 4.3 — Spot availability tracking.

Track eviction observations per (region, instance_type). Availability =
1 - eviction_rate over the observation window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SpotObservation:
    """One observed spot lifecycle event."""

    region: str
    instance_type: str
    timestamp: datetime
    evicted: bool


@dataclass
class VirtualInstance:
    """A logical spot capacity unit being tracked."""

    region: str
    instance_type: str
    launched_at: datetime
    instance_id: str = ""
    alive: bool = True
    evicted_at: datetime | None = None

    def mark_evicted(self, when: datetime | None = None) -> SpotObservation:
        self.alive = False
        self.evicted_at = when or datetime.now(UTC)
        return SpotObservation(
            region=self.region,
            instance_type=self.instance_type,
            timestamp=self.evicted_at,
            evicted=True,
        )


@dataclass
class AvailabilityTracker:
    """dict[(region, instance_type)] -> list[SpotObservation]."""

    observations: dict[tuple[str, str], list[SpotObservation]] = field(default_factory=dict)
    virtual_instances: list[VirtualInstance] = field(default_factory=list)

    def _key(self, region: str, instance_type: str) -> tuple[str, str]:
        return (region, instance_type)

    def add_observation(self, obs: SpotObservation) -> None:
        key = self._key(obs.region, obs.instance_type)
        self.observations.setdefault(key, []).append(obs)

    def record(
        self,
        region: str,
        instance_type: str,
        *,
        evicted: bool,
        timestamp: datetime | None = None,
    ) -> SpotObservation:
        obs = SpotObservation(
            region=region,
            instance_type=instance_type,
            timestamp=timestamp or datetime.now(UTC),
            evicted=evicted,
        )
        self.add_observation(obs)
        return obs

    def get_observations(self, region: str, instance_type: str) -> list[SpotObservation]:
        return list(self.observations.get(self._key(region, instance_type), []))

    def eviction_rate(self, region: str, instance_type: str) -> float:
        obs = self.get_observations(region, instance_type)
        if not obs:
            return 0.0
        evicted = sum(1 for o in obs if o.evicted)
        return evicted / len(obs)

    def get_availability(self, region: str, instance_type: str) -> float:
        """Availability ≈ 1 - eviction_rate. Clamped to [0, 1]."""
        rate = self.eviction_rate(region, instance_type)
        return max(0.0, min(1.0, 1.0 - rate))

    def at_risk_set(self, *, min_eviction_rate: float = 0.3) -> set[tuple[str, str]]:
        """Pairs currently considered at-risk based on eviction rate."""
        at_risk: set[tuple[str, str]] = set()
        for key, obs in self.observations.items():
            if not obs:
                continue
            rate = sum(1 for o in obs if o.evicted) / len(obs)
            if rate >= min_eviction_rate:
                at_risk.add(key)
        return at_risk

    def eviction_counts(self) -> dict[tuple[str, str], int]:
        return {
            key: sum(1 for o in obs if o.evicted) for key, obs in self.observations.items()
        }

    def seed_defaults(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        default_availability: float = 0.85,
        n: int = 20,
    ) -> None:
        """Seed synthetic observations so ranking works without live history."""
        # availability = 1 - eviction_rate => eviction_rate = 1 - avail
        er = max(0.0, min(1.0, 1.0 - default_availability))
        n_evicted = int(round(er * n))
        now = datetime.now(UTC)
        for region, itype in pairs:
            for i in range(n):
                self.record(
                    region,
                    itype,
                    evicted=(i < n_evicted),
                    timestamp=now,
                )
