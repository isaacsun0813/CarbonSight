"""
SkyNomad Sec 4.3 — spot availability probing.

A probe returns 1 (spot available / a request would succeed) or 0. From a probe
trace we recover *virtual instances*:

    0 -> 1                    starts one at the timestamp of the 1
    1 -> 0                    ends it at the timestamp of the 0, "preemption"
    trace ends while 1        ends it at the last observation, "censored"

Censored instances are right-censored observations for the Nelson-Aalen fit in
``spot/lifetime.py``. Timestamps are ``datetime`` throughout (naive values are
read as UTC); lifetimes are reported in hours to match the rest of the scheduler.

Pure Python — no boto3, no network.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

EndReason = Literal["preemption", "censored"]


def _as_utc(t: datetime) -> datetime:
    if not isinstance(t, datetime):
        raise TypeError(f"timestamp must be a datetime, got {type(t).__name__}")
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SpotObservation:
    """One probe result: outcome 1 = spot available, 0 = unavailable."""

    t: datetime
    region: str
    outcome: int

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {self.outcome!r}")
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("region must be a non-empty string")
        object.__setattr__(self, "t", _as_utc(self.t))

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t.isoformat(), "region": self.region, "outcome": self.outcome}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpotObservation:
        return cls(t=datetime.fromisoformat(d["t"]), region=d["region"], outcome=int(d["outcome"]))


@dataclass(frozen=True, slots=True)
class VirtualInstance:
    """An inferred spot opportunity: [start_t, end_t] plus how it ended."""

    id: str
    region: str
    start_t: datetime
    end_t: datetime
    end_reason: EndReason

    def __post_init__(self) -> None:
        if self.end_reason not in ("preemption", "censored"):
            raise ValueError(f"end_reason must be preemption|censored, got {self.end_reason!r}")
        object.__setattr__(self, "start_t", _as_utc(self.start_t))
        object.__setattr__(self, "end_t", _as_utc(self.end_t))

    @property
    def lifetime_hours(self) -> float:
        return (self.end_t - self.start_t).total_seconds() / 3600.0

    @property
    def preempted(self) -> bool:
        return self.end_reason == "preemption"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "region": self.region,
            "start_t": self.start_t.isoformat(),
            "end_t": self.end_t.isoformat(),
            "end_reason": self.end_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VirtualInstance:
        return cls(
            id=d["id"],
            region=d["region"],
            start_t=datetime.fromisoformat(d["start_t"]),
            end_t=datetime.fromisoformat(d["end_t"]),
            end_reason=d["end_reason"],
        )


class AvailabilityTracker:
    """Per-region probe log; turns probe traces into virtual instances."""

    def __init__(self, observations: Iterable[SpotObservation] | None = None) -> None:
        self._by_region: dict[str, list[SpotObservation]] = {}
        self._n = 0
        for obs in observations or ():
            self.record_observation(obs)

    def record_observation(
        self,
        t_or_obs: SpotObservation | datetime,
        region: str | None = None,
        outcome: int | None = None,
    ) -> None:
        """Record a probe, either as a ``SpotObservation`` or as ``(t, region, outcome)``."""
        if isinstance(t_or_obs, SpotObservation):
            if region is not None or outcome is not None:
                raise ValueError("pass either a SpotObservation or (t, region, outcome), not both")
            obs = t_or_obs
        else:
            if region is None or outcome is None:
                raise ValueError("region and outcome are required when passing a timestamp")
            obs = SpotObservation(t=t_or_obs, region=region, outcome=outcome)
        self._by_region.setdefault(obs.region, []).append(obs)
        self._n += 1

    def regions(self) -> list[str]:
        return sorted(self._by_region)

    def get_observations(self, region: str | None = None) -> list[SpotObservation]:
        """Observations sorted by time; all regions (region-then-time) when omitted."""
        if region is not None:
            return sorted(self._by_region.get(region, ()), key=lambda o: o.t)
        return [o for reg in self.regions() for o in self.get_observations(reg)]

    def extract_virtual_instances(self, region: str | None = None) -> list[VirtualInstance]:
        """Walk each region's probe trace and emit its virtual instances in order."""
        out: list[VirtualInstance] = []
        for reg in [region] if region is not None else self.regions():
            obs_list = self.get_observations(reg)
            if not obs_list:
                continue
            start: datetime | None = None
            prev: int | None = None
            for obs in obs_list:
                if obs.outcome == 1 and prev != 1:
                    start = obs.t
                elif obs.outcome == 0 and prev == 1 and start is not None:
                    out.append(
                        VirtualInstance(
                            id=f"{reg}-{len(out) + 1}",
                            region=reg,
                            start_t=start,
                            end_t=obs.t,
                            end_reason="preemption",
                        )
                    )
                    start = None
                prev = obs.outcome
            if start is not None:
                out.append(
                    VirtualInstance(
                        id=f"{reg}-{len(out) + 1}",
                        region=reg,
                        start_t=start,
                        end_t=obs_list[-1].t,
                        end_reason="censored",
                    )
                )
        return out

    def observed_lifetimes(self, region: str | None = None) -> list[tuple[float, bool]]:
        """``(lifetime_hours, preempted)`` pairs, ready for ``LifetimeStats``."""
        return [(vi.lifetime_hours, vi.preempted) for vi in self.extract_virtual_instances(region)]

    def is_available(self, region: str, at: datetime) -> bool:
        """True iff the latest probe at or before ``at`` returned 1."""
        at = _as_utc(at)
        last: SpotObservation | None = None
        for obs in self.get_observations(region):
            if obs.t > at:
                break
            last = obs
        return last is not None and last.outcome == 1

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [o.to_dict() for o in self.get_observations()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvailabilityTracker:
        return cls(SpotObservation.from_dict(o) for o in data.get("observations", ()))

    def clear(self) -> None:
        self._by_region.clear()
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"AvailabilityTracker(n_obs={self._n}, regions={self.regions()})"


def synthetic_uptime(region: str, *, low: float = 0.45, high: float = 0.95) -> float:
    """Deterministic per-region spot uptime in [low, high], derived from the region name.

    Stand-in for real probe telemetry so lifetimes vary across regions before a
    ledger exists; superseded by ``AvailabilityTracker`` data as soon as there is any.
    """
    h = int(hashlib.md5(region.encode("utf-8")).hexdigest()[:8], 16)
    return low + (high - low) * ((h % 1000) / 999.0)


def synthetic_probe_trace(
    region: str,
    *,
    start: datetime | None = None,
    hours: float = 168.0,
    step_hours: float = 1.0,
    uptime: float | None = None,
) -> list[SpotObservation]:
    """Deterministic Bernoulli probe trace for ``region`` on a fixed hourly grid.

    Hourly steps keep inferred lifetimes on an integer-hour grid, which is what
    the discrete Nelson-Aalen sum in ``spot/lifetime.py`` assumes.
    """
    t0 = _as_utc(start or datetime(2026, 1, 1, tzinfo=UTC))
    p_up = synthetic_uptime(region) if uptime is None else uptime
    rng = random.Random(int(hashlib.md5(region.encode("utf-8")).hexdigest()[:8], 16))
    n = max(1, int(hours / step_hours))
    return [
        SpotObservation(
            t=t0 + timedelta(hours=step_hours * i),
            region=region,
            outcome=1 if rng.random() < p_up else 0,
        )
        for i in range(n)
    ]


__all__ = [
    "AvailabilityTracker",
    "SpotObservation",
    "VirtualInstance",
    "synthetic_probe_trace",
    "synthetic_uptime",
]
