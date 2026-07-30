"""
Spot availability tracking per SkyNomad Sec 4.3.

SkyNomad infers spot availability via periodic probing:
  probe outcome 1 => spot available (or spot request succeeded)
  probe outcome 0 => unavailable

From a probe trace we extract "virtual instances":
  - 0->1 transition marks start of an availability window (virtual instance launch)
  - 1->0 transition marks end, interpreted as virtual preemption
  - If trace ends while still in 1-state, that instance is censored (still alive)

This module is pure Python, no boto3, no AWS calls.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class SpotObservation:
    """
    Single spot probe result.

    Attributes
    ----------
    t:
        Timestamp of the probe. Can be float (epoch seconds or logical time),
        int, or datetime. Must be comparable within a region's trace.
    region:
        Cloud region identifier, e.g. "us-east-1".
    outcome:
        0 = unavailable, 1 = available (probe succeeded).
    """

    t: Any
    region: str
    outcome: int

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {self.outcome!r}")
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("region must be non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": _serialize_t(self.t),
            "region": self.region,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpotObservation":
        return cls(
            t=_deserialize_t(d["t"]),
            region=d["region"],
            outcome=int(d["outcome"]),
        )


@dataclass(frozen=True, slots=True)
class VirtualInstance:
    """
    Inferred lifetime of a spot opportunity from probe sequence.

    Attributes
    ----------
    id:
        Unique identifier within the extraction run (e.g. "us-east-1-1").
    region:
        Region this virtual instance belongs to.
    start_t:
        Time of 0->1 transition (or first 1 if trace starts available).
    end_t:
        Time of 1->0 transition for preemption, or last observation time for
        censored. Can be None only if no observations.
    end_reason:
        "preemption" if ended by 1->0, "censored" if still available at end.
    """

    id: str
    region: str
    start_t: Any
    end_t: Any
    end_reason: Literal["preemption", "censored"]

    def __post_init__(self) -> None:
        if self.end_reason not in ("preemption", "censored"):
            raise ValueError(f"end_reason must be preemption|censored, got {self.end_reason!r}")

    @property
    def lifetime(self) -> Optional[float]:
        """Return lifetime as float seconds if both timestamps are convertible."""
        try:
            s = _as_float(self.start_t)
            e = _as_float(self.end_t)
            return e - s
        except Exception:
            # If datetimes or non-numeric, try timedelta via direct subtraction
            try:
                delta = self.end_t - self.start_t  # type: ignore
                # datetime timedelta
                if hasattr(delta, "total_seconds"):
                    return delta.total_seconds()  # type: ignore
                return float(delta)
            except Exception:
                return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "region": self.region,
            "start_t": _serialize_t(self.start_t),
            "end_t": _serialize_t(self.end_t),
            "end_reason": self.end_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VirtualInstance":
        return cls(
            id=d["id"],
            region=d["region"],
            start_t=_deserialize_t(d["start_t"]),
            end_t=_deserialize_t(d["end_t"]),
            end_reason=d["end_reason"],
        )


# ---------------------------------------------------------------------------
# Helpers for t serialization / comparison
# ---------------------------------------------------------------------------


def _serialize_t(t: Any) -> Any:
    """Serialize timestamp to JSON-friendly form."""
    if isinstance(t, datetime):
        return t.isoformat()
    return t


def _deserialize_t(v: Any) -> Any:
    """Deserialize timestamp; attempt ISO datetime parse."""
    if isinstance(v, str):
        # Try ISO datetime
        try:
            # fromisoformat handles most ISO8601 with timezone
            return datetime.fromisoformat(v)
        except Exception:
            # If not datetime, keep as string
            return v
    return v


def _as_float(t: Any) -> float:
    """Convert t to float for ordering when possible."""
    if isinstance(t, datetime):
        return t.timestamp()
    if isinstance(t, (int, float)):
        return float(t)
    # Try to coerce string that looks like number or iso datetime
    if isinstance(t, str):
        try:
            dt = datetime.fromisoformat(t)
            return dt.timestamp()
        except Exception:
            pass
        try:
            return float(t)
        except Exception:
            raise TypeError(f"Cannot convert t={t!r} to float")
    raise TypeError(f"Unsupported t type: {type(t)} {t!r}")


def _t_sort_key(obs: SpotObservation) -> float:
    """Sort key for observations: convert t to float if possible, else fallback."""
    try:
        return _as_float(obs.t)
    except Exception:
        # Fallback to string or direct comparison via id; still need deterministic
        # Use hash of repr as last resort but ensure stable ordering: use repr
        # However we still need to allow datetime direct comparison, so attempt
        # to return timestamp-like but keep original for tie-breaker.
        # For non-float-convertible, use 0 and rely on secondary key.
        return 0.0


def _compare_obs_lists(a: Any, b: Any) -> int:
    """Return -1/0/1 for t comparison using best-effort conversion."""
    try:
        fa = _as_float(a)
        fb = _as_float(b)
        if fa < fb:
            return -1
        if fa > fb:
            return 1
        return 0
    except Exception:
        # Fallback to direct comparable if same type
        try:
            if a < b:  # type: ignore
                return -1
            if a > b:  # type: ignore
                return 1
            return 0
        except Exception:
            # Last resort: compare string repr
            sa, sb = repr(a), repr(b)
            if sa < sb:
                return -1
            if sa > sb:
                return 1
            return 0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class AvailabilityTracker:
    """
    Tracks spot availability probes per region and extracts virtual instance lifetimes.

    Usage
    -----
    tracker = AvailabilityTracker()
    tracker.record_observation(SpotObservation(t=0, region="us-east-1", outcome=0))
    tracker.record_observation(SpotObservation(t=1, region="us-east-1", outcome=1))
    tracker.record_observation(SpotObservation(t=5, region="us-east-1", outcome=0))
    vis = tracker.extract_virtual_instances("us-east-1")
    # vis[0].end_reason == "preemption", lifetime 4

    Thread-safety: not thread-safe, caller must synchronize.
    """

    def __init__(self, observations: Optional[List[SpotObservation]] = None) -> None:
        # region -> list[SpotObservation]
        self._by_region: Dict[str, List[SpotObservation]] = {}
        self._all: List[SpotObservation] = []
        if observations:
            for o in observations:
                self._add_internal(o)

    # ---------------- internal ----------------

    def _add_internal(self, obs: SpotObservation) -> None:
        if not isinstance(obs, SpotObservation):
            raise TypeError("obs must be SpotObservation")
        self._all.append(obs)
        self._by_region.setdefault(obs.region, []).append(obs)

    @staticmethod
    def _sorted_region_obs(obs_list: List[SpotObservation]) -> List[SpotObservation]:
        # Sort by _as_float if possible, stable with original t for deterministic tie handling
        def key_fn(o: SpotObservation):
            try:
                return (_as_float(o.t), 0)
            except Exception:
                # If t not convertible, push by outcome and repr for stability
                return (0.0, 1)

        # We want primary sort by actual time value, using _compare logic.
        # If items have mixed types but convertible to float, use float.
        # Otherwise fallback to direct sort which requires comparable types (same-type per region assumed).
        try:
            return sorted(obs_list, key=lambda o: _as_float(o.t))
        except Exception:
            try:
                return sorted(obs_list, key=lambda o: o.t)  # type: ignore
            except Exception:
                # Slow but safe: sort by serialized repr
                return sorted(obs_list, key=lambda o: repr(o.t))

    # ---------------- public API ----------------

    def record_observation(
        self,
        t_or_obs: Union[SpotObservation, Any],
        region: Optional[str] = None,
        outcome: Optional[int] = None,
    ) -> None:
        """
        Record a probe outcome.

        Forms
        -----
        - record_observation(SpotObservation(...))
        - record_observation(t, region, outcome)

        Parameters
        ----------
        t_or_obs:
            Either SpotObservation or timestamp t.
        region:
            Required if first arg is timestamp.
        outcome:
            0/1, required if first arg is timestamp.
        """
        if isinstance(t_or_obs, SpotObservation):
            if region is not None or outcome is not None:
                raise ValueError(
                    "When passing SpotObservation, region/outcome kwargs must be None"
                )
            obs = t_or_obs
        else:
            if region is None or outcome is None:
                raise ValueError(
                    "When t is passed positionally, region and outcome are required"
                )
            obs = SpotObservation(t=t_or_obs, region=region, outcome=outcome)
        self._add_internal(obs)

    def extract_virtual_instances(
        self, region: Optional[str] = None
    ) -> List[VirtualInstance]:
        """
        Extract virtual instances from probe sequence.

        Pairing logic (per region, time-ordered):
        - 0->1 : start new virtual instance at the 1's timestamp
        - 1->0 : end active instance at the 0's timestamp, end_reason="preemption"
        - If trace ends while in 1-state, emit instance with end_reason="censored"
          and end_t = last observation's timestamp.

        If region is None, extracts across all regions.
        """
        result: List[VirtualInstance] = []
        # Determine regions to process
        if region is not None:
            regions = [region] if region in self._by_region else []
            # If region has no observations, return []
            if not regions:
                return []
        else:
            regions = sorted(self._by_region.keys())

        global_counter = itertools.count(1)

        for reg in regions:
            obs_list = self._sorted_region_obs(self._by_region.get(reg, []))
            if not obs_list:
                continue

            active_start: Any = None
            prev_outcome: Optional[int] = None

            for obs in obs_list:
                if prev_outcome is None:
                    # First observation in this region
                    if obs.outcome == 1:
                        active_start = obs.t
                else:
                    if prev_outcome == 0 and obs.outcome == 1:
                        # start
                        # If we already have an active_start (should not happen without intervening 1->0),
                        # we treat as continued, but overwrite to latest start for safety
                        active_start = obs.t
                    elif prev_outcome == 1 and obs.outcome == 0:
                        # preemption end
                        if active_start is not None:
                            vi = VirtualInstance(
                                id=f"{reg}-{next(global_counter)}",
                                region=reg,
                                start_t=active_start,
                                end_t=obs.t,
                                end_reason="preemption",
                            )
                            result.append(vi)
                            active_start = None
                        # else stray 1->0 without known start; ignore
                prev_outcome = obs.outcome

            # Handle censored tail
            if active_start is not None:
                last_t = obs_list[-1].t
                vi = VirtualInstance(
                    id=f"{reg}-{next(global_counter)}",
                    region=reg,
                    start_t=active_start,
                    end_t=last_t,
                    end_reason="censored",
                )
                result.append(vi)

        return result

    def is_available(self, region: str, at: Any) -> bool:
        """
        Whether region is considered available at time `at`.

        Returns True iff the latest observation with t <= at has outcome 1.
        If no observations <= at exist, returns False.
        """
        obs_list = self._sorted_region_obs(self._by_region.get(region, []))
        if not obs_list:
            return False

        # Find last observation <= at
        # Use _compare logic
        last_match: Optional[SpotObservation] = None
        for obs in obs_list:
            cmp = _compare_obs_lists(obs.t, at)
            if cmp <= 0:
                last_match = obs
            else:
                break

        if last_match is None:
            return False
        return last_match.outcome == 1

    def get_observations(
        self, region: Optional[str] = None
    ) -> List[SpotObservation]:
        """Return sorted observations, optionally filtered by region."""
        if region is not None:
            return self._sorted_region_obs(self._by_region.get(region, []))
        # All regions, sorted by region then time
        all_sorted = []
        for reg in sorted(self._by_region.keys()):
            all_sorted.extend(self._sorted_region_obs(self._by_region[reg]))
        return all_sorted

    # ---------------- persistence ----------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker to dict for JSON persistence."""
        return {
            "observations": [o.to_dict() for o in self._all],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvailabilityTracker":
        """Reconstruct tracker from dict produced by to_dict."""
        obs_data = data.get("observations", [])
        observations = [SpotObservation.from_dict(o) for o in obs_data]
        return cls(observations=observations)

    # ---------------- misc ----------------

    def __len__(self) -> int:
        return len(self._all)

    def clear(self) -> None:
        self._by_region.clear()
        self._all.clear()

    def __repr__(self) -> str:
        return f"AvailabilityTracker(n_obs={len(self._all)}, regions={list(self._by_region.keys())})"
