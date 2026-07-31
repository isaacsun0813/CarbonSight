"""
SkyNomad Sec 4.5 — Progress and urgency.

P = total progress target, p = current progress,
T = deadline (absolute time), t = now.
theta = (P - p) / (T - t)  — required progress rate (urgency).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class ProgressState:
    """Job progress relative to a wall-clock deadline."""

    P: float  # total work units (e.g. steps or normalized 1.0)
    p: float  # current completed work
    T: datetime  # absolute deadline
    t: datetime  # now

    def __post_init__(self) -> None:
        if self.T.tzinfo is None:
            self.T = self.T.replace(tzinfo=UTC)
        if self.t.tzinfo is None:
            self.t = self.t.replace(tzinfo=UTC)
        self.P = float(self.P)
        self.p = float(self.p)

    @classmethod
    def from_deadline_hours(
        cls,
        *,
        total_progress: float = 1.0,
        current_progress: float = 0.0,
        deadline_hours: float = 48.0,
        now: datetime | None = None,
    ) -> "ProgressState":
        t = now or datetime.now(UTC)
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        return cls(
            P=total_progress,
            p=current_progress,
            T=t + timedelta(hours=deadline_hours),
            t=t,
        )

    def remaining_work(self) -> float:
        return max(0.0, self.P - self.p)

    def time_left_hours(self) -> float:
        secs = (self.T - self.t).total_seconds()
        return max(secs / 3600.0, 1e-9)

    def theta(self) -> float:
        """Required progress rate: (P - p) / (T - t) in work-units per hour."""
        return self.remaining_work() / self.time_left_hours()

    def urgency(self, *, baseline_theta: float | None = None) -> float:
        """Urgency relative to a baseline rate. >1 means behind schedule."""
        th = self.theta()
        if baseline_theta is None or baseline_theta <= 0:
            # Baseline: finish exactly at deadline with steady rate from start
            # If no baseline, urgency = theta itself (higher = more urgent)
            return th
        return th / baseline_theta

    def fraction_complete(self) -> float:
        if self.P <= 0:
            return 1.0
        return max(0.0, min(1.0, self.p / self.P))

    def advance(self, delta_p: float, *, now: datetime | None = None) -> None:
        self.p = min(self.P, self.p + max(0.0, delta_p))
        if now is not None:
            self.t = now if now.tzinfo else now.replace(tzinfo=UTC)

    def is_complete(self) -> bool:
        return self.p >= self.P - 1e-12
