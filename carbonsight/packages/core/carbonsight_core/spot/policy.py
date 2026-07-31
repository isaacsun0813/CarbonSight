"""
SkyNomad Sec 4.7 — Policy loop.

Actions: PROBE, RUN, MIGRATE, WAIT, TERMINATE.
while p < P: probe every 2h + Delta heuristic for migrate/wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

from carbonsight_core.spot.availability import AvailabilityTracker
from carbonsight_core.spot.lifetime import LifetimeStats
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import ODCandidate, rank_candidates


class Action(str, Enum):
    PROBE = "PROBE"
    RUN = "RUN"
    MIGRATE = "MIGRATE"
    WAIT = "WAIT"
    TERMINATE = "TERMINATE"


@dataclass
class PolicyDecision:
    action: Action
    target: ODCandidate | None = None
    reason: str = ""
    ranked: list[ODCandidate] = field(default_factory=list)


@dataclass
class SkyNomadPolicy:
    """Multi-lever scheduler: cost + carbon + time + availability + migration."""

    tracker: AvailabilityTracker
    lifetime: LifetimeStats
    progress: ProgressState
    carbon_price_usd_per_ton: float = 50.0
    probe_interval_hours: float = 2.0
    delta_utility: float = 1.0  # migrate if best_U - current_U > delta
    last_probe_at: datetime | None = None
    current: ODCandidate | None = None

    def decide(
        self,
        candidates: list[ODCandidate],
        *,
        now: datetime | None = None,
    ) -> PolicyDecision:
        """One policy step given current candidates."""
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        self.progress.t = now

        if self.progress.is_complete():
            return PolicyDecision(Action.TERMINATE, reason="progress complete")

        if self.progress.time_left_hours() <= 0:
            return PolicyDecision(Action.TERMINATE, reason="deadline passed")

        # Probe cadence
        need_probe = (
            self.last_probe_at is None
            or (now - self.last_probe_at) >= timedelta(hours=self.probe_interval_hours)
        )
        if need_probe and self.current is None:
            self.last_probe_at = now
            ranked = rank_candidates(
                list(candidates),
                self.progress,
                carbon_price_usd_per_ton=self.carbon_price_usd_per_ton,
            )
            best = ranked[0] if ranked else None
            return PolicyDecision(
                Action.PROBE,
                target=best,
                reason="initial / periodic probe",
                ranked=ranked,
            )

        ranked = rank_candidates(
            list(candidates),
            self.progress,
            carbon_price_usd_per_ton=self.carbon_price_usd_per_ton,
        )
        if not ranked:
            return PolicyDecision(Action.WAIT, reason="no candidates")

        best = ranked[0]

        # Periodic re-probe
        if need_probe:
            self.last_probe_at = now
            if self.current is not None and best.region != self.current.region:
                return PolicyDecision(
                    Action.PROBE,
                    target=best,
                    reason="probe interval elapsed; evaluating migrate",
                    ranked=ranked,
                )

        # Delta heuristic: migrate if material utility gain
        if self.current is not None:
            # Ensure current has utility computed
            cur_list = rank_candidates(
                [self.current],
                self.progress,
                carbon_price_usd_per_ton=self.carbon_price_usd_per_ton,
            )
            cur_u = cur_list[0].utility if cur_list else self.current.utility
            if best.utility - cur_u > self.delta_utility and (
                best.region != self.current.region
                or best.instance_type != self.current.instance_type
            ):
                return PolicyDecision(
                    Action.MIGRATE,
                    target=best,
                    reason=f"delta U={best.utility - cur_u:.3f} > {self.delta_utility}",
                    ranked=ranked,
                )
            # At-risk check
            at_risk = self.tracker.at_risk_set()
            if (self.current.region, self.current.instance_type) in at_risk:
                return PolicyDecision(
                    Action.MIGRATE,
                    target=best,
                    reason="current placement at-risk",
                    ranked=ranked,
                )
            # Low survival -> wait or migrate
            if self.current.survival < 0.3 and best.survival > self.current.survival:
                return PolicyDecision(
                    Action.MIGRATE,
                    target=best,
                    reason="low survival on current",
                    ranked=ranked,
                )
            return PolicyDecision(
                Action.RUN,
                target=self.current,
                reason="continue current placement",
                ranked=ranked,
            )

        # No current placement: run best (or wait if urgency low and prices high)
        if self.progress.urgency() < 1e-6:
            return PolicyDecision(Action.WAIT, target=best, reason="no urgency", ranked=ranked)

        return PolicyDecision(Action.RUN, target=best, reason="start best candidate", ranked=ranked)

    def run_loop(
        self,
        candidates_fn,
        *,
        max_steps: int = 100,
        step_hours: float = 1.0,
        work_per_hour: float = 0.05,
        now: datetime | None = None,
    ) -> list[PolicyDecision]:
        """Simulate while p < P with probe every 2h + Delta heuristic.

        candidates_fn(now) -> list[ODCandidate]
        """
        t = now or datetime.now(UTC)
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        decisions: list[PolicyDecision] = []
        for _ in range(max_steps):
            if self.progress.is_complete():
                decisions.append(PolicyDecision(Action.TERMINATE, reason="done"))
                break
            cands = list(candidates_fn(t))
            d = self.decide(cands, now=t)
            decisions.append(d)
            if d.action == Action.TERMINATE:
                break
            if d.action in (Action.RUN, Action.MIGRATE, Action.PROBE) and d.target is not None:
                self.current = d.target
                self.progress.advance(work_per_hour * step_hours, now=t)
            elif d.action == Action.WAIT:
                pass
            t = t + timedelta(hours=step_hours)
            self.progress.t = t
        return decisions
