"""
SkyNomad Sec 4.7 — Algo 1, the decision loop, with a carbon lever.

One step:

    1. thrifty      p >= P                 -> idle
    2. safety net   T-t < P-p+2d           -> launch the cheapest on-demand region
    3. probe        every probe_interval_hr (recorded, does not change the action)
    4. rank         U = V*eta - C_total - E/Lbar over region x mode
    5. delta        launch the best only if U_best > U_current + delta, else stay

Step 5 is the anti-flapping rule: without it a candidate that is a cent better
would trigger a checkpoint migration every probe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from carbonsight_core.spot.progress import (
    ODCandidate,
    ProgressState,
    compute_safety_net_total_cost,
)
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    rank_candidates,
)

ActionKind = Literal["launch", "terminate", "idle", "stay"]


@dataclass(frozen=True, slots=True)
class PolicyState:
    """Everything the policy needs about the job and its current placement."""

    p: float
    P: float
    t: float
    T: float
    r0: str
    ckpt_size_gb: float = 0.0
    cold_start_hr: float = 0.1
    current_region: str = ""
    current_mode: str = "spot"
    current_utility: float = 0.0


@dataclass(frozen=True, slots=True)
class Action:
    """A policy decision, always with a human-readable reason."""

    kind: ActionKind
    region: str | None = None
    mode: str | None = None
    reason: str = ""

    @classmethod
    def launch(cls, region: str, mode: str = "spot", reason: str = "") -> Action:
        return cls(kind="launch", region=region, mode=mode, reason=reason)

    @classmethod
    def terminate(cls, reason: str = "") -> Action:
        return cls(kind="terminate", reason=reason)

    @classmethod
    def idle(cls, reason: str = "") -> Action:
        return cls(kind="idle", reason=reason)

    @classmethod
    def stay(cls, reason: str = "") -> Action:
        return cls(kind="stay", reason=reason)

    @property
    def is_launch(self) -> bool:
        return self.kind == "launch"

    @property
    def is_idle(self) -> bool:
        return self.kind == "idle"

    @property
    def is_stay(self) -> bool:
        return self.kind == "stay"

    @property
    def is_terminate(self) -> bool:
        return self.kind == "terminate"


class SkyNomadPolicy:
    """Algo 1 as a single-step decision function over a candidate list."""

    def __init__(
        self,
        spot_provider: object | None = None,
        carbon_provider: object | None = None,
        migration_estimator: MigrationCostEstimator | None = None,
        delta: float = 0.05,
        probe_interval_hr: float = 2.0,
        carbon_price_usd_per_ton: float = 50.0,
        carbon_weight: float = 1.0,
    ) -> None:
        self.spot_provider = spot_provider
        self.carbon_provider = carbon_provider
        self.migration_estimator = migration_estimator or MigrationCostEstimator()
        self.delta = float(delta)
        self.probe_interval_hr = float(probe_interval_hr)
        self.carbon_price_usd_per_ton = float(carbon_price_usd_per_ton)
        self.carbon_weight = float(carbon_weight)
        self._last_probe_t = 0.0

    def _probe_due(self, state: PolicyState) -> bool:
        return (state.t - self._last_probe_t) >= self.probe_interval_hr - 1e-9

    def _cheapest_od(
        self,
        candidates: list[CandidateState],
        progress: ProgressState,
        state: PolicyState,
    ) -> CandidateState | None:
        """argmin_r C_od(r)*(P-p+d) + E(r) + carbon$(r) over the on-demand candidates."""
        best: CandidateState | None = None
        best_cost = math.inf
        for cand in candidates:
            if not cand.is_od:
                continue
            cost = compute_safety_net_total_cost(
                ODCandidate(
                    region=cand.region,
                    od_price_per_hr=cand.price_per_hr,
                    migration_cost=cand.migration_cost,
                    carbon_kg_per_hr=cand.carbon_kg_per_hr,
                ),
                progress.remaining_work,
                state.cold_start_hr,
                self.carbon_price_usd_per_ton,
                self.carbon_weight,
            )
            if cost < best_cost:
                best, best_cost = cand, cost
        return best

    def _safety_net_action(
        self,
        state: PolicyState,
        candidates: list[CandidateState],
        progress: ProgressState,
    ) -> Action:
        threshold = progress.remaining_work + 2 * state.cold_start_hr
        why = f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d={threshold:.3f}"
        best = self._cheapest_od(candidates, progress, state)
        if best is not None:
            return Action.launch(best.region, best.mode, f"{why}, cheapest OD {best.region}")
        fallback = state.current_region or state.r0
        return Action.launch(
            fallback, "on_demand", f"{why}, no OD candidates - fallback {fallback} OD"
        )

    def _decide_ranked(
        self,
        state: PolicyState,
        ranked: list[tuple[CandidateState, float]],
        probing_due: bool,
    ) -> Action:
        probe_note = "; probing triggered" if probing_due else ""
        if not ranked:
            return Action.stay(f"no candidates{probe_note}")

        best, best_u = ranked[0]
        if best.region == state.current_region and best.mode == state.current_mode:
            return Action.stay(
                f"already in best {best.region}/{best.mode} U={best_u:.4f}{probe_note}"
            )
        if best_u > state.current_utility + self.delta:
            return Action.launch(
                best.region,
                best.mode,
                f"U_best {best_u:.4f} ({best.region}/{best.mode}) > "
                f"U_current {state.current_utility:.4f}+delta {self.delta}{probe_note}",
            )
        return Action.stay(
            f"U_best {best_u:.4f} <= U_current {state.current_utility:.4f}"
            f"+delta {self.delta}, no migration{probe_note}"
        )

    def rank_and_decide(
        self,
        state: PolicyState,
        candidates: list[CandidateState],
        progress: ProgressState,
        v: float,
    ) -> tuple[list[tuple[CandidateState, float]], Action]:
        """One Algo 1 step; returns the ranking as well as the chosen action."""
        ranked = rank_candidates(
            candidates, v, state.cold_start_hr, self.carbon_price_usd_per_ton, self.carbon_weight
        )

        probing_due = self._probe_due(state)
        if probing_due:
            self._last_probe_t = state.t

        if progress.is_thrifty():
            return ranked, Action.idle(f"thrifty: p={state.p} >= P={state.P}")
        if progress.is_safety_net(state.cold_start_hr):
            return ranked, self._safety_net_action(state, candidates, progress)
        return ranked, self._decide_ranked(state, ranked, probing_due)

    def decide(
        self,
        state: PolicyState,
        candidates: list[CandidateState],
        progress: ProgressState,
        v: float,
    ) -> Action:
        """One Algo 1 step."""
        return self.rank_and_decide(state, candidates, progress, v)[1]


__all__ = ["Action", "ActionKind", "PolicyState", "SkyNomadPolicy"]
