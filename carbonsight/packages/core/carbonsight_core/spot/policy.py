"""
SkyNomad policy (Algo 1) for CarbonSight.

Implements loop logic from paper Sec 4.7 extended with carbon lever:

- Thrifty: if p >= P -> Idle (job done)
- Safety net: if T-t < P-p + 2d -> Launch cheapest on-demand region
- Periodic probing trigger: every probe_interval_hr (default 2h)
- Rank candidates by utility U_s = V*η - C_total - E/L̄
- Launch if U_s > U_current + delta, else Stay

Uses existing spot modules:
- progress.ProgressState for thrifty/safety_net and V(t)
- unified_model.CandidateState, rank_candidates, effectiveness, total_cost_per_hr
- availability, lifetime (via mean_lifetime_hr already in CandidateState)
- providers via protocol (SpotPriceProvider, CarbonIntensityProvider) – pluggable
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    rank_candidates,
)

# Optional imports for provider protocols – best-effort, not required for core logic
try:
    from carbonsight_core.spot.unified_model import SpotPriceProvider  # type: ignore
except Exception:  # pragma: no cover
    SpotPriceProvider = object  # type: ignore

try:
    from carbonsight_core.providers.carbon import CarbonIntensityProvider  # type: ignore
except Exception:  # pragma: no cover
    CarbonIntensityProvider = object  # type: ignore


# ---------------------------------------------------------------------------
# PolicyState – input snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyState:
    """
    Policy decision snapshot.

    Attributes:
        p: progress done so far (e.g. hours of useful work)
        P: total progress required
        t: elapsed time since job start (hours)
        T: deadline absolute time since start (hours)
        r0: origin region (where current checkpoint resides) – used for egress cost
        ckpt_size_gb: checkpoint size in GB (for migration cost E)
        cold_start_hr: d – cold start + restore time (hours). Default 0.1h ~6min
        current_region: region where job currently runs (if any)
        current_mode: spot | on_demand | od | idle
        current_utility: utility U_current of current placement (for delta comparison)
    """

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


# ---------------------------------------------------------------------------
# Action – output
# ---------------------------------------------------------------------------

ActionKind = Literal["launch", "terminate", "idle", "stay"]
ModeKind = Literal["spot", "on_demand", "od", "idle"]


@dataclass(frozen=True, slots=True)
class Action:
    """
    Policy decision.

    Factory helpers mirror spec phrasing: Launch(region,mode), Terminate, Idle, Stay

    Attributes:
        kind: launch | terminate | idle | stay
        region: target region for launch (None for other kinds)
        mode: spot | on_demand | od | idle for launch (None for other kinds)
        reason: human-readable explanation (always present per spec)
    """

    kind: ActionKind
    region: Optional[str] = None
    mode: Optional[str] = None
    reason: str = ""

    # Factory helpers – Match spec naming Launch, Terminate, Idle, Stay

    @classmethod
    def Launch(cls, region: str, mode: str = "spot", reason: str = "") -> "Action":
        return cls(kind="launch", region=region, mode=mode, reason=reason)

    @classmethod
    def Terminate(cls, reason: str = "") -> "Action":
        return cls(kind="terminate", region=None, mode=None, reason=reason)

    @classmethod
    def Idle(cls, reason: str = "") -> "Action":
        return cls(kind="idle", region=None, mode=None, reason=reason)

    @classmethod
    def Stay(cls, reason: str = "") -> "Action":
        return cls(kind="stay", region=None, mode=None, reason=reason)

    # Convenience properties

    @property
    def is_launch(self) -> bool:
        return self.kind == "launch"

    @property
    def is_terminate(self) -> bool:
        return self.kind == "terminate"

    @property
    def is_idle(self) -> bool:
        return self.kind == "idle"

    @property
    def is_stay(self) -> bool:
        return self.kind == "stay"

    def __repr__(self) -> str:
        if self.kind == "launch":
            return f"Action.Launch(region={self.region!r}, mode={self.mode!r}, reason={self.reason!r})"
        return f"Action.{self.kind.capitalize()}(reason={self.reason!r})"


# ---------------------------------------------------------------------------
# SkyNomadPolicy
# ---------------------------------------------------------------------------


class SkyNomadPolicy:
    """
    SkyNomad scheduling policy (Algo 1) with carbon lever.

    Loop logic (single decision step):
    1. Thrifty: if p >= P -> Idle
    2. Safety net: if T-t < P-p + 2d -> Launch cheapest OD
    3. Periodic probing trigger: track last probe time, update when due
    4. Rank candidates by utility (via unified_model.rank_candidates)
    5. Launch if U_best > U_current + delta, else Stay

    Attributes:
        spot_provider: optional SpotPriceProvider (pluggable, not required for decide)
        carbon_provider: optional CarbonIntensityProvider
        migration_estimator: MigrationCostEstimator (egress $/GB)
        delta: anti-flapping threshold – require U improvement > delta to migrate
        probe_interval_hr: periodic probe interval, default 2h per paper Sec 4.3
        carbon_price_usd_per_ton: SCC for $ conversion
        carbon_weight: λ multiplier for carbon lever
    """

    def __init__(
        self,
        spot_provider: Optional[object] = None,
        carbon_provider: Optional[object] = None,
        migration_estimator: Optional[MigrationCostEstimator] = None,
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
        self._last_probe_t: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_probe(self, state: PolicyState) -> bool:
        """Periodic probing trigger – every probe_interval_hr."""
        return (state.t - self._last_probe_t) >= self.probe_interval_hr - 1e-9

    def _select_cheapest_od(
        self,
        candidates: List[CandidateState],
        progress: ProgressState,
        state: PolicyState,
    ) -> Optional[CandidateState]:
        """Safety-net cheapest OD: argmin_r [C_od*(P-p+d)+E+carbon_cost]"""
        od_cands = [c for c in candidates if c.is_od]
        if not od_cands:
            return None

        # Use progress module helpers for carbon-aware total cost
        try:
            from carbonsight_core.spot.progress import (
                ODCandidate,
                compute_safety_net_total_cost,
            )

            best: Optional[CandidateState] = None
            best_cost = math.inf
            for c in od_cands:
                od = ODCandidate(
                    region=c.region,
                    od_price_per_hr=c.price_per_hr,
                    migration_cost=c.migration_cost,
                    carbon_kg_per_hr=c.carbon_kg_per_hr,
                )
                cost = compute_safety_net_total_cost(
                    od,
                    progress.remaining_work,
                    state.cold_start_hr,
                    self.carbon_price_usd_per_ton,
                    self.carbon_weight,
                )
                if cost < best_cost:
                    best_cost = cost
                    best = c
            return best
        except Exception:
            # Fallback: cheapest per-hr price
            return min(od_cands, key=lambda c: c.price_per_hr + c.carbon_kg_per_hr)

    def _decide_from_ranked(
        self,
        state: PolicyState,
        progress: ProgressState,
        V: float,
        ranked: List[Tuple[CandidateState, float]],
        probing_due: bool,
    ) -> Action:
        """Core decision given pre-ranked list."""
        # Thrifty already handled upstream, but double-check p>=P
        if progress.is_thrifty() or state.p >= state.P:
            return Action.Idle(reason=f"thrifty: p={state.p} >= P={state.P}")

        # Safety net
        if progress.is_safety_net(state.cold_start_hr):
            # Re-derive cheapest OD from ranked list (filter OD)
            od_ranked = [(c, u) for c, u in ranked if c.is_od]
            if od_ranked:
                cheapest_od, _ = min(
                    od_ranked,
                    key=lambda x: x[0].price_per_hr,  # will be overridden by _select_cheapest_od in decide()
                )
                # Actually re-use _select_cheapest_od logic for accurate carbon-aware choice
                # Convert ranked back to candidates list
                cands = [c for c, _ in ranked]
                best_od = self._select_cheapest_od(cands, progress, state)
                if best_od:
                    cheapest_od = best_od
                return Action.Launch(
                    region=cheapest_od.region,
                    mode=cheapest_od.mode if isinstance(cheapest_od, CandidateState) else "on_demand",
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"cheapest OD {cheapest_od.region}"
                    ),
                )
            else:
                # No OD candidate – fallback to current or r0 as OD
                fallback_region = state.current_region or state.r0 or "us-east-1"
                return Action.Launch(
                    region=fallback_region,
                    mode="on_demand",
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"no OD candidates – fallback {fallback_region} OD"
                    ),
                )

        if not ranked:
            return Action.Stay(
                reason="no candidates"
                + ("; probing due" if probing_due else "")
            )

        best_candidate, best_utility = ranked[0]

        # If already in best placement, stay (avoid unnecessary self-migration)
        if (
            best_candidate.region == state.current_region
            and best_candidate.mode == state.current_mode
        ):
            return Action.Stay(
                reason=(
                    f"already in best {best_candidate.region}/{best_candidate.mode} "
                    f"U={best_utility:.4f} current U={state.current_utility:.4f}"
                    + ("; probing due" if probing_due else "")
                )
            )

        # Delta anti-flapping: require improvement over current + delta
        if best_utility > state.current_utility + self.delta:
            reason = (
                f"U_best {best_utility:.4f} ({best_candidate.region}/{best_candidate.mode}) "
                f"> U_current {state.current_utility:.4f}+delta {self.delta}"
            )
            if probing_due:
                reason += "; probing triggered"
            return Action.Launch(
                region=best_candidate.region,
                mode=best_candidate.mode,
                reason=reason,
            )
        else:
            reason = (
                f"U_best {best_utility:.4f} <= U_current {state.current_utility:.4f}"
                f"+delta {self.delta}, no migration"
            )
            if probing_due:
                reason += "; probing triggered"
            return Action.Stay(reason=reason)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        state: PolicyState,
        candidates: List[CandidateState],
        progress: ProgressState,
        V: float,
    ) -> Action:
        """
        Single decision step implementing Algo 1 loop body.

        Args:
            state: PolicyState snapshot (p,P,t,T,r0,ckpt,cold_start,current_*, etc.)
            candidates: list of CandidateState (region×mode with L̄, price, carbon, E)
            progress: ProgressState(p,P,t,T) – used for thrifty & safety_net checks
            V: future progress value V(t) from ProgressState.future_progress_value

        Returns:
            Action: Idle | Launch | Stay | Terminate (terminate not used in single step
                    but available for extension)
        """
        # Thrifty – paper: if p>=P idle
        if progress.is_thrifty() or state.p >= state.P:
            return Action.Idle(reason=f"thrifty: p={state.p} >= P={state.P}")

        # Safety net – must be checked before ranking
        if progress.is_safety_net(state.cold_start_hr):
            cheapest_od = self._select_cheapest_od(candidates, progress, state)
            if cheapest_od:
                return Action.Launch(
                    region=cheapest_od.region,
                    mode=cheapest_od.mode,
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"cheapest OD {cheapest_od.region}"
                    ),
                )
            else:
                fallback_region = state.current_region or state.r0 or "us-east-1"
                return Action.Launch(
                    region=fallback_region,
                    mode="on_demand",
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"no OD candidates – fallback {fallback_region} OD"
                    ),
                )

        # Periodic probing trigger
        probing_due = False
        if self._should_probe(state):
            probing_due = True
            self._last_probe_t = state.t

        # Rank candidates by utility using unified_model (carbon-aware)
        if not candidates:
            return Action.Stay(
                reason="no candidates" + ("; probing due" if probing_due else "")
            )

        ranked = rank_candidates(
            candidates,
            V,
            state.cold_start_hr,
            self.carbon_price_usd_per_ton,
            self.carbon_weight,
        )

        return self._decide_from_ranked(state, progress, V, ranked, probing_due)

    def rank_and_decide(
        self,
        state: PolicyState,
        candidates: List[CandidateState],
        progress: ProgressState,
        V: float,
    ) -> Tuple[List[Tuple[CandidateState, float]], Action]:
        """
        Rank candidates and decide action.

        Uses rank_candidates from unified_model and returns both sorted list
        and chosen action (sorted high→low utility).

        Returns:
            (ranked: list[(CandidateState, utility)], action: Action)
        """
        # Thrifty and safety-net still apply but we want ranking for observability
        ranked: List[Tuple[CandidateState, float]] = []
        if candidates:
            ranked = rank_candidates(
                candidates,
                V,
                state.cold_start_hr,
                self.carbon_price_usd_per_ton,
                self.carbon_weight,
            )

        # Probe handling – update timer if due (consistent with decide)
        probing_due = False
        if self._should_probe(state):
            probing_due = True
            self._last_probe_t = state.t

        # If thrifty, return idle regardless of ranking
        if progress.is_thrifty() or state.p >= state.P:
            return ranked, Action.Idle(reason=f"thrifty: p={state.p} >= P={state.P}")

        # If safety-net, pick cheapest OD but still return ranking
        if progress.is_safety_net(state.cold_start_hr):
            cheapest_od = self._select_cheapest_od(candidates, progress, state)
            if cheapest_od:
                action = Action.Launch(
                    region=cheapest_od.region,
                    mode=cheapest_od.mode,
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"cheapest OD {cheapest_od.region}"
                    ),
                )
            else:
                fallback_region = state.current_region or state.r0 or "us-east-1"
                action = Action.Launch(
                    region=fallback_region,
                    mode="on_demand",
                    reason=(
                        f"safety_net: T-t={progress.remaining_time:.3f} < P-p+2d="
                        f"{progress.remaining_work + 2*state.cold_start_hr:.3f}, "
                        f"no OD candidates – fallback {fallback_region} OD"
                    ),
                )
            return ranked, action

        # Normal path
        action = self._decide_from_ranked(state, progress, V, ranked, probing_due)
        return ranked, action


__all__ = [
    "PolicyState",
    "Action",
    "SkyNomadPolicy",
]
