"""Unit tests for spot/policy.py – SkyNomadPolicy (Algo1)."""

import math

import pytest

from carbonsight_core.spot.policy import Action, PolicyState, SkyNomadPolicy
from carbonsight_core.spot.progress import ProgressState
from carbonsight_core.spot.unified_model import CandidateState, rank_candidates


def _make_candidate(
    region: str,
    mode: str = "spot",
    Lbar: float = 5.0,
    price: float = 2.0,
    carbon: float = 0.0,
    migration: float = 0.0,
) -> CandidateState:
    return CandidateState(
        region=region,
        mode=mode,  # type: ignore
        mean_lifetime_hr=Lbar,
        price_per_hr=price,
        carbon_kg_per_hr=carbon,
        migration_cost=migration,
    )


def _progress(p: float, P: float, t: float, T: float) -> ProgressState:
    return ProgressState(p=p, P=P, t=t, T=T)


# ---------------------------------------------------------------------------
# 1. Thrifty idle
# ---------------------------------------------------------------------------


class TestThriftyIdle:
    def test_p_ge_P_triggers_idle(self):
        state = PolicyState(p=10, P=10, t=20, T=30, r0="us-east-1", current_utility=0.0)
        progress = _progress(p=10, P=10, t=20, T=30)
        V = 5.0
        cand = [_make_candidate("us-east-1")]
        policy = SkyNomadPolicy(delta=0.05)
        action = policy.decide(state, cand, progress, V)
        assert action.kind == "idle"
        assert "thrifty" in action.reason.lower()

    def test_p_gt_P_also_idle(self):
        state = PolicyState(p=12, P=10, t=5, T=20, r0="us-east-1", current_utility=0.0)
        progress = _progress(p=12, P=10, t=5, T=20)
        policy = SkyNomadPolicy()
        action = policy.decide(state, [_make_candidate("us-west-2")], progress, V=3.0)
        assert action.kind == "idle"

    def test_rank_and_decide_thrifty(self):
        state = PolicyState(p=10, P=10, t=20, T=30, r0="us-east-1")
        progress = _progress(p=10, P=10, t=20, T=30)
        policy = SkyNomadPolicy()
        ranked, action = policy.rank_and_decide(state, [_make_candidate("us-east-1")], progress, V=5.0)
        assert action.kind == "idle"
        # ranked may still be computed
        assert isinstance(ranked, list)


# ---------------------------------------------------------------------------
# 2. Safety net OD
# ---------------------------------------------------------------------------


class TestSafetyNet:
    def test_safety_net_triggers_cheapest_od(self):
        # T-t < P-p + 2d  -> safety net
        # P=10, p=5 => remaining 5, d=0.1 => 5+0.2=5.2, remaining time 2 => triggers
        cold = 0.1
        state = PolicyState(
            p=5, P=10, t=8, T=10, r0="us-east-1", cold_start_hr=cold, current_utility=0.0
        )
        progress = _progress(p=5, P=10, t=8, T=10)
        assert progress.is_safety_net(cold) is True

        cheap_od = _make_candidate("us-east-1", mode="on_demand", Lbar=math.inf, price=4.1, migration=0.0)
        expensive_od = _make_candidate("eu-west-1", mode="on_demand", Lbar=math.inf, price=10.0, migration=0.0)
        spot_candidate = _make_candidate("us-west-2", mode="spot", Lbar=10.0, price=1.0)

        policy = SkyNomadPolicy()
        action = policy.decide(state, [cheap_od, expensive_od, spot_candidate], progress, V=20.0)

        assert action.kind == "launch"
        assert action.mode in ("on_demand", "od")
        assert action.region == "us-east-1"  # cheapest OD
        assert "safety_net" in action.reason.lower()

    def test_safety_net_no_od_candidates_fallback(self):
        cold = 0.1
        state = PolicyState(
            p=5, P=10, t=8, T=10, r0="us-east-1", cold_start_hr=cold, current_region="us-east-1", current_utility=0.0
        )
        progress = _progress(p=5, P=10, t=8, T=10)
        # Only spot candidates – no OD
        spot = _make_candidate("us-west-2", mode="spot", Lbar=5.0, price=1.0)
        policy = SkyNomadPolicy()
        action = policy.decide(state, [spot], progress, V=20.0)
        assert action.kind == "launch"
        assert action.mode == "on_demand"  # fallback OD
        assert "safety_net" in action.reason.lower()


# ---------------------------------------------------------------------------
# 3. Prefers high Lbar region
# ---------------------------------------------------------------------------


class TestPrefersHighLbar:
    def test_high_lbar_wins_when_price_equal(self):
        state = PolicyState(
            p=2, P=10, t=2, T=20, r0="us-east-1", cold_start_hr=0.1, current_region="", current_utility=0.0
        )
        progress = _progress(p=2, P=10, t=2, T=20)
        V = 10.0

        low_lbar = _make_candidate("low-region", Lbar=2.0, price=2.0, migration=0.0)
        high_lbar = _make_candidate("high-region", Lbar=10.0, price=2.0, migration=0.0)

        # Direct ranking check
        ranked = rank_candidates([low_lbar, high_lbar], V, state.cold_start_hr)
        assert ranked[0][0].region == "high-region"
        assert ranked[0][1] > ranked[1][1]

        policy = SkyNomadPolicy(delta=0.05)
        action = policy.decide(state, [low_lbar, high_lbar], progress, V)

        assert action.kind == "launch"
        assert action.region == "high-region"

    def test_rank_and_decide_high_lbar(self):
        state = PolicyState(p=2, P=10, t=2, T=20, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=2, P=10, t=2, T=20)
        low = _make_candidate("r-low", Lbar=1.0, price=3.0)
        high = _make_candidate("r-high", Lbar=10.0, price=3.0)

        policy = SkyNomadPolicy(delta=0.01)
        ranked, action = policy.rank_and_decide(state, [low, high], progress, V=8.0)
        assert ranked[0][0].region == "r-high"
        assert action.region == "r-high"


# ---------------------------------------------------------------------------
# 4. Migration cost prevents flapping unless delta exceeded
# ---------------------------------------------------------------------------


class TestDeltaFlapping:
    def test_no_migration_if_improvement_less_than_delta(self):
        # current utility 5.0, best candidate utility only slightly higher
        state = PolicyState(
            p=2,
            P=10,
            t=2,
            T=20,
            r0="us-east-1",
            cold_start_hr=0.1,
            current_region="us-east-1",
            current_mode="spot",
            current_utility=5.0,
        )
        progress = _progress(p=2, P=10, t=2, T=20)
        V = 7.0

        # Candidate with utility 5.02 => improvement 0.02 < delta 0.05 => should Stay
        # We need to craft price to make utility ~5.02
        # Utility = V*eta - C - E/L. eta ~ (5-0.1)/5=0.98 => V*eta=6.86
        # So to get U~5.02, set C~1.84 etc.
        # Instead rely on migration cost to add: if migration high, U lower
        # Simpler: set up two candidates with same Lbar but different price/migration

        # Current placement utility is given, not recomputed; policy compares best utility vs current+delta
        # So create a candidate where utility will be 5.02
        # V=7, Lbar=5, eta=0.98 => V*eta=6.86
        # U = 6.86 - price - migration/Lbar
        # Want U=5.02 => price + migration/L ≈1.84
        # Choose price=1.5 migration=1.7 => migration/L=0.34 => total 1.84
        cand_small_improve = _make_candidate(
            "us-west-2", Lbar=5.0, price=1.5, migration=1.7
        )

        policy = SkyNomadPolicy(delta=0.05, probe_interval_hr=100)  # avoid probing interference
        action = policy.decide(state, [cand_small_improve], progress, V)

        # Compute expected utility to assert our math is correct
        from carbonsight_core.spot.unified_model import candidate_utility

        u = candidate_utility(cand_small_improve, V, state.cold_start_hr)
        # Should be ~5.02
        assert u == pytest.approx(5.02, abs=0.05)

        assert action.kind == "stay"
        assert "delta" in action.reason.lower() or "no migration" in action.reason.lower()

    def test_migration_when_delta_exceeded(self):
        state = PolicyState(
            p=2,
            P=10,
            t=2,
            T=20,
            r0="us-east-1",
            cold_start_hr=0.1,
            current_region="us-east-1",
            current_mode="spot",
            current_utility=5.0,
        )
        progress = _progress(p=2, P=10, t=2, T=20)
        V = 7.0

        # Candidate with utility 5.5 => improvement 0.5 > delta 0.05 => Launch
        # V*eta=6.86, need C+E/L = 1.36 => price 1.0 migration 1.8 => 1.0+0.36=1.36 => U=5.5
        cand_big_improve = _make_candidate("us-west-2", Lbar=5.0, price=1.0, migration=1.8)

        policy = SkyNomadPolicy(delta=0.05, probe_interval_hr=100)
        action = policy.decide(state, [cand_big_improve], progress, V)

        from carbonsight_core.spot.unified_model import candidate_utility

        u = candidate_utility(cand_big_improve, V, state.cold_start_hr)
        assert u == pytest.approx(5.5, abs=0.05)

        assert action.kind == "launch"
        assert action.region == "us-west-2"

    def test_migration_cost_reduces_utility(self):
        # Same Lbar and price, but one has high migration cost
        state = PolicyState(p=1, P=10, t=1, T=20, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=1, P=10, t=1, T=20)
        V = 10.0

        no_mig = _make_candidate("a", Lbar=5.0, price=2.0, migration=0.0)
        high_mig = _make_candidate("b", Lbar=5.0, price=2.0, migration=10.0)

        ranked = rank_candidates([no_mig, high_mig], V, state.cold_start_hr)
        # no migration should win
        assert ranked[0][0].region == "a"
        assert ranked[0][1] > ranked[1][1]

        policy = SkyNomadPolicy(delta=0.01)
        action = policy.decide(state, [no_mig, high_mig], progress, V)
        assert action.region == "a"


# ---------------------------------------------------------------------------
# 5. Prefers cheaper when Lbar equal
# ---------------------------------------------------------------------------


class TestCheaperWhenLbarEqual:
    def test_cheaper_wins(self):
        state = PolicyState(p=1, P=10, t=1, T=20, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=1, P=10, t=1, T=20)
        V = 10.0

        expensive = _make_candidate("expensive", Lbar=5.0, price=5.0, migration=0.0)
        cheap = _make_candidate("cheap", Lbar=5.0, price=1.0, migration=0.0)

        ranked = rank_candidates([expensive, cheap], V, state.cold_start_hr)
        assert ranked[0][0].region == "cheap"

        policy = SkyNomadPolicy(delta=0.01)
        action = policy.decide(state, [expensive, cheap], progress, V)
        assert action.kind == "launch"
        assert action.region == "cheap"

    def test_cheaper_wins_with_carbon_zero_and_same_migration(self):
        state = PolicyState(p=0, P=10, t=0, T=20, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=0, P=10, t=0, T=20)
        V = 6.0

        c1 = _make_candidate("r1", Lbar=4.0, price=3.0, carbon=0.0, migration=0.5)
        c2 = _make_candidate("r2", Lbar=4.0, price=2.0, carbon=0.0, migration=0.5)

        policy = SkyNomadPolicy(delta=0.0)
        ranked, action = policy.rank_and_decide(state, [c1, c2], progress, V)
        assert ranked[0][0].region == "r2"
        assert action.region == "r2"


# ---------------------------------------------------------------------------
# Additional sanity checks for rank_and_decide API
# ---------------------------------------------------------------------------


class TestRankAndDecideAPI:
    def test_returns_tuple(self):
        state = PolicyState(p=1, P=10, t=1, T=10, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=1, P=10, t=1, T=10)
        cands = [_make_candidate("us-east-1", price=2.0), _make_candidate("us-west-2", price=1.0)]
        policy = SkyNomadPolicy()

        ranked, action = policy.rank_and_decide(state, cands, progress, V=8.0)

        assert isinstance(ranked, list)
        assert len(ranked) == 2
        assert all(isinstance(t, tuple) and len(t) == 2 for t in ranked)
        # sorted descending
        assert ranked[0][1] >= ranked[1][1]
        assert isinstance(action, Action)

    def test_action_has_reason(self):
        state = PolicyState(p=0, P=5, t=0, T=10, r0="us-east-1", cold_start_hr=0.1, current_utility=0.0)
        progress = _progress(p=0, P=5, t=0, T=10)
        policy = SkyNomadPolicy()

        _, action = policy.rank_and_decide(state, [], progress, V=5.0)
        assert isinstance(action.reason, str)
        assert len(action.reason) > 0
