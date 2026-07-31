"""Numeric pins for the parts of the model that only monotonicity was guarding.

Mutation testing found these unprotected: the functional form of V (an
inequality passes for V = C_od*(theta/theta_tilde)**2 too), ``carbon_weight``
being ignored outright, the safety-net threshold being 1d instead of 2d, and
each of the three terms of the safety-net cost being deleted independently.
Every assertion here is a closed-form value derived from the definitions, not
from the implementation's own output.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from carbonsight_core.models import JobSpec
from carbonsight_core.spot.policy import PolicyState, SkyNomadPolicy
from carbonsight_core.spot.progress import (
    ODCandidate,
    ProgressState,
    carbon_usd_per_hr,
    compute_safety_net_total_cost,
)
from carbonsight_core.spot.scheduler_service import schedule_job
from carbonsight_core.spot.unified_model import (
    CandidateState,
    MigrationCostEstimator,
    candidate_utility,
    effectiveness,
    rank_candidates,
    total_cost_per_hr,
    utility,
)

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


# --- V(t): the functional form, not just its direction ----------------------


class TestFutureProgressValue:
    """V = C_od * theta / theta_tilde. Linear in the ratio, not squared."""

    @pytest.mark.parametrize(
        ("p", "big_p", "t", "big_t", "c_od", "expected"),
        [
            # theta = (10-0)/(20-0) = 0.5, theta~ = P/T = 0.5  -> V = C_od
            (0.0, 10.0, 0.0, 20.0, 4.0, 4.0),
            # theta = (10-2)/(20-8) = 0.6667, theta~ = 2/8 = 0.25 -> ratio 2.6667
            (2.0, 10.0, 8.0, 20.0, 3.0, 3.0 * (8 / 12) / 0.25),
            # ahead of schedule: theta = 4/12 = 0.3333, theta~ = 6/8 = 0.75
            (6.0, 10.0, 8.0, 20.0, 3.0, 3.0 * (4 / 12) / 0.75),
            # exactly on the target rate at t>0 -> back to C_od
            (5.0, 10.0, 10.0, 20.0, 7.5, 7.5),
        ],
    )
    def test_v_is_c_od_times_the_rate_ratio(self, p, big_p, t, big_t, c_od, expected):
        state = ProgressState(p=p, P=big_p, t=t, T=big_t)
        assert state.future_progress_value(c_od) == pytest.approx(expected, rel=1e-12)

    def test_v_is_linear_in_the_ratio_not_quadratic(self):
        """V must be C_od * r, not C_od * r**2, for a state where r != 1."""
        state = ProgressState(p=2.0, P=10.0, t=8.0, T=20.0)
        ratio = state.deadline_pressure / state.avg_progress
        assert ratio == pytest.approx((8 / 12) / 0.25, rel=1e-12)  # 2.6667
        value = state.future_progress_value(4.0)
        assert value == pytest.approx(4.0 * ratio, rel=1e-12)
        assert value != pytest.approx(4.0 * ratio**2, rel=1e-6)

    def test_v_ratio_holds_across_many_states(self):
        """Any functional form other than C_od*r has to break one of these."""
        for p, big_p, t, big_t in [
            (1.0, 10.0, 2.0, 20.0),
            (3.0, 10.0, 9.0, 20.0),
            (7.0, 10.0, 4.0, 20.0),
            (0.5, 4.0, 1.0, 6.0),
        ]:
            state = ProgressState(p=p, P=big_p, t=t, T=big_t)
            ratio = state.deadline_pressure / state.avg_progress
            assert state.future_progress_value(3.0) == pytest.approx(3.0 * ratio, rel=1e-12)

    def test_v_scales_linearly_with_c_od(self):
        state = ProgressState(p=1.0, P=10.0, t=5.0, T=20.0)
        assert state.future_progress_value(10.0) == pytest.approx(
            10 * state.future_progress_value(1.0), rel=1e-12
        )

    def test_v_is_scale_invariant_in_time(self):
        """Scaling p, P, t, T together leaves the ratio, hence V, unchanged."""
        base = ProgressState(p=2.0, P=10.0, t=8.0, T=20.0)
        scaled = ProgressState(p=20.0, P=100.0, t=80.0, T=200.0)
        assert scaled.future_progress_value(4.0) == pytest.approx(
            base.future_progress_value(4.0), rel=1e-12
        )

    def test_v_is_infinite_once_the_deadline_has_passed(self):
        assert math.isinf(ProgressState(p=1.0, P=10.0, t=20.0, T=20.0).future_progress_value(4.0))

    def test_v_is_zero_when_the_work_is_done(self):
        assert ProgressState(p=10.0, P=10.0, t=5.0, T=20.0).future_progress_value(4.0) == 0.0

    def test_zero_progress_after_real_elapsed_time_is_maximum_pressure(self):
        """p=0 with t>0 means the achieved rate is 0, so V must diverge.

        Falling back to P/T here would quietly reward a job that has done
        nothing with the same V as one exactly on plan.
        """
        stalled = ProgressState(p=0.0, P=10.0, t=10.0, T=20.0)
        assert stalled.avg_progress == 0.0
        assert math.isinf(stalled.future_progress_value(4.0))

    def test_t_zero_still_anchors_at_c_od(self):
        """Only t=0 gets the P/T fallback, because no rate has been observed yet."""
        assert ProgressState(p=0.0, P=10.0, t=0.0, T=20.0).future_progress_value(4.0) == 4.0


# --- carbon_weight ----------------------------------------------------------


class TestCarbonWeight:
    """A user-facing knob on JobSpec, the CLI and the API that had no coverage."""

    def test_carbon_usd_scales_linearly_with_weight(self):
        assert carbon_usd_per_hr(2.0, 50.0, 1.0) == pytest.approx(0.1)
        assert carbon_usd_per_hr(2.0, 50.0, 3.0) == pytest.approx(0.3)
        assert carbon_usd_per_hr(2.0, 50.0, 0.0) == 0.0

    def test_total_cost_adds_weighted_carbon(self):
        # 0.4 kg/hr at $50/t weight 2 -> 0.4/1000*50*2 = $0.04/hr
        assert total_cost_per_hr(1.0, 0.4, 50.0, 2.0) == pytest.approx(1.04)

    def test_weight_zero_disables_the_carbon_lever(self):
        dirty = CandidateState("dirty", "spot", 10.0, 1.0, 5.0)
        clean = CandidateState("clean", "spot", 10.0, 1.0, 0.0)
        off = rank_candidates([dirty, clean], 5.0, 0.1, 1000.0, carbon_weight=0.0)
        assert off[0][1] == pytest.approx(off[1][1])

    def test_weight_flips_the_ranking_at_fixed_carbon_price(self):
        green = CandidateState("green", "spot", 10.0, 1.20, 0.05)
        dirty = CandidateState("dirty", "spot", 10.0, 1.00, 0.90)
        # weight 1: carbon is 0.005 vs 0.09 $/hr, nowhere near the 0.20 price gap
        assert rank_candidates([green, dirty], 5.0, 0.1, 100.0, 1.0)[0][0].region == "dirty"
        # weight 1000: carbon is 5.00 vs 90.00 $/hr and swamps it
        assert rank_candidates([green, dirty], 5.0, 0.1, 100.0, 1000.0)[0][0].region == "green"

    def test_weight_reaches_the_migration_estimator(self):
        base = MigrationCostEstimator(0.02, egress_carbon_kg_per_gb=1.0, carbon_weight=1.0)
        heavy = MigrationCostEstimator(0.02, egress_carbon_kg_per_gb=1.0, carbon_weight=4.0)
        # 10 GB: $0.20 egress + 10 kg/1000*50*w
        assert base.estimate(10.0) == pytest.approx(0.2 + 0.5)
        assert heavy.estimate(10.0) == pytest.approx(0.2 + 2.0)

    def test_weight_reaches_the_safety_net(self):
        cand = ODCandidate("r", od_price_per_hr=1.0, carbon_kg_per_hr=10.0)
        light = compute_safety_net_total_cost(cand, 2.0, 0.0, 50.0, 1.0)
        heavy = compute_safety_net_total_cost(cand, 2.0, 0.0, 50.0, 10.0)
        assert light == pytest.approx((1.0 + 0.5) * 2.0)
        assert heavy == pytest.approx((1.0 + 5.0) * 2.0)

    def test_weight_reaches_schedule_job_end_to_end(self):
        """Ignoring job.carbon_weight anywhere in the chain must fail here."""
        common = dict(gpu_type="A100", gpu_count=1, duration_hours=1.0, deadline_hours=45.0)
        light = schedule_job(JobSpec(**common, carbon_weight=1.0), now=NOW)
        heavy = schedule_job(JobSpec(**common, carbon_weight=500.0), now=NOW)
        by_region_light = {c.region: u for c, u in light.ranked if c.is_spot}
        by_region_heavy = {c.region: u for c, u in heavy.ranked if c.is_spot}
        assert by_region_light != by_region_heavy
        # A dirtier region must be punished more than a cleaner one.
        dirty = max(light.inputs, key=lambda r: light.inputs[r].carbon_kg_per_hr)
        clean = min(light.inputs, key=lambda r: light.inputs[r].carbon_kg_per_hr)
        assert (by_region_light[dirty] - by_region_heavy[dirty]) > (
            by_region_light[clean] - by_region_heavy[clean]
        )


# --- safety net -------------------------------------------------------------


class TestSafetyNetThreshold:
    """Trigger is T-t < P-p + 2d. The 2 is load-bearing: d is paid twice, once
    to stop and once to start again."""

    def test_boundary_is_two_cold_starts_not_one(self):
        # P-p = 5, d = 0.5 -> threshold 6.0. A 1d threshold would be 5.5.
        state = ProgressState(p=0.0, P=5.0, t=0.0, T=5.8)
        assert state.is_safety_net(0.5) is True  # 5.8 < 6.0
        assert state.remaining_time > state.remaining_work + 1 * 0.5  # but > 1d

    @pytest.mark.parametrize(
        ("remaining_time", "expected"),
        [(5.99, True), (6.0, False), (6.01, False)],
    )
    def test_exact_threshold(self, remaining_time, expected):
        state = ProgressState(p=0.0, P=5.0, t=0.0, T=remaining_time)
        assert state.is_safety_net(0.5) is expected

    def test_zero_cold_start_reduces_to_work_versus_time(self):
        assert ProgressState(p=0.0, P=5.0, t=0.0, T=4.99).is_safety_net(0.0) is True
        assert ProgressState(p=0.0, P=5.0, t=0.0, T=5.01).is_safety_net(0.0) is False


class TestSafetyNetCost:
    """Cost(r) = (C_od + carbon$) * (P-p+d) + E. Each term pinned separately."""

    CAND = ODCandidate(
        region="r", od_price_per_hr=2.0, migration_cost=7.0, carbon_kg_per_hr=4.0
    )
    REMAINING = 3.0
    COLD = 1.0
    PRICE = 50.0
    # duration 4.0; carbon $ = 4/1000*50 = 0.2/hr; (2.0+0.2)*4 + 7 = 15.8
    EXPECTED = 15.8

    def test_all_three_terms(self):
        assert compute_safety_net_total_cost(
            self.CAND, self.REMAINING, self.COLD, self.PRICE
        ) == pytest.approx(self.EXPECTED)

    def test_price_term_is_present(self):
        without = ODCandidate("r", 0.0, 7.0, 4.0)
        assert compute_safety_net_total_cost(
            without, self.REMAINING, self.COLD, self.PRICE
        ) == pytest.approx(self.EXPECTED - 2.0 * 4.0)

    def test_migration_term_is_present(self):
        without = ODCandidate("r", 2.0, 0.0, 4.0)
        assert compute_safety_net_total_cost(
            without, self.REMAINING, self.COLD, self.PRICE
        ) == pytest.approx(self.EXPECTED - 7.0)

    def test_carbon_term_is_present(self):
        without = ODCandidate("r", 2.0, 7.0, 0.0)
        assert compute_safety_net_total_cost(
            without, self.REMAINING, self.COLD, self.PRICE
        ) == pytest.approx(self.EXPECTED - 0.2 * 4.0)

    def test_cold_start_extends_the_billed_duration(self):
        no_cold = compute_safety_net_total_cost(self.CAND, self.REMAINING, 0.0, self.PRICE)
        assert no_cold == pytest.approx((2.0 + 0.2) * 3.0 + 7.0)

    def test_carbon_can_decide_between_two_od_regions(self):
        """Same price and egress, different grids: the greener one must win."""
        dirty = ODCandidate("dirty", 2.0, 0.0, 40.0)
        green = ODCandidate("green", 2.0, 0.0, 1.0)
        policy = SkyNomadPolicy(carbon_price_usd_per_ton=5000.0)
        progress = ProgressState(p=0.0, P=3.0, t=0.0, T=3.0)
        state = PolicyState(p=0.0, P=3.0, t=0.0, T=3.0, r0="", cold_start_hr=1.0)
        candidates = [
            CandidateState(c.region, "on_demand", math.inf, c.od_price_per_hr, c.carbon_kg_per_hr)
            for c in (dirty, green)
        ]
        best, _cost = policy.cheapest_od(candidates, progress, state)
        assert best.region == "green"


# --- utility algebra --------------------------------------------------------


class TestUtilityTerms:
    def test_utility_closed_form(self):
        # V=10, Lbar=5, d=0.5 -> eta=0.9; C_total = 2 + 1/1000*50 = 2.05; E/L = 3/5
        eta = effectiveness(5.0, 0.5)
        assert eta == pytest.approx(0.9)
        assert utility(10.0, eta, 2.05, 3.0, 5.0) == pytest.approx(10 * 0.9 - 2.05 - 0.6)

    def test_candidate_utility_matches_the_closed_form(self):
        cand = CandidateState("r", "spot", 5.0, 2.0, 1.0, 3.0)
        assert candidate_utility(cand, 10.0, 0.5, 50.0, 1.0) == pytest.approx(
            10 * 0.9 - 2.05 - 0.6
        )

    def test_on_demand_drops_both_lifetime_terms(self):
        od = CandidateState("r", "on_demand", math.inf, 4.0, 0.0, 99.0)
        assert candidate_utility(od, 10.0, 0.5) == pytest.approx(10.0 - 4.0)

    def test_idle_is_the_zero_baseline(self):
        assert candidate_utility(CandidateState("idle", "idle", 0.0, 0.0, 0.0), 10.0, 0.5) == 0.0

    def test_effectiveness_is_zero_when_cold_start_eats_the_lifetime(self):
        assert effectiveness(0.4, 0.5) == 0.0
