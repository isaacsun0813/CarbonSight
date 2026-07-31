"""
Unit tests for spot lifetime prediction via Nelson-Aalen hazard.

Per task spec: hand-crafted e/c example verifying hazard and survival monotonicity.

We use a small dataset:
    l=1: e=1, c=0
    l=2: e=2, c=1
    l=3: e=1, c=0
    l=5: e=0, c=2

Expected:
    n(5)=2, n(3)=3, n(2)=6, n(1)=7
    h(1)=1/7≈0.1428, h(2)=2/6≈0.333, h(3)=1/3≈0.333, h(5)=0
    H monotonic increasing, S decreasing.

Tests cover:
- LifetimeStats aggregation
- compute_at_risk, compute_hazard, compute_cumulative_hazard, compute_survival
- expected_remaining decreasing tendency, volatility adjustment
- predict_remaining_lifetime
- compute_gamma and gamma_star
"""

import math

import pytest

from carbonsight_core.spot.lifetime import (
    LifetimeStats,
    compute_at_risk,
    compute_cumulative_hazard,
    compute_gamma,
    compute_gamma_star,
    compute_hazard,
    compute_survival,
    compute_volatility_adjusted_survival,
    compute_volatility_adjusted_survival_from_survival,
    expected_remaining,
    predict_remaining_lifetime,
)


# ---------------------------------------------------------------------------
# Hand-crafted fixture
# ---------------------------------------------------------------------------

HAND_STATS = {
    1: (1, 0),
    2: (2, 1),
    3: (1, 0),
    5: (0, 2),
}

# Manually derived at-risk:
# n(5)=0+2=2
# n(3)=1+2=3
# n(2)= (2+1)+3=6
# n(1)= (1+0)+6=7
EXPECTED_N = {
    1: 7,
    2: 6,
    3: 3,
    5: 2,
}

EXPECTED_H = {
    1: 1 / 7,
    2: 2 / 6,
    3: 1 / 3,
    5: 0.0,
}

# Cumulative hazard approx: sum
# H1=0.142857...
# H2≈0.47619
# H3≈0.8095238
# H5≈0.8095238
def _expected_cumulative():
    h = [EXPECTED_H[k] for k in sorted(EXPECTED_H)]
    cum = {}
    running = 0.0
    for k, hv in zip(sorted(EXPECTED_H), h):
        running += hv
        cum[k] = running
    return cum


EXPECTED_CUM = _expected_cumulative()
EXPECTED_S = {k: math.exp(-v) for k, v in EXPECTED_CUM.items()}


class TestLifetimeStats:
    def test_from_dict_and_aggregation(self):
        ls = LifetimeStats(HAND_STATS)
        assert ls.total_events() == 1 + 2 + 1 + 0
        assert ls.total_censored() == 0 + 1 + 0 + 2
        assert ls.total_observations() == 7
        assert set(ls.sorted_lifetimes()) == {1, 2, 3, 5}

    def test_add(self):
        ls = LifetimeStats()
        ls.add(2, preempted=True)
        ls.add(2, preempted=False)
        ls.add(2, preempted=True)
        assert ls.stats[2.0] == (2, 1)

    def test_from_observations(self):
        obs = [(1, True), (2, True), (2, True), (2, False), (3, True), (5, False), (5, False)]
        ls = LifetimeStats.from_observations(obs)
        assert ls.stats == {1.0: (1, 0), 2.0: (2, 1), 3.0: (1, 0), 5.0: (0, 2)}


class TestAtRiskAndHazard:
    def test_at_risk(self):
        n = compute_at_risk(HAND_STATS)
        assert n == EXPECTED_N

    def test_hazard_values(self):
        h = compute_hazard(HAND_STATS)
        for k in EXPECTED_H:
            assert h[k] == pytest.approx(EXPECTED_H[k], rel=1e-9)

    def test_hazard_bounds(self):
        h = compute_hazard(HAND_STATS)
        for v in h.values():
            assert 0.0 <= v <= 1.0

    def test_hazard_accepts_lifetime_stats_object(self):
        ls = LifetimeStats(HAND_STATS)
        h1 = compute_hazard(HAND_STATS)
        h2 = compute_hazard(ls)
        assert h1 == h2

    def test_empty_stats(self):
        assert compute_hazard({}) == {}
        assert compute_at_risk({}) == {}
        assert compute_cumulative_hazard({}) == {}
        assert compute_survival({}) == {}
        assert predict_remaining_lifetime({}, age=0) == 0.0


class TestCumulativeAndSurvival:
    def test_cumulative_monotonic_increasing(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        sorted_keys = sorted(cum.keys())
        for i in range(1, len(sorted_keys)):
            assert cum[sorted_keys[i]] >= cum[sorted_keys[i - 1]] - 1e-12

    def test_cumulative_values(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        for k in EXPECTED_CUM:
            assert cum[k] == pytest.approx(EXPECTED_CUM[k], rel=1e-9)

    def test_survival_decreasing(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)
        sorted_keys = sorted(surv.keys())
        # survival in (0,1]
        for v in surv.values():
            assert 0.0 < v <= 1.0 + 1e-12
        # non-increasing
        for i in range(1, len(sorted_keys)):
            assert surv[sorted_keys[i]] <= surv[sorted_keys[i - 1]] + 1e-12

    def test_survival_values(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)
        for k in EXPECTED_S:
            assert surv[k] == pytest.approx(EXPECTED_S[k], rel=1e-9)

    def test_survival_empty(self):
        assert compute_survival({}) == {}


class TestExpectedRemaining:
    """L(a) is the area under the survival step function, so gaps count.

    HAND_STATS has lifetimes 1, 2, 3, 5 — note the gap between 3 and 5. S is flat
    on each [l_i, l_{i+1}), so the rectangles are:

        [0,1) width 1 at S=1        [1,2) width 1 at S1
        [2,3) width 1 at S2         [3,5) width 2 at S3

    Values below are computed by hand from S_i = exp(-H_i) with
    H = 1/7, 1/7+1/3, 1/7+2/3 and pinned as literals, so that changing the
    weighting scheme has to change the numbers rather than the derivation.
    """

    S1 = 0.8668778997501816  # exp(-1/7)
    S2 = 0.6211451576154515  # exp(-10/21)
    S3 = 0.4450699538427624  # exp(-17/21)

    def test_expected_remaining_at_age_zero(self):
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        # 1 + S1 + S2 + 2*S3
        assert expected_remaining(surv, 0) == pytest.approx(3.3781629650511578, rel=1e-12)

    def test_expected_remaining_at_age_one(self):
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        # (S1 + S2 + 2*S3) / S1
        assert expected_remaining(surv, 1) == pytest.approx(2.7433655486389732, rel=1e-12)

    def test_expected_remaining_at_age_two(self):
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        # (S2 + 2*S3) / S2
        assert expected_remaining(surv, 2) == pytest.approx(2.4330626211475783, rel=1e-12)

    def test_final_gap_is_weighted_by_its_width(self):
        """At age 3 only the width-2 gap [3,5) is left, so L = 2 exactly."""
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        assert expected_remaining(surv, 3) == pytest.approx(2.0, rel=1e-12)

    def test_unit_width_sum_is_not_what_we_compute(self):
        """Guards the actual bug: sum(S) ignores the 3->5 gap and the [0,1) head."""
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        assert expected_remaining(surv, 0) != pytest.approx(sum(surv.values()), rel=1e-6)

    @pytest.mark.parametrize("last", [4, 40, 400])
    def test_lbar_scales_with_lifetime_magnitude(self, last):
        """The regression: [1,2,3,4], [1,2,3,40] and [1,2,3,400] all gave 1.7998.

        Four uncensored observations, so n = 4, 3, 2, 1 and h = 1/4, 1/3, 1/2, 1.
        Closed form, derived from the hazard definitions rather than from
        ``expected_remaining`` itself:

            L(0) = 1*[0,1) + S1*[1,2) + S2*[2,3) + S3*(last - 3)
        """
        s1 = math.exp(-1 / 4)
        s2 = math.exp(-(1 / 4 + 1 / 3))
        s3 = math.exp(-(1 / 4 + 1 / 3 + 1 / 2))
        expected = 1.0 + s1 + s2 + s3 * (last - 3)

        stats = LifetimeStats.from_observations([(x, True) for x in (1, 2, 3, last)])
        assert predict_remaining_lifetime(stats, age=0) == pytest.approx(expected, rel=1e-12)

    def test_lbar_tracks_the_sample_mean(self):
        """A long-tailed sample: restricted L(0) should land near the sample mean."""
        sample = [1, 1, 2, 2, 3, 4, 6, 9, 14, 20]
        stats = LifetimeStats.from_observations([(x, True) for x in sample])
        lbar = predict_remaining_lifetime(stats, age=0)
        assert lbar == pytest.approx(sum(sample) / len(sample), rel=0.35)

    def test_expected_remaining_decreasing_with_age_tendency(self):
        surv = compute_survival(compute_cumulative_hazard(compute_hazard(HAND_STATS)))
        assert expected_remaining(surv, 5) == 0.0  # past the last lifetime
        assert expected_remaining(surv, 0) > expected_remaining(surv, 2) >= 0

    def test_expected_remaining_age_before_first(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)
        # age -1 < min => S_a =1
        rem_neg = expected_remaining(surv, -1)
        rem0 = expected_remaining(surv, 0)
        assert rem_neg == pytest.approx(rem0)


class TestVolatilityGamma:
    def test_compute_gamma_neutral_when_zero_hazard(self):
        assert compute_gamma(3, 0.0) == 1.0
        assert compute_gamma(0, 0.0) == 1.0

    def test_compute_gamma_basic(self):
        # e_W=2, hazard_sum=1.0 => gamma=2
        assert compute_gamma(2, 1.0) == pytest.approx(2.0)
        assert compute_gamma(1, 2.0) == pytest.approx(0.5)

    def test_gamma_star_max(self):
        assert compute_gamma_star([]) == 1.0
        assert compute_gamma_star([0.5, 2.0, 1.2]) == pytest.approx(2.0)
        assert compute_gamma_star([0.8]) == pytest.approx(0.8)

    def test_volatility_adjusted_survival_more_pessimistic_when_gamma_gt1(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)

        gamma_star = 2.0  # volatile
        adj = compute_volatility_adjusted_survival(cum, gamma_star)
        # S̃ = S^{gamma*} => smaller when gamma>1 and S<1
        for l in surv:
            if surv[l] < 1.0:
                assert adj[l] <= surv[l] + 1e-12
            # check equivalence to S^{gamma}
            assert adj[l] == pytest.approx(surv[l] ** gamma_star, rel=1e-9)

    def test_volatility_adjusted_from_survival_helper(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)
        gamma_star = 1.5
        adj1 = compute_volatility_adjusted_survival(cum, gamma_star)
        adj2 = compute_volatility_adjusted_survival_from_survival(surv, gamma_star)
        for k in adj1:
            assert adj1[k] == pytest.approx(adj2[k], rel=1e-9)

    def test_volatility_gamma_star_zero_means_full_survival(self):
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        adj = compute_volatility_adjusted_survival(cum, gamma_star=0.0)
        for v in adj.values():
            assert v == pytest.approx(1.0)


class TestPredictRemainingLifetime:
    def test_predict_without_gamma(self):
        pred = predict_remaining_lifetime(HAND_STATS, age=0)
        h = compute_hazard(HAND_STATS)
        cum = compute_cumulative_hazard(h)
        surv = compute_survival(cum)
        expected = expected_remaining(surv, 0)
        assert pred == pytest.approx(expected, rel=1e-9)

    def test_predict_with_gamma_adjustment(self):
        # With high gamma*, expected remaining should be smaller
        pred_normal = predict_remaining_lifetime(HAND_STATS, age=1, gamma_star=None)
        pred_volatile = predict_remaining_lifetime(HAND_STATS, age=1, gamma_star=2.5)
        assert pred_volatile <= pred_normal + 1e-12
        assert pred_normal > 0

    def test_predict_age_beyond_max(self):
        assert predict_remaining_lifetime(HAND_STATS, age=10) == 0.0

    def test_predict_empty(self):
        assert predict_remaining_lifetime({}, age=0) == 0.0

    def test_predict_works_with_lifetime_stats_instance(self):
        ls = LifetimeStats(HAND_STATS)
        p1 = predict_remaining_lifetime(HAND_STATS, age=2)
        p2 = predict_remaining_lifetime(ls, age=2)
        assert p1 == pytest.approx(p2)

    def test_heavy_tailed_vs_short_lived(self):
        """
        Two regions: short-lived has early preemptions, long-lived has later.
        Long-lived should have larger expected remaining.
        """
        short = {1: (5, 0), 2: (2, 0)}  # most die at 1
        long = {5: (1, 0), 10: (2, 1), 20: (0, 3)}  # survive longer
        pred_short = predict_remaining_lifetime(short, age=0)
        pred_long = predict_remaining_lifetime(long, age=0)
        assert pred_long > pred_short
