"""Spot backtest harness: baselines, accounting, and that the policy is really driving.

The headline numbers come out of here, so these tests exist to make the harness
hard to flatter. Each of the following mutations must turn at least one of them
red: charging no egress, never idling, dropping the delta anti-flapping rule,
and swapping the baseline for a weaker one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from carbonsight_core.backtest.spot_runner import (
    CARBON_BREAKEVEN_USD_PER_TON,
    DEFAULT_REGIONS,
    DEFAULT_SEEDS,
    PATTERNS,
    _build_traces,
    _lifetimes,
    _run_skynomad,
    _run_up_multi,
    _run_up_single,
    _run_wait_enabled_pin,
    default_pinned_region,
    run_spot_backtest,
    run_spot_backtest_multi,
)
from carbonsight_core.spot.lifetime import MIN_LIFETIME_HR
from carbonsight_core.spot.unified_model import MigrationCostEstimator

START = datetime(2026, 1, 1, tzinfo=UTC)
WORK = 8.0
DEADLINE = 12.0        # tight: almost every hour must be spent running
SLACK_DEADLINE = 40.0  # generous: the job runs ahead of schedule and can wait


@pytest.fixture(scope="module")
def harness():
    """Small deterministic world: 3 regions, one per availability pattern."""
    regions = ["r-good", "r-flaky", "r-dead"]
    trackers = _build_traces(regions, days=3, seed=7, start=START)
    prices = {"r-good": (1.0, 3.0), "r-flaky": (0.8, 2.8), "r-dead": (0.5, 2.5)}
    carbon = {"r-good": 0.10, "r-flaky": 0.20, "r-dead": 0.30}
    return regions, trackers, prices, carbon, _lifetimes(trackers)


def _sky(harness, *, deadline=DEADLINE, checkpoint_gb=100.0, **kwargs):
    regions, trackers, prices, carbon, lifetimes = harness
    return _run_skynomad(
        regions,
        trackers,
        prices,
        carbon,
        lifetimes,
        START,
        WORK,
        deadline,
        MigrationCostEstimator(egress_usd_per_gb=0.02),
        checkpoint_gb,
        50.0,
        **kwargs,
    )


# --- trace generation -------------------------------------------------------


def test_patterns_produce_distinguishable_uptime():
    trackers = _build_traces(["a", "b", "c"], days=5, seed=3, start=START)
    up = {
        r: sum(o.outcome for o in t.get_observations(r)) / len(t.get_observations(r))
        for r, t in trackers.items()
    }
    assert up["a"] > up["b"] > up["c"]
    assert up["a"] > 0.85 and up["c"] < 0.35


def test_lifetimes_rank_with_uptime():
    trackers = _build_traces(["a", "b", "c"], days=5, seed=3, start=START)
    lifetimes = _lifetimes(trackers)
    assert lifetimes["a"] > lifetimes["b"] > lifetimes["c"]


def test_traces_are_seed_deterministic():
    a = _build_traces(["x"], days=2, seed=5, start=START)
    b = _build_traces(["x"], days=2, seed=5, start=START)
    c = _build_traces(["x"], days=2, seed=6, start=START)
    assert [o.outcome for o in a["x"].get_observations("x")] == [
        o.outcome for o in b["x"].get_observations("x")
    ]
    assert [o.outcome for o in a["x"].get_observations("x")] != [
        o.outcome for o in c["x"].get_observations("x")
    ]


# --- baselines --------------------------------------------------------------


def test_default_pin_is_a_high_uptime_region_not_a_dud():
    """The headline baseline must be a region someone would actually choose."""
    prices = {r: (1.0, 2.0 + i * 0.1) for i, r in enumerate(DEFAULT_REGIONS)}
    chosen = default_pinned_region(DEFAULT_REGIONS, prices)
    idx = DEFAULT_REGIONS.index(chosen)
    assert PATTERNS[idx % len(PATTERNS)][0] == 0.95


def test_default_pin_breaks_ties_deterministically():
    prices = {r: (1.0, 2.0) for r in DEFAULT_REGIONS}
    assert default_pinned_region(DEFAULT_REGIONS, prices) == default_pinned_region(
        DEFAULT_REGIONS, prices
    )


def test_up_single_pays_on_demand_when_spot_is_down(harness):
    regions, trackers, prices, carbon, _lt = harness
    cost, kg, met = _run_up_single(
        "r-dead", trackers, prices, carbon, START, WORK, DEADLINE
    )
    # r-dead is up ~20% of the time, so most hours are billed at on-demand.
    assert met is True
    assert cost > prices["r-dead"][0] * WORK
    assert kg == pytest.approx(carbon["r-dead"] * WORK)


def test_up_single_never_migrates_so_a_good_pin_is_cheap(harness):
    regions, trackers, prices, carbon, _lt = harness
    good, _kg, _met = _run_up_single("r-good", trackers, prices, carbon, START, WORK, DEADLINE)
    dead, _kg2, _met2 = _run_up_single("r-dead", trackers, prices, carbon, START, WORK, DEADLINE)
    assert good < dead


def test_up_multi_charges_egress_per_move(harness):
    regions, trackers, prices, carbon, _lt = harness
    free, _kg, moves_free, _m = _run_up_multi(
        regions, trackers, prices, carbon, START, WORK, DEADLINE, migration_cost=0.0
    )
    dear, _kg2, moves_dear, _m2 = _run_up_multi(
        regions, trackers, prices, carbon, START, WORK, DEADLINE, migration_cost=5.0
    )
    assert moves_free == moves_dear > 0  # same route, different bill
    assert dear == pytest.approx(free + 5.0 * moves_dear)


# --- SkyNomad accounting ----------------------------------------------------


def test_skynomad_pays_egress_when_it_migrates(harness):
    """Mutation guard: zeroing the egress charge must fail here."""
    run = _sky(harness, deadline=SLACK_DEADLINE, checkpoint_gb=100.0)
    assert run.migrations > 0
    assert run.egress > 0
    assert run.egress == pytest.approx(0.02 * 100.0 * run.migrations)


def test_skynomad_charges_no_egress_without_a_checkpoint(harness):
    run = _sky(harness, checkpoint_gb=0.0)
    assert run.egress == 0.0


def test_bigger_checkpoints_suppress_migration(harness):
    """E/Lbar is a real term: a huge checkpoint must deter moving."""
    cheap = _sky(harness, deadline=SLACK_DEADLINE, checkpoint_gb=1.0)
    dear = _sky(harness, deadline=SLACK_DEADLINE, checkpoint_gb=20_000.0)
    assert dear.migrations < cheap.migrations


def test_delta_rule_suppresses_flapping(harness):
    """Mutation guard: dropping the delta comparison must fail here.

    Checkpoint 0 so that E/Lbar is not already doing the suppressing and the
    delta rule is the only thing standing between the policy and a migration.
    """
    twitchy = _sky(harness, checkpoint_gb=0.0, delta=0.0)
    sticky = _sky(harness, checkpoint_gb=0.0, delta=1e6)
    assert twitchy.migrations > sticky.migrations
    assert sticky.migrations == 0


def test_skynomad_idles_when_it_is_ahead_of_schedule(harness):
    """Mutation guard: forcing it to run every hour must fail here.

    Running on schedule drives theta below theta-tilde, so V falls below what an
    hour costs and waiting scores better than working. Slack is what buys that
    option, so the generous deadline must idle far more than the tight one.
    """
    tight = _sky(harness, deadline=DEADLINE)
    slack = _sky(harness, deadline=SLACK_DEADLINE)
    assert slack.idle_hours > tight.idle_hours
    assert slack.idle_hours > 0
    assert slack.met_deadline is True  # idling spends slack, it does not miss


def test_idling_is_bounded_by_the_deadline(harness):
    """Idle hours come out of the same clock as working hours."""
    run = _sky(harness, deadline=SLACK_DEADLINE)
    assert run.idle_hours + WORK <= SLACK_DEADLINE


def test_progress_only_accrues_on_hours_actually_run(harness):
    run = _sky(harness)
    if run.met_deadline:
        # Cost is charged per running hour, so it cannot be below the cheapest
        # spot rate times the work actually required.
        assert run.cost >= 0.5 * WORK * 0.999


def test_carbon_tracks_the_regions_chosen(harness):
    regions, trackers, prices, _carbon, lifetimes = harness
    clean = dict.fromkeys(regions, 0.01)
    dirty = dict.fromkeys(regions, 5.0)
    est = MigrationCostEstimator(egress_usd_per_gb=0.02)
    args = (trackers, prices)
    low = _run_skynomad(
        regions, *args, clean, lifetimes, START, WORK, DEADLINE, est, 100.0, 50.0
    )
    high = _run_skynomad(
        regions, *args, dirty, lifetimes, START, WORK, DEADLINE, est, 50.0, 50.0
    )
    assert high.carbon_kg > low.carbon_kg


# --- end-to-end aggregation -------------------------------------------------


def test_run_spot_backtest_is_seed_deterministic():
    a = run_spot_backtest(n=4, days=4, seed=1)
    b = run_spot_backtest(n=4, days=4, seed=1)
    assert a.skynomad_cost_mean == b.skynomad_cost_mean
    assert a.migrations_mean == b.migrations_mean


def test_run_spot_backtest_varies_with_seed():
    a = run_spot_backtest(n=4, days=4, seed=1)
    b = run_spot_backtest(n=4, days=4, seed=2)
    assert a.skynomad_cost_mean != b.skynomad_cost_mean


def test_savings_are_computed_against_the_stated_baseline():
    r = run_spot_backtest(n=4, days=4, seed=1)
    for baseline, pct in (
        (r.up_single_default_cost_mean, r.cost_savings_vs_up_single_default_pct),
        (r.up_single_best_cost_mean, r.cost_savings_vs_up_single_best_pct),
        (r.up_multi_cost_mean, r.cost_savings_vs_up_multi_pct),
    ):
        assert pct == pytest.approx((1 - r.skynomad_cost_mean / baseline) * 100)


def test_best_single_pin_is_no_worse_than_the_default_pin():
    """Otherwise 'best' is mislabelled and the stricter baseline is not stricter."""
    r = run_spot_backtest(n=4, days=4, seed=1)
    assert r.up_single_best_cost_mean <= r.up_single_default_cost_mean + 1e-9
    assert r.cost_savings_vs_up_single_best_pct <= (
        r.cost_savings_vs_up_single_default_pct + 1e-9
    )


def test_rotating_baseline_is_reported_but_not_the_headline():
    """Rotating over all pins includes 20%-uptime regions and flatters the result."""
    r = run_spot_backtest(n=8, days=4, seed=1)
    assert r.up_single_rotating_cost_mean > r.up_single_best_cost_mean


def test_result_carries_the_constant_carbon_caveat():
    r = run_spot_backtest(n=2, days=4, seed=1)
    assert any("constant per region" in c for c in r.caveats)


def test_multi_seed_reports_mean_and_stdev():
    multi = run_spot_backtest_multi(n=3, days=4, seeds=(1, 2, 3))
    assert multi.seeds == [1, 2, 3]
    assert len(multi.per_seed) == 3
    mean, sd = multi.metrics["skynomad_cost_mean"]
    values = [r.skynomad_cost_mean for r in multi.per_seed]
    assert mean == pytest.approx(sum(values) / 3)
    assert sd > 0
    assert multi.mean("skynomad_cost_mean") == mean
    assert multi.stdev("skynomad_cost_mean") == sd


def test_single_seed_multi_has_zero_spread():
    multi = run_spot_backtest_multi(n=2, days=4, seeds=(9,))
    assert multi.stdev("skynomad_cost_mean") == 0.0


def test_deadline_met_is_a_percentage():
    r = run_spot_backtest(n=4, days=4, seed=1)
    assert 0.0 <= r.deadline_met_pct <= 100.0
    assert 0.0 <= r.up_multi_deadline_met_pct <= 100.0


# --- the wait-enabled pin: the honest floor ---------------------------------


class TestWaitEnabledPin:
    """UP-single and UP-multi run every hour unconditionally, so they buy $4.10
    on-demand whenever spot is down and never touch their slack. SkyNomad may
    idle for free. That asymmetry, not region selection, produced the old
    headline. This pin closes it while staying strictly less informed."""

    def test_pin_waits_instead_of_buying_on_demand(self, harness):
        regions, trackers, prices, carbon, _lt = harness
        cost, _kg, idle, met = _run_wait_enabled_pin(
            "r-flaky", trackers, prices, carbon, START, WORK, SLACK_DEADLINE
        )
        always_on, _kg2, _met2 = _run_up_single(
            "r-flaky", trackers, prices, carbon, START, WORK, SLACK_DEADLINE
        )
        assert idle > 0
        assert met is True
        assert cost < always_on

    def test_with_ample_slack_the_pin_reaches_the_pure_spot_floor(self, harness):
        """Given enough slack it never touches on-demand, so cost is exactly
        work_hours x spot price — a provable floor, identical on every seed."""
        regions, trackers, prices, carbon, _lt = harness
        cost, _kg, _idle, met = _run_wait_enabled_pin(
            "r-good", trackers, prices, carbon, START, WORK, SLACK_DEADLINE
        )
        assert met is True
        assert cost == pytest.approx(prices["r-good"][0] * WORK)

    def test_pin_still_buys_on_demand_when_the_safety_net_fires(self, harness):
        """It waits, but not off a cliff: a tight deadline forces on-demand."""
        regions, trackers, prices, carbon, _lt = harness
        cost, _kg, _idle, met = _run_wait_enabled_pin(
            "r-dead", trackers, prices, carbon, START, WORK, WORK + 1.0
        )
        assert met is True
        assert cost > prices["r-dead"][0] * WORK  # some hours billed at OD

    def test_pin_is_reported_and_is_the_headline(self):
        r = run_spot_backtest(n=4, days=4, seed=1)
        assert r.wait_pin_cost_mean > 0
        assert r.cost_savings_vs_wait_pin_pct == pytest.approx(
            (1 - r.skynomad_cost_mean / r.wait_pin_cost_mean) * 100
        )

    def test_pin_is_a_tighter_baseline_than_the_always_on_pins(self):
        """If it were not, it would not be worth reporting."""
        r = run_spot_backtest(n=8, days=6, seed=3)
        assert r.wait_pin_cost_mean <= r.up_single_default_cost_mean + 1e-9
        assert r.cost_savings_vs_wait_pin_pct <= r.cost_savings_vs_up_single_default_pct + 1e-9


# --- causal lifetime fitting ------------------------------------------------


class TestCausalLifetimeFit:
    def test_fit_ignores_history_after_the_cutoff(self):
        trackers = _build_traces(["a", "b", "c"], days=6, seed=4, start=START)
        early = _lifetimes(trackers, cutoff=START + timedelta(hours=24))
        full = _lifetimes(trackers)
        assert early != full

    def test_a_workload_at_hour_zero_gets_no_signal(self):
        """No history means the floor, not a lifetime read off the future."""
        trackers = _build_traces(["a", "b", "c"], days=6, seed=4, start=START)
        cold = _lifetimes(trackers, cutoff=START)
        assert set(cold.values()) == {MIN_LIFETIME_HR}

    def test_more_history_sharpens_the_estimate(self):
        trackers = _build_traces(["a", "b", "c"], days=8, seed=4, start=START)
        short = _lifetimes(trackers, cutoff=START + timedelta(hours=12))
        long = _lifetimes(trackers, cutoff=START + timedelta(hours=168))
        # The high-uptime region should separate from the low-uptime one once
        # there is enough history to see it.
        assert long["a"] > long["c"]
        assert (long["a"] - long["c"]) > (short["a"] - short["c"])


# --- seed selection ---------------------------------------------------------


def test_default_seeds_are_a_contiguous_range():
    """A hand-picked tuple that beats other seed sets 5/5 is selection bias."""
    assert DEFAULT_SEEDS == tuple(range(1, len(DEFAULT_SEEDS) + 1))
    assert len(DEFAULT_SEEDS) >= 8


# --- carbon breakeven -------------------------------------------------------


def test_carbon_breakeven_price():
    """Derives CARBON_BREAKEVEN_USD_PER_TON rather than trusting the constant.

    Carbon can only change a migration decision once the dollarised spread
    between the dirtiest and cleanest region exceeds the anti-flapping delta.
    """
    from datetime import UTC

    from carbonsight_core.models import JobSpec
    from carbonsight_core.providers.carbon import SyntheticCarbonProvider

    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=30.0)
    provider = SyntheticCarbonProvider()
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    kg = [
        provider.get_kg_per_hr(job, [(r, 1.0)], t0, t0 + timedelta(hours=45))
        for r in DEFAULT_REGIONS
    ]
    spread = max(kg) - min(kg)
    delta = 0.05
    breakeven = delta * 1000.0 / spread
    assert breakeven == pytest.approx(CARBON_BREAKEVEN_USD_PER_TON, rel=0.02)
    # And confirm the consequence: at the shipped default the lever is inert.
    assert spread / 1000.0 * 50.0 < delta
