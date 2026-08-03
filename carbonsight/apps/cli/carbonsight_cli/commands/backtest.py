"""carbonsight backtest run --days 180 --n 1000"""

import json

import typer
from carbonsight_core.backtest.runner import run_backtest
from carbonsight_core.backtest.spot_runner import (
    CARBON_BREAKEVEN_USD_PER_TON,
    DEFAULT_SEEDS,
    run_spot_backtest_multi,
)

backtest_group = typer.Typer(help="Backtesting harness")


@backtest_group.command("run")
def run_backtest_cmd(
    days: int = typer.Option(
        180, "--days", help="Backtest period in days (--spot uses 14 unless overridden)"
    ),
    n: int = typer.Option(1000, "--n", help="Number of synthetic workloads (per seed for --spot)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    json_out: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
    spot: bool = typer.Option(False, "--spot", help="Use spot-aware runner (evictions + carbon)"),
    carbon_price: float = typer.Option(50.0, "--carbon-price", help="USD/ton for spot backtest"),
    seeds: str = typer.Option(
        "",
        "--seeds",
        help=(
            f"Comma-separated seeds for --spot. Default is {len(DEFAULT_SEEDS)} seeds "
            "so the reported spread is real; pass one seed to disable it."
        ),
    ),
) -> None:
    """Generate workloads and output savings, regret, rank accuracy."""
    if spot:
        # A single seed reports +/- 0.00, which reads as precision it does not
        # have. Default to the full seed set unless the caller asks for one.
        seed_list = tuple(int(x) for x in seeds.split(",")) if seeds else DEFAULT_SEEDS
        spot_days = days if days != 180 else 14  # 180d of hourly traces is not the intent
        multi = run_spot_backtest_multi(
            n=n, days=spot_days, seeds=seed_list, carbon_price_usd_per_ton=carbon_price
        )
        first = multi.per_seed[0]
        payload = {
            "n_workloads": multi.n_workloads,
            "days": multi.days,
            "seeds": multi.seeds,
            "deadline_ratio": first.deadline_ratio,
            "checkpoint_gb": first.checkpoint_gb,
            "up_single_default_region": multi.up_single_default_region,
            "up_single_best_region": first.up_single_best_region,
            "metrics_mean_stdev": {k: list(v) for k, v in multi.metrics.items()},
            "caveats": multi.caveats,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2))
            return

        def line(label: str, key: str, unit: str = "") -> None:
            mean, sd = multi.metrics[key]
            typer.echo(f"  {label:<34} {mean:8.2f}{unit} +/- {sd:.2f}")

        typer.echo(
            f"Spot backtest: n={multi.n_workloads}/seed, days={multi.days}, "
            f"seeds={len(multi.seeds)} (mean +/- sigma across seeds)"
        )
        line("SkyNomad cost", "skynomad_cost_mean")
        line(f"wait-enabled pin {multi.up_single_default_region}", "wait_pin_cost_mean")
        line(f"UP-single pinned {multi.up_single_default_region}", "up_single_default_cost_mean")
        line(
            f"UP-single best ({first.up_single_best_region}, hindsight)",
            "up_single_best_cost_mean",
        )
        line("UP-multi cost", "up_multi_cost_mean")
        typer.echo("")
        line("Savings vs wait-enabled pin", "cost_savings_vs_wait_pin_pct", "%")
        line("Savings vs pinned default", "cost_savings_vs_up_single_default_pct", "%")
        line("Savings vs best single pin", "cost_savings_vs_up_single_best_pct", "%")
        line("Savings vs UP-multi", "cost_savings_vs_up_multi_pct", "%")

        headline, _sd = multi.metrics["cost_savings_vs_wait_pin_pct"]
        losses = sum(1 for r in multi.per_seed if r.cost_savings_vs_wait_pin_pct < 0)
        typer.echo("")
        if headline < 0:
            typer.echo(
                f"  VERDICT: SkyNomad LOSES to the wait-enabled pin by "
                f"{-headline:.2f}% ({losses}/{len(multi.seeds)} seeds). That pin is one\n"
                f"           region, never migrates, and has no lifetime, effectiveness or\n"
                f"           carbon model — it is only allowed to wait for spot. Treat the\n"
                f"           other baselines below as upper bounds, not as the result."
            )
        else:
            typer.echo(
                f"  VERDICT: SkyNomad beats the wait-enabled pin by {headline:.2f}% "
                f"({len(multi.seeds) - losses}/{len(multi.seeds)} seeds).\n"
                f"           UP-single and UP-multi are forbidden to wait, so their deltas\n"
                f"           overstate the advantage; this line is the one that counts."
            )

        typer.echo("")
        line("Deadline met", "deadline_met_pct", "%")
        line("Migrations", "migrations_mean")
        line("Idle hours SkyNomad", "idle_hours_mean")
        line("Idle hours wait-pin", "wait_pin_idle_hours_mean")
        line("Egress share of cost", "egress_cost_pct_mean", "%")
        typer.echo("")
        line("CO2 kg SkyNomad", "carbon_kg_mean")
        line("CO2 kg wait-enabled pin", "wait_pin_carbon_kg_mean")
        line("CO2 kg UP-multi", "up_multi_carbon_kg_mean")
        sky_kg, _ = multi.metrics["carbon_kg_mean"]
        pin_kg, _ = multi.metrics["wait_pin_carbon_kg_mean"]
        if pin_kg > 0:
            typer.echo(
                f"  CO2 vs wait-enabled pin:        {(sky_kg / pin_kg - 1) * 100:+8.1f}%"
            )
        typer.echo(
            f"  Carbon breakeven: ~${CARBON_BREAKEVEN_USD_PER_TON:.0f}/ton. Below that the "
            f"whole dirtiest-to-cleanest\n"
            f"           spread is smaller than the ${first.delta:.2f}/hr anti-flapping "
            f"delta, so carbon\n"
            f"           cannot move a migration decision at the ${carbon_price:.0f}/ton default."
        )
        for caveat in multi.caveats:
            typer.echo(f"  NOTE: {caveat}")
        return

    result = run_backtest(n=n, days=days, seed=seed)
    if json_out:
        typer.echo(json.dumps({
            "n_workloads": result.n_workloads,
            "days": result.days,
            "carbon_savings_pct_mean": result.carbon_savings_pct_mean,
            "cost_savings_pct_mean": result.cost_savings_pct_mean,
            "regret_mean": result.regret_mean,
            "regret_p95": result.regret_p95,
            "rank_top1_pct": result.rank_top1_pct,
            "rank_top3_pct": result.rank_top3_pct,
        }, indent=2))
    else:
        typer.echo(f"Backtest: n={result.n_workloads}, days={result.days}")
        typer.echo(f"  Carbon savings vs baseline (mean %): {result.carbon_savings_pct_mean:.1f}")
        typer.echo(f"  Cost savings vs baseline (mean %):   {result.cost_savings_pct_mean:.1f}")
        typer.echo(f"  Regret (mean / p95):                 {result.regret_mean:.3f} / {result.regret_p95:.3f}")
        typer.echo(f"  Rank accuracy: top-1 {result.rank_top1_pct:.1f}%, top-3 {result.rank_top3_pct:.1f}%")


if __name__ == "__main__":
    typer.run(backtest_group)
