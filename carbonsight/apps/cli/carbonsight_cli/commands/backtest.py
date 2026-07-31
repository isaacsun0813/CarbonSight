"""carbonsight backtest run --days 180 --n 1000"""

import json

import typer
from carbonsight_core.backtest.runner import run_backtest
from carbonsight_core.backtest.spot_runner import run_spot_backtest_multi

backtest_group = typer.Typer(help="Backtesting harness")


@backtest_group.command("run")
def run_backtest_cmd(
    days: int = typer.Option(180, "--days", help="Backtest period in days"),
    n: int = typer.Option(1000, "--n", help="Number of synthetic workloads"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    json_out: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
    spot: bool = typer.Option(False, "--spot", help="Use spot-aware runner (evictions + carbon)"),
    carbon_price: float = typer.Option(50.0, "--carbon-price", help="USD/ton for spot backtest"),
    seeds: str = typer.Option(
        "", "--seeds", help="Comma-separated seeds for --spot; defaults to --seed alone"
    ),
) -> None:
    """Generate workloads and output savings, regret, rank accuracy."""
    if spot:
        seed_list = tuple(int(x) for x in seeds.split(",")) if seeds else (seed,)
        multi = run_spot_backtest_multi(
            n=n, days=days, seeds=seed_list, carbon_price_usd_per_ton=carbon_price
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
        line(f"UP-single pinned {multi.up_single_default_region}", "up_single_default_cost_mean")
        line(f"UP-single best ({first.up_single_best_region}, hindsight)", "up_single_best_cost_mean")
        line("UP-multi cost", "up_multi_cost_mean")
        typer.echo("")
        line("Savings vs pinned default", "cost_savings_vs_up_single_default_pct", "%")
        line("Savings vs best single pin", "cost_savings_vs_up_single_best_pct", "%")
        line("Savings vs UP-multi", "cost_savings_vs_up_multi_pct", "%")
        typer.echo("")
        line("Deadline met", "deadline_met_pct", "%")
        line("Migrations", "migrations_mean")
        line("Idle hours", "idle_hours_mean")
        line("Egress share of cost", "egress_cost_pct_mean", "%")
        line("CO2 kg SkyNomad", "carbon_kg_mean")
        line("CO2 kg UP-multi", "up_multi_carbon_kg_mean")
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
