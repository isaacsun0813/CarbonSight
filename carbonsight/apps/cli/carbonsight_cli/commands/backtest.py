"""carbonsight backtest run --days 180 --n 1000"""

import json

import typer
from carbonsight_core.backtest.runner import run_backtest
from carbonsight_core.backtest.spot_runner import run_spot_backtest

backtest_group = typer.Typer(help="Backtesting harness")


@backtest_group.command("run")
def run_backtest_cmd(
    days: int = typer.Option(180, "--days", help="Backtest period in days"),
    n: int = typer.Option(1000, "--n", help="Number of synthetic workloads"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    json_out: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
    spot: bool = typer.Option(False, "--spot", help="Use spot-aware runner (evictions + carbon)"),
    carbon_price: float = typer.Option(50.0, "--carbon-price", help="USD/ton for spot backtest"),
) -> None:
    """Generate workloads and output savings, regret, rank accuracy."""
    if spot:
        result = run_spot_backtest(
            n=n, days=days, seed=seed, carbon_price_usd_per_ton=carbon_price
        )
        payload = {
            "n_workloads": result.n_workloads,
            "days": result.days,
            "carbon_savings_pct_mean": result.carbon_savings_pct_mean,
            "cost_savings_pct_mean": result.cost_savings_pct_mean,
            "eviction_rate": result.eviction_rate,
            "joint_utility_mean": result.joint_utility_mean,
            "regret_mean": result.regret_mean,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2))
        else:
            typer.echo(f"Spot backtest: n={result.n_workloads}, days={result.days}")
            typer.echo(f"  Carbon savings (mean %): {result.carbon_savings_pct_mean:.1f}")
            typer.echo(f"  Cost savings (mean %):   {result.cost_savings_pct_mean:.1f}")
            typer.echo(f"  Eviction rate:           {result.eviction_rate:.3f}")
            typer.echo(f"  Joint U mean:            {result.joint_utility_mean:.3f}")
            typer.echo(f"  Regret mean:             {result.regret_mean:.3f}")
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
