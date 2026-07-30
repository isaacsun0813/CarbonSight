"""carbonsight backtest run --days 180 --n 1000 + spot backtest"""

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
) -> None:
    """Generate workloads and output savings, regret, rank accuracy."""
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


@backtest_group.command("spot")
def run_spot_backtest_cmd(
    n: int = typer.Option(20, "--n", help="Number of synthetic workloads"),
    days: int = typer.Option(14, "--days", help="Trace days"),
    deadline_ratio: float = typer.Option(1.5, "--deadline-ratio", help="T/P ratio, e.g. 45/30=1.5"),
    checkpoint_gb: float = typer.Option(100.0, "--checkpoint-gb", help="Checkpoint size GB for migration cost"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """SkyNomad-inspired spot+carbon backtest: multi-region availability traces, cost savings vs UP baselines."""
    result = run_spot_backtest(n=n, days=days, deadline_ratio=deadline_ratio, checkpoint_gb=checkpoint_gb, seed=seed)
    if json_out:
        typer.echo(json.dumps({
            "n_workloads": result.n_workloads,
            "days": result.days,
            "deadline_ratio": result.deadline_ratio,
            "checkpoint_gb": result.checkpoint_gb,
            "skynomad_cost_mean": result.skynomad_cost_mean,
            "up_single_cost_mean": result.up_single_cost_mean,
            "up_multi_cost_mean": result.up_multi_cost_mean,
            "cost_savings_vs_up_single_pct": result.cost_savings_vs_up_single_pct,
            "cost_savings_vs_up_multi_pct": result.cost_savings_vs_up_multi_pct,
            "deadline_met_pct": result.deadline_met_pct,
            "migrations_mean": result.migrations_mean,
            "egress_cost_pct_mean": result.egress_cost_pct_mean,
            "selection_accuracy_pct": result.selection_accuracy_pct,
        }, indent=2))
    else:
        typer.echo(f"Spot Backtest: n={result.n_workloads}, days={result.days}, T/P={result.deadline_ratio}, ckpt={result.checkpoint_gb}GB")
        typer.echo(f"  SkyNomad cost mean: ${result.skynomad_cost_mean:.2f}")
        typer.echo(f"  UP single-region mean: ${result.up_single_cost_mean:.2f} (savings {result.cost_savings_vs_up_single_pct:.1f}%)")
        typer.echo(f"  UP multi failover mean: ${result.up_multi_cost_mean:.2f} (savings {result.cost_savings_vs_up_multi_pct:.1f}%)")
        typer.echo(f"  Deadline met: {result.deadline_met_pct:.1f}%")
        typer.echo(f"  Migrations mean: {result.migrations_mean:.1f}, egress cost {result.egress_cost_pct_mean:.1f}%")
        typer.echo(f"  Selection accuracy: {result.selection_accuracy_pct:.1f}%")


if __name__ == "__main__":
    typer.run(backtest_group)
