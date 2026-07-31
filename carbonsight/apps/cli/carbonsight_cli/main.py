"""CarbonSight CLI entrypoint."""

import typer

from carbonsight_cli.commands import (
    advise,
    backtest,
    dashboard,
    history,
    mappings,
    report,
    run,
    schedule,
    train,
)

app = typer.Typer(
    name="carbonsight",
    help="Greenest cloud advisor + SkyNomad multi-lever scheduler (WattTime + SkyPilot)",
)

app.command("advise")(advise.advise)
app.command("schedule")(schedule.schedule)
app.command("train")(train.train_cmd)
app.command("run")(run.run_cmd)
app.command("history")(history.history_cmd)
app.command("report")(report.report_cmd)
app.command("dashboard")(dashboard.dashboard_cmd)
app.add_typer(mappings.mappings_group, name="mappings")
app.add_typer(backtest.backtest_group, name="backtest")


if __name__ == "__main__":
    app()
