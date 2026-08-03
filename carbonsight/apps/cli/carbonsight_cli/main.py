"""CarbonSight CLI entrypoint. Commands: advise, train, run, history, report, dashboard, mappings validate/refresh, backtest run."""

import typer

from carbonsight_cli.commands import (
    advise,
    backtest,
    dashboard,
    history,
    mappings,
    report,
    run,
    train,
)

app = typer.Typer(name="carbonsight", help="Greenest cloud advisor and launcher (WattTime + SkyPilot)")

app.command("advise")(advise.advise)
app.command("train")(train.train_cmd)
app.command("run")(run.run_cmd)
app.command("history")(history.history_cmd)
app.command("report")(report.report_cmd)
app.command("dashboard")(dashboard.dashboard_cmd)
app.add_typer(mappings.mappings_group, name="mappings")
app.add_typer(backtest.backtest_group, name="backtest")


if __name__ == "__main__":
    app()
