"""CarbonSight CLI entrypoint. Commands: advise, run, mappings validate, backtest run."""

import typer

from carbonsight_cli.commands import advise, backtest, mappings, run

app = typer.Typer(name="carbonsight", help="Greenest cloud advisor and launcher (WattTime + SkyPilot)")

app.command("advise")(advise.advise)
app.command("run")(run.run_cmd)
app.add_typer(mappings.mappings_group, name="mappings")
app.add_typer(backtest.backtest_group, name="backtest")


if __name__ == "__main__":
    app()
