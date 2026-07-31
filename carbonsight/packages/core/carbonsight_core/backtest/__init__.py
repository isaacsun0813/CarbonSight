"""Backtest harnesses (baseline + spot-aware)."""

from carbonsight_core.backtest.runner import BacktestResult, run_backtest
from carbonsight_core.backtest.spot_runner import SpotBacktestResult, run_spot_backtest

__all__ = [
    "BacktestResult",
    "SpotBacktestResult",
    "run_backtest",
    "run_spot_backtest",
]
