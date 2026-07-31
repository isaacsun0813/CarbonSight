"""Backtest harnesses (baseline + spot-aware)."""

from carbonsight_core.backtest.runner import BacktestResult, run_backtest
from carbonsight_core.backtest.spot_runner import (
    MultiSeedBacktestResult,
    SpotBacktestResult,
    run_spot_backtest,
    run_spot_backtest_multi,
)

__all__ = [
    "BacktestResult",
    "MultiSeedBacktestResult",
    "SpotBacktestResult",
    "run_backtest",
    "run_spot_backtest",
    "run_spot_backtest_multi",
]
