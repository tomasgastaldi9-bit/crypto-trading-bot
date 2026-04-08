from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from backtester import Backtester
from config import DEFAULT_CONFIG, BinanceConfig, EngineConfig, OutputConfig
from data_loader import BinanceDataLoader
from indicators import add_indicators
from performance import PerformanceAnalyzer, format_summary_report
from risk_management import RiskManager
from strategy import TrendMomentumStrategy


def build_config(args: argparse.Namespace) -> EngineConfig:
    """Create a runtime config with CLI overrides applied."""

    config = EngineConfig(
        binance=replace(DEFAULT_CONFIG.binance),
        indicators=replace(DEFAULT_CONFIG.indicators),
        strategy=replace(DEFAULT_CONFIG.strategy),
        risk=replace(DEFAULT_CONFIG.risk),
        execution=replace(DEFAULT_CONFIG.execution),
        output=replace(DEFAULT_CONFIG.output),
    )

    config.binance.symbol = args.symbol or config.binance.symbol
    config.binance.start_date = args.start or config.binance.start_date
    config.binance.end_date = args.end if args.end is not None else config.binance.end_date
    config.output.output_dir = Path(args.output_dir) if args.output_dir else config.output.output_dir
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Professional crypto trading backtesting engine")
    parser.add_argument("--symbol", type=str, default=None, help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--start", type=str, default=None, help="Backtest start date, e.g. 2022-01-01")
    parser.add_argument("--end", type=str, default=None, help="Backtest end date, e.g. 2024-01-01")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for reports and logs")
    return parser.parse_args()


def main() -> None:
    """Run the full backtesting workflow."""

    args = parse_args()
    config = build_config(args)

    loader = BinanceDataLoader(config.binance)
    market_data = loader.load_klines()

    indicator_data = add_indicators(market_data, config.indicators)
    strategy = TrendMomentumStrategy(config.strategy)
    signal_data = strategy.generate_signals(indicator_data)
    signal_data = signal_data.dropna().reset_index(drop=True)

    risk_manager = RiskManager(config.risk)
    backtester = Backtester(
        risk_manager=risk_manager,
        strategy_config=config.strategy,
        risk_config=config.risk,
        execution_config=config.execution,
    )
    result = backtester.run(signal_data)

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)
    summary = analyzer.summarize(
        equity_curve=result.equity_curve,
        trades=result.trades,
        initial_capital=config.risk.initial_capital,
    )
    output_paths = analyzer.save_outputs(
        equity_curve=result.equity_curve,
        trades=result.trades,
        summary=summary,
        output_config=config.output,
    )

    if config.output.print_summary:
        print(format_summary_report(summary))
        print(f"Trade log    : {output_paths['trade_log']}")
        print(f"Equity curve : {output_paths['equity_curve']}")
        print(f"Summary JSON : {output_paths['summary']}")
        if output_paths["equity_plot"].exists():
            print(f"Equity plot  : {output_paths['equity_plot']}")


if __name__ == "__main__":
    main()
