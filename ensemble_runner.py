from optimizer import WalkForwardOptimizer
from optimizer_runner import BacktesterWrapper
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer

import pandas as pd
import numpy as np


def run_single_config(wrapper, data, params):
    result = wrapper.run(data, params)
    return result


def run_ensemble(wrapper, data, top_configs):
    equity_curves = []

    for i, config in enumerate(top_configs):
        print(f"Running config {i+1}")

        res = wrapper.run(data, config["params"])

        # ⚠️ necesitamos equity curve real → modificamos wrapper después
        equity_curves.append(res["equity_curve"])

    # =========================
    # COMBINAR (promedio simple)
    # =========================
    combined = equity_curves[0].copy()

    for i in range(1, len(equity_curves)):
        combined["equity"] += equity_curves[i]["equity"]

    combined["equity"] /= len(equity_curves)

    return combined


def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    # ⚠️ PEGÁ ACA TUS TOP CONFIGS
    top_configs = [
        {'atr_period': 12, 'rsi_period': 19, 'sl_atr': 1.89, 'ts_atr': 2.44},
        {'atr_period': 10, 'rsi_period': 16, 'sl_atr': 2.40, 'ts_atr': 3.30},
        {'atr_period': 12, 'rsi_period': 13, 'sl_atr': 2.16, 'ts_atr': 3.36},
        {'atr_period': 10, 'rsi_period': 19, 'sl_atr': 2.44, 'ts_atr': 3.13},
        {'atr_period': 13, 'rsi_period': 16, 'sl_atr': 2.77, 'ts_atr': 2.20},
    ]

    equity = run_ensemble(wrapper, data, [{"params": p} for p in top_configs])

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    summary = analyzer.summarize(
        equity_curve=equity,
        trades=pd.DataFrame(),  # opcional mejorar después
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== ENSEMBLE RESULTS ===")
    print(summary)


if __name__ == "__main__":
    main()