from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from optimizer import WalkForwardOptimizer
from ensemble_runner import run_ensemble_weighted
from optimizer_runner import BacktesterWrapper
from performance import PerformanceAnalyzer

import pandas as pd
import numpy as np


def generate_walkforward_splits(data, train_years=2, test_years=1):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    start = data["timestamp"].min()
    end = data["timestamp"].max()

    splits = []

    current_start = start

    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        train = data[
            (data["timestamp"] >= current_start)
            & (data["timestamp"] < train_end)
        ]

        test = data[
            (data["timestamp"] >= train_end)
            & (data["timestamp"] < test_end)
        ]

        if len(train) == 0 or len(test) == 0:
            break

        splits.append((train.copy(), test.copy(), current_start, train_end, test_end))

        current_start += pd.DateOffset(years=1)

    return splits


def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    param_space = {
        "atr_period": (10, 25),
        "rsi_period": (12, 20),
        "sl_atr": (1.8, 3.0),
        "ts_atr": (2.2, 3.5),
        "mr_window": (20, 60),
        "mr_z": (2.0, 3.5),
        "mr_rsi": (20, 35),
    }

    splits = generate_walkforward_splits(data)

    print(f"\nTotal splits: {len(splits)}")

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    all_results = []

    for i, (train_data, test_data, start, train_end, test_end) in enumerate(splits):
        print("\n" + "=" * 60)
        print(f"SPLIT {i+1}")
        print(f"Train: {start.date()} → {train_end.date()}")
        print(f"Test : {train_end.date()} → {test_end.date()}")

        optimizer = WalkForwardOptimizer(
            backtester=wrapper,
            param_space=param_space,
            n_trials=50,
        )

        results = optimizer.optimize(train_data)

        if results is None:
            print("No results")
            continue

        top_configs = results[:1]

        # =========================
        # TRAIN
        # =========================
        train_equity, train_trades = run_ensemble_weighted(
            wrapper, train_data, top_configs
        )

        train_summary = analyzer.summarize(
            equity_curve=train_equity,
            trades=train_trades,
            initial_capital=config.risk.initial_capital,
        )

        # =========================
        # TEST (OOS)
        # =========================
        test_equity, test_trades = run_ensemble_weighted(
            wrapper, test_data, top_configs
        )

        test_summary = analyzer.summarize(
            equity_curve=test_equity,
            trades=test_trades,
            initial_capital=config.risk.initial_capital,
        )

        print("\nTRAIN:", train_summary)
        print("TEST :", test_summary)

        all_results.append(test_summary)

    # =========================
    # AGREGADO FINAL
    # =========================
    print("\n" + "=" * 60)
    print("FINAL OOS SUMMARY")

    sharpe_list = [r["sharpe_ratio"] for r in all_results]
    return_list = [r["total_return"] for r in all_results]
    dd_list = [r["max_drawdown"] for r in all_results]

    print(f"Avg Sharpe: {np.mean(sharpe_list):.3f}")
    print(f"Avg Return: {np.mean(return_list):.3f}")
    print(f"Avg DD    : {np.mean(dd_list):.3f}")
    print(f"Sharpe Std: {np.std(sharpe_list):.3f}")


if __name__ == "__main__":
    main()