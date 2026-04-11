from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from optimizer import WalkForwardOptimizer
from ensemble_runner import run_ensemble_weighted
from optimizer_runner import BacktesterWrapper
from performance import PerformanceAnalyzer

import pandas as pd


# =========================
# 🔥 SPLIT POR FECHA (PRO)
# =========================
def split_data_by_date(data, split_date="2023-01-01"):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    train = data[data["timestamp"] < split_date].copy()
    test = data[data["timestamp"] >= split_date].copy()

    return train, test


def main():
    config = DEFAULT_CONFIG

    # =========================
    # LOAD DATA
    # =========================
    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    # 🔥 CAMBIÁS SOLO ESTA FECHA
    train_data, test_data = split_data_by_date(data, "2023-01-01")

    print("\n=== DATA SPLIT ===")
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    print(f"Train: {train_data['timestamp'].min()} → {train_data['timestamp'].max()}")
    print(f"Test : {test_data['timestamp'].min()} → {test_data['timestamp'].max()}")

    wrapper = BacktesterWrapper(config)

    # =========================
    # PARAM SPACE
    # =========================
    param_space = {
        "atr_period": (10, 25),
        "rsi_period": (12, 20),
        "sl_atr": (1.8, 3.0),
        "ts_atr": (2.2, 3.5),
        "mr_window": (20, 60),
        "mr_z": (2.0, 3.5),
        "mr_rsi": (20, 35),
    }

    # =========================
    # OPTIMIZER (TRAIN ONLY)
    # =========================
    optimizer = WalkForwardOptimizer(
        backtester=wrapper,
        param_space=param_space,
        n_trials=50,
    )

    results = optimizer.optimize(train_data)

    if results is None:
        print("No results")
        return

    optimizer.report(results)

    # 🔥 podés cambiar esto
    top_configs = results[:2]

    print(f"\nUsing {len(top_configs)} configs")

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    # =========================
    # TRAIN PERFORMANCE
    # =========================
    print("\n=== TRAIN PERFORMANCE ===")

    train_equity, train_trades = run_ensemble_weighted(
        wrapper, train_data, top_configs
    )

    train_summary = analyzer.summarize(
        equity_curve=train_equity,
        trades=train_trades,
        initial_capital=config.risk.initial_capital,
    )

    print(train_summary)

    # =========================
    # OOS PERFORMANCE
    # =========================
    print("\n=== OOS (TEST) PERFORMANCE ===")

    test_equity, test_trades = run_ensemble_weighted(
        wrapper, test_data, top_configs
    )

    test_summary = analyzer.summarize(
        equity_curve=test_equity,
        trades=test_trades,
        initial_capital=config.risk.initial_capital,
    )

    print(test_summary)

    # =========================
    # SAVE TEST OUTPUTS
    # =========================
    output_paths = analyzer.save_outputs(
        equity_curve=test_equity,
        trades=test_trades,
        summary=test_summary,
        output_config=config.output,
    )

    print("\n=== FILES SAVED (OOS) ===")
    for k, v in output_paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()