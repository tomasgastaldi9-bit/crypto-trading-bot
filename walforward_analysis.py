import pandas as pd
import numpy as np
import json

from ensemble_runner import run_ensemble_weighted
from optimizer_runner import BacktesterWrapper
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer


def split_data(df, n_splits=5):
    chunk_size = len(df) // n_splits
    splits = []

    for i in range(n_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_splits - 1 else len(df)
        splits.append(df.iloc[start:end].copy())

    return splits


def main():
    print("=== LOADING DATA ===")

    config = DEFAULT_CONFIG
    data = BinanceDataLoader(config.binance).load_klines()

    print(f"Total rows: {len(data)}")

    wrapper = BacktesterWrapper(config)

    # =========================
    # LOAD CONFIGS (FIJAS)
    # =========================
    with open("top_configs.json", "r") as f:
        configs = json.load(f)

    configs = sorted(configs, key=lambda x: x["mean_test"], reverse=True)
    configs = [c for c in configs if c["overfit"] < 0.8 and c["mean_test"] > 0.3]

    top_configs = configs[:2]  # podés cambiar

    print(f"Using {len(top_configs)} configs")

    # =========================
    # SPLITS
    # =========================
    splits = split_data(data, n_splits=5)

    results = []

    print("\n=== WALKFORWARD TEST ===")

    for i, split in enumerate(splits):
        print(f"\n--- SPLIT {i+1} ---")

        portfolio = run_ensemble_weighted(wrapper, split, top_configs)

        summary = PerformanceAnalyzer(annualization_factor=365*24).summarize(
            equity_curve=portfolio,
            trades=pd.DataFrame(),
            initial_capital=config.risk.initial_capital,
        )

        print(f"Sharpe: {summary['sharpe_ratio']:.3f}")
        print(f"Return: {summary['total_return']:.3f}")
        print(f"DD: {summary['max_drawdown']:.3f}")

        results.append(summary)

    # =========================
    # RESUMEN
    # =========================
    sharpes = [r["sharpe_ratio"] for r in results]
    returns = [r["total_return"] for r in results]
    dds = [r["max_drawdown"] for r in results]

    print("\n=== FINAL SUMMARY ===")
    print(f"Avg Sharpe: {np.mean(sharpes):.3f}")
    print(f"Std Sharpe: {np.std(sharpes):.3f}")
    print(f"Avg Return: {np.mean(returns):.3f}")
    print(f"Avg DD: {np.mean(dds):.3f}")


if __name__ == "__main__":
    main()