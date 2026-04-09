from optimizer_runner import BacktesterWrapper
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer

import pandas as pd
import numpy as np


# =========================
# SOFTMAX
# =========================
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def run_ensemble_weighted(wrapper, data, top_configs):
    equity_curves = []
    sharpes = []

    print("\n=== RUNNING CONFIGS ===")

    for i, config in enumerate(top_configs):
        print(f"Running config {i+1} | Sharpe: {config['mean_test']:.3f}")

        res = wrapper.run(data, config["params"])

        equity_curves.append(res["equity_curve"])
        sharpes.append(config["mean_test"])

    # =========================
    # PESOS BASE (SOFTMAX)
    # =========================
    sharpes = np.array(sharpes)
    sharpes = np.clip(sharpes, 0.01, None)

    weights = softmax(sharpes)

    # =========================
    # VOLATILITY SCALING
    # =========================
    vols = []

    for ec in equity_curves:
        returns = ec["equity"].pct_change().dropna()
        vol = returns.std() if len(returns) > 0 else 1.0
        vols.append(vol)

    vols = np.array(vols)
    inv_vol = 1 / (vols + 1e-8)

    weights = weights * inv_vol
    weights = weights / weights.sum()

    # =========================
    # 🔥 CORRELATION-AWARE 🔥
    # =========================
    returns_matrix = []

    for ec in equity_curves:
        r = ec["equity"].pct_change().fillna(0)
        returns_matrix.append(r.values)

    returns_matrix = np.array(returns_matrix)

    # matriz de correlación
    corr_matrix = np.corrcoef(returns_matrix)

    # promedio de correlaciones por estrategia (excluyendo self=1)
    avg_corr = (corr_matrix.sum(axis=1) - 1) / (len(corr_matrix) - 1)

    # score de diversificación (menos correlación = mejor)
    diversification = 1 - avg_corr

    # evitar valores raros
    diversification = np.clip(diversification, 0.1, None)

    # aplicar penalización
    weights = weights * diversification
    weights = weights / weights.sum()

    print("\n=== FINAL WEIGHTS (CORR ADJUSTED) ===")
    for i, w in enumerate(weights):
        print(f"Config {i+1}: {w:.3f}")

    # =========================
    # COMBINAR EQUITY
    # =========================
    combined = equity_curves[0].copy()
    combined["equity"] *= weights[0]

    for i in range(1, len(equity_curves)):
        combined["equity"] += equity_curves[i]["equity"] * weights[i]

    return combined


def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    # =========================
    # CONFIGS
    # =========================
    raw_configs = [
        {"params": {'atr_period': 12, 'rsi_period': 19, 'sl_atr': 1.89, 'ts_atr': 2.44}, "mean_test": 0.516},
        {"params": {'atr_period': 10, 'rsi_period': 16, 'sl_atr': 2.40, 'ts_atr': 3.30}, "mean_test": 0.606},
        {"params": {'atr_period': 12, 'rsi_period': 13, 'sl_atr': 2.16, 'ts_atr': 3.36}, "mean_test": 0.317},
        {"params": {'atr_period': 10, 'rsi_period': 19, 'sl_atr': 2.44, 'ts_atr': 3.13}, "mean_test": 0.399},
        {"params": {'atr_period': 13, 'rsi_period': 16, 'sl_atr': 2.77, 'ts_atr': 2.20}, "mean_test": 0.259},
    ]

    # =========================
    # SORT + FILTER 🔥
    # =========================
    raw_configs = sorted(raw_configs, key=lambda x: x["mean_test"], reverse=True)

    top_configs = [c for c in raw_configs if c["mean_test"] > 0.3][:6]

    print(f"\nUsing {len(top_configs)} configs after filtering")

    equity = run_ensemble_weighted(wrapper, data, top_configs)

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    summary = analyzer.summarize(
        equity_curve=equity,
        trades=pd.DataFrame(),
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== ENSEMBLE RESULTS ===")
    print(summary)


if __name__ == "__main__":
    main()