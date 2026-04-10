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
    scores = []

    print("\n=== RUNNING CONFIGS ===")

    for i, config in enumerate(top_configs):
        print(
            f"Running config {i+1} | "
            f"Sharpe: {config['mean_test']:.3f} | "
            f"Std: {config['std_test']:.3f} | "
            f"Overfit: {config['overfit']:.3f}"
        )

        res = wrapper.run(data, config["params"])

        equity_curves.append(res["equity_curve"])

        # =========================
        # 🔥 SCORE MEJORADO
        # =========================
        score = (
            config["mean_test"]
            - 0.5 * config["std_test"]
            - 0.7 * config["overfit"]
        )
        print(f"Score: {score:.3f}")

        scores.append(score)

    # =========================
    # FILTRO NEGATIVOS
    # =========================
    scores = np.array(scores)
    scores = np.clip(scores, 0.01, None)

    # =========================
    # SOFTMAX
    # =========================
    exp_scores = np.exp(scores - np.max(scores))
    weights = exp_scores / exp_scores.sum()

    print("\n=== FINAL WEIGHTS ===")
    for i, w in enumerate(weights):
        print(f"Config {i+1}: {w:.3f}")

    # =========================
    # COMBINAR
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
    # 🔥 CORRER OPTIMIZER
    # =========================
    from optimizer import WalkForwardOptimizer

    param_space = {
        "atr_period": (10, 25),
        "rsi_period": (12, 20),
        "sl_atr": (1.8, 3.0),
        "ts_atr": (2.2, 3.5),

        # MR params
        "mr_window": (20, 60),
        "mr_z": (2.0, 3.5),
        "mr_rsi": (20, 35),
    }

    optimizer = WalkForwardOptimizer(
        backtester=wrapper,
        param_space=param_space,
        n_trials=50,
    )

    results = optimizer.optimize(data)

    if results is None:
        print("No results")
        return

    optimizer.report(results)

    # =========================
    # 🔥 USAR TOP CONFIGS REALES
    # =========================
    top_configs = [
        r for r in results
        if r["mean_test"] > 0.4 and r["std_test"] < 0.6
    ][:5]

    print(f"\nUsing {len(top_configs)} configs")

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