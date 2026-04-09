from optimizer_runner import BacktesterWrapper
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer
from copy import deepcopy

import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def run_asset(wrapper, base_config, symbol):
    print(f"\n=== RUNNING {symbol} ===")

    config = deepcopy(base_config)

    # 👇 ACA CAMBIÁS EL ASSET
    config.binance.symbol = symbol

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    return data


def run_multi_asset(wrapper, base_config, top_configs):
    asset_equities = []

    for symbol in ASSETS:
        data = run_asset(wrapper, base_config, symbol)

        # =========================
        # ENSEMBLE DENTRO DEL ASSET
        # =========================
        equity_curves = []
        sharpes = []

        for cfg in top_configs:
            res = wrapper.run(data, cfg["params"])
            equity_curves.append(res["equity_curve"])
            sharpes.append(cfg["mean_test"])

        sharpes = np.array(sharpes)
        sharpes = np.clip(sharpes, 0.01, None)

        weights = softmax(sharpes)

        combined = equity_curves[0].copy()
        combined["equity"] *= weights[0]

        for i in range(1, len(equity_curves)):
            combined["equity"] += equity_curves[i]["equity"] * weights[i]

        # 🔥 normalizar (importante para comparar assets)
        combined["equity"] = combined["equity"] / combined["equity"].iloc[0]

        asset_equities.append(combined)

        # =========================
    # CALCULAR SHARPE POR ASSET
    # =========================
    asset_sharpes = []

    for ec in asset_equities:
        returns = ec["equity"].pct_change().dropna()

        if len(returns) == 0:
            sharpe = 0.01
        else:
            sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

        asset_sharpes.append(sharpe)

    asset_sharpes = np.array(asset_sharpes)

    # =========================
    # 🔥 FILTRO DE ASSETS MALOS
    # =========================
    threshold = 0.08  # 👈 clave (ajustable)

    valid_idx = [i for i, s in enumerate(asset_sharpes) if s > threshold]

    # si todos son malos, usamos al menos el mejor
    if len(valid_idx) == 0:
        valid_idx = [int(np.argmax(asset_sharpes))]

    asset_equities = [asset_equities[i] for i in valid_idx]
    asset_sharpes = asset_sharpes[valid_idx]
    selected_assets = [ASSETS[i] for i in valid_idx]

    # =========================
    # PESOS (SOFTMAX + TEMPERATURE AGRESIVA)
    # =========================
    temperature = 8  # 👈 MUY IMPORTANTE

    scaled = asset_sharpes * temperature
    weights = np.exp(scaled - np.max(scaled))
    weights = weights / weights.sum()

    print("\n=== ASSET WEIGHTS (FILTERED) ===")
    for i, w in enumerate(weights):
        print(f"{selected_assets[i]}: {w:.3f} | Sharpe: {asset_sharpes[i]:.2f}")

    # =========================
    # COMBINAR ENTRE ASSETS
    # =========================
    final = asset_equities[0].copy()
    final["equity"] *= weights[0]

    for i in range(1, len(asset_equities)):
        final["equity"] += asset_equities[i]["equity"] * weights[i]

    # escalar capital (una sola vez)
    initial_capital = base_config.risk.initial_capital
    final["equity"] *= initial_capital

    return final


def main():
    config = DEFAULT_CONFIG
    wrapper = BacktesterWrapper(config)

    # 👉 reutilizamos tus mejores configs
    top_configs = [
        {"params": {'atr_period': 12, 'rsi_period': 19, 'sl_atr': 1.89, 'ts_atr': 2.44}, "mean_test": 0.516},
        {"params": {'atr_period': 10, 'rsi_period': 16, 'sl_atr': 2.40, 'ts_atr': 3.30}, "mean_test": 0.606},
        {"params": {'atr_period': 10, 'rsi_period': 19, 'sl_atr': 2.44, 'ts_atr': 3.13}, "mean_test": 0.399},
        {"params": {'atr_period': 12, 'rsi_period': 13, 'sl_atr': 2.16, 'ts_atr': 3.36}, "mean_test": 0.317},
    ]

    equity = run_multi_asset(wrapper, config, top_configs)

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    summary = analyzer.summarize(
        equity_curve=equity,
        trades=pd.DataFrame(),
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== MULTI-ASSET RESULTS ===")
    print(summary)


if __name__ == "__main__":
    main()