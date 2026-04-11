from optimizer_runner import BacktesterWrapper, build_signal
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer
from volatility_breakout import VolatilityBreakoutStrategy
from indicators import add_indicators
from risk_management import RiskManager
from backtester import Backtester

import pandas as pd
import numpy as np


# =========================
# VOLATILITY BREAKOUT RUN
# =========================
def run_vb_strategy(wrapper, data):
    config = wrapper.base_config

    indicator_data = add_indicators(data, config.indicators)

    vb = VolatilityBreakoutStrategy(config.strategy)
    vb_data = vb.generate_signals(indicator_data.copy())

    vb_data["signal"] = build_signal(vb_data)

    signal_data = vb_data.copy().fillna(0).reset_index(drop=True)

    risk_manager = RiskManager(config.risk)

    backtester = Backtester(
        risk_manager=risk_manager,
        strategy_config=config.strategy,
        risk_config=config.risk,
        execution_config=config.execution,
    )

    result = backtester.run(signal_data)

    equity = result.equity_curve.copy()

    # limpiar
    equity["equity"] = equity["equity"].replace([np.inf, -np.inf], np.nan)
    equity["equity"] = equity["equity"].ffill().bfill()

    if equity["equity"].isna().all():
        equity["equity"] = 1.0

    return equity


# =========================
# ENSEMBLE
# =========================
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
        eq = res["equity_curve"].copy()

        eq["equity"] = eq["equity"].replace([np.inf, -np.inf], np.nan)
        eq["equity"] = eq["equity"].ffill().bfill()

        equity_curves.append(eq)

        score = (
            config["mean_test"]
            - 0.3 * config["std_test"]
            - 0.4 * config["overfit"]
        )

        print(f"Score: {score:.3f}")
        scores.append(score)

    # =========================
    # VOLATILITY BREAKOUT
    # =========================
    print("\n=== RUNNING VOLATILITY BREAKOUT ===")

    vb_equity = run_vb_strategy(wrapper, data)
    equity_curves.append(vb_equity)

    vb_score = np.mean(scores)
    scores.append(vb_score)

    # =========================
    # BASE WEIGHTS
    # =========================
    scores = np.array(scores)

    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    softmax_w = np.exp(scores)
    softmax_w /= softmax_w.sum()

    uniform_w = np.ones_like(softmax_w) / len(softmax_w)

    alpha = 0.7
    base_weights = alpha * softmax_w + (1 - alpha) * uniform_w

    print("\n=== BASE WEIGHTS ===")
    for i, w in enumerate(base_weights):
        if i < len(top_configs):
            print(f"Config {i+1}: {w:.3f}")
        else:
            print(f"VB Strategy: {w:.3f}")

    # =========================
    # VOL REGIME
    # =========================
    returns_price = data["close"].pct_change().fillna(0)
    vol = returns_price.rolling(50).std()
    vol_norm = vol / (vol.rolling(200).mean() + 1e-8)

    vb_regime = (vol_norm > 1.2).astype(float)

    # =========================
    # ALIGN SERIES
    # =========================
    min_len = min(len(eq) for eq in equity_curves)

    aligned_equities = []

    for i, eq in enumerate(equity_curves):
        e = eq.copy().reset_index(drop=True)
        e = e.iloc[-min_len:]

        e["equity"] = e["equity"].replace([np.inf, -np.inf], np.nan)
        e["equity"] = e["equity"].ffill().bfill()

        values = e["equity"].values

        # aplicar regime a VB
        if i == len(equity_curves) - 1:
            regime = vb_regime.values[-min_len:]

            rets = np.diff(values) / (values[:-1] + 1e-8)

            regime = regime[-len(rets):]  # 🔥 alinear tamaños

            rets = rets * regime

            values = np.concatenate([
                [values[0]],
                np.cumprod(1 + rets) * values[0]
            ])

        aligned_equities.append(values)

    equity_matrix = np.column_stack(aligned_equities)

    # =========================
    # RETURNS MATRIX
    # =========================
    returns_matrix = np.diff(equity_matrix, axis=0) / (equity_matrix[:-1] + 1e-8)

    # =========================
    # 🔥 DYNAMIC + CORRELATION
    # =========================
    window = 80
    alpha_dyn = 0.6
    beta = 0.2
    corr_penalty = 0.5   # 🔥 intensidad penalización

    dynamic_weights = []
    prev_w = base_weights.copy()

    for i in range(len(returns_matrix)):

        if i < window:
            dynamic_weights.append(base_weights)
            continue

        window_returns = returns_matrix[i - window:i]

        # =========================
        # SHARPE
        # =========================
        sharpe = window_returns.mean(axis=0) / (window_returns.std(axis=0) + 1e-8)
        sharpe = np.clip(sharpe, -3, 3)

        # =========================
        # CORRELATION ROBUSTA 🔥
        # =========================
        stds = window_returns.std(axis=0)

        # evitar columnas sin varianza
        valid = stds > 1e-6

        if valid.sum() < 2:
            penalty = np.ones_like(sharpe)
        else:
            safe_returns = window_returns[:, valid]

            corr_matrix = np.corrcoef(safe_returns.T)

            # limpiar NaNs
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            avg_corr_valid = np.mean(
                np.abs(corr_matrix - np.eye(len(corr_matrix))),
                axis=1
            )

            penalty = np.ones_like(sharpe)
            penalty[valid] = 1 - corr_penalty * avg_corr_valid
            penalty = np.clip(penalty, 0.5, 1.0)

        sharpe_adj = sharpe * penalty

        # =========================
        # WEIGHTS
        # =========================
        w_dyn = np.exp(sharpe_adj)
        w_dyn /= w_dyn.sum()

        # mezcla con base
        w = alpha_dyn * w_dyn + (1 - alpha_dyn) * base_weights

        # smoothing temporal
        w = beta * w + (1 - beta) * prev_w

        prev_w = w
        dynamic_weights.append(w)

    dynamic_weights = np.array(dynamic_weights)

    print("\n=== FINAL WEIGHTS (LAST STEP) ===")
    for i, w in enumerate(dynamic_weights[-1]):
        if i < len(top_configs):
            print(f"Config {i+1}: {w:.3f}")
        else:
            print(f"VB Strategy: {w:.3f}")

    # =========================
    # COMBINAR
    # =========================
    combined_equity = [equity_matrix[0, 0]]

    for i in range(1, len(equity_matrix)):
        w = dynamic_weights[i - 1]
        ret = returns_matrix[i - 1]

        portfolio_ret = np.dot(ret, w)
        combined_equity.append(combined_equity[-1] * (1 + portfolio_ret))

    combined_equity = np.array(combined_equity)

    combined = equity_curves[0].copy().reset_index(drop=True).iloc[-min_len:]
    combined["equity"] = combined_equity

    return combined


# =========================
# MAIN
# =========================
def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    from optimizer import WalkForwardOptimizer

    param_space = {
        "atr_period": (10, 25),
        "rsi_period": (12, 20),
        "sl_atr": (1.8, 3.0),
        "ts_atr": (2.2, 3.5),
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

    top_configs = results[:3]

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