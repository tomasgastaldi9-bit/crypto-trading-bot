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
    trades = result.trades.copy()

    equity["equity"] = equity["equity"].replace([np.inf, -np.inf], np.nan)
    equity["equity"] = equity["equity"].ffill().bfill()

    if equity["equity"].isna().all():
        equity["equity"] = 1.0

    return equity, trades


# =========================
# ENSEMBLE
# =========================
def run_ensemble_weighted(wrapper, data, top_configs):
    equity_curves = []
    trades_list = []
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
        trades_list.append(res["trades"])

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
    # VB STRATEGY
    # =========================
    print("\n=== RUNNING VOLATILITY BREAKOUT ===")

    vb_equity, vb_trades = run_vb_strategy(wrapper, data)

    equity_curves.append(vb_equity)
    trades_list.append(vb_trades)

    vb_score = np.mean(scores)
    scores.append(vb_score)

    # =========================
    # ALIGN SERIES
    # =========================
    min_len = min(len(eq) for eq in equity_curves)

    aligned = []
    for eq in equity_curves:
        e = eq.copy().reset_index(drop=True).iloc[-min_len:]
        e["equity"] = e["equity"].replace([np.inf, -np.inf], np.nan)
        e["equity"] = e["equity"].ffill().bfill()
        aligned.append(e["equity"].values)

    equity_matrix = np.column_stack(aligned)

    returns_matrix = np.diff(equity_matrix, axis=0) / (equity_matrix[:-1] + 1e-8)

    # =========================
    # 🔥 DYNAMIC WEIGHTS (ROLLING SHARPE)
    # =========================
    window = 200
    dynamic_weights = []

    for i in range(len(returns_matrix)):
        start = max(0, i - window)
        window_returns = returns_matrix[start:i+1]

        if len(window_returns) < 20:
            w = np.ones(len(equity_curves)) / len(equity_curves)
        else:
            mean = window_returns.mean(axis=0)
            std = window_returns.std(axis=0) + 1e-8

            sharpe = mean / std
            sharpe = np.clip(sharpe, 0, None)

            if sharpe.sum() == 0:
                w = np.ones(len(sharpe)) / len(sharpe)
            else:
                w = sharpe / sharpe.sum()

        dynamic_weights.append(w)

    dynamic_weights = np.array(dynamic_weights)

    # =========================
    # 🔥 CORRELATION PENALTY
    # =========================
    corr_matrix = np.corrcoef(returns_matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix)

    avg_corr = np.mean(corr_matrix, axis=1)

    penalty = 1 - avg_corr
    penalty = np.clip(penalty, 0.3, 1.0)

    # =========================
    # COMBINAR
    # =========================
    combined_equity = [equity_matrix[0, 0]]

    for i in range(1, len(equity_matrix)):
        w = dynamic_weights[i - 1] * penalty
        w = w / (w.sum() + 1e-8)

        ret = returns_matrix[i - 1]
        portfolio_ret = np.dot(ret, w)

        combined_equity.append(combined_equity[-1] * (1 + portfolio_ret))

    combined_equity = np.array(combined_equity)

    combined = equity_curves[0].copy().reset_index(drop=True).iloc[-min_len:]
    combined["equity"] = combined_equity

    # =========================
    # MERGE TRADES
    # =========================
    combined_trades = pd.concat(trades_list, ignore_index=True)

    print("\n=== FINAL WEIGHTS (LAST STEP) ===")
    final_w = dynamic_weights[-1] * penalty
    final_w = final_w / (final_w.sum() + 1e-8)

    for i, w in enumerate(final_w):
        if i < len(top_configs):
            print(f"Config {i+1}: {w:.3f}")
        else:
            print(f"VB Strategy: {w:.3f}")

    return combined, combined_trades


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

    top_configs = results[:1]

    print(f"\nUsing {len(top_configs)} configs")

    equity, trades = run_ensemble_weighted(wrapper, data, top_configs)

    analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

    summary = analyzer.summarize(
        equity_curve=equity,
        trades=trades,
        initial_capital=config.risk.initial_capital,
    )

    # =========================
    # SAVE OUTPUTS
    # =========================
    output_paths = analyzer.save_outputs(
        equity_curve=equity,
        trades=trades,
        summary=summary,
        output_config=config.output,
    )

    print("\n=== FILES SAVED ===")
    for k, v in output_paths.items():
        print(f"{k}: {v}")

    print("\n=== ENSEMBLE RESULTS ===")
    print(summary)


if __name__ == "__main__":
    main()