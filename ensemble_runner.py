from optimizer_runner import BacktesterWrapper, build_signal
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer
from volatility_breakout import VolatilityBreakoutStrategy
from range_strategy import RangeStrategy
from indicators import add_indicators
from risk_management import RiskManager
from backtester import Backtester

import pandas as pd
import numpy as np
import json


# =========================
# HELPERS
# =========================
def clean_equity(eq):
    eq = eq.copy()
    eq["equity"] = eq["equity"].replace([np.inf, -np.inf], np.nan)
    eq["equity"] = eq["equity"].ffill().bfill()

    if eq["equity"].isna().all():
        eq["equity"] = 1.0

    return eq


def compute_turnover(eq):
    returns = eq["equity"].pct_change(fill_method=None).fillna(0)
    return np.mean(np.abs(returns))


def combine_equities_return_based(equity_curves, weights):
    returns_list = []

    for eq in equity_curves:
        r = eq["equity"].pct_change(fill_method=None).fillna(0).values
        returns_list.append(r)

    returns_matrix = np.vstack(returns_list)

    portfolio_returns = np.sum(weights[:, None] * returns_matrix, axis=0)

    equity = 100000 * np.cumprod(1 + portfolio_returns)

    return pd.DataFrame({"equity": equity})


def apply_vol_targeting(equity, target_vol=0.15, window=50):
    returns = equity["equity"].pct_change(fill_method=None).fillna(0)

    realized_vol = returns.rolling(window).std()
    scaling = target_vol / (realized_vol + 1e-8)

    scaling = scaling.clip(0.5, 2.0)

    scaled_returns = returns * scaling.shift(1).fillna(1)

    new_equity = equity["equity"].iloc[0] * np.cumprod(1 + scaled_returns)

    return pd.DataFrame({"equity": new_equity})


def compute_correlation_penalty(aligned_equities):
    returns_list = [
        eq["equity"].pct_change(fill_method=None).fillna(0).values
        for eq in aligned_equities
    ]

    returns_matrix = np.vstack(returns_list)

    corr_matrix = np.corrcoef(returns_matrix)

    penalty = corr_matrix.mean(axis=1)
    penalty = (penalty - penalty.min()) / (penalty.max() - penalty.min() + 1e-8)

    return penalty


# =========================
# VOLATILITY BREAKOUT
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

    return clean_equity(result.equity_curve.copy())


# =========================
# 🔥 RANGE STRATEGY (NUEVO EDGE)
# =========================
def run_range_strategy(wrapper, data):
    config = wrapper.base_config

    indicator_data = add_indicators(data, config.indicators)

    rs = RangeStrategy(config.strategy)
    rs_data = rs.generate_signals(indicator_data.copy())

    rs_data["signal"] = build_signal(rs_data)

    signal_data = rs_data.copy().fillna(0).reset_index(drop=True)

    risk_manager = RiskManager(config.risk)

    backtester = Backtester(
        risk_manager=risk_manager,
        strategy_config=config.strategy,
        risk_config=config.risk,
        execution_config=config.execution,
    )

    result = backtester.run(signal_data)

    return clean_equity(result.equity_curve.copy())


# =========================
# ENSEMBLE
# =========================
def run_ensemble_weighted(wrapper, data, top_configs):
    equity_curves = []
    scores = []

    print("\n=== RUNNING CONFIGS ===")

    for i, config in enumerate(top_configs):
        print(
            f"Config {i+1} | Sharpe: {config['mean_test']:.3f} | "
            f"Std: {config['std_test']:.3f} | Overfit: {config['overfit']:.3f}"
        )

        res = wrapper.run(data, config["params"])
        eq = clean_equity(res["equity_curve"])

        equity_curves.append(eq)

        turnover = compute_turnover(eq)

        score = (
            config["mean_test"]
            - 0.3 * config["std_test"]
            - 0.4 * config["overfit"]
            - 0.2 * turnover
        )

        print(f"Score: {score:.3f} | Turnover: {turnover:.5f}")
        scores.append(score)

    # =========================
    # VOL BREAKOUT
    # =========================
    print("\n=== RUNNING VB ===")
    vb_eq = run_vb_strategy(wrapper, data)
    equity_curves.append(vb_eq)
    scores.append(np.mean(scores))

    # =========================
    # 🔥 RANGE STRATEGY
    # =========================
    print("\n=== RUNNING RANGE STRATEGY ===")
    rs_eq = run_range_strategy(wrapper, data)
    equity_curves.append(rs_eq)

    rs_score = np.mean(scores) * 0.5  # 🔥 bajar influencia del range
    scores.append(rs_score)

    # =========================
    # ALIGN
    # =========================
    min_len = min(len(eq) for eq in equity_curves)

    aligned = []
    for eq in equity_curves:
        e = eq.copy().reset_index(drop=True).iloc[-min_len:]
        aligned.append(clean_equity(e))

    # =========================
    # CORRELATION
    # =========================
    penalty = compute_correlation_penalty(aligned)

    scores = np.array(scores)
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    adjusted_scores = scores - 0.5 * penalty

    softmax_w = np.exp(adjusted_scores)
    softmax_w /= softmax_w.sum()

    uniform_w = np.ones_like(softmax_w) / len(softmax_w)

    alpha = 0.7
    weights = alpha * softmax_w + (1 - alpha) * uniform_w

    print("\n=== WEIGHTS (FULL SYSTEM) ===")
    for i, w in enumerate(weights):
        if i < len(top_configs):
            name = f"Config {i+1}"
        elif i == len(top_configs):
            name = "VB"
        else:
            name = "RANGE"
        print(f"{name}: {w:.3f} | Penalty: {penalty[i]:.3f}")

    # =========================
    # ENSEMBLE
    # =========================
    portfolio = combine_equities_return_based(aligned, weights)

    portfolio = apply_vol_targeting(portfolio)

    return portfolio


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== LOADING DATA ===")

    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    print(f"Data loaded: {len(data)} rows")

    wrapper = BacktesterWrapper(config)

    print("\n=== LOADING CONFIGS ===")

    with open("top_configs.json", "r") as f:
        configs = json.load(f)

    configs = sorted(configs, key=lambda x: x["mean_test"], reverse=True)
    configs = [
        c for c in configs
        if c["overfit"] < 0.8 and c["mean_test"] > 0.3
    ]

    top_configs = configs[:5]

    print(f"Using {len(top_configs)} configs")

    for i, c in enumerate(top_configs):
        print(f"{i+1} | Sharpe: {c['mean_test']:.3f} | Overfit: {c['overfit']:.3f}")

    print("\n=== RUNNING ENSEMBLE ===")

    portfolio = run_ensemble_weighted(wrapper, data, top_configs)

    analyzer = PerformanceAnalyzer(annualization_factor=365*24)

    summary = analyzer.summarize(
        equity_curve=portfolio,
        trades=pd.DataFrame(),
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== RESULTS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")