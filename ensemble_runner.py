from optimizer_runner import BacktesterWrapper, build_signal
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer
from volatility_breakout import VolatilityBreakoutStrategy
from range_strategy import RangeStrategy
from volatility_mr_strategy import VolatilityMeanReversionStrategy
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


def combine_equities_return_based(equity_curves, weights, fee=0.0004, slippage=0.0003):
    returns_list = []

    for eq in equity_curves:
        r = eq["equity"].pct_change(fill_method=None).fillna(0).values
        returns_list.append(r)

    returns_matrix = np.vstack(returns_list)

    # 🔥 portfolio returns base
    portfolio_returns = np.sum(weights[:, None] * returns_matrix, axis=0)

    # =========================
    # 🔥 TURNOVER COST (CLAVE)
    # =========================
    # aproximación: cambio en contribución de cada estrategia
    weighted_returns = weights[:, None] * returns_matrix

    # turnover proxy
    turnover = np.sum(np.abs(np.diff(weighted_returns, axis=1)), axis=0)

    # agregar 0 al inicio para alinear
    turnover = np.insert(turnover, 0, 0)

    # costo total
    cost_per_trade = fee + slippage
    costs = turnover * cost_per_trade

    # aplicar costos
    net_returns = portfolio_returns - costs

    equity = 100000 * np.cumprod(1 + net_returns)

    return pd.DataFrame({"equity": equity})


# =========================
# PERFORMANCE WEIGHTS
# =========================
def compute_performance_weights(equity_curves, lookback=200):
    perf_scores = []

    for eq in equity_curves:
        returns = eq["equity"].pct_change(fill_method=None).fillna(0)
        recent = returns.iloc[-lookback:]

        mean_ret = recent.mean()
        std_ret = recent.std() + 1e-8

        sharpe_like = mean_ret / std_ret
        perf_scores.append(sharpe_like)

    perf_scores = np.array(perf_scores)
    perf_scores = (perf_scores - perf_scores.mean()) / (perf_scores.std() + 1e-8)

    weights = np.exp(perf_scores)
    weights /= weights.sum()

    return weights


# =========================
# VOL TARGETING
# =========================
def apply_vol_targeting(equity, target_vol=0.18, window=50):
    returns = equity["equity"].pct_change(fill_method=None).fillna(0)

    realized_vol = returns.rolling(window).std()
    vol_scaling = target_vol / (realized_vol + 1e-8)
    vol_scaling = vol_scaling.clip(0.7, 1.8)

    eq = equity["equity"]
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / (rolling_max + 1e-8)

    dd_scaling = 1 + 0.5 * drawdown
    dd_scaling = dd_scaling.clip(0.7, 1.0)

    vol_long = realized_vol.rolling(100).mean()
    vol_regime = realized_vol / (vol_long + 1e-8)
    regime_scaling = np.where(vol_regime > 1.3, 0.85, 1.05)

    final_scaling = vol_scaling * dd_scaling * regime_scaling
    final_scaling = np.clip(final_scaling, 0.5, 2.0)

    scaled_returns = returns * pd.Series(final_scaling, index=returns.index).shift(1).fillna(1)
    new_equity = equity["equity"].iloc[0] * np.cumprod(1 + scaled_returns)

    return pd.DataFrame({"equity": new_equity})


# =========================
# CORRELATION PENALTY
# =========================
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
# REGIME FILTER
# =========================
def compute_regime_filter(data):
    price = data["close"]

    ema_fast = price.ewm(span=50).mean()
    ema_slow = price.ewm(span=200).mean()
    trend = (ema_fast > ema_slow).astype(int)

    returns = price.pct_change().fillna(0)
    vol = returns.rolling(50).std()
    vol_threshold = vol.rolling(200).mean()

    high_vol = (vol > vol_threshold).astype(int)

    regime = trend & high_vol
    return regime


# =========================
# STRATEGY RUNNERS
# =========================
def run_strategy(wrapper, data, strategy_class):
    config = wrapper.base_config
    indicator_data = add_indicators(data, config.indicators)

    strat = strategy_class(config.strategy)
    strat_data = strat.generate_signals(indicator_data.copy())
    strat_data["signal"] = build_signal(strat_data)

    signal_data = strat_data.copy().fillna(0).reset_index(drop=True)

    result = Backtester(
        RiskManager(config.risk),
        config.strategy,
        config.risk,
        config.execution
    ).run(signal_data)

    return clean_equity(result.equity_curve.copy())


# =========================
# ENSEMBLE
# =========================
def run_ensemble_weighted(wrapper, data, top_configs):
    equity_curves = []
    scores = []

    print("\n=== RUNNING CONFIGS ===")

    for i, config in enumerate(top_configs):
        print(f"Config {i+1} | Sharpe: {config['mean_test']:.3f}")

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

        scores.append(score)

    print("\n=== RUNNING VB ===")
    equity_curves.append(run_strategy(wrapper, data, VolatilityBreakoutStrategy))
    scores.append(np.mean(scores))

    print("\n=== RUNNING RANGE STRATEGY ===")
    equity_curves.append(run_strategy(wrapper, data, RangeStrategy))
    scores.append(np.mean(scores) * 0.5)

    print("\n=== RUNNING VOL MR STRATEGY ===")
    equity_curves.append(run_strategy(wrapper, data, VolatilityMeanReversionStrategy))
    scores.append(np.mean(scores) * 0.05)

    min_len = min(len(eq) for eq in equity_curves)
    aligned = [clean_equity(eq.iloc[-min_len:].reset_index(drop=True)) for eq in equity_curves]

    penalty = compute_correlation_penalty(aligned)

    scores = np.array(scores)
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    adjusted_scores = scores - 0.5 * penalty

    softmax_w = np.exp(adjusted_scores)
    softmax_w /= softmax_w.sum()

    uniform_w = np.ones_like(softmax_w) / len(softmax_w)

    perf_w = compute_performance_weights(aligned)
    perf_w = np.clip(perf_w, 0.05, 0.5)
    perf_w[-1] *= 0.15
    perf_w = perf_w / perf_w.sum()

    weights = 0.6 * softmax_w + 0.3 * uniform_w + 0.1 * perf_w

    print("\n=== WEIGHTS ===")
    print(weights)

    portfolio = combine_equities_return_based(aligned, weights)
    portfolio = apply_vol_targeting(portfolio)

    # 🔥 REGIME FILTER
    regime = compute_regime_filter(data).iloc[-len(portfolio):].reset_index(drop=True)

    returns = portfolio["equity"].pct_change(fill_method=None).fillna(0)
    filtered_returns = returns * regime

    portfolio["equity"] = portfolio["equity"].iloc[0] * (1 + filtered_returns).cumprod()

    # =========================
    # GENERATE ENSEMBLE SIGNALS
    # =========================
    returns = portfolio["equity"].pct_change(fill_method=None).fillna(0)

    # señal base
    signals = np.sign(returns)

    # evitar ruido (micro movimientos)
    threshold = returns.rolling(10).std() * 0.5
    signals = np.where(np.abs(returns) > threshold, signals, 0)

    signals = pd.Series(signals, index=portfolio.index)

    return portfolio, signals


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== LOADING DATA ===")

    config = DEFAULT_CONFIG
    data = BinanceDataLoader(config.binance).load_klines()

    wrapper = BacktesterWrapper(config)

    with open("top_configs.json", "r") as f:
        configs = json.load(f)

    configs = sorted(configs, key=lambda x: x["mean_test"], reverse=True)
    configs = [c for c in configs if c["overfit"] < 0.8 and c["mean_test"] > 0.3]

    top_configs = configs[:2]

    portfolio = run_ensemble_weighted(wrapper, data, top_configs)

    summary = PerformanceAnalyzer(annualization_factor=365*24).summarize(
        equity_curve=portfolio,
        trades=pd.DataFrame(),
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== RESULTS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")