from optimizer import WalkForwardOptimizer
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy
from risk_management import RiskManager
from backtester import Backtester
from performance import PerformanceAnalyzer
from mean_reversion import MeanReversionStrategy
from volatility_breakout import VolatilityBreakoutStrategy

from copy import deepcopy
import numpy as np
import pandas as pd


# =========================
# BUILD SIGNAL (entry/exit → position)
# =========================
def build_signal(df):
    # fallback si no existen señales
    if "entry_signal" not in df.columns or "exit_signal" not in df.columns:
        return np.zeros(len(df))

    signal = np.zeros(len(df))
    in_position = False

    for i in range(len(df)):
        entry = df["entry_signal"].iloc[i]
        exit_ = df["exit_signal"].iloc[i]

        if not in_position and entry:
            signal[i] = 1
            in_position = True

        elif in_position and exit_:
            signal[i] = 0
            in_position = False

        else:
            signal[i] = signal[i-1] if i > 0 else 0

    return signal


class BacktesterWrapper:
    def __init__(self, base_config):
        self.base_config = base_config

    def run(self, data, params):
        config = deepcopy(self.base_config)

        # =========================
        # PARAMS
        # =========================
        config.indicators.atr_period = int(params["atr_period"])
        config.indicators.rsi_period = int(params["rsi_period"])
        config.strategy.stop_loss_atr_multiple = params["sl_atr"]
        config.strategy.trailing_stop_atr_multiple = params["ts_atr"]

        config.strategy.mr_window = int(params["mr_window"])
        config.strategy.mr_z = params["mr_z"]
        config.strategy.mr_rsi = params["mr_rsi"]

        # =========================
        # INDICATORS
        # =========================
        indicator_data = add_indicators(data, config.indicators)

        # =========================
        # STRATEGIES
        # =========================
        trend_strategy = TrendMomentumStrategy(config.strategy)
        mr_strategy = MeanReversionStrategy(config.strategy)

        trend_data = trend_strategy.generate_signals(indicator_data.copy())
        mr_data = mr_strategy.generate_signals(indicator_data.copy())

        # =========================
        # BUILD SIGNALS
        # =========================
        trend_data["signal"] = build_signal(trend_data)
        mr_data["signal"] = build_signal(mr_data)

        # =========================
        # CONVICTION
        # =========================
        trend_strength = indicator_data["close"].pct_change(20).abs()

        if "zscore" in mr_data.columns:
            mr_strength = mr_data["zscore"].abs() / 3.0
        else:
            mr_strength = np.zeros(len(mr_data))

        trend_weight = np.clip(trend_strength.fillna(0), 0, 1)
        mr_weight = np.clip(mr_strength.fillna(0), 0, 1)

        # =========================
        # 🔥 REGIME DETECTION (MEJORADO)
        # =========================
        returns_50 = indicator_data["close"].pct_change(50).abs()
        volatility = indicator_data["close"].pct_change().rolling(50).std()

        trend_score = returns_50 / (volatility + 1e-8)
        trend_score = trend_score.fillna(0)

        strong_trend = trend_score > 1.2
        sideways = trend_score < 0.8

        # dynamic weights
        trend_w = np.where(strong_trend, 1.0, 0.4)
        mr_w = np.where(sideways, 1.0, 0.3)

        trend_w = pd.Series(trend_w).rolling(10).mean().fillna(0.5)
        mr_w = pd.Series(mr_w).rolling(10).mean().fillna(0.5)

        # =========================
        # 🔥 COMBINACIÓN FINAL (LONG + SHORT)
        # =========================
        combined_signal = (
            trend_data["signal"] * trend_weight * trend_w +
            mr_data["signal"] * mr_weight * mr_w
        )

        combined_signal = np.clip(combined_signal, -1, 1)

        # =========================
        # 🔥 ENTRY / EXIT LONG + SHORT
        # =========================
        signal_data = trend_data.copy()

        # LONG
        signal_data["entry_long"] = (
            (combined_signal > 0.5) &
            (combined_signal.shift(1) <= 0.5)
        )

        signal_data["exit_long"] = (
            (combined_signal < 0.2) &
            (combined_signal.shift(1) >= 0.2)
        )

        # SHORT
        signal_data["entry_short"] = (
            (combined_signal < -0.5) &
            (combined_signal.shift(1) >= -0.5)
        )

        signal_data["exit_short"] = (
            (combined_signal > -0.2) &
            (combined_signal.shift(1) <= -0.2)
        )

        signal_data = signal_data.fillna(False).reset_index(drop=True)

        # =========================
        # BACKTEST
        # =========================
        risk_manager = RiskManager(config.risk)

        backtester = Backtester(
            risk_manager=risk_manager,
            strategy_config=config.strategy,
            risk_config=config.risk,
            execution_config=config.execution,
        )

        result = backtester.run(signal_data)

        equity = result.equity_curve.copy()

        # =========================
        # VOL TARGET
        # =========================
        returns = equity["equity"].pct_change().fillna(0)

        rolling_vol = returns.rolling(24).std()
        target_vol = 0.015

        scaling = target_vol / (rolling_vol + 1e-8)
        scaling = scaling.clip(0.5, 1.2)

        scaled_returns = returns * scaling
        equity["equity"] = (1 + scaled_returns).cumprod() * equity["equity"].iloc[0]

        result.equity_curve = equity

        # =========================
        # PERFORMANCE
        # =========================
        analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

        summary = analyzer.summarize(
            equity_curve=result.equity_curve,
            trades=result.trades,
            initial_capital=config.risk.initial_capital,
        )

        return {
            "sharpe": summary["sharpe_ratio"],
            "max_drawdown": summary["max_drawdown"],
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }
        
def run_vb_strategy(wrapper, data):
    config = wrapper.base_config

    indicator_data = add_indicators(data, config.indicators)

    vb = VolatilityBreakoutStrategy(config.strategy)
    vb_data = vb.generate_signals(indicator_data.copy())

    vb_data["signal"] = build_signal(vb_data)

    signal_data = vb_data.copy()
    signal_data = signal_data.fillna(0).reset_index(drop=True)

    risk_manager = RiskManager(config.risk)

    backtester = Backtester(
        risk_manager=risk_manager,
        strategy_config=config.strategy,
        risk_config=config.risk,
        execution_config=config.execution,
    )

    result = backtester.run(signal_data)

    return {
    "equity_curve": result.equity_curve,
    "trades": result.trades
    }


def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    # =========================
    # PARAM SPACE
    # =========================
    param_space = {
        "atr_period": (10, 25),
        "rsi_period": (12, 20),
        "sl_atr": (1.8, 3.0),
        "ts_atr": (2.2, 3.5),

        # 🔥 ESTO ES LO QUE TE FALTABA
        "mr_window": (25, 45),
        "mr_z": (2.0, 2.8),
        "mr_rsi": (22, 30),
    }
    # =========================
    # OPTIMIZER
    # =========================
    optimizer = WalkForwardOptimizer(
        backtester=wrapper,
        param_space=param_space,
        n_trials=100,
        train_size=0.5,
        test_size=0.2,
        step_size=0.1,
        min_folds=2
    )

    results = optimizer.optimize(data)

    if results:
        optimizer.report(results)


if __name__ == "__main__":
    main()