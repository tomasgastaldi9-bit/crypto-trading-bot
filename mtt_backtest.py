import pandas as pd
from copy import deepcopy

from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy
from risk_management import RiskManager
from backtester import Backtester


SYMBOL = "BTCUSDT"
TIMEFRAMES = ["1h", "4h"]


def run_single_backtest(interval: str):
    config = deepcopy(DEFAULT_CONFIG)
    config.binance.symbol = SYMBOL
    config.binance.interval = interval

    print(f"\n🚀 Running {SYMBOL} - {interval}")

    # === DATA ===
    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    # === INDICATORS ===
    data = add_indicators(data, config.indicators)

    # === STRATEGY ===
    strategy = TrendMomentumStrategy(config.strategy)
    data = strategy.generate_signals(data)

    # === BACKTEST ===
    risk_manager = RiskManager(config.risk)
    backtester = Backtester(
        risk_manager,
        config.strategy,
        config.risk,
        config.execution,
    )

    result = backtester.run(data)

    equity = result.equity_curve.copy()
    equity = equity[["timestamp", "equity"]]

    # 🔥 NORMALIZAR
    equity["equity"] = equity["equity"] / equity["equity"].iloc[0]

    equity = equity.rename(columns={"equity": interval})

    return equity


def run_portfolio():
    equity_curves = []

    for tf in TIMEFRAMES:
        equity = run_single_backtest(tf)
        equity_curves.append(equity)

    # === MERGE ===
    portfolio = equity_curves[0]
    for eq in equity_curves[1:]:
        portfolio = pd.merge(portfolio, eq, on="timestamp", how="outer")

    portfolio = portfolio.sort_values("timestamp").ffill().dropna()

    # === PORTFOLIO ===
    portfolio["total_equity"] = portfolio[TIMEFRAMES].mean(axis=1)

    portfolio["returns"] = portfolio["total_equity"].pct_change().fillna(0)

    sharpe = (
        portfolio["returns"].mean()
        / portfolio["returns"].std()
        * (24 * 365) ** 0.5
    )

    total_return = (
        portfolio["total_equity"].iloc[-1]
        / portfolio["total_equity"].iloc[0]
        - 1
    )

    print("\n================ MULTI-TF RESULTS ================")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("=================================================")

    return portfolio


if __name__ == "__main__":
    portfolio = run_portfolio()