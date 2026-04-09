import pandas as pd
from copy import deepcopy

from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy
from risk_management import RiskManager
from backtester import Backtester


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def run_single_backtest(symbol: str):
    config = deepcopy(DEFAULT_CONFIG)
    config.binance.symbol = symbol

    print(f"\n🚀 Running backtest for {symbol}")

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

    # 🔥 NORMALIZACIÓN CORRECTA (CRÍTICO)
    equity["equity"] = equity["equity"] / equity["equity"].iloc[0]

    equity = equity.rename(columns={"equity": symbol})

    return equity


def run_portfolio():
    equity_curves = []

    for symbol in SYMBOLS:
        equity = run_single_backtest(symbol)
        equity_curves.append(equity)

    # === MERGE ===
    portfolio = equity_curves[0]
    for eq in equity_curves[1:]:
        portfolio = pd.merge(portfolio, eq, on="timestamp", how="outer")

    # ordenar + forward fill
    portfolio = portfolio.sort_values("timestamp").ffill().dropna()

    # === PORTFOLIO EQUITY ===
    portfolio["total_equity"] = portfolio[SYMBOLS].mean(axis=1)

    # === RETURNS ===
    portfolio["returns"] = portfolio["total_equity"].pct_change().fillna(0)

    # === METRICS ===
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

    print("\n================ PORTFOLIO RESULTS ================")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("===================================================")

    return portfolio


if __name__ == "__main__":
    portfolio = run_portfolio()