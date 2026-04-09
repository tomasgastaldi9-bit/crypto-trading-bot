import pandas as pd
from copy import deepcopy

from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy
from risk_management import RiskManager
from backtester import Backtester


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1h", "4h"]


def run_single(symbol: str, timeframe: str):
    config = deepcopy(DEFAULT_CONFIG)
    config.binance.symbol = symbol
    config.binance.interval = timeframe

    print(f"\n🚀 {symbol} - {timeframe}")

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    data = add_indicators(data, config.indicators)

    strategy = TrendMomentumStrategy(config.strategy)
    data = strategy.generate_signals(data)

    risk_manager = RiskManager(config.risk)
    backtester = Backtester(
        risk_manager,
        config.strategy,
        config.risk,
        config.execution,
    )

    result = backtester.run(data)

    equity = result.equity_curve.copy()[["timestamp", "equity"]]

    # 🔥 normalizar
    equity["equity"] = equity["equity"] / equity["equity"].iloc[0]

    name = f"{symbol}_{timeframe}"
    equity = equity.rename(columns={"equity": name})

    return equity


def run_portfolio():
    curves = []

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            curves.append(run_single(symbol, tf))

    # merge
    portfolio = curves[0]
    for c in curves[1:]:
        portfolio = pd.merge(portfolio, c, on="timestamp", how="outer")

    portfolio = portfolio.sort_values("timestamp").ffill().dropna()

    cols = [col for col in portfolio.columns if col != "timestamp"]

    # 🔥 WEIGHTS INTELIGENTES
    weights = {
        "BTCUSDT_1h": 0.25,
        "BTCUSDT_4h": 0.25,
        "ETHUSDT_1h": 0.15,
        "ETHUSDT_4h": 0.15,
        "SOLUSDT_1h": 0.10,
        "SOLUSDT_4h": 0.10,
    }

    # asegurar que todos los pesos existan
    missing = [col for col in cols if col not in weights]
    if missing:
        raise ValueError(f"Faltan weights para: {missing}")

    # 🔥 portfolio ponderado
    portfolio["total_equity"] = sum(
        portfolio[col] * weights[col] for col in cols
    )

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

    print("\n================ PORTFOLIO RESULTS ================")
    print(f"Strategies: {len(cols)}")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("===================================================")

    return portfolio


if __name__ == "__main__":
    run_portfolio()