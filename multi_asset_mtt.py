import pandas as pd
from copy import deepcopy

from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy, MeanReversionStrategy
from risk_management import RiskManager
from backtester import Backtester


ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", "XRPUSDT"]
TIMEFRAMES = ["1h", "4h"]
STRATEGIES = ["trend", "mean"]


# 🔥 FILTRO AUTOMÁTICO DE ASSETS
def filter_assets(symbols):
    CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    valid_symbols = []

    for symbol in symbols:
        print(f"\n🔍 Evaluating {symbol}")

        config = deepcopy(DEFAULT_CONFIG)
        config.binance.symbol = symbol

        loader = BinanceDataLoader(config.binance)
        data = loader.load_klines()

        data = add_indicators(data, config.indicators)

        atr_ratio = (data["atr"] / data["close"]).mean()
        ema_slope = (data["ema_slow"].diff()).mean()
        volatility = data["close"].pct_change().std()

        print(f"ATR: {atr_ratio:.4f} | slope: {ema_slope:.6f} | vol: {volatility:.4f}")

        # 🔥 criterio más inteligente
        if (
            atr_ratio > 0.003
            and volatility > 0.008
        ):
            valid_symbols.append(symbol)

    # 🔥 SIEMPRE incluir BTC/ETH
    final_symbols = list(set(valid_symbols + CORE_SYMBOLS))

    print("\n✅ FINAL SYMBOLS:", final_symbols)

    return final_symbols


def run_single(symbol: str, timeframe: str, strategy_type: str):
    config = deepcopy(DEFAULT_CONFIG)
    config.binance.symbol = symbol
    config.binance.interval = timeframe

    print(f"\n🚀 {symbol} - {timeframe} - {strategy_type}")

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    data = add_indicators(data, config.indicators)

    if strategy_type == "trend":
        strategy = TrendMomentumStrategy(config.strategy)
    else:
        strategy = MeanReversionStrategy(config.strategy)

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
    equity["equity"] = equity["equity"] / equity["equity"].iloc[0]

    name = f"{symbol}_{timeframe}_{strategy_type}"
    equity = equity.rename(columns={"equity": name})

    return equity


def run_portfolio():
    # 🔥 seleccionar assets automáticamente
    SYMBOLS = filter_assets(ALL_SYMBOLS)

    if not SYMBOLS:
        raise ValueError("❌ No assets passed the filter")

    curves = []

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            for strat in STRATEGIES:
                curves.append(run_single(symbol, tf, strat))

    # merge
    portfolio = curves[0]
    for c in curves[1:]:
        portfolio = pd.merge(portfolio, c, on="timestamp", how="outer")

    portfolio = portfolio.sort_values("timestamp").ffill().dropna()

    cols = [col for col in portfolio.columns if col != "timestamp"]

    # 🔥 separar estrategias
    trend_cols = [col for col in cols if "trend" in col]
    mean_cols = [col for col in cols if "mean" in col]

    # 🔥 separar core vs alt
    core_assets = ["BTCUSDT", "ETHUSDT"]

    core_trend = [col for col in trend_cols if any(c in col for c in core_assets)]
    core_mean = [col for col in mean_cols if any(c in col for c in core_assets)]

    alt_trend = [col for col in trend_cols if col not in core_trend]
    alt_mean = [col for col in mean_cols if col not in core_mean]

    portfolio["total_equity"] = 0.0

    # 🔥 CORE (70%)
    if core_trend:
        portfolio["total_equity"] += (
            sum(portfolio[col] for col in core_trend)
            * (0.5 / len(core_trend))   # trend domina
        )

    if core_mean:
        portfolio["total_equity"] += (
            sum(portfolio[col] for col in core_mean)
            * (0.2 / len(core_mean))
        )

    # 🔥 ALT (30%)
    if alt_trend:
        portfolio["total_equity"] += (
            sum(portfolio[col] for col in alt_trend)
            * (0.2 / len(alt_trend))
        )

    if alt_mean:
        portfolio["total_equity"] += (
            sum(portfolio[col] for col in alt_mean)
            * (0.1 / len(alt_mean))
        )

    # returns
    portfolio["returns"] = portfolio["total_equity"].pct_change().fillna(0)

    # sharpe
    sharpe = (
        portfolio["returns"].mean()
        / portfolio["returns"].std()
        * (24 * 365) ** 0.5
    )

    # total return
    total_return = (
        portfolio["total_equity"].iloc[-1]
        / portfolio["total_equity"].iloc[0]
        - 1
    )

    print("\n================ PORTFOLIO RESULTS ================")
    print(f"Assets: {SYMBOLS}")
    print(f"Strategies: {len(cols)}")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("===================================================")

    return portfolio


if __name__ == "__main__":
    run_portfolio()