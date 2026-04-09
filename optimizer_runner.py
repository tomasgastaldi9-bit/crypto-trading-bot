from optimizer import WalkForwardOptimizer
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from indicators import add_indicators
from strategy import TrendMomentumStrategy
from risk_management import RiskManager
from backtester import Backtester
from performance import PerformanceAnalyzer

from copy import deepcopy


class BacktesterWrapper:
    def __init__(self, base_config):
        self.base_config = base_config

    def run(self, data, params):
        config = deepcopy(self.base_config)

        config.indicators.atr_period = int(params["atr_period"])
        config.indicators.rsi_period = int(params["rsi_period"])
        config.strategy.stop_loss_atr_multiple = params["sl_atr"]
        config.strategy.trailing_stop_atr_multiple = params["ts_atr"]

        indicator_data = add_indicators(data, config.indicators)

        strategy = TrendMomentumStrategy(config.strategy)
        signal_data = strategy.generate_signals(indicator_data)
        signal_data = signal_data.dropna().reset_index(drop=True)

        risk_manager = RiskManager(config.risk)

        backtester = Backtester(
            risk_manager=risk_manager,
            strategy_config=config.strategy,
            risk_config=config.risk,
            execution_config=config.execution,
        )

        result = backtester.run(signal_data)

        analyzer = PerformanceAnalyzer(config.execution.annualization_factor)

        summary = analyzer.summarize(
            equity_curve=result.equity_curve,
            trades=result.trades,
            initial_capital=config.risk.initial_capital,
        )

        return {
            "sharpe": summary["sharpe_ratio"],
            "max_drawdown": summary["max_drawdown"],
            "equity_curve": result.equity_curve   # 👈 AGREGAR
        }


def main():
    config = DEFAULT_CONFIG

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    wrapper = BacktesterWrapper(config)

    param_space = {
    "atr_period": (10, 25),      # ↓ menos rango
    "rsi_period": (12, 20),

    "sl_atr": (1.8, 3.0),
    "ts_atr": (2.2, 3.5),
    }

    optimizer = WalkForwardOptimizer(
        backtester=wrapper,
        param_space=param_space,
        n_trials=100,
        train_size=0.5,
        test_size=0.2,
        step_size=0.1,
        min_folds=3
    )

    results = optimizer.optimize(data)

    if results:
        optimizer.report(results)


if __name__ == "__main__":
    main()