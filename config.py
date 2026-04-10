from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class BinanceConfig:
    """Configuration for Binance kline retrieval."""

    base_url: str = "https://fapi.binance.com"
    klines_endpoint: str = "/fapi/v1/klines"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_date: str = "2022-01-01"
    end_date: str | None = None
    request_limit: int = 1000
    timeout_seconds: int = 30
    use_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: BASE_DIR / "data_cache")


@dataclass
class IndicatorConfig:
    """Technical indicator parameters."""

    ema_fast_period: int = 50
    ema_slow_period: int = 200
    rsi_period: int = 14
    atr_period: int = 14


@dataclass
class StrategyConfig:
    """Strategy thresholds and filters."""

    momentum_threshold: float = 65.0
    exit_rsi_threshold: float = 50.0
    atr_min_ratio: float = 0.005
    breakout_window: int = 20
    extended_pct: float = 0.05

    # 🔥 NECESARIOS PARA EL BACKTESTER
    stop_loss_atr_multiple: float = 2.5
    trailing_stop_atr_multiple: float = 3.0

    # control de overtrading
    cooldown_bars: int = 5
    # mean reversion tuning
    mean_reversion_rsi_entry: int = 25
    mean_reversion_rsi_exit: int = 55
    mean_reversion_deviation_pct: float = 0.02
    mean_reversion_max_ema_distance_ratio: float = 0.01
    mean_reversion_max_slope_ratio: float = 0.0004
    mean_reversion_max_atr_expansion: float = 1.03
    mr_window: int = 30
    mr_z: float = 2.5
    mr_rsi: float = 25


@dataclass
class RiskConfig:
    """Portfolio and position risk controls."""

    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 3
    max_leverage: float = 1.0
    quantity_precision: int = 6
    min_quantity: float = 0.000001


@dataclass
class ExecutionConfig:
    """Execution cost assumptions."""

    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    annualization_factor: int = 24 * 365
    close_positions_at_end: bool = True


@dataclass
class OutputConfig:
    """Output locations and reporting preferences."""

    output_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    trade_log_filename: str = "trade_log.csv"
    equity_curve_filename: str = "equity_curve.csv"
    summary_filename: str = "summary_report.json"
    equity_plot_filename: str = "equity_curve.png"
    print_summary: bool = True


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    binance: BinanceConfig = field(default_factory=BinanceConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_CONFIG = EngineConfig()
