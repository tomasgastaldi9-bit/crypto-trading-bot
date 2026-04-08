from __future__ import annotations

import pandas as pd

from config import IndicatorConfig


def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute exponential moving average."""

    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss.replace(0.0, pd.NA)
    indicator = 100.0 - (100.0 / (1.0 + relative_strength))
    return indicator.fillna(100.0).where(average_loss != 0.0, 100.0)


def atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing."""

    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(data: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    """Return a copy of the input data enriched with the required indicators."""

    enriched = data.copy()
    enriched["ema_fast"] = ema(enriched["close"], config.ema_fast_period)
    enriched["ema_slow"] = ema(enriched["close"], config.ema_slow_period)
    enriched["rsi"] = rsi(enriched["close"], config.rsi_period)
    enriched["atr"] = atr(enriched, config.atr_period)
    return enriched
