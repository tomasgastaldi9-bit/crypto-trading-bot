from __future__ import annotations

from typing import Any

import pandas as pd

from config import StrategyConfig


class TrendMomentumStrategy:
    """Trend-following strategy that outputs backtester-compatible signals."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Trend confirmation: fast EMA above slow EMA and the slow EMA rising.
        df["trend_up"] = (
            (df["ema_fast"] > df["ema_slow"])
            & (df["ema_slow"].diff() > df["atr"] * 0.05)
        )

        # Breakout over the highest high of the lookback window.
        df["breakout"] = (
            df["close"] > df["high"].shift(1).rolling(self.config.breakout_window).max()
        )
        df["strong_candle"] = df["close"] > df["open"]

        # Require expanding participation and sufficient volatility.
        df["volume_ok"] = (
            df["volume"] > df["volume"].rolling(self.config.breakout_window).mean()
        )
        df["momentum"] = df["rsi"] > self.config.momentum_threshold
        df["vol_ok"] = (df["atr"] / df["close"]) > self.config.atr_min_ratio

        # Avoid buying after the move is already too stretched.
        df["not_extended"] = (
            df["close"] < df["ema_fast"] * (1 + self.config.extended_pct)
        )

        df["entry_signal"] = (
            df["trend_up"]
            & df["breakout"]
            & df["volume_ok"]
            & df["momentum"]
            & df["vol_ok"]
            & df["not_extended"]
            & df["strong_candle"]
        )

        df["exit_signal"] = (
            (df["rsi"] < self.config.exit_rsi_threshold)
            | (df["ema_fast"] < df["ema_slow"])
            | (df["close"] < df["ema_fast"])
        )

        return df


class MeanReversionStrategy:
    """Long-only mean reversion strategy for range-bound market regimes.

    The class preserves the existing architecture by only enriching the input
    dataframe and emitting the same two columns expected by the backtester:
    `entry_signal` and `exit_signal`.

    Stop loss handling remains delegated to the current backtester/risk manager
    via `StrategyConfig.stop_loss_atr_multiple`.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

        # Optional strategy parameters are resolved dynamically so the current
        # config structure does not need to change for this strategy to work.
        self.mean_period = int(self._param("mean_reversion_ema_period", 20))
        self.bb_period = int(self._param("mean_reversion_bb_period", 20))
        self.bb_std = float(self._param("mean_reversion_bb_std", 2.0))
        self.rsi_entry = float(self._param("mean_reversion_rsi_entry", 30.0))
        self.rsi_exit = float(self._param("mean_reversion_rsi_exit", 50.0))
        self.mean_deviation_pct = float(
            self._param("mean_reversion_deviation_pct", 0.015)
        )
        self.signal_lookback = int(self._param("mean_reversion_signal_lookback", 3))
        self.cooldown_bars = int(getattr(config, "cooldown_bars", 0))

        # Regime filter thresholds: only trade when trend characteristics stay muted.
        self.max_ema_distance_ratio = float(
            self._param("mean_reversion_max_ema_distance_ratio", 0.015)
        )
        self.max_slope_ratio = float(
            self._param("mean_reversion_max_slope_ratio", 0.0008)
        )
        self.max_atr_expansion = float(
            self._param("mean_reversion_max_atr_expansion", 1.15)
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a signal dataframe compatible with the existing backtester."""

        df = data.copy()

        # Mean anchor and Bollinger Bands define the reversion zone.
        df["mean_ema"] = df["close"].ewm(
            span=self.mean_period,
            adjust=False,
            min_periods=self.mean_period,
        ).mean()
        rolling_std = df["close"].rolling(self.bb_period, min_periods=self.bb_period).std()
        df["bb_mid"] = df["close"].rolling(self.bb_period, min_periods=self.bb_period).mean()
        df["bb_upper"] = df["bb_mid"] + (rolling_std * self.bb_std)
        df["bb_lower"] = df["bb_mid"] - (rolling_std * self.bb_std)

        # Bonus regime filter: avoid taking mean reversion entries in strong trends.
        df["non_trending"] = self._build_non_trending_regime(df)

        # Setup condition: price is stretched below the mean in a quiet regime.
        df["below_mean"] = df["close"] <= (
            df["mean_ema"] * (1.0 - self.mean_deviation_pct)
        )
        df["lower_band_touch"] = df["low"] <= df["bb_lower"]
        df["oversold_setup"] = (
            df["non_trending"]
            & df["below_mean"]
            & (df["rsi"] < self.rsi_entry)
            & df["lower_band_touch"]
        )

        # Entry trigger: momentum begins to recover or price closes back inside the band.
        df["rsi_cross_back"] = (
            df["rsi"].shift(1).lt(self.rsi_entry) & df["rsi"].ge(self.rsi_entry)
        )
        df["band_reentry"] = (
            df["close"].shift(1).le(df["bb_lower"].shift(1))
            & df["close"].gt(df["bb_lower"])
        )

        recent_setup = (
            df["oversold_setup"]
            .rolling(self.signal_lookback, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )

        raw_entry = df["non_trending"] & recent_setup & (
            df["rsi_cross_back"] | df["band_reentry"]
        )

        # Cooldown keeps the strategy from repeatedly firing in the same local range.
        if self.cooldown_bars > 0:
            recent_entry = (
                raw_entry.shift(1)
                .rolling(self.cooldown_bars, min_periods=1)
                .max()
                .fillna(0)
                .astype(bool)
            )
            df["entry_signal"] = raw_entry & ~recent_entry
        else:
            df["entry_signal"] = raw_entry

        # Exit into the mean or once momentum has normalized.
        df["exit_signal"] = (
            (df["close"] >= df["mean_ema"])
            | (df["rsi"] >= self.rsi_exit)
        )

        return df

    def _build_non_trending_regime(self, df: pd.DataFrame) -> pd.Series:
        """Return True when trend strength is muted enough for reversion trades."""

        ema_distance_ratio = (df["ema_fast"] - df["ema_slow"]).abs() / df["close"]
        slow_slope_ratio = df["ema_slow"].diff().abs() / df["close"]
        atr_expansion = df["atr"] / df["atr"].rolling(self.bb_period, min_periods=1).mean()

        quiet_signals = pd.DataFrame(
            {
                "distance_quiet": ema_distance_ratio <= self.max_ema_distance_ratio,
                "slope_quiet": slow_slope_ratio <= self.max_slope_ratio,
                "atr_quiet": atr_expansion <= self.max_atr_expansion,
            }
        )

        # Requiring 2 of 3 conditions keeps the filter robust without being brittle.
        return quiet_signals.sum(axis=1) >= 2

    def _param(self, name: str, default: Any) -> Any:
        """Resolve optional config fields without forcing a config schema change."""

        return getattr(self.config, name, default)
