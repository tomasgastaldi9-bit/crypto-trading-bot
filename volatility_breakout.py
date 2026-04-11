import pandas as pd
import numpy as np


class VolatilityBreakoutStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, df):
        data = df.copy()

        atr = data["atr"]
        atr_ma = atr.rolling(20).mean()

        # =========================
        # 🔥 VOL EXPANSION (mejor calibrado)
        # =========================
        vol_expansion = (atr / (atr_ma + 1e-8)) > 1.2

        # =========================
        # 🔥 TREND
        # =========================
        trend = data["ema_fast"] > data["ema_slow"]

        # =========================
        # 🔥 BREAKOUT
        # =========================
        high_break = data["high"].rolling(20).max()

        breakout = data["close"] > high_break.shift(1)

        # =========================
        # 🔥 CONFIRMACIÓN (NUEVO)
        # =========================
        continuation = data["close"] > data["close"].shift(1)

        # =========================
        # 🔥 ANTI-WICK (MUY IMPORTANTE)
        # =========================
        candle_range = data["high"] - data["low"]
        body = (data["close"] - data["open"]).abs()

        strong_candle = (body / (candle_range + 1e-8)) > 0.6

        # =========================
        # 🔥 MOMENTUM
        # =========================
        momentum = data["rsi"] > 55

        # =========================
        # ENTRY (MEJORADO)
        # =========================
        data["entry_signal"] = (
            breakout
            & continuation
            & strong_candle
            & vol_expansion
            & trend
            & momentum
        )

        # =========================
        # 🔥 EXIT MEJORADO
        # =========================
        data["exit_signal"] = (
            (data["rsi"] < 48)  # sale antes
            | (data["close"] < data["ema_fast"])
        )

        return data