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
        # 🔥 VOL EXPANSION REAL
        # =========================
        vol_expansion = (atr / (atr_ma + 1e-8)) > 1.3

        # =========================
        # 🔥 TREND FILTER
        # =========================
        trend = data["ema_fast"] > data["ema_slow"]

        # =========================
        # 🔥 BREAKOUT
        # =========================
        high_break = data["high"].rolling(20).max()

        breakout = data["close"] > high_break.shift(1)

        # =========================
        # 🔥 MOMENTUM CONFIRMATION
        # =========================
        momentum = data["rsi"] > 55

        # =========================
        # ENTRY (SOLO LONG)
        # =========================
        data["entry_signal"] = (
            breakout
            & vol_expansion
            & trend
            & momentum
        )

        # =========================
        # 🔥 EXIT INTELIGENTE
        # =========================
        data["exit_signal"] = (
            (data["rsi"] < 50)
            | (data["close"] < data["ema_fast"])
        )

        return data