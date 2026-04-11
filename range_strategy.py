import pandas as pd
import numpy as np


class RangeStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, df):
        data = df.copy()

        # =========================
        # 🔥 VOL BAJA
        # =========================
        atr = data["atr"]
        atr_ma = atr.rolling(20).mean()

        low_vol = (atr / (atr_ma + 1e-8)) < 1.08

        # =========================
        # 🔥 NO TREND
        # =========================
        trend_strength = (data["ema_fast"] - data["ema_slow"]).abs() / data["close"]
        sideways = trend_strength < 0.0025

        # =========================
        # 🔥 Z-SCORE
        # =========================
        window = 20
        mean = data["close"].rolling(window).mean()
        std = data["close"].rolling(window).std()

        zscore = (data["close"] - mean) / (std + 1e-8)

        # =========================
        # 🔥 RSI
        # =========================
        rsi = data["rsi"]

        # =========================
        # 🔥 🔥 NUEVO: FILTRO DE RANGO REAL
        # =========================
        range_width = std / (mean + 1e-8)
        valid_range = range_width > 0.004  # clave

        # =========================
        # 🔥 ENTRY LONG
        # =========================
        long_entry = (
            (zscore < -2.0)
            & (rsi < 32)
            & low_vol
            & sideways
            & valid_range
        )

        # =========================
        # 🔥 ENTRY SHORT
        # =========================
        short_entry = (
            (zscore > 2.0)
            & (rsi > 68)
            & low_vol
            & sideways
            & valid_range
        )

        # =========================
        # 🔥 EXIT LONG
        # =========================
        exit_long = (
            (zscore > -0.3)
            | (~low_vol)
            | (~sideways)
        )

        # =========================
        # 🔥 EXIT SHORT
        # =========================
        exit_short = (
            (zscore < 0.3)
            | (~low_vol)
            | (~sideways)
        )

        # =========================
        # 🔥 COMBINAR
        # =========================
        entry_signal = long_entry | short_entry
        exit_signal = exit_long | exit_short

        # =========================
        # CLEAN
        # =========================
        data["entry_signal"] = entry_signal.fillna(False)
        data["exit_signal"] = exit_signal.fillna(False)

        return data