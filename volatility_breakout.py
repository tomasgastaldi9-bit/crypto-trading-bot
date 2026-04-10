import pandas as pd
import numpy as np


class VolatilityBreakoutStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, df):
        data = df.copy()

        # =========================
        # VOLATILITY EXPANSION
        # =========================
        atr = data["atr"]
        atr_ma = atr.rolling(20).mean()

        vol_expansion = atr > atr_ma * 1.5

        # =========================
        # BREAKOUT LEVELS
        # =========================
        high_break = data["high"].rolling(20).max()
        low_break = data["low"].rolling(20).min()

        # =========================
        # ENTRY
        # =========================
        data["entry_signal"] = False
        data["exit_signal"] = False

        data.loc[
            (data["close"] > high_break.shift(1)) & vol_expansion,
            "entry_signal"
        ] = True

        data.loc[
            (data["close"] < low_break.shift(1)) & vol_expansion,
            "entry_signal"
        ] = True

        # =========================
        # EXIT (vol collapse)
        # =========================
        data.loc[
            atr < atr_ma,
            "exit_signal"
        ] = True

        return data