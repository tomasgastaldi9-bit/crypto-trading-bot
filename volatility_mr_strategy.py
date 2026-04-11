import pandas as pd
import numpy as np


class VolatilityMeanReversionStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, df):
        data = df.copy()

        # =========================
        # 🔥 VOL SPIKE DETECTION
        # =========================
        returns = data["close"].pct_change().fillna(0)

        vol = returns.rolling(20).std()
        vol_ma = vol.rolling(50).mean()

        vol_spike = vol > (vol_ma * 1.5)

        # =========================
        # 🔥 PRICE EXTENSION
        # =========================
        window = 20
        mean = data["close"].rolling(window).mean()
        std = data["close"].rolling(window).std()

        zscore = (data["close"] - mean) / (std + 1e-8)

        # =========================
        # 🔥 ENTRY (FADE SPIKE)
        # =========================
        long_entry = (
            (zscore < -2.0)
            & vol_spike
        )

        short_entry = (
            (zscore > 2.0)
            & vol_spike
        )

        # =========================
        # 🔥 EXIT
        # =========================
        exit_signal = (
            (zscore.abs() < 0.5)
            | (~vol_spike)
        )

        # =========================
        # 🔥 COMBINAR
        # =========================
        entry_signal = long_entry | short_entry

        data["entry_signal"] = entry_signal.fillna(False)
        data["exit_signal"] = exit_signal.fillna(False)

        return data