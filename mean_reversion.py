import numpy as np
import pandas as pd


class MeanReversionStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, df):
        data = df.copy()

        # =========================
        # INDICADORES
        # =========================
        window = self.config.mr_window

        data["mean"] = data["close"].rolling(window).mean()
        data["std"] = data["close"].rolling(window).std()

        data["zscore"] = (data["close"] - data["mean"]) / (data["std"] + 1e-8)

        rsi = data["rsi"]

        # =========================
        # FILTRO ANTI-TREND
        # =========================
        data["ema"] = data["close"].ewm(span=100).mean()
        no_trend = abs(data["close"] - data["ema"]) / data["ema"] < 0.02

        # =========================
        # FILTRO DE VOLUMEN
        # =========================
        data["vol_ma"] = data["volume"].rolling(20).mean()
        high_volume = data["volume"] > data["vol_ma"]

        # =========================
        # PARAMS DINÁMICOS 🔥
        # =========================
        z_thr = getattr(self.config, "mr_z", 2.5)
        rsi_thr = getattr(self.config, "mr_rsi", 25)

        # =========================
        # SEÑALES
        # =========================
        data["entry_signal"] = False
        data["exit_signal"] = False

        # =========================
        # ENTRY LONG
        # =========================
        data.loc[
            (data["zscore"] < -z_thr) &
            (rsi < rsi_thr) &
            no_trend &
            high_volume,
            "entry_signal"
        ] = True

        # =========================
        # ENTRY SHORT
        # =========================
        data.loc[
            (data["zscore"] > z_thr) &
            (rsi > (100 - rsi_thr)) &
            no_trend &
            high_volume,
            "entry_signal"
        ] = True

        # =========================
        # EXIT
        # =========================
        data.loc[
            (data["zscore"].abs() < 0.5),
            "exit_signal"
        ] = True

        return data