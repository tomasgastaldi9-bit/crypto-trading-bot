from __future__ import annotations
import pandas as pd
import numpy as np
from config import StrategyConfig

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores: EMA50, EMA200, RSI14, ATR14, pendiente de EMA200."""
    df = df.copy()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA200_slope'] = df['EMA200'] - df['EMA200'].shift(1)
    # RSI(14) Wilder
    delta = df['close'].diff()
    up = np.where(delta>0, delta, 0.0)
    down = np.where(delta<0, -delta, 0.0)
    avg_up = pd.Series(up).ewm(alpha=1/14, adjust=False).mean()
    avg_down = pd.Series(down).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    df['RSI'] = 100 - (100/(1+rs))
    # ATR(14)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df

def is_trend(df: pd.DataFrame, idx: int, config: StrategyConfig) -> bool:
    """Verificar tendencia alcista o bajista en el índice dado."""
    ema50 = df.at[idx, 'EMA50']
    ema200 = df.at[idx, 'EMA200']
    slope = df.at[idx, 'EMA200_slope']
    atr = df.at[idx, 'ATR']
    price = df.at[idx, 'close']
    if price == 0 or (atr/price) < config.atr_min_ratio:
        return False
    # Tendencia alcista: EMA50>EMA200 y EMA200 ascendente
    return (ema50 > ema200) and (slope > 0)

def is_breakout(df: pd.DataFrame, idx: int, window: int) -> bool:
    """Detectar breakout en el índice dado: cierre > máximo previo de N velas."""
    if idx < window: 
        return False
    prev_high = df['high'].shift(1).rolling(window=window).max().iat[idx]
    return df.at[idx, 'close'] > prev_high

class TrendMomentumStrategy:
    """Estrategia swing/intradía con EMA, RSI, ATR y filtros de fuerza."""
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # === TREND ===
        df["trend_up"] = (
        (df["ema_fast"] > df["ema_slow"]) &
        (df["ema_slow"].diff() > df["atr"] * 0.05)
)

        # === BREAKOUT ===
        df["breakout"] = df["close"] > df["high"].shift(1).rolling(self.config.breakout_window).max()

        df["strong_candle"] = df["close"] > df["open"]
        
        # === VOLUME ===
        df["volume_ok"] = df["volume"] > df["volume"].rolling(self.config.breakout_window).mean()

        # === MOMENTUM ===
        df["momentum"] = df["rsi"] > self.config.momentum_threshold

        # === VOLATILITY ===
        df["vol_ok"] = (df["atr"] / df["close"]) > self.config.atr_min_ratio

        # === NOT EXTENDED ===
        df["not_extended"] = df["close"] < df["ema_fast"] * (1 + self.config.extended_pct)

        # === ENTRY ===
        df["entry_signal"] = (
            df["trend_up"]
            & df["breakout"]
            & df["volume_ok"]
            & df["momentum"]
            & df["vol_ok"]
            & df["not_extended"]
            & df["strong_candle"]
        )

        # === EXIT ===
        df["exit_signal"] = (
            (df["rsi"] < self.config.exit_rsi_threshold)
            | (df["ema_fast"] < df["ema_slow"])
            | (df["close"] < df["ema_fast"])
        )

        return df

if __name__ == "__main__":
    import config
    df_sample = pd.read_csv(config.sample_data_path, parse_dates=['datetime'], index_col='datetime')
    strat = TrendMomentumStrategy(config.StrategyConfig())
    sigs = strat.generate_signals(df_sample)
    print(sigs[['signal','entry_price','stop_price']].dropna().head(10))

