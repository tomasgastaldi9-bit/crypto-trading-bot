import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from ensemble_runner import run_ensemble_weighted
from optimizer_runner import BacktesterWrapper

# =========================
# CONFIG
# =========================
SYMBOL = "BTCUSDT"
CAPITAL = 100000
SLEEP_TIME = 60 * 5  # 5 min

# =========================
# STATE
# =========================
position = 0
entry_price = 0
equity = CAPITAL

# =========================
# LOGGING
# =========================
LOG_FILE = "paper_log.txt"
TRADES_FILE = "trades.csv"
EQUITY_FILE = "equity_curve.csv"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def log_trade(data):
    df = pd.DataFrame([data])
    if not os.path.exists(TRADES_FILE):
        df.to_csv(TRADES_FILE, index=False)
    else:
        df.to_csv(TRADES_FILE, mode="a", header=False, index=False)

def log_equity(timestamp, equity):
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "equity": equity
    }])
    if not os.path.exists(EQUITY_FILE):
        df.to_csv(EQUITY_FILE, index=False)
    else:
        df.to_csv(EQUITY_FILE, mode="a", header=False, index=False)

# =========================
# INIT
# =========================
config = DEFAULT_CONFIG
loader = BinanceDataLoader(config.binance)
wrapper = BacktesterWrapper(config)

with open("top_configs.json", "r") as f:
    configs = json.load(f)

configs = sorted(configs, key=lambda x: x["mean_test"], reverse=True)
configs = configs[:2]

log("🚀 PAPER TRADING BOT STARTED")

# =========================
# LOOP
# =========================
while True:
    try:
        now = datetime.now()
        log(f"\n⏱ {now}")

        # =========================
        # LOAD DATA
        # =========================
        data = loader.load_klines()

        # =========================
        # RUN MODEL
        # =========================
        portfolio, signals = run_ensemble_weighted(wrapper, data, configs)
        signal = signals.iloc[-1]
        print("Last 5 signals:", signals.tail())
        returns = portfolio["equity"].pct_change().fillna(0)

        signal = np.sign(returns.iloc[-1])
        current_price = loader.get_latest_price()

        log(f"Price: {current_price:.2f} | Signal: {signal}")

        # =========================
        # EXECUTION
        # =========================
        if signal == 1 and position == 0:
            position = 1
            entry_price = current_price

            log(f"🟢 BUY at {entry_price:.2f}")

            log_trade({
                "timestamp": now,
                "type": "BUY",
                "price": entry_price,
                "equity": equity
            })

        elif signal <= 0 and position == 1:
            pnl = (current_price - entry_price) / entry_price
            equity *= (1 + pnl)

            log(f"🔴 SELL at {current_price:.2f}")
            log(f"PnL: {pnl:.4f} | Equity: {equity:.2f}")

            log_trade({
                "timestamp": now,
                "type": "SELL",
                "price": current_price,
                "pnl": pnl,
                "equity": equity
            })

            position = 0

        # =========================
        # EQUITY TRACKING
        # =========================
        log_equity(now, equity)

        # =========================
        # SLEEP
        # =========================
        time.sleep(SLEEP_TIME)

    except Exception as e:
        log(f"❌ ERROR: {e}")
        time.sleep(60)