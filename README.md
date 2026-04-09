# 📈 Crypto Trading Bot — Multi-Asset & Multi-Timeframe System

## 🚀 Overview

This project implements a **systematic crypto trading strategy** using:

* Trend-following logic
* Multi-timeframe confirmation (1H + 4H)
* Multi-asset diversification (BTC, ETH, SOL)
* Risk-managed position sizing
* Portfolio-level aggregation

The goal is to build a **robust, scalable quantitative trading system** rather than a single-strategy bot.

---

## 🧠 Strategy Logic

The system is based on a **trend-following breakout model**:

### Entry Conditions

* EMA Fast > EMA Slow (trend confirmation)
* EMA Slow slope > 0 (uptrend strength)
* Breakout above recent highs
* RSI momentum filter
* ATR-based volatility filter
* Price not overextended

### Exit Conditions

* Price falls below EMA Slow
* Chandelier trailing stop (ATR-based)

---

## ⚙️ Features

* ✅ Binance Futures data integration
* ✅ Indicator engine (EMA, RSI, ATR)
* ✅ Backtesting engine with:

  * Fees
  * Slippage
  * Position sizing
* ✅ Risk management:

  * Fixed % risk per trade
  * Volatility-adjusted sizing
* ✅ Multi-timeframe portfolio
* ✅ Multi-asset portfolio
* ✅ Weighted allocation system

---

## 📊 Portfolio Architecture

The system combines:

* Assets: BTCUSDT, ETHUSDT, SOLUSDT
* Timeframes: 1H and 4H

Total strategies running simultaneously:

```
BTC_1H
BTC_4H
ETH_1H
ETH_4H
SOL_1H
SOL_4H
```

Each strategy is normalized and combined into a **single portfolio equity curve**.

---

## 🧮 Performance (Sample Backtest)

| Metric        | Value    |
| ------------- | -------- |
| Total Return  | ~15–23%  |
| Sharpe Ratio  | ~0.7–0.8 |
| Max Drawdown  | ~15%     |
| Win Rate      | ~35%     |
| Profit Factor | ~1.2     |

> ⚠️ Results depend on data range and parameters.

---

## 📁 Project Structure

```
TradingBot/
│
├── backtester.py
├── strategy.py
├── risk_management.py
├── indicators.py
├── data_loader.py
├── config.py
│
├── multi_timeframe_backtest.py
├── multi_asset_multitf.py
│
├── outputs/
├── data_cache/
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Run single backtest

```
python main.py
```

---

### 3. Run multi-timeframe

```
python multi_timeframe_backtest.py
```

---

### 4. Run full portfolio (recommended)

```
python multi_asset_multitf.py
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

* Not financial advice
* Past performance ≠ future results
* Trading involves risk

---

## 🧠 Future Improvements

* Portfolio-level risk management
* Capital allocation optimization
* Additional strategies (mean reversion)
* Live trading integration (Binance API)
* Walk-forward optimization

---

## 👤 Author

Developed by Tomas Gastaldi

---

## ⭐ Final Note

This project evolves from a single trading bot into a **portfolio-level quantitative system**, focusing on:

> **robustness, diversification, and risk management over overfitting**

---
