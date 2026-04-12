import pandas as pd
import numpy as np
import json

from ensemble_runner import run_ensemble_weighted
from optimizer_runner import BacktesterWrapper
from config import DEFAULT_CONFIG
from data_loader import BinanceDataLoader
from performance import PerformanceAnalyzer

def apply_correlation_penalty(portfolios, base_weights):
    returns_matrix = []

    for p in portfolios:
        r = p["equity"].pct_change(fill_method=None).fillna(0)
        returns_matrix.append(r.values)

    returns_matrix = np.array(returns_matrix)

    # matriz de correlación
    corr = np.corrcoef(returns_matrix)

    penalties = []

    for i in range(len(base_weights)):
        # promedio de correlación con otros
        avg_corr = np.mean(np.abs(corr[i]))
        penalty = 1 - avg_corr  # más correlación → menos peso
        penalties.append(penalty)

    penalties = np.array(penalties)

    # aplicar penalty
    new_weights = base_weights * penalties

    # normalizar
    new_weights = new_weights / new_weights.sum()

    return new_weights, penalties

def apply_vol_target_to_asset(equity, target_vol=0.15):
    returns = equity["equity"].pct_change(fill_method=None).fillna(0)

    # volatilidad rolling
    vol = returns.rolling(100).std() * np.sqrt(252)
    vol = vol.replace(0, np.nan).bfill()

    # scaling dinámico
    scaling = target_vol / (vol + 1e-8)

    # 🔥 limitar leverage
    scaling = scaling.clip(0.3, 2.0)

    # aplicar
    scaled_returns = returns * scaling
    new_equity = (1 + scaled_returns).cumprod()

    return new_equity.to_frame(name="equity")

def compute_asset_weights(portfolios, lookback=500):
    scores = []

    for p in portfolios:
        returns = p["equity"].pct_change(fill_method=None).fillna(0)

        # 🔥 usar ventana más larga
        recent = returns.iloc[-lookback:]

        # 🔥 suavizar (EWMA)
        mean = recent.ewm(span=100).mean().iloc[-1]
        vol = recent.ewm(span=100).std().iloc[-1] + 1e-8

        sharpe_like = mean / vol
        scores.append(sharpe_like)

    scores = np.array(scores)

    # 🔥 rankear en vez de usar valores crudos
    ranks = np.argsort(np.argsort(scores))
    ranks = ranks / (len(scores) - 1)

    # 🔥 convertir a weights suaves
    weights = 0.5 + ranks  # rango [0.5, 1.5]
    weights = weights / weights.sum()

    # 🔥 mezcla leve con uniforme
    uniform = np.ones_like(weights) / len(weights)
    weights = 0.8 * weights + 0.2 * uniform

    return weights

# =========================
# COMBINE PORTFOLIOS
# =========================
def combine_portfolios(portfolios, weights):
    returns_list = []

    for p in portfolios:
        r = p["equity"].pct_change(fill_method=None).fillna(0).values
        returns_list.append(r)

    returns_matrix = np.vstack(returns_list)

    portfolio_returns = np.sum(weights[:, None] * returns_matrix, axis=0)

    equity = 100000 * np.cumprod(1 + portfolio_returns)

    return pd.DataFrame({"equity": equity})


# =========================
# LOAD DATA PER ASSET
# =========================
def load_asset_data(symbol, config):
    # cambiar symbol dinámicamente
    config.binance.symbol = symbol

    loader = BinanceDataLoader(config.binance)
    data = loader.load_klines()

    return data


# =========================
# MAIN
# =========================
def main():
    print("=== MULTI-ASSET SYSTEM ===")

    config = DEFAULT_CONFIG
    wrapper = BacktesterWrapper(config)

    # =========================
    # LOAD CONFIGS
    # =========================
    with open("top_configs.json", "r") as f:
        configs = json.load(f)

    configs = sorted(configs, key=lambda x: x["mean_test"], reverse=True)
    configs = [c for c in configs if c["overfit"] < 0.8 and c["mean_test"] > 0.3]

    top_configs = configs[:2]

    print(f"Using {len(top_configs)} configs")

    # =========================
    # ASSETS
    # =========================
    assets = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
    ]

    portfolios = []

    for asset in assets:
        print(f"\n=== RUNNING {asset} ===")

        data = load_asset_data(asset, config)

        portfolio_equity = run_ensemble_weighted(wrapper, data, top_configs)

        # 🔥 aplicar vol targeting por asset
        portfolio_equity = apply_vol_target_to_asset(portfolio_equity)

        portfolios.append(portfolio_equity)

    # =========================
    # ALIGN
    # =========================
    min_len = min(len(p) for p in portfolios)
    aligned = [p.iloc[-min_len:].reset_index(drop=True) for p in portfolios]

    # 🔥 dynamic weights entre assets
    weights = compute_asset_weights(aligned, lookback=500)

# 🔥 aplicar correlation penalty
    weights, penalties = apply_correlation_penalty(aligned, weights)
    print("\n=== DYNAMIC ASSET WEIGHTS ===")
    for i, w in enumerate(weights):
        print(f"{assets[i]}: {w:.3f}")
    
    print("\n=== CORRELATION PENALTIES ===")
    for i, p in enumerate(penalties):
        print(f"{assets[i]}: {p:.3f}")

    portfolio = combine_portfolios(aligned, weights)

    # =========================
    # PERFORMANCE
    # =========================
    summary = PerformanceAnalyzer(annualization_factor=365*24).summarize(
        equity_curve=portfolio,
        trades=pd.DataFrame(),
        initial_capital=config.risk.initial_capital,
    )

    print("\n=== FINAL MULTI-ASSET RESULTS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()