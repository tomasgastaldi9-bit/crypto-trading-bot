from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OutputConfig


class PerformanceAnalyzer:
    """Compute performance statistics and persist outputs."""

    def __init__(self, annualization_factor: int) -> None:
        self.annualization_factor = annualization_factor

    def summarize(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
    ) -> dict[str, float]:
        """Calculate the required summary metrics."""

        final_equity = float(equity_curve["equity"].iloc[-1])
        total_return = (final_equity / initial_capital) - 1.0

        returns = equity_curve["equity"].pct_change().dropna()
        sharpe_ratio = 0.0
        if not returns.empty and returns.std(ddof=0) > 0.0:
            sharpe_ratio = np.sqrt(self.annualization_factor) * (
                returns.mean() / returns.std(ddof=0)
            )

        rolling_peak = equity_curve["equity"].cummax()
        drawdowns = (equity_curve["equity"] / rolling_peak) - 1.0
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

        if trades.empty:
            win_rate = 0.0
            profit_factor = 0.0
        else:
            winning_trades = trades[trades["net_pnl"] > 0.0]
            losing_trades = trades[trades["net_pnl"] < 0.0]
            win_rate = len(winning_trades) / len(trades)
            gross_profit = float(winning_trades["net_pnl"].sum())
            gross_loss = float(losing_trades["net_pnl"].sum())
            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0.0 else float("inf")

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": max_drawdown,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": float(len(trades)),
        }

    def save_outputs(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        summary: dict[str, float],
        output_config: OutputConfig,
    ) -> dict[str, Path]:
        """Persist backtest artifacts to disk."""

        output_config.output_dir.mkdir(parents=True, exist_ok=True)
        trade_log_path = output_config.output_dir / output_config.trade_log_filename
        equity_curve_path = output_config.output_dir / output_config.equity_curve_filename
        summary_path = output_config.output_dir / output_config.summary_filename
        equity_plot_path = output_config.output_dir / output_config.equity_plot_filename

        equity_curve.to_csv(equity_curve_path, index=False)
        trades.to_csv(trade_log_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._save_equity_plot(equity_curve, equity_plot_path)

        return {
            "trade_log": trade_log_path,
            "equity_curve": equity_curve_path,
            "summary": summary_path,
            "equity_plot": equity_plot_path,
        }

    def _save_equity_plot(self, equity_curve: pd.DataFrame, target_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plot_data = equity_curve.copy()
        plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"], utc=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data["timestamp"], plot_data["equity"], linewidth=1.5)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(target_path, dpi=150)
        plt.close(fig)


def format_summary_report(summary: dict[str, float]) -> str:
    """Return a console-friendly summary report."""

    profit_factor = "inf" if np.isinf(summary["profit_factor"]) else f"{summary['profit_factor']:.2f}"
    lines = [
        "=" * 68,
        "CRYPTO BACKTEST SUMMARY",
        "=" * 68,
        f"Initial Capital : {summary['initial_capital']:,.2f}",
        f"Final Equity    : {summary['final_equity']:,.2f}",
        f"Total Return    : {summary['total_return'] * 100:.2f}%",
        f"Sharpe Ratio    : {summary['sharpe_ratio']:.2f}",
        f"Max Drawdown    : {summary['max_drawdown'] * 100:.2f}%",
        f"Win Rate        : {summary['win_rate'] * 100:.2f}%",
        f"Profit Factor   : {profit_factor}",
        f"Total Trades    : {int(summary['total_trades'])}",
        "=" * 68,
    ]
    return "\n".join(lines)
