from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from config import ExecutionConfig, RiskConfig, StrategyConfig
from risk_management import RiskManager


@dataclass
class Position:
    """Represents an open long position."""

    id: int
    entry_time: pd.Timestamp
    entry_bar_index: int
    entry_price: float
    quantity: float
    initial_stop_price: float
    stop_price: float
    atr_at_entry: float
    entry_fee: float


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame


class Backtester:
    """Run the bar-by-bar execution model for the strategy."""

    def __init__(
        self,
        risk_manager: RiskManager,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        execution_config: ExecutionConfig,
    ) -> None:
        self.risk_manager = risk_manager
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.execution_config = execution_config

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Execute the backtest and return equity and trade history."""

        self._validate_inputs(data)

        cash = self.risk_config.initial_capital
        positions: list[Position] = []
        trades: list[dict[str, object]] = []
        equity_records: list[dict[str, object]] = [
            {
                "timestamp": data.iloc[0]["timestamp"],
                "cash": cash,
                "unrealized_pnl": 0.0,
                "equity": cash,
                "open_positions": 0,
            }
        ]
        next_position_id = 1

        for bar_index in range(1, len(data)):
            previous_bar = data.iloc[bar_index - 1]
            bar = data.iloc[bar_index]

            if positions and bool(previous_bar["exit_signal"]):
                remaining_positions: list[Position] = []
                for position in positions:
                    cash = self._close_position(
                        position=position,
                        cash=cash,
                        exit_time=bar["timestamp"],
                        exit_price=bar["open"],
                        execution_bar=bar_index,
                        reason="rsi_exit",
                        trades=trades,
                    )
                positions = remaining_positions

            if bool(previous_bar["entry_signal"]) and self.risk_manager.can_open_position(len(positions)):
                entry_price = self._apply_slippage(float(bar["open"]), side="buy")
                atr_value = float(previous_bar["atr"])
                stop_price = entry_price - (atr_value * self.strategy_config.stop_loss_atr_multiple)
                current_open_notional = sum(position.quantity * float(bar["open"]) for position in positions)
                equity_before_entry = cash + self._mark_to_market(positions, float(bar["open"]))
                quantity = self.risk_manager.calculate_position_size(
                    equity=equity_before_entry,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    current_open_notional=current_open_notional,
                )
                if quantity > 0.0 and atr_value > 0.0:
                    entry_fee = entry_price * quantity * self.execution_config.fee_rate
                    cash -= entry_fee
                    positions.append(
                        Position(
                            id=next_position_id,
                            entry_time=bar["timestamp"],
                            entry_bar_index=bar_index,
                            entry_price=entry_price,
                            quantity=quantity,
                            initial_stop_price=stop_price,
                            stop_price=stop_price,
                            atr_at_entry=atr_value,
                            entry_fee=entry_fee,
                        )
                    )
                    next_position_id += 1

            active_positions: list[Position] = []
            for position in positions:
                if float(bar["low"]) <= position.stop_price:
                    stop_reference_price = min(float(bar["open"]), position.stop_price)
                    reason = (
                        "trailing_stop"
                        if position.stop_price > position.initial_stop_price
                        else "stop_loss"
                    )
                    cash = self._close_position(
                        position=position,
                        cash=cash,
                        exit_time=bar["timestamp"],
                        exit_price=stop_reference_price,
                        execution_bar=bar_index,
                        reason=reason,
                        trades=trades,
                    )
                else:
                    active_positions.append(position)
            positions = active_positions

            current_atr = float(bar["atr"])
            if current_atr > 0.0:
                for position in positions:
                    candidate_stop = float(bar["close"]) - (
                        current_atr * self.strategy_config.trailing_stop_atr_multiple
                    )
                    position.stop_price = max(position.stop_price, candidate_stop)

            unrealized_pnl = self._mark_to_market(positions, float(bar["close"]))
            equity = cash + unrealized_pnl
            equity_records.append(
                {
                    "timestamp": bar["timestamp"],
                    "cash": cash,
                    "unrealized_pnl": unrealized_pnl,
                    "equity": equity,
                    "open_positions": len(positions),
                }
            )

        if positions and self.execution_config.close_positions_at_end:
            final_bar_index = len(data) - 1
            final_bar = data.iloc[final_bar_index]
            for position in positions:
                cash = self._close_position(
                    position=position,
                    cash=cash,
                    exit_time=final_bar["timestamp"],
                    exit_price=float(final_bar["close"]),
                    execution_bar=final_bar_index,
                    reason="end_of_backtest",
                    trades=trades,
                )
            positions = []
            equity_records.append(
                {
                    "timestamp": final_bar["timestamp"],
                    "cash": cash,
                    "unrealized_pnl": 0.0,
                    "equity": cash,
                    "open_positions": 0,
                }
            )

        equity_curve = pd.DataFrame(equity_records)
        trades_frame = pd.DataFrame(trades)
        if not trades_frame.empty:
            trades_frame = trades_frame.sort_values("entry_time").reset_index(drop=True)

        return BacktestResult(equity_curve=equity_curve, trades=trades_frame)

    def _close_position(
        self,
        position: Position,
        cash: float,
        exit_time: pd.Timestamp,
        exit_price: float,
        execution_bar: int,
        reason: str,
        trades: list[dict[str, object]],
    ) -> float:
        execution_price = self._apply_slippage(exit_price, side="sell")
        exit_fee = execution_price * position.quantity * self.execution_config.fee_rate
        gross_pnl = (execution_price - position.entry_price) * position.quantity
        cash += gross_pnl - exit_fee

        net_pnl = gross_pnl - position.entry_fee - exit_fee
        trade_return = net_pnl / (position.entry_price * position.quantity)
        bars_held = execution_bar - position.entry_bar_index
        trades.append(
            {
                "position_id": position.id,
                "entry_time": position.entry_time,
                "exit_time": exit_time,
                "entry_price": position.entry_price,
                "exit_price": execution_price,
                "quantity": position.quantity,
                "initial_stop_price": position.initial_stop_price,
                "final_stop_price": position.stop_price,
                "entry_fee": position.entry_fee,
                "exit_fee": exit_fee,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "return_pct": trade_return,
                "bars_held": bars_held,
                "exit_reason": reason,
            }
        )
        return cash

    def _apply_slippage(self, price: float, side: Literal["buy", "sell"]) -> float:
        if side == "buy":
            return price * (1.0 + self.execution_config.slippage_rate)
        return price * (1.0 - self.execution_config.slippage_rate)

    @staticmethod
    def _mark_to_market(positions: list[Position], mark_price: float) -> float:
        return sum((mark_price - position.entry_price) * position.quantity for position in positions)

    @staticmethod
    def _validate_inputs(data: pd.DataFrame) -> None:
        required_columns = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_fast",
            "ema_slow",
            "rsi",
            "atr",
            "entry_signal",
            "exit_signal",
        }
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required data columns: {sorted(missing_columns)}")

        if data.empty:
            raise ValueError("Backtest input data is empty.")
