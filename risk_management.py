from __future__ import annotations

import math

from config import RiskConfig


class RiskManager:
    """Calculate position sizes under portfolio and trade risk constraints."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        current_open_notional: float,
    ) -> float:
        """Return the tradable quantity that satisfies risk and leverage caps."""

        stop_distance = entry_price - stop_price
        if equity <= 0.0 or entry_price <= 0.0 or stop_distance <= 0.0:
            return 0.0

        # =========================
        # 🔥 VOLATILITY-BASED SIZING
        # =========================
        volatility = abs(entry_price - stop_price) / entry_price

        vol_target = 0.02
        vol_factor = vol_target / (volatility + 1e-8)

        # límites para no descontrolar
        vol_factor = max(0.5, min(vol_factor, 1.5))

        risk_budget = equity * self.config.risk_per_trade * vol_factor

        quantity_from_risk = risk_budget / stop_distance

        # =========================
        # 🔥 LEVERAGE CONSTRAINT
        # =========================
        max_notional = equity * self.config.max_leverage
        available_notional = max(max_notional - current_open_notional, 0.0)

        quantity_from_notional = (
            available_notional / entry_price if entry_price > 0.0 else 0.0
        )

        # =========================
        # FINAL SIZE
        # =========================
        raw_quantity = min(quantity_from_risk, quantity_from_notional)

        quantity = self._round_down(raw_quantity)

        if quantity < self.config.min_quantity:
            return 0.0

        return quantity

    def can_open_position(self, open_positions: int) -> bool:
        """Check concurrent position capacity."""

        return open_positions < self.config.max_concurrent_positions

    def _round_down(self, quantity: float) -> float:
        factor = 10**self.config.quantity_precision
        return math.floor(quantity * factor) / factor
