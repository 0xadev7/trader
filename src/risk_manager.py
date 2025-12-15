"""Risk management and position sizing."""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class RiskManager:
    """Manages risk, position sizing, and trade execution limits."""

    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_drawdown_pct: float = 0.15,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct

        self.peak_equity = initial_capital
        self.positions = (
            {}
        )  # {pair: {'size': float, 'entry_price': float, 'stop_loss': float, 'take_profit': float}}

    def calculate_position_size(
        self, pair: str, price: float, confidence: float = 1.0
    ) -> float:
        """Calculate optimal position size based on risk parameters.

        Args:
            pair: Trading pair
            price: Current price
            confidence: Model confidence (0-1), scales position size

        Returns:
            Position size in base currency
        """
        # Base position size from risk per trade
        risk_amount = self.current_capital * self.risk_per_trade

        # Calculate position size based on stop loss distance
        stop_loss_distance = price * self.stop_loss_pct
        base_position_size = risk_amount / stop_loss_distance

        # Apply confidence scaling (reduce size if low confidence)
        adjusted_position_size = base_position_size * confidence

        # Cap at maximum position size
        max_position_value = self.current_capital * self.max_position_size
        max_position_size = max_position_value / price

        final_position_size = min(adjusted_position_size, max_position_size)

        # Ensure minimum viable position
        if final_position_size * price < 10:  # Minimum $10 trade
            return 0.0

        return final_position_size

    def calculate_stop_loss(self, entry_price: float, side: str = "long") -> float:
        """Calculate stop loss price."""
        if side == "long":
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str = "long") -> float:
        """Calculate take profit price."""
        if side == "long":
            return entry_price * (1 + self.take_profit_pct)
        else:  # short
            return entry_price * (1 - self.take_profit_pct)

    def check_drawdown(self, current_equity: float) -> bool:
        """Check if current drawdown exceeds maximum.

        Returns:
            True if trading should be halted due to drawdown
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= self.max_drawdown_pct:
            logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            return True

        return False

    def validate_trade(
        self, pair: str, side: str, size: float, price: float
    ) -> Dict[str, any]:
        """Validate a trade before execution.

        Returns:
            Dict with 'allowed': bool, 'reason': str, 'adjusted_size': float
        """
        # Check drawdown
        if self.check_drawdown(self.current_capital):
            return {
                "allowed": False,
                "reason": "Maximum drawdown exceeded",
                "adjusted_size": 0.0,
            }

        # Check if already have position
        if pair in self.positions:
            existing_position = self.positions[pair]
            # Allow closing or reversing
            if side == existing_position.get("side"):
                return {
                    "allowed": False,
                    "reason": f"Already have {side} position",
                    "adjusted_size": 0.0,
                }

        # Calculate position size
        position_size = self.calculate_position_size(pair, price)
        if size > position_size:
            size = position_size

        # Check available capital
        trade_value = size * price
        if trade_value > self.current_capital * self.max_position_size:
            size = (self.current_capital * self.max_position_size) / price

        if trade_value > self.current_capital * 0.95:  # Leave 5% buffer
            size = (self.current_capital * 0.95) / price

        return {"allowed": True, "reason": "Trade validated", "adjusted_size": size}

    def open_position(
        self,
        pair: str,
        side: str,
        size: float,
        entry_price: float,
        confidence: float = 1.0,
    ):
        """Record opened position."""
        stop_loss = self.calculate_stop_loss(entry_price, side)
        take_profit = self.calculate_take_profit(entry_price, side)

        self.positions[pair] = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
        }

        # Update capital (reserve for position)
        trade_value = size * entry_price
        self.current_capital -= trade_value

        logger.info(
            f"Opened {side} position: {pair}, size: {size:.6f}, price: {entry_price:.2f}"
        )

    def close_position(self, pair: str, exit_price: float) -> Dict[str, float]:
        """Close position and calculate P&L.

        Returns:
            Dict with 'pnl', 'pnl_pct', 'size'
        """
        if pair not in self.positions:
            return {"pnl": 0.0, "pnl_pct": 0.0, "size": 0.0}

        position = self.positions[pair]
        size = position["size"]
        entry_price = position["entry_price"]
        side = position["side"]

        # Calculate P&L
        if side == "long":
            pnl = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price

        # Update capital
        # When we opened the position, we deducted: size * entry_price
        # When closing, we receive: size * exit_price
        # The P&L is already included in the difference, so we just add back the exit value
        exit_value = size * exit_price
        self.current_capital += exit_value

        # Remove position
        del self.positions[pair]

        logger.info(f"Closed {side} position: {pair}, P&L: {pnl:.2f} ({pnl_pct:.2%})")

        return {"pnl": pnl, "pnl_pct": pnl_pct, "size": size}

    def check_stop_loss_take_profit(
        self, pair: str, current_price: float
    ) -> Optional[str]:
        """Check if stop loss or take profit should be triggered.

        Returns:
            'stop_loss', 'take_profit', or None
        """
        if pair not in self.positions:
            return None

        position = self.positions[pair]
        side = position["side"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]

        if side == "long":
            if current_price <= stop_loss:
                return "stop_loss"
            if current_price >= take_profit:
                return "take_profit"
        else:  # short
            if current_price >= stop_loss:
                return "stop_loss"
            if current_price <= take_profit:
                return "take_profit"

        return None

    def get_current_positions(self) -> Dict:
        """Get current open positions."""
        return self.positions.copy()

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity including unrealized P&L.
        
        current_capital is the cash remaining after opening positions.
        For each position, we need to add the current market value.
        """
        equity = self.current_capital

        for pair, position in self.positions.items():
            if pair in current_prices:
                current_price = current_prices[pair]
                entry_price = position["entry_price"]
                size = position["size"]
                side = position["side"]

                if side == "long":
                    # Long position: we own the asset
                    # We deducted size * entry_price when opening (spent cash to buy)
                    # Current value is size * current_price
                    # Equity = cash + current_position_value
                    equity += size * current_price
                else:  # short
                    # Short position: we sold the asset
                    # When opening, we deducted size * entry_price (treated as margin/collateral)
                    # To close short, we'd buy back at current_price
                    # Profit = (entry_price - current_price) * size (profit if price went down)
                    # Since we deducted entry_value as margin, we need to:
                    # - Add back the margin: size * entry_price
                    # - Add the profit: (entry_price - current_price) * size
                    # Total: size * entry_price + (entry_price - current_price) * size
                    # = size * (2 * entry_price - current_price)
                    # Or simpler: we get back margin + profit = entry_value + (entry_price - current_price) * size
                    entry_value = size * entry_price
                    profit = (entry_price - current_price) * size
                    equity += entry_value + profit

        return equity
