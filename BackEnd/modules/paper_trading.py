"""
Paper Trading Executor - Simulates order execution with realistic market conditions.
Includes slippage, spread simulation, and partial fills.
"""

import random
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class SimulatedFill:
    """Result of a simulated order execution"""
    order_id: str
    filled_quantity: int
    fill_price: float
    slippage_applied: float
    spread_applied: float
    is_partial: bool
    timestamp: datetime


class PaperTradingExecutor:
    """Simulates order execution with realistic market conditions"""
    
    def __init__(self, config: Dict[str, Any], balance_tracker: Any,
                 database: Any, price_fetcher: Any,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize paper trading executor.
        
        Args:
            config: Paper trading configuration
            balance_tracker: BalanceTracker instance
            database: Database instance
            price_fetcher: Price fetcher for market orders
            logger: Optional logger instance
        """
        self.config = config
        self.balance_tracker = balance_tracker
        self.database = database
        self.price_fetcher = price_fetcher
        self.logger = logger or logging.getLogger(__name__)
        
        # Simulation settings
        self.slippage_enabled = config.get('simulate_slippage', True)
        self.slippage_pct = config.get('slippage_percent', 0.15) / 100
        
        self.spread_enabled = config.get('simulate_spread', True)
        self.spread_pct = config.get('spread_percent', 0.05) / 100
        
        self.latency_ms = config.get('simulate_latency_ms', 100)
        
        self.partial_fills_enabled = config.get('simulate_partial_fills', False)
        self.partial_prob = config.get('partial_fill_probability', 0.1)
        
        self.logger.info(
            f"Paper trading initialized: slippage={self.slippage_pct*100:.2f}%, "
            f"spread={self.spread_pct*100:.2f}%, latency={self.latency_ms}ms"
        )
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Fetch current market price for market orders.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price
            
        Raises:
            ValueError: If price cannot be fetched
        """
        price_data = self.price_fetcher.get_price(symbol)
        
        if price_data is None:
            raise ValueError(f"Cannot get current price for {symbol}")
        
        # Use close price as current price
        return price_data.get('close', price_data.get('price', 0))
    
    def _apply_slippage(self, price: float, side: str) -> tuple[float, float]:
        """
        Apply slippage to price.
        
        Args:
            price: Base price
            side: 'buy' or 'sell'
            
        Returns:
            Tuple of (adjusted_price, slippage_amount)
        """
        if not self.slippage_enabled:
            return price, 0.0
        
        # Random slippage between 50% and 100% of max
        slippage_factor = random.uniform(0.5, 1.0)
        slippage = price * self.slippage_pct * slippage_factor
        
        # Slippage is always unfavorable
        if side.lower() == 'buy':
            return price + slippage, slippage
        else:
            return price - slippage, -slippage
    
    def _apply_spread(self, price: float, side: str) -> tuple[float, float]:
        """
        Apply bid-ask spread to price.
        
        Args:
            price: Base price
            side: 'buy' or 'sell'
            
        Returns:
            Tuple of (adjusted_price, spread_amount)
        """
        if not self.spread_enabled:
            return price, 0.0
        
        spread = price * self.spread_pct
        
        # Buy at ask (higher), sell at bid (lower)
        if side.lower() == 'buy':
            return price + spread, spread
        else:
            return price - spread, -spread
    
    def _simulate_partial_fill(self, quantity: int) -> tuple[int, bool]:
        """
        Simulate partial fill.
        
        Args:
            quantity: Requested quantity
            
        Returns:
            Tuple of (filled_quantity, is_partial)
        """
        if not self.partial_fills_enabled:
            return quantity, False
        
        if random.random() < self.partial_prob:
            # Partial fill: random amount between 1 and quantity-1
            filled = random.randint(1, max(1, quantity - 1))
            return filled, filled < quantity
        
        return quantity, False
    
    def execute_order(self, order: Any) -> SimulatedFill:
        """
        Simulate order execution with slippage and spread.
        
        Args:
            order: Order object with symbol, side, quantity, order_type, limit_price
            
        Returns:
            SimulatedFill result
        """
        # Simulate network latency (intentionally blocking for paper trading)
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
        
        symbol = order.symbol
        side = order.side
        quantity = order.quantity
        order_type = getattr(order, 'order_type', 'market')
        order_id = getattr(order, 'id', str(random.randint(100000, 999999)))
        
        # Get base price based on order type
        if order_type == 'market':
            base_price = self._get_current_price(symbol)
            self.logger.debug(f"Market order: fetched price {base_price:.2f} for {symbol}")
        else:
            # Limit order - use the limit price
            limit_price = getattr(order, 'limit_price', None)
            if limit_price is None:
                raise ValueError("Limit order must have limit_price set")
            base_price = limit_price
            
            # For limit orders, check if order would fill at current market
            current_price = self._get_current_price(symbol)
            
            if side.lower() == 'buy' and current_price > limit_price:
                # Buy limit above market - use market price (better)
                base_price = current_price
            elif side.lower() == 'sell' and current_price < limit_price:
                # Sell limit below market - use market price (worse)
                base_price = current_price
        
        # Apply spread
        price_after_spread, spread_adj = self._apply_spread(base_price, side)
        
        # Apply slippage
        fill_price, slippage_adj = self._apply_slippage(price_after_spread, side)
        
        # Simulate partial fills
        filled_qty, is_partial = self._simulate_partial_fill(quantity)
        
        # Calculate total value
        total_value = filled_qty * fill_price
        
        # Update balance tracker
        if side.lower() == 'buy':
            # Check if we have enough cash
            available_cash = self.balance_tracker.get_available_cash_usd()
            if total_value > available_cash:
                raise ValueError(
                    f"Insufficient cash: need {total_value:.2f} USD, "
                    f"have {available_cash:.2f} USD"
                )
            
            # Add position
            stop_loss = getattr(order, 'stop_loss', None)
            take_profit = getattr(order, 'take_profit', None)
            self.balance_tracker.add_position(
                symbol, filled_qty, fill_price, stop_loss, take_profit
            )
        else:
            # Reduce/close position
            self.balance_tracker.reduce_position(symbol, filled_qty, fill_price)
        
        # Log the execution
        self.logger.info(
            f"Paper trade executed: {side.upper()} {filled_qty} {symbol} "
            f"@ {fill_price:.2f} (slippage: {slippage_adj:.4f}, spread: {spread_adj:.4f})"
            f"{' [PARTIAL]' if is_partial else ''}"
        )
        
        # Save to database if available
        if self.database:
            try:
                self.database.save_trade(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=filled_qty,
                    price=fill_price,
                    is_paper=True
                )
            except Exception as e:
                self.logger.warning(f"Failed to save trade to database: {e}")
        
        return SimulatedFill(
            order_id=order_id,
            filled_quantity=filled_qty,
            fill_price=fill_price,
            slippage_applied=slippage_adj,
            spread_applied=spread_adj,
            is_partial=is_partial,
            timestamp=datetime.now()
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order (in paper trading, always succeeds).
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True (always succeeds in paper trading)
        """
        self.logger.info(f"Paper order cancelled: {order_id}")
        return True
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status (in paper trading, orders fill immediately).
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        return {
            'order_id': order_id,
            'status': 'filled',
            'message': 'Paper trading - orders fill immediately'
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of paper trading settings"""
        return {
            'mode': 'paper',
            'slippage_enabled': self.slippage_enabled,
            'slippage_percent': self.slippage_pct * 100,
            'spread_enabled': self.spread_enabled,
            'spread_percent': self.spread_pct * 100,
            'latency_ms': self.latency_ms,
            'partial_fills_enabled': self.partial_fills_enabled,
            'partial_fill_probability': self.partial_prob
        }


