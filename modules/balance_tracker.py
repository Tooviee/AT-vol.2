"""
Balance Tracker - Tracks account balance and positions.
Supports both paper trading and live trading modes.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Represents a stock position"""
    symbol: str
    quantity: int
    avg_price: float  # USD
    current_price: float = 0.0  # USD
    side: str = "long"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L in USD"""
        if self.side == "long":
            return (self.current_price - self.avg_price) * self.quantity
        else:
            return (self.avg_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.avg_price <= 0:
            return 0.0
        return ((self.current_price - self.avg_price) / self.avg_price) * 100
    
    @property
    def market_value(self) -> float:
        """Calculate current market value in USD"""
        return self.current_price * self.quantity
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis in USD"""
        return self.avg_price * self.quantity


class BalanceTracker:
    """Tracks account balance and positions"""
    
    def __init__(self, config: Dict[str, Any], 
                 exchange_rate_tracker: Any = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize balance tracker.
        
        Args:
            config: Configuration dictionary
            exchange_rate_tracker: Exchange rate tracker for USD/KRW conversion
            logger: Optional logger instance
        """
        self.config = config
        self.exchange_rate_tracker = exchange_rate_tracker
        self.logger = logger or logging.getLogger(__name__)
        
        # Mode
        self.is_paper = config.get('mode', 'paper') == 'paper'
        
        # Paper trading initial balance
        paper_config = config.get('paper_trading', {})
        initial_balance_krw = paper_config.get('initial_balance_krw', 10000000)
        
        # Convert to USD for internal tracking (approximate)
        fallback_rate = config.get('exchange_rate', {}).get('fallback_rate', 1350.0)
        self._cash_usd = initial_balance_krw / fallback_rate if self.is_paper else 0.0
        
        # Positions
        self._positions: Dict[str, Position] = {}
        
        # Track peak balance for drawdown calculations
        self._peak_balance_krw = initial_balance_krw if self.is_paper else 0.0
        
        # Trade history for P&L tracking
        self._realized_pnl_today = 0.0
    
    def get_exchange_rate(self) -> float:
        """Get current USD/KRW exchange rate"""
        if self.exchange_rate_tracker:
            return self.exchange_rate_tracker.get_rate()
        return self.config.get('exchange_rate', {}).get('fallback_rate', 1350.0)
    
    def get_total_balance(self) -> float:
        """Get total balance in KRW"""
        exchange_rate = self.get_exchange_rate()
        total_usd = self._cash_usd + self.get_positions_value_usd()
        return total_usd * exchange_rate
    
    def get_available_cash(self) -> float:
        """Get available cash in KRW"""
        return self._cash_usd * self.get_exchange_rate()
    
    def get_available_cash_usd(self) -> float:
        """Get available cash in USD"""
        return self._cash_usd
    
    def get_positions_value(self) -> float:
        """Get total positions value in KRW"""
        return self.get_positions_value_usd() * self.get_exchange_rate()
    
    def get_positions_value_usd(self) -> float:
        """Get total positions value in USD"""
        return sum(pos.market_value for pos in self._positions.values())
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self._positions)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self._positions.copy()
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol"""
        return symbol in self._positions
    
    def add_position(self, symbol: str, quantity: int, price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        Add or update a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to add
            price: Purchase price (USD)
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            
        Returns:
            True if successful
        """
        try:
            cost = quantity * price
            
            if symbol in self._positions:
                # Average into existing position
                existing = self._positions[symbol]
                total_quantity = existing.quantity + quantity
                total_cost = existing.cost_basis + cost
                new_avg_price = total_cost / total_quantity
                
                existing.quantity = total_quantity
                existing.avg_price = new_avg_price
                existing.current_price = price
                
                if stop_loss:
                    existing.stop_loss = stop_loss
                if take_profit:
                    existing.take_profit = take_profit
                
                self.logger.info(f"Added to position: {symbol} +{quantity} @ {price:.2f}")
            else:
                # New position
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                self.logger.info(f"New position: {symbol} {quantity} @ {price:.2f}")
            
            # Deduct cash
            self._cash_usd -= cost
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False
    
    def reduce_position(self, symbol: str, quantity: int, price: float) -> Optional[float]:
        """
        Reduce or close a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Sale price (USD)
            
        Returns:
            Realized P&L in USD, or None if failed
        """
        if symbol not in self._positions:
            self.logger.warning(f"No position to reduce: {symbol}")
            return None
        
        try:
            position = self._positions[symbol]
            
            if quantity > position.quantity:
                quantity = position.quantity  # Can't sell more than we have
            
            # Calculate realized P&L
            cost_basis = position.avg_price * quantity
            proceeds = price * quantity
            realized_pnl = proceeds - cost_basis
            
            # Update position
            position.quantity -= quantity
            position.current_price = price
            
            if position.quantity <= 0:
                del self._positions[symbol]
                self.logger.info(f"Closed position: {symbol} @ {price:.2f}, P&L: {realized_pnl:.2f} USD")
            else:
                self.logger.info(f"Reduced position: {symbol} -{quantity} @ {price:.2f}, P&L: {realized_pnl:.2f} USD")
            
            # Add cash
            self._cash_usd += proceeds
            
            # Track realized P&L
            self._realized_pnl_today += realized_pnl * self.get_exchange_rate()
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"Error reducing position: {e}")
            return None
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a position"""
        if symbol in self._positions:
            self._positions[symbol].current_price = price
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update prices for multiple symbols"""
        for symbol, price in prices.items():
            self.update_price(symbol, price)
    
    def deduct_cash(self, amount_usd: float) -> bool:
        """Deduct cash (for paper trading)"""
        if amount_usd > self._cash_usd:
            return False
        self._cash_usd -= amount_usd
        return True
    
    def add_cash(self, amount_usd: float) -> None:
        """Add cash (for paper trading)"""
        self._cash_usd += amount_usd
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L in KRW"""
        return self.get_unrealized_pnl_usd() * self.get_exchange_rate()
    
    def get_unrealized_pnl_usd(self) -> float:
        """Get total unrealized P&L in USD"""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    def get_realized_pnl_today(self) -> float:
        """Get today's realized P&L in KRW"""
        return self._realized_pnl_today
    
    def reset_daily_pnl(self) -> None:
        """Reset daily P&L tracking (call at market open)"""
        self._realized_pnl_today = 0.0
    
    def update_peak_balance(self) -> None:
        """Update peak balance for drawdown tracking"""
        current_balance = self.get_total_balance()
        if current_balance > self._peak_balance_krw:
            self._peak_balance_krw = current_balance
    
    def get_peak_balance(self) -> float:
        """Get peak balance in KRW"""
        return self._peak_balance_krw
    
    def get_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if self._peak_balance_krw <= 0:
            return 0.0
        current = self.get_total_balance()
        return ((self._peak_balance_krw - current) / self._peak_balance_krw) * 100
    
    def sync_with_broker(self, positions: Dict[str, Dict], cash: float) -> None:
        """
        Sync local state with broker data.
        
        Args:
            positions: Positions from broker API
            cash: Cash balance in USD
        """
        self._cash_usd = cash
        self._positions.clear()
        
        for symbol, pos_data in positions.items():
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=pos_data.get('quantity', 0),
                avg_price=pos_data.get('avg_price', 0),
                current_price=pos_data.get('current_price', pos_data.get('avg_price', 0))
            )
        
        self.logger.info(f"Synced with broker: {len(positions)} positions, {cash:.2f} USD cash")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get balance summary"""
        exchange_rate = self.get_exchange_rate()
        total_balance = self.get_total_balance()
        positions_value = self.get_positions_value()
        cash = self.get_available_cash()
        
        return {
            "mode": "paper" if self.is_paper else "live",
            "exchange_rate": exchange_rate,
            "total_balance_krw": total_balance,
            "total_balance_usd": total_balance / exchange_rate,
            "positions_value_krw": positions_value,
            "positions_value_usd": positions_value / exchange_rate,
            "cash_krw": cash,
            "cash_usd": self._cash_usd,
            "unrealized_pnl_krw": self.get_unrealized_pnl(),
            "unrealized_pnl_usd": self.get_unrealized_pnl_usd(),
            "realized_pnl_today_krw": self._realized_pnl_today,
            "position_count": len(self._positions),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl_usd": pos.unrealized_pnl,
                    "unrealized_pnl_percent": pos.unrealized_pnl_percent
                }
                for symbol, pos in self._positions.items()
            },
            "peak_balance_krw": self._peak_balance_krw,
            "drawdown_percent": self.get_drawdown()
        }


