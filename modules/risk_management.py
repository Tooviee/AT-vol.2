"""
Risk Management - Per-trade and portfolio-level risk management.
Calculates position sizes, stop-losses, and enforces risk limits.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    allowed: bool
    reason: str


class RiskManager:
    """Per-trade and portfolio-level risk management"""
    
    def __init__(self, config: Dict[str, Any], balance_tracker: Any,
                 database: Any = None, logger: Optional[logging.Logger] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration dictionary
            balance_tracker: BalanceTracker instance
            database: Database instance for historical data
            logger: Optional logger instance
        """
        self.config = config
        self.balance_tracker = balance_tracker
        self.database = database
        self.logger = logger or logging.getLogger(__name__)
        
        # Per-trade limits
        self.risk_per_trade_percent = config.get('risk_per_trade_percent', 1.0)
        self.max_position_size_percent = config.get('max_position_size_percent', 10.0)
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_multiplier = config.get('take_profit_atr_multiplier', 3.0)
        
        # Portfolio-level limits
        self.max_total_exposure_percent = config.get('max_total_exposure_percent', 80.0)
        self.max_drawdown_percent = config.get('max_drawdown_percent', 15.0)
        self.max_correlated_positions = config.get('max_correlated_positions', 3)
    
    def check_portfolio_limits(self) -> Tuple[bool, str]:
        """
        Check if portfolio-level limits allow new trades.
        
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            total_balance = self.balance_tracker.get_total_balance()
            positions_value = self.balance_tracker.get_positions_value()
            
            if total_balance <= 0:
                return False, "No available balance"
            
            # Check total exposure
            exposure_percent = (positions_value / total_balance) * 100
            if exposure_percent >= self.max_total_exposure_percent:
                return False, f"Max exposure reached: {exposure_percent:.1f}% >= {self.max_total_exposure_percent}%"
            
            # Check drawdown from peak
            if self.database:
                peak_balance = self.database.get_peak_balance()
                if peak_balance and peak_balance > 0:
                    drawdown_percent = ((peak_balance - total_balance) / peak_balance) * 100
                    if drawdown_percent >= self.max_drawdown_percent:
                        return False, f"Max drawdown reached: {drawdown_percent:.1f}% >= {self.max_drawdown_percent}%"
            
            # Check number of positions
            num_positions = self.balance_tracker.get_position_count()
            if num_positions >= self.max_correlated_positions:
                return False, f"Max positions reached: {num_positions} >= {self.max_correlated_positions}"
            
            return True, "OK"
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio limits: {e}")
            return False, f"Error: {e}"
    
    def calculate_position_size(self, symbol: str, entry_price: float,
                                 stop_loss_price: float,
                                 exchange_rate: float = 1.0) -> PositionSizeResult:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Stock symbol
            entry_price: Proposed entry price (USD)
            stop_loss_price: Stop-loss price (USD)
            exchange_rate: USD to KRW exchange rate
            
        Returns:
            PositionSizeResult with calculated position size
        """
        # Check portfolio limits first
        allowed, reason = self.check_portfolio_limits()
        if not allowed:
            self.logger.warning(f"Position blocked for {symbol}: {reason}")
            return PositionSizeResult(
                shares=0,
                position_value=0,
                risk_amount=0,
                risk_percent=0,
                allowed=False,
                reason=reason
            )
        
        try:
            total_balance = self.balance_tracker.get_total_balance()
            
            if total_balance <= 0:
                return PositionSizeResult(
                    shares=0, position_value=0, risk_amount=0,
                    risk_percent=0, allowed=False, reason="No balance available"
                )
            
            # Risk amount in KRW
            risk_amount = total_balance * (self.risk_per_trade_percent / 100)
            
            # Max position value based on portfolio percentage
            max_position_value = total_balance * (self.max_position_size_percent / 100)
            
            # Calculate risk per share (in USD)
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share <= 0:
                return PositionSizeResult(
                    shares=0, position_value=0, risk_amount=0,
                    risk_percent=0, allowed=False, 
                    reason="Invalid stop-loss (risk per share <= 0)"
                )
            
            # Convert risk amount to USD for calculation
            risk_amount_usd = risk_amount / exchange_rate if exchange_rate > 0 else risk_amount
            
            # Calculate shares based on risk
            shares_by_risk = int(risk_amount_usd / risk_per_share)
            
            # Calculate shares based on max position size
            max_position_value_usd = max_position_value / exchange_rate if exchange_rate > 0 else max_position_value
            shares_by_max = int(max_position_value_usd / entry_price)
            
            # Take the smaller of the two
            shares = min(shares_by_risk, shares_by_max)
            
            if shares <= 0:
                return PositionSizeResult(
                    shares=0, position_value=0, risk_amount=0,
                    risk_percent=0, allowed=False,
                    reason="Calculated position size is 0"
                )
            
            # Calculate actual position value and risk
            position_value_usd = shares * entry_price
            position_value_krw = position_value_usd * exchange_rate
            actual_risk_usd = shares * risk_per_share
            actual_risk_krw = actual_risk_usd * exchange_rate
            actual_risk_percent = (actual_risk_krw / total_balance) * 100
            
            return PositionSizeResult(
                shares=shares,
                position_value=position_value_krw,
                risk_amount=actual_risk_krw,
                risk_percent=actual_risk_percent,
                allowed=True,
                reason=f"Position sized: {shares} shares, {actual_risk_percent:.2f}% risk"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return PositionSizeResult(
                shares=0, position_value=0, risk_amount=0,
                risk_percent=0, allowed=False, reason=f"Error: {e}"
            )
    
    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: 'buy' or 'sell'
            
        Returns:
            Stop-loss price
        """
        if side.lower() == 'buy':
            return entry_price - (atr * self.stop_loss_atr_multiplier)
        else:
            return entry_price + (atr * self.stop_loss_atr_multiplier)
    
    def calculate_take_profit(self, entry_price: float, atr: float, side: str) -> float:
        """
        Calculate take-profit price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: 'buy' or 'sell'
            
        Returns:
            Take-profit price
        """
        if side.lower() == 'buy':
            return entry_price + (atr * self.take_profit_atr_multiplier)
        else:
            return entry_price - (atr * self.take_profit_atr_multiplier)
    
    def calculate_risk_reward_ratio(self, entry_price: float, 
                                     stop_loss: float, take_profit: float) -> float:
        """
        Calculate risk/reward ratio.
        
        Returns:
            Risk/reward ratio (e.g., 2.0 means 2:1 reward to risk)
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk <= 0:
            return 0
        
        return reward / risk
    
    def validate_trade(self, symbol: str, side: str, quantity: int,
                       entry_price: float, stop_loss: float,
                       exchange_rate: float = 1.0) -> Tuple[bool, str]:
        """
        Validate a proposed trade against risk rules.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop-loss price
            exchange_rate: USD to KRW rate
            
        Returns:
            Tuple of (valid, reason)
        """
        # Check portfolio limits
        allowed, reason = self.check_portfolio_limits()
        if not allowed:
            return False, reason
        
        # Calculate position value
        position_value_usd = quantity * entry_price
        position_value_krw = position_value_usd * exchange_rate
        total_balance = self.balance_tracker.get_total_balance()
        
        if total_balance <= 0:
            return False, "No balance available"
        
        # Check max position size
        position_percent = (position_value_krw / total_balance) * 100
        if position_percent > self.max_position_size_percent:
            return False, f"Position too large: {position_percent:.1f}% > {self.max_position_size_percent}%"
        
        # Check risk per trade
        risk_per_share = abs(entry_price - stop_loss)
        trade_risk_usd = quantity * risk_per_share
        trade_risk_krw = trade_risk_usd * exchange_rate
        trade_risk_percent = (trade_risk_krw / total_balance) * 100
        
        if trade_risk_percent > self.risk_per_trade_percent * 1.5:  # Allow 50% buffer
            return False, f"Trade risk too high: {trade_risk_percent:.2f}% > {self.risk_per_trade_percent * 1.5:.2f}%"
        
        # Check if we have enough cash
        available_cash = self.balance_tracker.get_available_cash()
        if side.lower() == 'buy' and position_value_krw > available_cash:
            return False, f"Insufficient cash: need {position_value_krw:,.0f} KRW, have {available_cash:,.0f} KRW"
        
        return True, "Trade validated"
    
    def get_max_shares(self, symbol: str, entry_price: float,
                        exchange_rate: float = 1.0) -> int:
        """
        Get maximum shares that can be purchased.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price (USD)
            exchange_rate: USD to KRW rate
            
        Returns:
            Maximum number of shares
        """
        try:
            available_cash = self.balance_tracker.get_available_cash()
            total_balance = self.balance_tracker.get_total_balance()
            
            # Convert cash to USD
            available_cash_usd = available_cash / exchange_rate if exchange_rate > 0 else available_cash
            
            # Max by cash
            max_by_cash = int(available_cash_usd / entry_price) if entry_price > 0 else 0
            
            # Max by position limit
            max_position_value_krw = total_balance * (self.max_position_size_percent / 100)
            max_position_value_usd = max_position_value_krw / exchange_rate if exchange_rate > 0 else max_position_value_krw
            max_by_limit = int(max_position_value_usd / entry_price) if entry_price > 0 else 0
            
            return min(max_by_cash, max_by_limit)
            
        except Exception as e:
            self.logger.error(f"Error calculating max shares: {e}")
            return 0
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        try:
            total_balance = self.balance_tracker.get_total_balance()
            positions_value = self.balance_tracker.get_positions_value()
            available_cash = self.balance_tracker.get_available_cash()
            
            exposure_percent = (positions_value / total_balance * 100) if total_balance > 0 else 0
            
            return {
                "total_balance_krw": total_balance,
                "positions_value_krw": positions_value,
                "available_cash_krw": available_cash,
                "exposure_percent": exposure_percent,
                "max_exposure_percent": self.max_total_exposure_percent,
                "exposure_remaining": self.max_total_exposure_percent - exposure_percent,
                "position_count": self.balance_tracker.get_position_count(),
                "max_positions": self.max_correlated_positions,
                "can_open_position": exposure_percent < self.max_total_exposure_percent
            }
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {"error": str(e)}


