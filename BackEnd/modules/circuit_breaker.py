"""
Circuit Breaker - Safety controls with configurable thresholds.
Prevents runaway losses and handles API errors.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Tripped, no trading
    HALF_OPEN = "half_open"  # Testing if can resume


@dataclass
class CircuitBreakerEvent:
    """Circuit breaker event record"""
    event_type: str
    reason: str
    timestamp: datetime
    consecutive_losses: int
    daily_loss_percent: float
    daily_loss_krw: float


class CircuitBreaker:
    """Safety controls with configurable thresholds"""
    
    def __init__(self, config: Dict[str, Any], balance_tracker: Any,
                 database: Any = None, notifier: Any = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            balance_tracker: Balance tracker instance
            database: Optional database instance
            notifier: Optional notifier instance
            logger: Optional logger instance
        """
        self.config = config
        self.balance_tracker = balance_tracker
        self.database = database
        self.notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        
        # Thresholds
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.max_daily_loss_percent = config.get('max_daily_loss_percent', 5.0)
        self.max_daily_loss_krw = config.get('max_daily_loss_krw', 500000)
        self.loss_type = config.get('loss_type', 'realized')
        self.api_error_threshold = config.get('api_error_threshold', 5)
        self.cooldown_minutes = config.get('cooldown_minutes', 30)
        
        # State
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_losses = 0
        self.api_errors_this_minute = 0
        self.api_error_minute: Optional[datetime] = None
        self.tripped_at: Optional[datetime] = None
        self.trip_reason: Optional[str] = None
        
        # Event history
        self.events: List[CircuitBreakerEvent] = []
    
    def record_trade_result(self, is_win: bool, pnl_krw: float = 0) -> None:
        """
        Record a trade result.
        
        Args:
            is_win: True if trade was profitable
            pnl_krw: P&L in KRW
        """
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._trip(f"Max consecutive losses reached: {self.consecutive_losses}")
    
    def record_api_error(self) -> None:
        """Record an API error"""
        now = datetime.now()
        
        # Reset counter if new minute
        if self.api_error_minute is None or (now - self.api_error_minute).seconds >= 60:
            self.api_errors_this_minute = 0
            self.api_error_minute = now
        
        self.api_errors_this_minute += 1
        
        if self.api_errors_this_minute >= self.api_error_threshold:
            self._trip(f"API error threshold reached: {self.api_errors_this_minute}/min")
    
    def get_daily_loss(self) -> float:
        """Get daily loss based on configured loss type"""
        if self.loss_type == 'realized':
            if self.database:
                return abs(min(0, self.database.get_realized_pnl_today()))
            return 0
        elif self.loss_type == 'unrealized':
            return abs(min(0, self.balance_tracker.get_unrealized_pnl()))
        else:  # 'both'
            realized = 0
            if self.database:
                realized = min(0, self.database.get_realized_pnl_today())
            unrealized = min(0, self.balance_tracker.get_unrealized_pnl())
            return abs(realized + unrealized)
    
    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been exceeded.
        
        Returns:
            True if limit exceeded
        """
        daily_loss = self.get_daily_loss()
        total_balance = self.balance_tracker.get_total_balance()
        
        # Check absolute limit
        if daily_loss >= self.max_daily_loss_krw:
            self._trip(f"Daily loss limit reached: {daily_loss:,.0f} KRW")
            return True
        
        # Check percentage limit
        if total_balance > 0:
            loss_percent = (daily_loss / total_balance) * 100
            if loss_percent >= self.max_daily_loss_percent:
                self._trip(f"Daily loss limit reached: {loss_percent:.1f}%")
                return True
        
        return False
    
    def _trip(self, reason: str) -> None:
        """Trip the circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            return  # Already tripped
        
        self.state = CircuitBreakerState.OPEN
        self.tripped_at = datetime.now()
        self.trip_reason = reason
        
        self.logger.warning(f"Circuit breaker TRIPPED: {reason}")
        
        # Record event
        event = CircuitBreakerEvent(
            event_type="tripped",
            reason=reason,
            timestamp=datetime.now(),
            consecutive_losses=self.consecutive_losses,
            daily_loss_percent=self._get_daily_loss_percent(),
            daily_loss_krw=self.get_daily_loss()
        )
        self.events.append(event)
        
        # Save to database
        if self.database:
            try:
                self.database.log_circuit_breaker_event(
                    event_type="tripped",
                    reason=reason,
                    consecutive_losses=self.consecutive_losses,
                    daily_loss_percent=self._get_daily_loss_percent(),
                    daily_loss_krw=self.get_daily_loss()
                )
            except Exception as e:
                self.logger.warning(f"Failed to log circuit breaker event: {e}")
        
        # Send notification
        if self.notifier:
            try:
                self.notifier.send_circuit_breaker_alert(reason)
            except Exception as e:
                self.logger.warning(f"Failed to send circuit breaker alert: {e}")
    
    def _get_daily_loss_percent(self) -> float:
        """Get daily loss as percentage"""
        total = self.balance_tracker.get_total_balance()
        if total <= 0:
            return 0
        return (self.get_daily_loss() / total) * 100
    
    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if tripped
        if self.state == CircuitBreakerState.OPEN:
            # Check cooldown
            if self.tripped_at:
                elapsed = datetime.now() - self.tripped_at
                if elapsed < timedelta(minutes=self.cooldown_minutes):
                    remaining = timedelta(minutes=self.cooldown_minutes) - elapsed
                    return False, f"Circuit breaker active. Resume in {remaining.seconds // 60}m"
                else:
                    # Cooldown expired, try half-open
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker entering half-open state")
        
        # Check daily loss before each trade
        if self.check_daily_loss_limit():
            return False, f"Daily loss limit reached"
        
        return True, "OK"
    
    def reset(self, force: bool = False) -> None:
        """
        Reset the circuit breaker.
        
        Args:
            force: Force reset even if conditions aren't met
        """
        if force or self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.consecutive_losses = 0
            self.api_errors_this_minute = 0
            self.tripped_at = None
            self.trip_reason = None
            
            self.logger.info("Circuit breaker RESET")
            
            # Record event
            self.events.append(CircuitBreakerEvent(
                event_type="reset",
                reason="Manual reset" if force else "Cooldown expired",
                timestamp=datetime.now(),
                consecutive_losses=0,
                daily_loss_percent=self._get_daily_loss_percent(),
                daily_loss_krw=self.get_daily_loss()
            ))
    
    def reset_daily(self) -> None:
        """Reset for new trading day"""
        self.consecutive_losses = 0
        self.api_errors_this_minute = 0
        self.api_error_minute = None
        
        # Only reset if tripped for daily loss
        if self.state == CircuitBreakerState.OPEN and self.trip_reason:
            if "Daily loss" in self.trip_reason:
                self.reset(force=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        daily_loss = self.get_daily_loss()
        daily_loss_percent = self._get_daily_loss_percent()
        
        return {
            "state": self.state.value,
            "can_trade": self.can_trade()[0],
            "reason": self.can_trade()[1],
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "daily_loss_krw": daily_loss,
            "max_daily_loss_krw": self.max_daily_loss_krw,
            "daily_loss_percent": daily_loss_percent,
            "max_daily_loss_percent": self.max_daily_loss_percent,
            "loss_type": self.loss_type,
            "api_errors_this_minute": self.api_errors_this_minute,
            "api_error_threshold": self.api_error_threshold,
            "tripped_at": self.tripped_at.isoformat() if self.tripped_at else None,
            "trip_reason": self.trip_reason,
            "cooldown_minutes": self.cooldown_minutes
        }


