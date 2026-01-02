"""
Market Hours Manager - Manages NYSE market hours with calendar integration.
Handles holidays, early close days, and trading windows.
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .timezone_utils import TimezoneManager, get_timezone_manager


class MarketStatus(Enum):
    """Market status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    EARLY_CLOSE = "early_close"


@dataclass
class TradingWindow:
    """Trading window information"""
    status: MarketStatus
    can_trade: bool
    open_time: Optional[datetime]
    close_time: Optional[datetime]
    time_until_open: Optional[timedelta]
    time_until_close: Optional[timedelta]
    is_early_close: bool
    message: str


class MarketHoursManager:
    """Manages NYSE market hours with calendar integration"""
    
    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize market hours manager.
        
        Args:
            config: Market hours configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Get timezone manager
        self.tz_manager = get_timezone_manager()
        
        # Regular hours
        self.regular_open = self._parse_time(config.get('regular_open', '09:30'))
        self.regular_close = self._parse_time(config.get('regular_close', '16:00'))
        
        # Extended hours settings
        self.trade_premarket = config.get('trade_premarket', False)
        self.trade_afterhours = config.get('trade_afterhours', False)
        self.premarket_start = self._parse_time(config.get('premarket_start', '04:00'))
        self.afterhours_end = self._parse_time(config.get('afterhours_end', '20:00'))
        
        # Safety buffer
        self.early_close_buffer = timedelta(
            minutes=config.get('early_close_buffer_minutes', 30)
        )
        
        self.skip_holidays = config.get('skip_holidays', True)
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object"""
        parts = time_str.split(':')
        return time(int(parts[0]), int(parts[1]))
    
    def get_current_status(self) -> MarketStatus:
        """Get current market status"""
        now = self.tz_manager.now_market()
        current_time = now.time()
        
        # Check if today is a trading day
        if not self.tz_manager.is_trading_day(now):
            return MarketStatus.CLOSED
        
        # Check regular hours
        if self.regular_open <= current_time <= self.regular_close:
            if self.tz_manager.is_early_close_day(now):
                return MarketStatus.EARLY_CLOSE
            return MarketStatus.OPEN
        
        # Check pre-market
        if self.premarket_start <= current_time < self.regular_open:
            return MarketStatus.PRE_MARKET
        
        # Check after-hours
        if self.regular_close < current_time <= self.afterhours_end:
            return MarketStatus.AFTER_HOURS
        
        return MarketStatus.CLOSED
    
    def is_market_open(self) -> bool:
        """Check if market is currently open for regular trading"""
        return self.tz_manager.is_market_open()
    
    def can_trade_now(self) -> bool:
        """Check if trading is allowed right now based on config"""
        status = self.get_current_status()
        
        if status == MarketStatus.OPEN:
            return True
        if status == MarketStatus.EARLY_CLOSE:
            # Check if we're within the buffer before close
            close_time = self.tz_manager.get_market_close_time()
            now = self.tz_manager.now_utc()
            if close_time - now < self.early_close_buffer:
                return False
            return True
        if status == MarketStatus.PRE_MARKET:
            return self.trade_premarket
        if status == MarketStatus.AFTER_HOURS:
            return self.trade_afterhours
        
        return False
    
    def get_trading_window(self) -> TradingWindow:
        """Get current trading window information"""
        status = self.get_current_status()
        can_trade = self.can_trade_now()
        now = self.tz_manager.now_utc()
        
        # Get times
        is_early = self.tz_manager.is_early_close_day(now)
        
        if self.tz_manager.is_trading_day(now):
            open_time = self.tz_manager.get_market_open_time(now)
            close_time = self.tz_manager.get_market_close_time(now)
        else:
            open_time = self.tz_manager.next_market_open(now)
            close_time = None
        
        # Calculate time until
        if status == MarketStatus.OPEN or status == MarketStatus.EARLY_CLOSE:
            time_until_open = None
            time_until_close = close_time - now if close_time else None
        else:
            time_until_open = open_time - now if open_time and open_time > now else None
            if time_until_open and time_until_open.total_seconds() < 0:
                # Next day
                time_until_open = self.tz_manager.next_market_open(now) - now
            time_until_close = None
        
        # Generate message
        if status == MarketStatus.OPEN:
            hours = int(time_until_close.total_seconds() // 3600) if time_until_close else 0
            mins = int((time_until_close.total_seconds() % 3600) // 60) if time_until_close else 0
            message = f"Market open. Closes in {hours}h {mins}m"
        elif status == MarketStatus.EARLY_CLOSE:
            hours = int(time_until_close.total_seconds() // 3600) if time_until_close else 0
            mins = int((time_until_close.total_seconds() % 3600) // 60) if time_until_close else 0
            message = f"Early close day. Closes in {hours}h {mins}m"
        elif status == MarketStatus.PRE_MARKET:
            message = "Pre-market session"
        elif status == MarketStatus.AFTER_HOURS:
            message = "After-hours session"
        else:
            if time_until_open:
                hours = int(time_until_open.total_seconds() // 3600)
                mins = int((time_until_open.total_seconds() % 3600) // 60)
                message = f"Market closed. Opens in {hours}h {mins}m"
            else:
                message = "Market closed"
        
        return TradingWindow(
            status=status,
            can_trade=can_trade,
            open_time=open_time,
            close_time=close_time,
            time_until_open=time_until_open,
            time_until_close=time_until_close,
            is_early_close=is_early,
            message=message
        )
    
    def wait_for_market_open(self, check_interval: int = 60) -> None:
        """
        Block until market opens.
        
        Args:
            check_interval: Seconds between checks
        """
        import time as time_module
        
        while not self.can_trade_now():
            window = self.get_trading_window()
            self.logger.info(window.message)
            
            if window.time_until_open:
                # Sleep for minimum of check_interval or time until open
                sleep_seconds = min(
                    check_interval,
                    window.time_until_open.total_seconds()
                )
                time_module.sleep(max(1, sleep_seconds))
            else:
                time_module.sleep(check_interval)
    
    def get_seconds_until_market_open(self) -> float:
        """Get seconds until market opens"""
        return self.tz_manager.seconds_until_market_open()
    
    def get_seconds_until_market_close(self) -> float:
        """Get seconds until market closes"""
        return self.tz_manager.seconds_until_market_close()
    
    def is_trading_day(self) -> bool:
        """Check if today is a trading day"""
        return self.tz_manager.is_trading_day()
    
    def is_early_close_day(self) -> bool:
        """Check if today is an early close day"""
        return self.tz_manager.is_early_close_day()
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get status as dictionary"""
        window = self.get_trading_window()
        
        return {
            "status": window.status.value,
            "can_trade": window.can_trade,
            "is_early_close": window.is_early_close,
            "message": window.message,
            "open_time": window.open_time.isoformat() if window.open_time else None,
            "close_time": window.close_time.isoformat() if window.close_time else None,
            "seconds_until_open": window.time_until_open.total_seconds() if window.time_until_open else None,
            "seconds_until_close": window.time_until_close.total_seconds() if window.time_until_close else None,
            "local_time": self.tz_manager.now_local().isoformat(),
            "market_time": self.tz_manager.now_market().isoformat()
        }


