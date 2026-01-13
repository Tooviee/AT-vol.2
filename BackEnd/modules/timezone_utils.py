"""
Timezone Utilities - Centralized timezone handling for US stock trading from Korea.
Uses exchange-calendars for accurate market hours including holidays.
"""

from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional
import logging

try:
    import exchange_calendars as xcals
    import pandas as pd
    HAS_EXCHANGE_CALENDARS = True
except ImportError:
    HAS_EXCHANGE_CALENDARS = False


# Timezone definitions
UTC = ZoneInfo("UTC")
KST = ZoneInfo("Asia/Seoul")
EST = ZoneInfo("America/New_York")


class TimezoneManager:
    """Centralized timezone handling with exchange calendar integration"""
    
    def __init__(self, local_tz: str = "Asia/Seoul", 
                 market_tz: str = "America/New_York",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize timezone manager.
        
        Args:
            local_tz: Local timezone string
            market_tz: Market timezone string
            logger: Optional logger instance
        """
        self.local_tz = ZoneInfo(local_tz)
        self.market_tz = ZoneInfo(market_tz)
        self.logger = logger or logging.getLogger(__name__)
        
        # NYSE calendar for accurate market hours
        self.calendar = None
        if HAS_EXCHANGE_CALENDARS:
            try:
                self.calendar = xcals.get_calendar("XNYS")
                self.logger.info("Exchange calendar loaded for NYSE")
            except Exception as e:
                self.logger.warning(f"Failed to load exchange calendar: {e}")
    
    # === Current Time Methods ===
    
    def now_utc(self) -> datetime:
        """Current time in UTC (use for all internal timestamps)"""
        return datetime.now(UTC)
    
    def now_local(self) -> datetime:
        """Current time in local timezone (Korea)"""
        return datetime.now(self.local_tz)
    
    def now_market(self) -> datetime:
        """Current time in market timezone (US Eastern)"""
        return datetime.now(self.market_tz)
    
    # === Conversion Methods ===
    
    def to_utc(self, dt: datetime) -> datetime:
        """Convert any datetime to UTC"""
        if dt.tzinfo is None:
            raise ValueError("Cannot convert naive datetime. Add timezone first.")
        return dt.astimezone(UTC)
    
    def to_local(self, dt: datetime) -> datetime:
        """Convert any datetime to local (Korea) time"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)  # Assume UTC if naive
        return dt.astimezone(self.local_tz)
    
    def to_market(self, dt: datetime) -> datetime:
        """Convert any datetime to market (US Eastern) time"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(self.market_tz)
    
    def local_to_market(self, dt: datetime) -> datetime:
        """Convert Korea time to US Eastern time"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.local_tz)
        return dt.astimezone(self.market_tz)
    
    def market_to_local(self, dt: datetime) -> datetime:
        """Convert US Eastern time to Korea time"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.market_tz)
        return dt.astimezone(self.local_tz)
    
    # === Display Formatting ===
    
    def format_local(self, dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S KST") -> str:
        """Format datetime for display in local timezone"""
        return self.to_local(dt).strftime(fmt)
    
    def format_market(self, dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S ET") -> str:
        """Format datetime for display in market timezone"""
        return self.to_market(dt).strftime(fmt)
    
    def format_both(self, dt: datetime) -> str:
        """Format datetime showing both local and market time"""
        local = self.to_local(dt)
        market = self.to_market(dt)
        return f"{local.strftime('%Y-%m-%d %H:%M KST')} / {market.strftime('%H:%M ET')}"
    
    # === Market Status Methods ===
    
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is currently open (handles holidays)"""
        if dt is None:
            dt = self.now_utc()
        
        market_dt = self.to_market(dt)
        
        # Use exchange calendar if available
        if self.calendar:
            try:
                ts = pd.Timestamp(market_dt)
                return self.calendar.is_open_on_minute(ts)
            except Exception:
                pass
        
        # Fallback to simple check
        return self._simple_market_check(market_dt)
    
    def _simple_market_check(self, market_dt: datetime) -> bool:
        """Fallback market check without calendar"""
        # Weekend check
        if market_dt.weekday() >= 5:
            return False
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        market_time = market_dt.time()
        return time(9, 30) <= market_time <= time(16, 0)
    
    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        """Check if given date is a trading day"""
        if dt is None:
            dt = self.now_utc()
        
        market_dt = self.to_market(dt)
        
        if self.calendar:
            try:
                ts = pd.Timestamp(market_dt.date())
                return self.calendar.is_session(ts)
            except Exception:
                pass
        
        # Fallback: weekdays only
        return market_dt.weekday() < 5
    
    def next_market_open(self, dt: Optional[datetime] = None) -> datetime:
        """Get next market open time (handles weekends and holidays)"""
        if dt is None:
            dt = self.now_utc()
        
        if self.calendar:
            try:
                ts = pd.Timestamp(dt)
                next_open = self.calendar.next_open(ts)
                return next_open.to_pydatetime().replace(tzinfo=self.market_tz)
            except Exception as e:
                self.logger.warning(f"Calendar next_open failed: {e}")
        
        return self._fallback_next_open(dt)
    
    def _fallback_next_open(self, dt: datetime) -> datetime:
        """Fallback next open calculation"""
        market_dt = self.to_market(dt)
        next_open = market_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If already past today's open or market is open, go to next day
        if market_dt.time() >= time(9, 30):
            next_open += timedelta(days=1)
        
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        return next_open
    
    def next_market_close(self, dt: Optional[datetime] = None) -> datetime:
        """Get next market close time"""
        if dt is None:
            dt = self.now_utc()
        
        if self.calendar:
            try:
                ts = pd.Timestamp(dt)
                next_close = self.calendar.next_close(ts)
                return next_close.to_pydatetime().replace(tzinfo=self.market_tz)
            except Exception:
                pass
        
        return self._fallback_next_close(dt)
    
    def _fallback_next_close(self, dt: datetime) -> datetime:
        """Fallback next close calculation"""
        market_dt = self.to_market(dt)
        next_close = market_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If already past today's close, go to next trading day
        if market_dt.time() >= time(16, 0):
            next_close += timedelta(days=1)
            while next_close.weekday() >= 5:
                next_close += timedelta(days=1)
        
        return next_close
    
    def seconds_until_market_open(self) -> float:
        """Seconds until next market open (accurate with holidays)"""
        now = self.now_utc()
        
        if self.is_market_open(now):
            return 0.0
        
        next_open = self.next_market_open(now)
        return max(0.0, (next_open - now).total_seconds())
    
    def seconds_until_market_close(self) -> float:
        """Seconds until market close"""
        now = self.now_utc()
        
        if not self.is_market_open(now):
            return 0.0
        
        next_close = self.next_market_close(now)
        return max(0.0, (next_close - now).total_seconds())
    
    def is_early_close_day(self, dt: Optional[datetime] = None) -> bool:
        """Check if today is an early close day (1:00 PM close)"""
        if dt is None:
            dt = self.now_utc()
        
        if not self.calendar:
            return False
        
        try:
            ts = pd.Timestamp(self.to_market(dt).date())
            
            if self.calendar.is_session(ts):
                close_time = self.calendar.session_close(ts)
                # Early close if before 4:00 PM
                return close_time.hour < 16
        except Exception:
            pass
        
        return False
    
    def get_market_close_time(self, dt: Optional[datetime] = None) -> datetime:
        """Get market close time for given date (handles early close)"""
        if dt is None:
            dt = self.now_utc()
        
        if self.calendar:
            try:
                ts = pd.Timestamp(self.to_market(dt).date())
                
                if self.calendar.is_session(ts):
                    close_time = self.calendar.session_close(ts)
                    return close_time.to_pydatetime().replace(tzinfo=self.market_tz)
            except Exception:
                pass
        
        # Fallback: regular close at 4:00 PM
        market_dt = self.to_market(dt)
        return market_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    def get_market_open_time(self, dt: Optional[datetime] = None) -> datetime:
        """Get market open time for given date"""
        if dt is None:
            dt = self.now_utc()
        
        if self.calendar:
            try:
                ts = pd.Timestamp(self.to_market(dt).date())
                
                if self.calendar.is_session(ts):
                    open_time = self.calendar.session_open(ts)
                    return open_time.to_pydatetime().replace(tzinfo=self.market_tz)
            except Exception:
                pass
        
        # Fallback: regular open at 9:30 AM
        market_dt = self.to_market(dt)
        return market_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # === Time Until Methods ===
    
    def time_until_open(self) -> timedelta:
        """Get timedelta until next market open"""
        seconds = self.seconds_until_market_open()
        return timedelta(seconds=seconds)
    
    def time_until_close(self) -> timedelta:
        """Get timedelta until market close"""
        seconds = self.seconds_until_market_close()
        return timedelta(seconds=seconds)
    
    # === Market Date Methods ===
    
    def get_market_date(self, dt: Optional[datetime] = None) -> datetime:
        """Get the market date (in US Eastern) for a given datetime"""
        if dt is None:
            dt = self.now_utc()
        market_dt = self.to_market(dt)
        return market_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def get_previous_trading_day(self, dt: Optional[datetime] = None) -> datetime:
        """Get previous trading day"""
        if dt is None:
            dt = self.now_utc()
        
        market_date = self.get_market_date(dt)
        prev_day = market_date - timedelta(days=1)
        
        if self.calendar:
            try:
                ts = pd.Timestamp(prev_day.date())
                prev_session = self.calendar.previous_session(ts)
                return prev_session.to_pydatetime().replace(tzinfo=self.market_tz)
            except Exception:
                pass
        
        # Fallback: skip weekends
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_status_string(self) -> str:
        """Get human-readable market status"""
        now = self.now_utc()
        
        if self.is_market_open(now):
            close_time = self.get_market_close_time(now)
            time_left = close_time - now
            hours, remainder = divmod(int(time_left.total_seconds()), 3600)
            minutes = remainder // 60
            
            if self.is_early_close_day(now):
                return f"Market OPEN (early close in {hours}h {minutes}m)"
            return f"Market OPEN (closes in {hours}h {minutes}m)"
        else:
            next_open = self.next_market_open(now)
            time_until = next_open - now
            hours, remainder = divmod(int(time_until.total_seconds()), 3600)
            minutes = remainder // 60
            
            return f"Market CLOSED (opens in {hours}h {minutes}m)"


# Global singleton instance
_tz_manager: Optional[TimezoneManager] = None


def get_timezone_manager(local_tz: str = "Asia/Seoul",
                          market_tz: str = "America/New_York") -> TimezoneManager:
    """Get or create the timezone manager singleton"""
    global _tz_manager
    if _tz_manager is None:
        _tz_manager = TimezoneManager(local_tz, market_tz)
    return _tz_manager


# Convenience functions
def now_utc() -> datetime:
    """Get current UTC time"""
    return get_timezone_manager().now_utc()


def now_kst() -> datetime:
    """Get current Korea time"""
    return get_timezone_manager().now_local()


def now_est() -> datetime:
    """Get current US Eastern time"""
    return get_timezone_manager().now_market()


def kst_to_est(dt: datetime) -> datetime:
    """Convert Korea time to US Eastern time"""
    return get_timezone_manager().local_to_market(dt)


def est_to_kst(dt: datetime) -> datetime:
    """Convert US Eastern time to Korea time"""
    return get_timezone_manager().market_to_local(dt)


def is_market_open() -> bool:
    """Check if market is currently open"""
    return get_timezone_manager().is_market_open()


def get_market_status() -> str:
    """Get market status string"""
    return get_timezone_manager().get_status_string()


