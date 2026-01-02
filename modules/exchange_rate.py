"""
Exchange Rate Tracker - Tracks USD/KRW exchange rate for accurate P&L calculation.
"""

import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ExchangeRate:
    """Exchange rate data"""
    usd_to_krw: float
    updated_at: datetime
    source: str


class ExchangeRateTracker:
    """Tracks USD/KRW exchange rate for accurate P&L calculation"""
    
    # Free API endpoint
    API_URL = "https://api.exchangerate-api.com/v4/latest/USD"
    
    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize exchange rate tracker.
        
        Args:
            config: Exchange rate configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.update_interval = timedelta(
            minutes=config.get('update_interval_minutes', 60)
        )
        self.fallback_rate = config.get('fallback_rate', 1350.0)
        
        self._current_rate: Optional[ExchangeRate] = None
        self._last_update: Optional[datetime] = None
        
        # Initial fetch
        self._fetch_rate()
    
    def get_rate(self) -> float:
        """
        Get current USD/KRW rate, fetching if stale.
        
        Returns:
            Current exchange rate
        """
        if self._should_update():
            self._fetch_rate()
        
        if self._current_rate:
            return self._current_rate.usd_to_krw
        return self.fallback_rate
    
    def _should_update(self) -> bool:
        """Check if rate should be updated"""
        if self._last_update is None:
            return True
        return datetime.now() - self._last_update > self.update_interval
    
    def _fetch_rate(self) -> None:
        """Fetch current exchange rate from API"""
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            krw_rate = data.get('rates', {}).get('KRW')
            if krw_rate:
                self._current_rate = ExchangeRate(
                    usd_to_krw=krw_rate,
                    updated_at=datetime.now(),
                    source="exchangerate-api"
                )
                self._last_update = datetime.now()
                self.logger.info(f"Exchange rate updated: 1 USD = {krw_rate:,.2f} KRW")
            else:
                self.logger.warning("KRW rate not found in API response")
                
        except requests.exceptions.Timeout:
            self.logger.warning("Exchange rate API timeout, using fallback")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to fetch exchange rate: {e}, using fallback")
        except Exception as e:
            self.logger.warning(f"Unexpected error fetching exchange rate: {e}")
    
    def usd_to_krw(self, usd_amount: float) -> float:
        """
        Convert USD to KRW.
        
        Args:
            usd_amount: Amount in USD
            
        Returns:
            Amount in KRW
        """
        return usd_amount * self.get_rate()
    
    def krw_to_usd(self, krw_amount: float) -> float:
        """
        Convert KRW to USD.
        
        Args:
            krw_amount: Amount in KRW
            
        Returns:
            Amount in USD
        """
        rate = self.get_rate()
        if rate <= 0:
            return 0
        return krw_amount / rate
    
    def format_krw(self, usd_amount: float) -> str:
        """
        Format USD amount as KRW string.
        
        Args:
            usd_amount: Amount in USD
            
        Returns:
            Formatted string
        """
        krw = self.usd_to_krw(usd_amount)
        return f"{krw:,.0f} KRW"
    
    def format_both(self, usd_amount: float) -> str:
        """
        Format amount showing both USD and KRW.
        
        Args:
            usd_amount: Amount in USD
            
        Returns:
            Formatted string
        """
        krw = self.usd_to_krw(usd_amount)
        return f"${usd_amount:,.2f} ({krw:,.0f} KRW)"
    
    def get_status(self) -> Dict[str, Any]:
        """Get exchange rate status"""
        return {
            "current_rate": self._current_rate.usd_to_krw if self._current_rate else self.fallback_rate,
            "source": self._current_rate.source if self._current_rate else "fallback",
            "updated_at": self._current_rate.updated_at.isoformat() if self._current_rate else None,
            "fallback_rate": self.fallback_rate,
            "update_interval_minutes": self.update_interval.total_seconds() / 60,
            "is_stale": self._should_update()
        }
    
    def force_update(self) -> bool:
        """
        Force an immediate update.
        
        Returns:
            True if update succeeded
        """
        self._fetch_rate()
        return self._current_rate is not None


# Global instance
_exchange_rate_tracker: Optional[ExchangeRateTracker] = None


def init_exchange_rate_tracker(config: Dict[str, Any],
                                logger: Optional[logging.Logger] = None) -> ExchangeRateTracker:
    """Initialize global exchange rate tracker"""
    global _exchange_rate_tracker
    _exchange_rate_tracker = ExchangeRateTracker(config, logger)
    return _exchange_rate_tracker


def get_exchange_rate() -> float:
    """Get current exchange rate"""
    if _exchange_rate_tracker:
        return _exchange_rate_tracker.get_rate()
    return 1350.0  # Fallback


def usd_to_krw(usd_amount: float) -> float:
    """Convert USD to KRW"""
    if _exchange_rate_tracker:
        return _exchange_rate_tracker.usd_to_krw(usd_amount)
    return usd_amount * 1350.0


