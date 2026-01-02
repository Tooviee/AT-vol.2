"""
Rate Limiter - API rate limiting with exponential backoff retry.
Prevents API rate limit violations.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from datetime import datetime, timedelta


class RateLimiter:
    """Prevents API rate limit violations with exponential backoff"""
    
    def __init__(self, calls_per_second: float = 5.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
            logger: Logger instance
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_time = 0.0
        self.logger = logger or logging.getLogger(__name__)
    
    def acquire(self) -> None:
        """Block until rate limit allows next call"""
        if self.min_interval <= 0:
            return
        
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def try_acquire(self) -> bool:
        """
        Try to acquire without blocking.
        
        Returns:
            True if acquired, False if rate limited
        """
        if self.min_interval <= 0:
            return True
        
        elapsed = time.time() - self.last_call_time
        if elapsed >= self.min_interval:
            self.last_call_time = time.time()
            return True
        return False
    
    def reset(self) -> None:
        """Reset the rate limiter"""
        self.last_call_time = 0.0
    
    def with_retry(self, max_attempts: int = 3, 
                   base_delay: float = 1.0,
                   max_delay: float = 60.0,
                   exceptions: tuple = (Exception,)):
        """
        Decorator for exponential backoff retry.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            exceptions: Tuple of exceptions to catch and retry
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        self.acquire()
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            self.logger.error(
                                f"Failed after {max_attempts} attempts: {e}"
                            )
                            raise
                        
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        
                        self.logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                
                if last_exception:
                    raise last_exception
                    
            return wrapper
        return decorator
    
    def get_status(self) -> dict:
        """Get rate limiter status"""
        elapsed = time.time() - self.last_call_time
        can_call = elapsed >= self.min_interval
        
        return {
            "calls_per_second": self.calls_per_second,
            "min_interval_ms": self.min_interval * 1000,
            "last_call_ago_ms": elapsed * 1000,
            "can_call_now": can_call,
            "wait_time_ms": max(0, (self.min_interval - elapsed) * 1000)
        }


class MultiRateLimiter:
    """Manages multiple rate limiters for different APIs"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize multi rate limiter.
        
        Args:
            config: Configuration with rate limits per API
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.limiters: dict[str, RateLimiter] = {}
        
        # Create limiters from config
        self.limiters['kis'] = RateLimiter(
            config.get('kis_api_calls_per_second', 5.0),
            self.logger
        )
        self.limiters['yfinance'] = RateLimiter(
            config.get('yfinance_calls_per_second', 2.0),
            self.logger
        )
        
        # Default limiter for unknown APIs
        self.default_limiter = RateLimiter(1.0, self.logger)
    
    def get_limiter(self, api_name: str) -> RateLimiter:
        """Get rate limiter for an API"""
        return self.limiters.get(api_name, self.default_limiter)
    
    def acquire(self, api_name: str) -> None:
        """Acquire rate limit for an API"""
        self.get_limiter(api_name).acquire()
    
    def with_retry(self, api_name: str, max_attempts: int = 3,
                   base_delay: float = 1.0):
        """Get retry decorator for an API"""
        return self.get_limiter(api_name).with_retry(
            max_attempts=max_attempts,
            base_delay=base_delay
        )


# Global rate limiter instance
_rate_limiters: Optional[MultiRateLimiter] = None


def init_rate_limiters(config: dict, logger: Optional[logging.Logger] = None) -> MultiRateLimiter:
    """Initialize global rate limiters"""
    global _rate_limiters
    _rate_limiters = MultiRateLimiter(config, logger)
    return _rate_limiters


def get_rate_limiter(api_name: str = 'default') -> RateLimiter:
    """Get rate limiter for an API"""
    if _rate_limiters:
        return _rate_limiters.get_limiter(api_name)
    return RateLimiter(1.0)


