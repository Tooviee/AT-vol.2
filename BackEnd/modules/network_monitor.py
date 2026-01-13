"""
Network Monitor - Monitors internet connectivity and pauses trading on outages.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests


class NetworkMonitor:
    """Monitors internet connectivity and pauses trading on outages"""
    
    DEFAULT_CHECK_URLS = [
        "https://www.google.com",
        "https://finance.yahoo.com"
    ]
    
    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize network monitor.
        
        Args:
            config: Network configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.check_urls = config.get('check_urls', self.DEFAULT_CHECK_URLS)
        self.timeout = config.get('timeout_seconds', 5)
        self.failure_threshold = config.get('failure_threshold', 3)
        
        self.consecutive_failures = 0
        self.last_check_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self._is_connected = True
    
    def is_connected(self) -> bool:
        """
        Check if internet is available.
        
        Returns:
            True if connected
        """
        self.last_check_time = datetime.now()
        
        for url in self.check_urls:
            try:
                response = requests.head(url, timeout=self.timeout)
                if response.status_code < 500:
                    self.consecutive_failures = 0
                    self.last_success_time = datetime.now()
                    self._is_connected = True
                    return True
            except requests.exceptions.RequestException:
                continue
        
        self.consecutive_failures += 1
        self._is_connected = False
        
        self.logger.warning(
            f"Network check failed ({self.consecutive_failures}/{self.failure_threshold})"
        )
        
        return False
    
    def should_pause_trading(self) -> bool:
        """
        Check if trading should be paused due to network issues.
        
        Returns:
            True if trading should be paused
        """
        return self.consecutive_failures >= self.failure_threshold
    
    def wait_for_connection(self, check_interval: int = 30,
                            max_wait: int = 300) -> bool:
        """
        Wait for connection to restore.
        
        Args:
            check_interval: Seconds between checks
            max_wait: Maximum seconds to wait
            
        Returns:
            True if connection restored
        """
        waited = 0
        
        while waited < max_wait:
            if self.is_connected():
                self.logger.info("Network connection restored")
                return True
            
            time.sleep(check_interval)
            waited += check_interval
            
            self.logger.info(f"Waiting for network... ({waited}/{max_wait}s)")
        
        self.logger.error(f"Network not restored after {max_wait}s")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get network status"""
        return {
            "is_connected": self._is_connected,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "should_pause_trading": self.should_pause_trading(),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "check_urls": self.check_urls
        }
    
    def reset(self) -> None:
        """Reset failure count"""
        self.consecutive_failures = 0
        self._is_connected = True


class NetworkAwareExecutor:
    """Wrapper that adds network checking to any executor"""
    
    def __init__(self, executor: Any, network_monitor: NetworkMonitor,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize network-aware executor.
        
        Args:
            executor: Underlying executor
            network_monitor: Network monitor instance
            logger: Optional logger instance
        """
        self.executor = executor
        self.network_monitor = network_monitor
        self.logger = logger or logging.getLogger(__name__)
    
    def execute_order(self, order: Any) -> Any:
        """Execute order with network check"""
        if self.network_monitor.should_pause_trading():
            self.logger.warning("Network issues - waiting for connection")
            
            if not self.network_monitor.wait_for_connection():
                raise ConnectionError("Network unavailable")
        
        return self.executor.execute_order(order)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other methods to underlying executor"""
        return getattr(self.executor, name)


