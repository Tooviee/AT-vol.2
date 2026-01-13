"""
Shutdown Handler - Cross-platform graceful shutdown handler.
Windows-compatible using SIGBREAK.
"""

import signal
import sys
import platform
import logging
from typing import Optional, Callable, List, Any


class ShutdownHandler:
    """Cross-platform graceful shutdown handler"""
    
    def __init__(self, trader: Any = None, notifier: Any = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize shutdown handler.
        
        Args:
            trader: Main trader instance with save_state and close_connections methods
            notifier: Notifier for shutdown alerts
            logger: Optional logger instance
        """
        self.trader = trader
        self.notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        self.shutting_down = False
        
        # Cleanup callbacks
        self._cleanup_callbacks: List[Callable] = []
        
        # Register signal handlers
        self._register_signals()
        
        self.logger.info("Shutdown handler initialized")
    
    def _register_signals(self) -> None:
        """Register signal handlers for different platforms"""
        # SIGINT (Ctrl+C) works on all platforms
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # Platform-specific signals
        if platform.system() == 'Windows':
            # Windows: SIGTERM doesn't work properly, use SIGBREAK (Ctrl+Break)
            try:
                signal.signal(signal.SIGBREAK, self._handle_signal)
                self.logger.debug("Registered SIGBREAK handler for Windows")
            except (AttributeError, ValueError):
                self.logger.warning("SIGBREAK not available on this system")
        else:
            # Linux/Mac: Use SIGTERM
            try:
                signal.signal(signal.SIGTERM, self._handle_signal)
                self.logger.debug("Registered SIGTERM handler for Unix")
            except (AttributeError, ValueError):
                self.logger.warning("SIGTERM not available on this system")
    
    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal"""
        try:
            signal_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            signal_name = str(signum)
        
        if self.shutting_down:
            self.logger.warning(f"Force exit requested ({signal_name})")
            sys.exit(1)
        
        self.shutting_down = True
        self.logger.info(f"Shutdown signal received: {signal_name}")
        
        try:
            self._perform_cleanup()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    def _perform_cleanup(self) -> None:
        """Perform cleanup operations"""
        # Run registered callbacks first
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback error: {e}")
        
        # Save trader state
        if self.trader:
            try:
                if hasattr(self.trader, 'save_state'):
                    self.trader.save_state()
                    self.logger.info("State saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
        
        # Send shutdown notification
        if self.notifier:
            try:
                if hasattr(self.notifier, 'send_shutdown_alert'):
                    self.notifier.send_shutdown_alert()
            except Exception as e:
                self.logger.warning(f"Failed to send shutdown alert: {e}")
        
        # Close connections
        if self.trader:
            try:
                if hasattr(self.trader, 'close_connections'):
                    self.trader.close_connections()
                    self.logger.info("Connections closed")
            except Exception as e:
                self.logger.error(f"Failed to close connections: {e}")
    
    def register_cleanup(self, callback: Callable) -> None:
        """
        Register a cleanup callback to run on shutdown.
        
        Args:
            callback: Function to call during shutdown
        """
        self._cleanup_callbacks.append(callback)
    
    def unregister_cleanup(self, callback: Callable) -> bool:
        """
        Unregister a cleanup callback.
        
        Args:
            callback: Function to remove
            
        Returns:
            True if callback was removed
        """
        try:
            self._cleanup_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    @property
    def should_stop(self) -> bool:
        """Check if shutdown has been requested"""
        return self.shutting_down
    
    def request_shutdown(self, reason: str = "Programmatic shutdown") -> None:
        """
        Request a graceful shutdown programmatically.
        
        Args:
            reason: Reason for shutdown
        """
        self.logger.info(f"Shutdown requested: {reason}")
        self.shutting_down = True
        self._perform_cleanup()
        sys.exit(0)
    
    def check_shutdown(self) -> None:
        """
        Check if shutdown was requested and exit if so.
        Call this periodically in the main loop.
        """
        if self.shutting_down:
            self._perform_cleanup()
            sys.exit(0)


# Global shutdown handler instance
_shutdown_handler: Optional[ShutdownHandler] = None


def init_shutdown_handler(trader: Any = None, notifier: Any = None,
                          logger: Optional[logging.Logger] = None) -> ShutdownHandler:
    """Initialize global shutdown handler"""
    global _shutdown_handler
    _shutdown_handler = ShutdownHandler(trader, notifier, logger)
    return _shutdown_handler


def get_shutdown_handler() -> Optional[ShutdownHandler]:
    """Get the global shutdown handler"""
    return _shutdown_handler


def should_stop() -> bool:
    """Check if shutdown has been requested"""
    if _shutdown_handler:
        return _shutdown_handler.should_stop
    return False


def register_cleanup(callback: Callable) -> None:
    """Register a cleanup callback"""
    if _shutdown_handler:
        _shutdown_handler.register_cleanup(callback)


