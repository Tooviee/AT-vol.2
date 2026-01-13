"""
Structured JSON Logging - Provides structured logging for the trading system.
Supports JSON format, log rotation, and separate log files.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                    'message', 'taskName'
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Format: [HH:MM:SS] LEVEL    | logger: message
        formatted = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} | {record.name}: {record.getMessage()}"
        
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


class TradingLogger:
    """
    Trading system logger with structured JSON output.
    Creates separate log files for different categories.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        """
        Initialize trading logger.
        
        Args:
            config: Logging configuration dictionary
            log_dir: Directory for log files
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings from config
        self.log_level = getattr(logging, config.get('level', 'INFO').upper())
        self.use_json = config.get('format', 'json').lower() == 'json'
        self.rotate_size_mb = config.get('rotate_size_mb', 10)
        self.keep_days = config.get('keep_days', 30)
        
        # Calculate max bytes and backup count
        self.max_bytes = self.rotate_size_mb * 1024 * 1024
        self.backup_count = self.keep_days
        
        # Create formatters
        self.json_formatter = JSONFormatter()
        self.console_formatter = ConsoleFormatter()
        
        # Set up root logger
        self._setup_root_logger()
        
        # Create specialized loggers
        self.system_logger = self._create_logger("system", "system.jsonl")
        self.trade_logger = self._create_logger("trades", "trades.jsonl")
        self.error_logger = self._create_logger("errors", "errors.jsonl")
    
    def _setup_root_logger(self) -> None:
        """Set up the root logger with console handler"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.console_formatter)
        root_logger.addHandler(console_handler)
    
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a specialized logger with file handler"""
        logger = logging.getLogger(f"trading.{name}")
        logger.setLevel(self.log_level)
        logger.propagate = True  # Also log to root logger
        
        # File handler with rotation
        file_path = self.log_dir / filename
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.json_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        return logger
    
    # === System Logging ===
    
    def log_system(self, message: str, level: str = "INFO", **extra) -> None:
        """Log a system message"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.system_logger.log(log_level, message, extra=extra)
    
    def log_startup(self, config_summary: Dict[str, Any]) -> None:
        """Log system startup"""
        self.system_logger.info(
            "Trading system starting",
            extra={"event": "startup", "config": config_summary}
        )
    
    def log_shutdown(self, reason: str) -> None:
        """Log system shutdown"""
        self.system_logger.info(
            f"Trading system shutting down: {reason}",
            extra={"event": "shutdown", "reason": reason}
        )
    
    # === Trade Logging ===
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log a trade execution"""
        self.trade_logger.info(
            f"Trade executed: {trade_data.get('side', '').upper()} "
            f"{trade_data.get('quantity', 0)} {trade_data.get('symbol', '')} "
            f"@ {trade_data.get('price', 0):.2f}",
            extra={"event": "trade", "trade": trade_data}
        )
    
    def log_order(self, order_data: Dict[str, Any], status: str) -> None:
        """Log an order status change"""
        self.trade_logger.info(
            f"Order {status}: {order_data.get('order_id', '')} "
            f"{order_data.get('symbol', '')}",
            extra={"event": "order", "status": status, "order": order_data}
        )
    
    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        """Log a trading signal"""
        self.trade_logger.info(
            f"Signal: {signal_data.get('signal', '')} for {signal_data.get('symbol', '')}",
            extra={"event": "signal", "signal": signal_data}
        )
    
    def log_position(self, action: str, position_data: Dict[str, Any]) -> None:
        """Log a position change"""
        self.trade_logger.info(
            f"Position {action}: {position_data.get('symbol', '')}",
            extra={"event": "position", "action": action, "position": position_data}
        )
    
    # === Error Logging ===
    
    def log_error(self, message: str, error: Optional[Exception] = None, 
                  **extra) -> None:
        """Log an error"""
        if error:
            self.error_logger.error(
                message,
                exc_info=error,
                extra={"event": "error", **extra}
            )
        else:
            self.error_logger.error(
                message,
                extra={"event": "error", **extra}
            )
    
    def log_warning(self, message: str, **extra) -> None:
        """Log a warning"""
        self.error_logger.warning(
            message,
            extra={"event": "warning", **extra}
        )
    
    def log_api_error(self, api_name: str, error: str, 
                      request_data: Optional[Dict] = None) -> None:
        """Log an API error"""
        self.error_logger.error(
            f"API error ({api_name}): {error}",
            extra={
                "event": "api_error",
                "api": api_name,
                "error": error,
                "request": request_data
            }
        )
    
    # === Performance Logging ===
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics"""
        self.system_logger.info(
            "Performance metrics",
            extra={"event": "performance", "metrics": metrics}
        )
    
    def log_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Log daily trading summary"""
        self.trade_logger.info(
            f"Daily summary: P&L {summary.get('realized_pnl', 0):,.0f} KRW",
            extra={"event": "daily_summary", "summary": summary}
        )


# Global logger instance
_trading_logger: Optional[TradingLogger] = None


def init_logging(config: Dict[str, Any], log_dir: str = "logs") -> TradingLogger:
    """Initialize the trading logger"""
    global _trading_logger
    _trading_logger = TradingLogger(config, log_dir)
    return _trading_logger


def get_trading_logger() -> Optional[TradingLogger]:
    """Get the trading logger instance"""
    return _trading_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name"""
    if _trading_logger:
        return _trading_logger.get_logger(name)
    return logging.getLogger(name)


