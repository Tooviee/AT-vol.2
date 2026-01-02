"""
Notifier - Discord notifications for trades, alerts, and confirmations.
Supports both webhook and bot-based notifications.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    description: str
    color: int
    fields: List[Dict[str, str]]
    footer: Optional[str] = None
    timestamp: Optional[datetime] = None


class Notifier:
    """
    Notification handler for the trading system.
    Provides notification methods that can be implemented with Discord bot or webhooks.
    """
    
    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize notifier.
        
        Args:
            config: Notifications configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.large_order_threshold = config.get('large_order_threshold_krw', 5000000)
        self.confirmation_timeout = config.get('confirmation_timeout_seconds', 60)
        
        # Discord bot/webhook will be set externally
        self.discord_bot: Any = None
        self.enabled = True
    
    def set_discord_bot(self, bot: Any) -> None:
        """Set the Discord bot instance"""
        self.discord_bot = bot
    
    def _send(self, message: NotificationMessage) -> bool:
        """
        Send a notification.
        
        Args:
            message: Notification message
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        if self.discord_bot:
            try:
                # Use Discord bot to send
                self.discord_bot.send_embed(
                    title=message.title,
                    description=message.description,
                    color=message.color,
                    fields=message.fields,
                    footer=message.footer
                )
                return True
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")
                return False
        else:
            # Log only if no bot
            self.logger.info(f"Notification: {message.title} - {message.description}")
            return True
    
    def requires_confirmation(self, order: Any) -> bool:
        """Check if order needs manual confirmation"""
        try:
            order_value = order.quantity * getattr(order, 'avg_fill_price', 0) or 0
            # Convert to KRW (approximate)
            order_value_krw = order_value * 1350
            return order_value_krw >= self.large_order_threshold
        except Exception:
            return False
    
    def send_trade_notification(self, order: Any, result: Any) -> None:
        """Send trade execution notification"""
        side = order.side.upper() if hasattr(order, 'side') else 'TRADE'
        color = 0x00FF00 if side == 'BUY' else 0xFF0000  # Green for buy, red for sell
        emoji = "📈" if side == 'BUY' else "📉"
        
        message = NotificationMessage(
            title=f"{emoji} Trade Executed",
            description=f"{side} {order.quantity} {order.symbol}",
            color=color,
            fields=[
                {"name": "Symbol", "value": order.symbol},
                {"name": "Side", "value": side},
                {"name": "Quantity", "value": str(order.quantity)},
                {"name": "Price", "value": f"${result.fill_price:.2f}"},
                {"name": "Total", "value": f"${order.quantity * result.fill_price:,.2f}"}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_partial_fill_notification(self, order: Any, result: Any) -> None:
        """Send partial fill notification"""
        message = NotificationMessage(
            title="⚠️ Partial Fill",
            description=f"{order.symbol} partially filled",
            color=0xFFA500,
            fields=[
                {"name": "Symbol", "value": order.symbol},
                {"name": "Filled", "value": f"{result.filled_quantity}/{order.quantity}"},
                {"name": "Price", "value": f"${result.fill_price:.2f}"}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_order_rejected_notification(self, order: Any, reason: str) -> None:
        """Send order rejection notification"""
        message = NotificationMessage(
            title="❌ Order Rejected",
            description=f"{order.symbol} order rejected",
            color=0xFF0000,
            fields=[
                {"name": "Symbol", "value": order.symbol},
                {"name": "Side", "value": order.side.upper()},
                {"name": "Reason", "value": reason[:200]}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_order_cancelled_notification(self, order: Any, reason: str) -> None:
        """Send order cancellation notification"""
        message = NotificationMessage(
            title="🚫 Order Cancelled",
            description=f"{order.symbol} order cancelled",
            color=0x808080,
            fields=[
                {"name": "Symbol", "value": order.symbol},
                {"name": "Reason", "value": reason}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_order_timeout_notification(self, order: Any) -> None:
        """Send order timeout notification"""
        message = NotificationMessage(
            title="⏰ Order Timeout",
            description=f"{order.symbol} order timed out",
            color=0xFFA500,
            fields=[
                {"name": "Symbol", "value": order.symbol},
                {"name": "Order ID", "value": order.id[:8]}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_circuit_breaker_alert(self, reason: str) -> None:
        """Send circuit breaker alert"""
        message = NotificationMessage(
            title="🛑 Circuit Breaker Tripped",
            description="Trading has been paused",
            color=0xFF0000,
            fields=[
                {"name": "Reason", "value": reason}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_health_alert(self, health: Any) -> None:
        """Send health alert"""
        issues = health.issues if hasattr(health, 'issues') else []
        
        message = NotificationMessage(
            title="⚠️ Health Alert",
            description="System health issues detected",
            color=0xFF0000,
            fields=[
                {"name": "Issues", "value": "\n".join(issues) if issues else "Unknown"}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_daily_summary(self, stats: Dict[str, Any]) -> None:
        """Send daily P&L summary"""
        pnl = stats.get('realized_pnl_krw', 0)
        emoji = "✅" if pnl >= 0 else "❌"
        color = 0x00FF00 if pnl >= 0 else 0xFF0000
        
        win_count = stats.get('win_count', 0)
        loss_count = stats.get('loss_count', 0)
        total = win_count + loss_count
        win_rate = (win_count / total * 100) if total > 0 else 0
        
        message = NotificationMessage(
            title=f"📊 Daily Summary",
            description=f"Trading results for {stats.get('date', 'today')}",
            color=color,
            fields=[
                {"name": "Realized P&L", "value": f"{emoji} {pnl:+,.0f} KRW"},
                {"name": "Trades", "value": f"{win_count}W / {loss_count}L"},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%"}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_shutdown_alert(self) -> None:
        """Send shutdown notification"""
        message = NotificationMessage(
            title="🛑 Trading Bot Stopped",
            description="Bot has been shut down gracefully",
            color=0x808080,
            fields=[],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_startup_alert(self, mode: str) -> None:
        """Send startup notification"""
        message = NotificationMessage(
            title="🚀 Trading Bot Started",
            description=f"Bot is now running in {mode} mode",
            color=0x00FF00,
            fields=[
                {"name": "Mode", "value": mode.upper()},
                {"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)
    
    def send_signal_alert(self, signal: Any) -> None:
        """Send trading signal notification"""
        signal_type = signal.signal.value if hasattr(signal.signal, 'value') else str(signal.signal)
        color = 0x00FF00 if signal_type == 'buy' else 0xFF0000 if signal_type == 'sell' else 0x808080
        
        message = NotificationMessage(
            title=f"📊 Signal: {signal_type.upper()}",
            description=f"{signal.symbol} - {signal.reason}",
            color=color,
            fields=[
                {"name": "Symbol", "value": signal.symbol},
                {"name": "Price", "value": f"${signal.price:.2f}"},
                {"name": "Confidence", "value": f"{signal.confidence*100:.0f}%"}
            ],
            timestamp=datetime.now()
        )
        
        self._send(message)


