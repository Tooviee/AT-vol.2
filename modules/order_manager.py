"""
Order Manager - Manages order lifecycle with state machine and notifications.
Includes timeouts, retry logic, and notifier integration.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from uuid import uuid4


class OrderState(Enum):
    """Order lifecycle states"""
    CREATED = "created"
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# Valid state transitions
VALID_TRANSITIONS = {
    OrderState.CREATED: [
        OrderState.PENDING,
        OrderState.CANCELLED,
    ],
    OrderState.PENDING: [
        OrderState.SUBMITTED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
    ],
    OrderState.SUBMITTED: [
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.TIMEOUT,
    ],
    OrderState.PARTIAL_FILL: [
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
    ],
    # Terminal states
    OrderState.FILLED: [],
    OrderState.CANCELLED: [],
    OrderState.REJECTED: [],
    OrderState.TIMEOUT: [],
}


class InvalidStateTransition(Exception):
    """Raised when attempting an invalid state transition"""
    pass


@dataclass
class Order:
    """Order with state machine"""
    symbol: str
    side: str
    quantity: int
    order_type: str = 'market'
    limit_price: Optional[float] = None
    
    # Auto-generated
    id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderState = OrderState.CREATED
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    
    # Risk levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # State history
    state_history: List[tuple] = field(default_factory=list)
    
    # Metadata
    reason: str = ""
    is_paper: bool = True
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.TIMEOUT
        )
    
    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIAL_FILL
        )
    
    def transition_to(self, new_state: OrderState, reason: str = "") -> None:
        """Transition to a new state with validation"""
        valid_next_states = VALID_TRANSITIONS.get(self.status, [])
        
        if new_state not in valid_next_states:
            raise InvalidStateTransition(
                f"Cannot transition from {self.status.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_next_states]}"
            )
        
        old_state = self.status
        self.status = new_state
        self.reason = reason
        self.state_history.append((
            datetime.utcnow(),
            old_state.value,
            new_state.value,
            reason
        ))
        
        # Update timestamps
        if new_state == OrderState.SUBMITTED:
            self.submitted_at = datetime.utcnow()
        elif new_state == OrderState.FILLED:
            self.filled_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'limit_price': self.limit_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'reason': self.reason,
            'is_paper': self.is_paper
        }


class OrderManager:
    """Manages order lifecycle with state machine and notifications"""
    
    def __init__(self, database: Any, executor: Any, notifier: Any,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize order manager.
        
        Args:
            database: Database instance
            executor: Order executor (paper or live)
            notifier: Notifier for alerts
            logger: Optional logger instance
            config: Order management configuration
        """
        self.database = database
        self.executor = executor
        self.notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        self.active_orders: Dict[str, Order] = {}
        
        # Order timeout settings
        self.order_timeout_seconds = self.config.get('order_timeout_seconds', 60)
        self.stale_check_interval = self.config.get('stale_check_interval_seconds', 30)
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
    
    def create_order(self, symbol: str, side: str, quantity: int,
                     order_type: str = 'market', limit_price: float = None,
                     stop_loss: float = None, take_profit: float = None,
                     is_paper: bool = True) -> Order:
        """Create a new order"""
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_paper=is_paper
        )
        
        self.active_orders[order.id] = order
        
        if self.database:
            try:
                self.database.save_order(order)
            except Exception as e:
                self.logger.warning(f"Failed to save order to database: {e}")
        
        self.logger.info(f"Order created: {order.id} - {side} {quantity} {symbol}")
        
        return order
    
    def submit_order(self, order: Order) -> None:
        """Submit order to broker with notifications"""
        try:
            order.transition_to(OrderState.PENDING, "Preparing to submit")
            
            # Execute via paper or live executor
            result = self.executor.execute_order(order)
            
            order.transition_to(OrderState.SUBMITTED, "Sent to broker")
            
            # Handle fill
            if result.filled_quantity == order.quantity:
                order.filled_quantity = result.filled_quantity
                order.avg_fill_price = result.fill_price
                order.transition_to(OrderState.FILLED, "Completely filled")
                
                # Send notification
                if self.notifier:
                    try:
                        self.notifier.send_trade_notification(order, result)
                    except Exception as e:
                        self.logger.warning(f"Failed to send notification: {e}")
                
            elif result.filled_quantity > 0:
                order.filled_quantity = result.filled_quantity
                order.avg_fill_price = result.fill_price
                order.transition_to(
                    OrderState.PARTIAL_FILL,
                    f"Filled {result.filled_quantity}/{order.quantity}"
                )
                
                if self.notifier:
                    try:
                        self.notifier.send_partial_fill_notification(order, result)
                    except Exception as e:
                        self.logger.warning(f"Failed to send notification: {e}")
            
            if self.database:
                self.database.update_order(order)
            
        except Exception as e:
            order.transition_to(OrderState.REJECTED, str(e))
            self.logger.error(f"Order rejected: {order.id} - {e}")
            
            if self.database:
                self.database.update_order(order)
            
            if self.notifier:
                try:
                    self.notifier.send_order_rejected_notification(order, str(e))
                except Exception:
                    pass
            
            raise
    
    def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an active order with notification"""
        order = self.active_orders.get(order_id)
        if not order:
            self.logger.warning(f"Order not found: {order_id}")
            return False
        
        if order.is_terminal:
            self.logger.warning(f"Cannot cancel terminal order: {order_id}")
            return False
        
        try:
            order.transition_to(OrderState.CANCELLED, reason)
            
            if self.database:
                self.database.update_order(order)
            
            if self.notifier:
                try:
                    self.notifier.send_order_cancelled_notification(order, reason)
                except Exception:
                    pass
            
            self.logger.info(f"Order cancelled: {order_id} - {reason}")
            return True
            
        except InvalidStateTransition as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    def check_order_timeouts(self) -> List[Order]:
        """Check for timed-out orders and transition them"""
        timed_out = []
        now = datetime.utcnow()
        
        for order in list(self.active_orders.values()):
            if order.status == OrderState.SUBMITTED:
                if order.submitted_at:
                    elapsed = (now - order.submitted_at).total_seconds()
                    if elapsed > self.order_timeout_seconds:
                        try:
                            order.transition_to(
                                OrderState.TIMEOUT,
                                f"No response after {elapsed:.0f}s"
                            )
                            
                            if self.database:
                                self.database.update_order(order)
                            
                            if self.notifier:
                                try:
                                    self.notifier.send_order_timeout_notification(order)
                                except Exception:
                                    pass
                            
                            timed_out.append(order)
                            self.logger.warning(f"Order timed out: {order.id}")
                            
                        except InvalidStateTransition:
                            pass
        
        return timed_out
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID"""
        return self.active_orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all non-terminal orders"""
        return [o for o in self.active_orders.values() if o.is_active]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for a symbol"""
        return [o for o in self.active_orders.values() if o.symbol == symbol]
    
    def cleanup_terminal_orders(self, max_age_hours: int = 24) -> int:
        """Remove old terminal orders from memory"""
        now = datetime.utcnow()
        removed = 0
        
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            if order.is_terminal:
                age = (now - order.created_at).total_seconds() / 3600
                if age > max_age_hours:
                    del self.active_orders[order_id]
                    removed += 1
        
        if removed > 0:
            self.logger.info(f"Cleaned up {removed} old orders")
        
        return removed
    
    def recover_orders(self) -> int:
        """Recover order state from database on startup"""
        if not self.database:
            return 0
        
        try:
            pending_orders = self.database.get_pending_orders()
            
            for db_order in pending_orders:
                order = Order(
                    symbol=db_order.symbol,
                    side=db_order.side.value,
                    quantity=db_order.quantity,
                    order_type=db_order.order_type.value if hasattr(db_order.order_type, 'value') else db_order.order_type,
                    limit_price=db_order.limit_price
                )
                order.id = db_order.id
                order.status = OrderState(db_order.status.value if hasattr(db_order.status, 'value') else db_order.status)
                order.filled_quantity = db_order.filled_quantity or 0
                order.created_at = db_order.created_at
                order.submitted_at = db_order.submitted_at
                
                self.active_orders[order.id] = order
                self.logger.info(f"Recovered order: {order.id} - {order.status.value}")
            
            return len(pending_orders)
            
        except Exception as e:
            self.logger.error(f"Failed to recover orders: {e}")
            return 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get order manager summary"""
        active = self.get_active_orders()
        
        return {
            'total_orders': len(self.active_orders),
            'active_orders': len(active),
            'pending': len([o for o in active if o.status == OrderState.PENDING]),
            'submitted': len([o for o in active if o.status == OrderState.SUBMITTED]),
            'partial_fill': len([o for o in active if o.status == OrderState.PARTIAL_FILL]),
            'timeout_seconds': self.order_timeout_seconds
        }


