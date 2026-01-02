"""
Tests for order manager module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.order_manager import (
    OrderManager, Order, OrderState, 
    InvalidStateTransition, VALID_TRANSITIONS
)


@pytest.fixture
def mock_database():
    """Create a mock database"""
    db = Mock()
    db.save_order = Mock()
    db.update_order = Mock()
    db.get_pending_orders = Mock(return_value=[])
    return db


@pytest.fixture
def mock_executor():
    """Create a mock executor"""
    executor = Mock()
    result = Mock()
    result.filled_quantity = 10
    result.fill_price = 150.0
    result.is_partial = False
    executor.execute_order = Mock(return_value=result)
    return executor


@pytest.fixture
def mock_notifier():
    """Create a mock notifier"""
    notifier = Mock()
    return notifier


@pytest.fixture
def order_manager(mock_database, mock_executor, mock_notifier):
    """Create an order manager instance"""
    config = {
        'order_timeout_seconds': 60,
        'stale_check_interval_seconds': 30,
        'max_retry_attempts': 3
    }
    return OrderManager(
        mock_database, mock_executor, mock_notifier,
        config=config
    )


class TestOrder:
    """Test Order class"""
    
    def test_order_creation(self):
        """Test basic order creation"""
        order = Order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        
        assert order.symbol == 'AAPL'
        assert order.side == 'buy'
        assert order.quantity == 10
        assert order.status == OrderState.CREATED
    
    def test_order_id_generated(self):
        """Test order ID is auto-generated"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        
        assert order.id is not None
        assert len(order.id) > 0
    
    def test_remaining_quantity(self):
        """Test remaining quantity calculation"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.filled_quantity = 3
        
        assert order.remaining_quantity == 7
    
    def test_is_terminal_for_filled(self):
        """Test is_terminal for filled orders"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.status = OrderState.FILLED
        
        assert order.is_terminal
    
    def test_is_active_for_submitted(self):
        """Test is_active for submitted orders"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.status = OrderState.SUBMITTED
        
        assert order.is_active


class TestStateTransitions:
    """Test order state transitions"""
    
    def test_valid_transition(self):
        """Test valid state transition"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.transition_to(OrderState.PENDING, "Test")
        
        assert order.status == OrderState.PENDING
    
    def test_invalid_transition_raises_error(self):
        """Test invalid transition raises error"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.status = OrderState.FILLED  # Terminal state
        
        with pytest.raises(InvalidStateTransition):
            order.transition_to(OrderState.SUBMITTED, "Test")
    
    def test_state_history_recorded(self):
        """Test state changes are recorded"""
        order = Order(symbol='AAPL', side='buy', quantity=10)
        order.transition_to(OrderState.PENDING, "Step 1")
        order.transition_to(OrderState.SUBMITTED, "Step 2")
        
        assert len(order.state_history) == 2
    
    def test_all_valid_transitions(self):
        """Test all defined valid transitions work"""
        for from_state, valid_to_states in VALID_TRANSITIONS.items():
            for to_state in valid_to_states:
                order = Order(symbol='AAPL', side='buy', quantity=10)
                order.status = from_state
                
                # Should not raise
                order.transition_to(to_state, "Test")
                assert order.status == to_state


class TestOrderManager:
    """Test OrderManager class"""
    
    def test_create_order(self, order_manager):
        """Test order creation through manager"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        
        assert order.symbol == 'AAPL'
        assert order.id in order_manager.active_orders
    
    def test_submit_order(self, order_manager, mock_executor):
        """Test order submission"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        
        order_manager.submit_order(order)
        
        mock_executor.execute_order.assert_called_once()
        assert order.status == OrderState.FILLED
    
    def test_cancel_order(self, order_manager):
        """Test order cancellation"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        order.transition_to(OrderState.PENDING, "Test")
        
        result = order_manager.cancel_order(order.id, "Test cancel")
        
        assert result
        assert order.status == OrderState.CANCELLED
    
    def test_cancel_terminal_order_fails(self, order_manager):
        """Test cancelling terminal order fails"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        order.status = OrderState.FILLED
        
        result = order_manager.cancel_order(order.id, "Test")
        
        assert not result
    
    def test_get_active_orders(self, order_manager):
        """Test getting active orders"""
        order1 = order_manager.create_order(symbol='AAPL', side='buy', quantity=10)
        order1.status = OrderState.SUBMITTED
        
        order2 = order_manager.create_order(symbol='MSFT', side='buy', quantity=5)
        order2.status = OrderState.FILLED
        
        active = order_manager.get_active_orders()
        
        assert len(active) == 1
        assert active[0].symbol == 'AAPL'


class TestOrderTimeouts:
    """Test order timeout handling"""
    
    def test_check_order_timeouts(self, order_manager):
        """Test timeout detection"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        order.status = OrderState.SUBMITTED
        order.submitted_at = datetime.utcnow() - timedelta(seconds=120)  # 2 min ago
        
        timed_out = order_manager.check_order_timeouts()
        
        assert len(timed_out) == 1
        assert timed_out[0].status == OrderState.TIMEOUT
    
    def test_non_expired_order_not_timed_out(self, order_manager):
        """Test non-expired orders aren't timed out"""
        order = order_manager.create_order(
            symbol='AAPL',
            side='buy',
            quantity=10
        )
        order.status = OrderState.SUBMITTED
        order.submitted_at = datetime.utcnow()  # Just now
        
        timed_out = order_manager.check_order_timeouts()
        
        assert len(timed_out) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


