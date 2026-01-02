"""
Tests for circuit breaker module.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.fixture
def mock_balance_tracker():
    """Create mock balance tracker"""
    tracker = Mock()
    tracker.get_total_balance.return_value = 10000000  # 10M KRW
    tracker.get_unrealized_pnl.return_value = 0
    return tracker


@pytest.fixture
def mock_database():
    """Create mock database"""
    db = Mock()
    db.get_realized_pnl_today.return_value = 0
    db.log_circuit_breaker_event = Mock()
    return db


@pytest.fixture
def circuit_breaker_config():
    """Default circuit breaker config"""
    return {
        'max_consecutive_losses': 3,
        'max_daily_loss_percent': 5.0,
        'max_daily_loss_krw': 500000,
        'loss_type': 'realized',
        'api_error_threshold': 5,
        'cooldown_minutes': 30
    }


@pytest.fixture
def circuit_breaker(circuit_breaker_config, mock_balance_tracker, mock_database):
    """Create circuit breaker instance"""
    return CircuitBreaker(
        circuit_breaker_config,
        mock_balance_tracker,
        mock_database
    )


class TestCircuitBreakerState:
    """Test circuit breaker state management"""
    
    def test_initial_state_closed(self, circuit_breaker):
        """Test initial state is closed"""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    def test_can_trade_when_closed(self, circuit_breaker):
        """Test can trade when closed"""
        can_trade, reason = circuit_breaker.can_trade()
        assert can_trade
        assert reason == "OK"


class TestConsecutiveLosses:
    """Test consecutive loss tracking"""
    
    def test_record_win_resets_losses(self, circuit_breaker):
        """Test winning trade resets consecutive losses"""
        circuit_breaker.consecutive_losses = 2
        circuit_breaker.record_trade_result(is_win=True)
        
        assert circuit_breaker.consecutive_losses == 0
    
    def test_record_loss_increments(self, circuit_breaker):
        """Test losing trade increments counter"""
        circuit_breaker.record_trade_result(is_win=False)
        
        assert circuit_breaker.consecutive_losses == 1
    
    def test_max_losses_trips_breaker(self, circuit_breaker):
        """Test max consecutive losses trips circuit breaker"""
        for _ in range(3):
            circuit_breaker.record_trade_result(is_win=False)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        can_trade, _ = circuit_breaker.can_trade()
        assert not can_trade


class TestDailyLossLimit:
    """Test daily loss limit"""
    
    def test_daily_loss_trips_breaker(self, circuit_breaker, mock_database):
        """Test exceeding daily loss trips breaker"""
        mock_database.get_realized_pnl_today.return_value = -600000  # 600K loss
        
        circuit_breaker.check_daily_loss_limit()
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_percentage_loss_trips_breaker(self, circuit_breaker, mock_database, mock_balance_tracker):
        """Test exceeding percentage loss trips breaker"""
        mock_database.get_realized_pnl_today.return_value = -600000  # 6% of 10M
        
        circuit_breaker.check_daily_loss_limit()
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestAPIErrors:
    """Test API error handling"""
    
    def test_api_errors_trip_breaker(self, circuit_breaker):
        """Test too many API errors trips breaker"""
        for _ in range(5):
            circuit_breaker.record_api_error()
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestCooldown:
    """Test cooldown functionality"""
    
    def test_cooldown_prevents_trading(self, circuit_breaker):
        """Test cooldown prevents immediate trading"""
        # Trip the breaker
        for _ in range(3):
            circuit_breaker.record_trade_result(is_win=False)
        
        can_trade, reason = circuit_breaker.can_trade()
        assert not can_trade
        assert "resume" in reason.lower() or "active" in reason.lower()
    
    def test_cooldown_expires(self, circuit_breaker):
        """Test cooldown expires after timeout"""
        # Trip the breaker
        for _ in range(3):
            circuit_breaker.record_trade_result(is_win=False)
        
        # Simulate time passing
        circuit_breaker.tripped_at = datetime.now() - timedelta(minutes=35)
        
        can_trade, _ = circuit_breaker.can_trade()
        assert can_trade


class TestReset:
    """Test reset functionality"""
    
    def test_force_reset(self, circuit_breaker):
        """Test force reset works"""
        # Trip the breaker
        for _ in range(3):
            circuit_breaker.record_trade_result(is_win=False)
        
        circuit_breaker.reset(force=True)
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.consecutive_losses == 0
    
    def test_daily_reset(self, circuit_breaker):
        """Test daily reset clears losses"""
        circuit_breaker.consecutive_losses = 2
        
        circuit_breaker.reset_daily()
        
        assert circuit_breaker.consecutive_losses == 0


class TestLossTypes:
    """Test different loss type configurations"""
    
    def test_realized_loss_type(self, mock_balance_tracker, mock_database):
        """Test realized loss type"""
        config = {
            'max_consecutive_losses': 3,
            'max_daily_loss_percent': 5.0,
            'max_daily_loss_krw': 500000,
            'loss_type': 'realized',
            'api_error_threshold': 5,
            'cooldown_minutes': 30
        }
        
        mock_database.get_realized_pnl_today.return_value = -300000
        mock_balance_tracker.get_unrealized_pnl.return_value = -500000
        
        cb = CircuitBreaker(config, mock_balance_tracker, mock_database)
        loss = cb.get_daily_loss()
        
        assert loss == 300000  # Only realized
    
    def test_both_loss_type(self, mock_balance_tracker, mock_database):
        """Test both loss type"""
        config = {
            'max_consecutive_losses': 3,
            'max_daily_loss_percent': 5.0,
            'max_daily_loss_krw': 500000,
            'loss_type': 'both',
            'api_error_threshold': 5,
            'cooldown_minutes': 30
        }
        
        mock_database.get_realized_pnl_today.return_value = -200000
        mock_balance_tracker.get_unrealized_pnl.return_value = -300000
        
        cb = CircuitBreaker(config, mock_balance_tracker, mock_database)
        loss = cb.get_daily_loss()
        
        assert loss == 500000  # Both combined


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


