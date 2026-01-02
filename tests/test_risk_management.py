"""
Tests for risk management module.
"""

import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.risk_management import RiskManager, PositionSizeResult


@pytest.fixture
def mock_balance_tracker():
    """Create a mock balance tracker"""
    tracker = Mock()
    tracker.get_total_balance.return_value = 10000000  # 10M KRW
    tracker.get_available_cash.return_value = 5000000  # 5M KRW
    tracker.get_positions_value.return_value = 5000000  # 5M KRW
    tracker.get_position_count.return_value = 2
    return tracker


@pytest.fixture
def risk_config():
    """Default risk configuration"""
    return {
        'risk_per_trade_percent': 1.0,
        'max_position_size_percent': 10.0,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'max_total_exposure_percent': 80.0,
        'max_drawdown_percent': 15.0,
        'max_correlated_positions': 5
    }


@pytest.fixture
def risk_manager(risk_config, mock_balance_tracker):
    """Create a risk manager instance"""
    return RiskManager(risk_config, mock_balance_tracker)


class TestPositionSizing:
    """Test position size calculations"""
    
    def test_calculate_position_size(self, risk_manager):
        """Test basic position size calculation"""
        result = risk_manager.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss_price=145.0,
            exchange_rate=1350.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.shares > 0
        assert result.allowed
    
    def test_position_size_respects_risk_limit(self, risk_manager):
        """Test that position size respects risk per trade"""
        result = risk_manager.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss_price=145.0,  # $5 risk per share
            exchange_rate=1350.0
        )
        
        # Risk should be approximately 1% of 10M KRW = 100,000 KRW
        assert result.risk_amount <= 100000 * 1.5  # Allow some buffer
    
    def test_position_size_respects_max_size(self, risk_manager):
        """Test that position size respects max position percentage"""
        result = risk_manager.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss_price=149.0,  # Very tight stop
            exchange_rate=1350.0
        )
        
        # Max position is 10% of 10M KRW = 1M KRW
        assert result.position_value <= 1000000 * 1.1  # Allow some buffer
    
    def test_invalid_stop_loss_returns_zero(self, risk_manager):
        """Test that invalid stop-loss returns 0 shares"""
        result = risk_manager.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss_price=150.0,  # Same as entry
            exchange_rate=1350.0
        )
        
        assert result.shares == 0
        assert not result.allowed


class TestPortfolioLimits:
    """Test portfolio-level risk limits"""
    
    def test_check_portfolio_limits_pass(self, risk_manager):
        """Test portfolio limits pass when under limits"""
        allowed, reason = risk_manager.check_portfolio_limits()
        
        assert allowed
        assert reason == "OK"
    
    def test_max_exposure_blocks_new_position(self, risk_config, mock_balance_tracker):
        """Test that max exposure blocks new positions"""
        mock_balance_tracker.get_positions_value.return_value = 8500000  # 85% exposure
        
        risk_manager = RiskManager(risk_config, mock_balance_tracker)
        allowed, reason = risk_manager.check_portfolio_limits()
        
        assert not allowed
        assert "exposure" in reason.lower()
    
    def test_max_positions_blocks_new_position(self, risk_config, mock_balance_tracker):
        """Test that max positions blocks new positions"""
        mock_balance_tracker.get_position_count.return_value = 5  # At max
        
        risk_manager = RiskManager(risk_config, mock_balance_tracker)
        allowed, reason = risk_manager.check_portfolio_limits()
        
        assert not allowed
        assert "positions" in reason.lower()


class TestTradeValidation:
    """Test trade validation"""
    
    def test_validate_valid_trade(self, risk_manager):
        """Test validation of a valid trade"""
        is_valid, reason = risk_manager.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=4,
            entry_price=150.0,
            stop_loss=145.0,
            exchange_rate=1350.0
        )
        
        assert is_valid
    
    def test_validate_oversized_trade(self, risk_manager):
        """Test validation rejects oversized trade"""
        is_valid, reason = risk_manager.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=1000,  # Way too large
            entry_price=150.0,
            stop_loss=145.0,
            exchange_rate=1350.0
        )
        
        assert not is_valid


class TestStopLossCalculations:
    """Test stop-loss and take-profit calculations"""
    
    def test_buy_stop_loss(self, risk_manager):
        """Test buy stop-loss calculation"""
        stop_loss = risk_manager.calculate_stop_loss(100.0, 2.0, 'buy')
        
        assert stop_loss == 96.0  # 100 - (2 * 2.0)
    
    def test_sell_stop_loss(self, risk_manager):
        """Test sell stop-loss calculation"""
        stop_loss = risk_manager.calculate_stop_loss(100.0, 2.0, 'sell')
        
        assert stop_loss == 104.0  # 100 + (2 * 2.0)
    
    def test_buy_take_profit(self, risk_manager):
        """Test buy take-profit calculation"""
        take_profit = risk_manager.calculate_take_profit(100.0, 2.0, 'buy')
        
        assert take_profit == 106.0  # 100 + (2 * 3.0)
    
    def test_risk_reward_ratio(self, risk_manager):
        """Test risk/reward ratio calculation"""
        ratio = risk_manager.calculate_risk_reward_ratio(
            entry_price=100.0,
            stop_loss=96.0,
            take_profit=106.0
        )
        
        assert ratio == 1.5  # 6 reward / 4 risk


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


