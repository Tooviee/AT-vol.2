"""
Tests for the trading strategy module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.strategy import USAStrategy, Signal


@pytest.fixture
def strategy():
    """Create a strategy instance for testing"""
    config = {
        'sma_short': 10,
        'sma_long': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_period': 14,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0
    }
    return USAStrategy(config)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Create realistic price data with trend
    np.random.seed(42)
    base_price = 100
    returns = np.random.randn(100) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(100) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df


class TestStrategyIndicators:
    """Test indicator calculations"""
    
    def test_calculate_indicators(self, strategy, sample_data):
        """Test that all indicators are calculated"""
        df = strategy.calculate_indicators(sample_data)
        
        assert 'SMA_short' in df.columns
        assert 'SMA_long' in df.columns
        assert 'MACD' in df.columns
        assert 'MACD_signal' in df.columns
        assert 'RSI' in df.columns
        assert 'ATR' in df.columns
    
    def test_sma_values(self, strategy, sample_data):
        """Test SMA calculation correctness"""
        df = strategy.calculate_indicators(sample_data)
        
        # SMA short should be average of last 10 closes
        expected_sma = sample_data['Close'].iloc[-10:].mean()
        actual_sma = df['SMA_short'].iloc[-1]
        
        assert abs(expected_sma - actual_sma) < 0.01
    
    def test_rsi_bounds(self, strategy, sample_data):
        """Test RSI is within valid bounds"""
        df = strategy.calculate_indicators(sample_data)
        
        rsi_values = df['RSI'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_atr_positive(self, strategy, sample_data):
        """Test ATR is always positive"""
        df = strategy.calculate_indicators(sample_data)
        
        atr_values = df['ATR'].dropna()
        assert (atr_values > 0).all()


class TestSignalGeneration:
    """Test signal generation"""
    
    def test_generate_signal_returns_signal(self, strategy, sample_data):
        """Test that generate_signal returns a TradeSignal"""
        signal = strategy.generate_signal(sample_data, 'TEST')
        
        assert signal is not None
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
        assert signal.symbol == 'TEST'
        assert signal.price > 0
    
    def test_insufficient_data_returns_hold(self, strategy):
        """Test that insufficient data returns HOLD signal"""
        # Create minimal data
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000000] * 5
        }, index=dates)
        
        signal = strategy.generate_signal(df, 'TEST')
        
        assert signal.signal == Signal.HOLD
        assert 'Insufficient data' in signal.reason
    
    def test_signal_includes_stop_loss(self, strategy, sample_data):
        """Test that BUY signals include stop-loss"""
        signal = strategy.generate_signal(sample_data, 'TEST')
        
        if signal.signal == Signal.BUY:
            assert signal.stop_loss > 0
            assert signal.stop_loss < signal.price
            assert signal.take_profit > signal.price


class TestStopLossCalculation:
    """Test stop-loss and take-profit calculations"""
    
    def test_buy_stop_loss_below_price(self, strategy):
        """Test buy stop-loss is below entry price"""
        entry_price = 100
        atr = 2.0
        
        stop_loss = strategy.calculate_stop_loss(entry_price, atr, 'buy')
        
        assert stop_loss < entry_price
        assert stop_loss == entry_price - (atr * 2.0)  # 2.0 ATR multiplier
    
    def test_buy_take_profit_above_price(self, strategy):
        """Test buy take-profit is above entry price"""
        entry_price = 100
        atr = 2.0
        
        take_profit = strategy.calculate_take_profit(entry_price, atr, 'buy')
        
        assert take_profit > entry_price
        assert take_profit == entry_price + (atr * 3.0)  # 3.0 ATR multiplier
    
    def test_sell_stop_loss_above_price(self, strategy):
        """Test sell stop-loss is above entry price"""
        entry_price = 100
        atr = 2.0
        
        stop_loss = strategy.calculate_stop_loss(entry_price, atr, 'sell')
        
        assert stop_loss > entry_price


class TestIndicatorSummary:
    """Test indicator summary"""
    
    def test_get_indicator_summary(self, strategy, sample_data):
        """Test indicator summary generation"""
        summary = strategy.get_indicator_summary(sample_data)
        
        assert 'price' in summary
        assert 'sma_short' in summary
        assert 'sma_long' in summary
        assert 'rsi' in summary
        assert 'atr' in summary
        assert 'sma_trend' in summary
        assert summary['sma_trend'] in ['bullish', 'bearish']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


