"""
Tests for the trading strategy module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.strategy import USAStrategy, Signal, TradeSignal


@pytest.fixture
def strategy():
    """Create a strategy instance for testing (hybrid MACD/RSI/ATR)"""
    config = {
        'sma_50': 50,
        'sma_200': 200,
        'ema_fast': 12,
        'ema_slow': 26,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_period': 14,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'risk_reward_ratio': 2.0,
    }
    return USAStrategy(config)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing (min_data_points >= 210 for sma_200)"""
    n = 250
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
    np.random.seed(42)
    base_price = 100
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(n) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    return df


@pytest.fixture
def strategy_rr15():
    """Strategy with risk_reward_ratio=1.5 for take-profit 1.5:1 tests."""
    config = {
        'sma_50': 50,
        'sma_200': 200,
        'ema_fast': 12,
        'ema_slow': 26,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_period': 14,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'risk_reward_ratio': 1.5,
    }
    return USAStrategy(config)


class TestStrategyIndicators:
    """Test indicator calculations"""
    
    def test_calculate_indicators(self, strategy, sample_data):
        """Test that all indicators are calculated"""
        df = strategy.calculate_indicators(sample_data)
        assert 'SMA_50' in df.columns
        assert 'SMA_200' in df.columns
        assert 'EMA_12' in df.columns
        assert 'EMA_26' in df.columns
        assert 'MACD' in df.columns
        assert 'MACD_signal' in df.columns
        assert 'RSI' in df.columns
        assert 'ATR' in df.columns
    
    def test_sma_values(self, strategy, sample_data):
        """Test SMA 50 calculation correctness"""
        df = strategy.calculate_indicators(sample_data)
        expected_sma = sample_data['Close'].iloc[-50:].mean()
        actual_sma = df['SMA_50'].iloc[-1]
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


class TestIndicatorCalculations:
    """Indicator calculation tests: MACD (12/26/9), RSI (14), ATR (14), SMA 50/200, EMA 12/26."""

    def test_macd_calculation(self, strategy, sample_data):
        """MACD = EMA_fast - EMA_slow; Signal = EMA(9) of MACD; fast=12, slow=26, signal=9."""
        df = strategy.calculate_indicators(sample_data)
        exp_fast = sample_data['Close'].ewm(span=12, adjust=False).mean()
        exp_slow = sample_data['Close'].ewm(span=26, adjust=False).mean()
        expected_macd = exp_fast - exp_slow
        macd_series = df['MACD']
        expected_signal = macd_series.ewm(span=9, adjust=False).mean()
        # Last valid row
        i = -1
        assert abs(df['MACD'].iloc[i] - expected_macd.iloc[i]) < 1e-6
        assert abs(df['MACD_signal'].iloc[i] - expected_signal.iloc[i]) < 1e-6
        assert abs(df['MACD_hist'].iloc[i] - (df['MACD'].iloc[i] - df['MACD_signal'].iloc[i])) < 1e-6

    def test_rsi_calculation(self, strategy, sample_data):
        """RSI period=14; formula 100 - 100/(1+RS), RS = avg_gain/avg_loss."""
        df = strategy.calculate_indicators(sample_data)
        rsi = df['RSI'].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()
        # rolling(14) yields first valid at index 13, so we expect at least n-14 valid
        assert len(rsi) >= len(sample_data) - 14

    def test_atr_calculation(self, strategy, sample_data):
        """ATR(14) = SMA(14) of True Range; TR = max(H-L, |H-C_prev|, |L-C_prev|)."""
        df = strategy.calculate_indicators(sample_data)
        hl = sample_data['High'] - sample_data['Low']
        hc = (sample_data['High'] - sample_data['Close'].shift(1)).abs()
        lc = (sample_data['Low'] - sample_data['Close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        expected_atr = tr.rolling(14).mean()
        valid = expected_atr.notna()
        assert (abs(df.loc[valid, 'ATR'] - expected_atr[valid]) < 1e-6).all()

    def test_sma_50_calculation(self, strategy, sample_data):
        """SMA 50 = rolling mean of Close over 50."""
        df = strategy.calculate_indicators(sample_data)
        expected = sample_data['Close'].rolling(50).mean()
        valid = expected.notna()
        assert (abs(df.loc[valid, 'SMA_50'] - expected[valid]) < 1e-6).all()

    def test_sma_200_calculation(self, strategy, sample_data):
        """SMA 200 = rolling mean of Close over 200."""
        df = strategy.calculate_indicators(sample_data)
        expected = sample_data['Close'].rolling(200).mean()
        valid = expected.notna()
        assert (abs(df.loc[valid, 'SMA_200'] - expected[valid]) < 1e-6).all()

    def test_ema_12_calculation(self, strategy, sample_data):
        """EMA 12 = ewm(span=12, adjust=False) of Close."""
        df = strategy.calculate_indicators(sample_data)
        expected = sample_data['Close'].ewm(span=12, adjust=False).mean()
        assert (abs(df['EMA_12'] - expected) < 1e-6).all()

    def test_ema_26_calculation(self, strategy, sample_data):
        """EMA 26 = ewm(span=26, adjust=False) of Close."""
        df = strategy.calculate_indicators(sample_data)
        expected = sample_data['Close'].ewm(span=26, adjust=False).mean()
        assert (abs(df['EMA_26'] - expected) < 1e-6).all()


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

    def test_signal_timestamp_not_none(self, strategy, sample_data):
        """Generate_signal must set signal.timestamp."""
        signal = strategy.generate_signal(sample_data, 'TEST')
        assert signal.timestamp is not None


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
        """Test buy take-profit is above entry (R:R = 2:1 vs 2*ATR stop => 4*ATR target)"""
        entry_price = 100
        atr = 2.0
        take_profit = strategy.calculate_take_profit(entry_price, atr, 'buy')
        assert take_profit > entry_price
        # risk=2*ATR=4, reward=4*2=8 => entry + 8
        assert take_profit == entry_price + (atr * 2.0 * strategy.risk_reward_ratio)
    
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
        assert 'sma_50' in summary
        assert 'sma_200' in summary
        assert 'ema_12' in summary
        assert 'ema_26' in summary
        assert 'rsi' in summary
        assert 'atr' in summary
        assert 'sma_trend' in summary
        assert summary['sma_trend'] in ['bullish', 'bearish']


class TestCheckExitConditions:
    """Tests for check_exit_conditions (trailing stop, R:R take-profit, fixed stop)."""

    def test_trailing_stop_triggered_long(self, strategy, sample_data):
        """Long: price drops to high_since_entry - 2*ATR -> (True, 'Trailing stop-loss triggered')."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        entry = 100.0
        atr = 2.0
        risk = atr * strategy.stop_loss_atr_multiplier  # 4.0
        high_since = 110.0
        trailing = high_since - risk  # 106
        df.loc[last, 'High'] = high_since
        df.loc[last, 'Close'] = 105.0  # below 106
        df.loc[last, 'ATR'] = atr
        pos = {'avg_price': entry, 'side': 'buy', 'entry_time': df.index[-3]}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert ok is True
        assert "Trailing stop-loss" in msg

    def test_take_profit_triggered_long(self, strategy, sample_data):
        """Long: entry + (2*ATR * risk_reward_ratio) reached -> (True, 'Take-profit (R:R) triggered')."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        entry, atr = 100.0, 2.0
        reward = atr * strategy.stop_loss_atr_multiplier * strategy.risk_reward_ratio  # 8
        tp = entry + reward  # 108
        df.loc[last, 'Close'] = tp
        df.loc[last, 'ATR'] = atr
        pos = {'avg_price': entry, 'side': 'buy'}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert ok is True
        assert "Take-profit (R:R)" in msg

    def test_take_profit_triggered_short(self, strategy, sample_data):
        """Short: entry - (2*ATR * risk_reward_ratio) reached -> (True, 'Take-profit (R:R) triggered')."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        entry, atr = 100.0, 2.0
        reward = atr * strategy.stop_loss_atr_multiplier * strategy.risk_reward_ratio
        tp = entry - reward  # 92
        df.loc[last, 'Close'] = tp
        df.loc[last, 'ATR'] = atr
        pos = {'avg_price': entry, 'side': 'sell'}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert ok is True
        assert "Take-profit (R:R)" in msg

    def test_take_profit_1_5_rr_long(self, strategy_rr15, sample_data):
        """Take-profit at 1.5:1 R:R for long."""
        df = strategy_rr15.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        entry, atr = 100.0, 2.0
        risk = atr * 2.0
        reward = risk * 1.5  # 6
        tp = entry + reward  # 106
        df.loc[last, 'Close'] = tp
        df.loc[last, 'ATR'] = atr
        pos = {'avg_price': entry, 'side': 'buy'}
        with patch.object(strategy_rr15, 'calculate_indicators', return_value=df):
            ok, msg = strategy_rr15.check_exit_conditions(df, pos)
        assert ok is True
        assert "Take-profit (R:R)" in msg

    def test_fixed_stop_loss_from_position(self, strategy, sample_data):
        """Fixed stop from position['stop_loss']: long price<=stop -> (True, 'Stop-loss triggered')."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        # Trailing runs before fixed stop. To hit fixed stop: trailing must not fire.
        # trailing = max(entry-risk, high_since-risk). Use entry=97, risk=4 => 93.
        # Cap High in last 60 bars to 96 so high_since=96, high_since-risk=92, max(93,92)=93.
        # Close=94 > 93 so no trailing. stop_loss=95, 94<=95 -> fixed stop triggers.
        df.loc[last, 'Close'] = 94.0
        df.loc[last, 'High'] = 96.0
        df.loc[last, 'ATR'] = 2.0
        for idx in df.index[-60:]:
            if df.loc[idx, 'High'] > 96:
                df.loc[idx, 'High'] = 96.0
        pos = {'avg_price': 97.0, 'side': 'buy', 'stop_loss': 95.0}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert ok is True
        assert "Stop-loss triggered" in msg

    def test_fixed_stop_loss_short(self, strategy, sample_data):
        """Short: price >= stop_loss -> (True, 'Stop-loss triggered')."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        # Trailing runs before fixed stop. For short: trailing = min(entry+risk, low_since+risk).
        # Use entry=110, risk=4 so entry+risk=114. Set Low in last 60 bars > 102 so
        # low_since+risk > 106; then trailing > 106 and Close=106 does not trigger trailing.
        # stop_loss=105, 106 >= 105 -> fixed stop triggers.
        df.loc[last, 'Close'] = 106.0
        df.loc[last, 'ATR'] = 2.0
        for idx in df.index[-60:]:
            if df.loc[idx, 'Low'] <= 102:
                df.loc[idx, 'Low'] = 103.0
        pos = {'avg_price': 110.0, 'side': 'sell', 'stop_loss': 105.0}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert ok is True
        assert "Stop-loss triggered" in msg

    def test_entry_time_none_uses_fallback(self, strategy, sample_data):
        """entry_time None: uses last ~60 bars for high_since; no crash."""
        df = strategy.calculate_indicators(sample_data).copy()
        pos = {'avg_price': 100.0, 'side': 'buy'}  # no entry_time
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        # Should not crash; may or may not trigger
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_entry_date_as_fallback(self, strategy, sample_data):
        """entry_date used when entry_time absent for mask."""
        df = strategy.calculate_indicators(sample_data).copy()
        pos = {'avg_price': 100.0, 'side': 'buy', 'entry_date': df.index[-5]}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_side_buy_vs_sell(self, strategy, sample_data):
        """Direction of trailing and take-profit respects side 'buy' vs 'sell'."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        entry, atr = 100.0, 2.0
        reward = atr * 2.0 * strategy.risk_reward_ratio
        # Long TP: entry+reward. Short TP: entry-reward. We already test both in
        # test_take_profit_triggered_long and test_take_profit_triggered_short.
        # Here: ensure sell trailing uses low_since + risk (price >= trailing).
        df.loc[last, 'High'] = 105
        df.loc[last, 'Low'] = 90.0
        df.loc[last, 'Close'] = 112.0  # above low_since+risk to trigger short trailing
        df.loc[last, 'ATR'] = atr
        low_since = 90.0
        risk = 4.0
        trailing_sell = low_since + risk  # 94
        pos = {'avg_price': 100.0, 'side': 'sell', 'entry_time': df.index[-2]}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            ok, msg = strategy.check_exit_conditions(df, pos)
        # Close 112 >= 94 -> trailing triggers for short
        assert ok is True
        assert "Trailing stop-loss" in msg


class TestEntryLogic:
    """BUY/NO-BUY with controlled data (patch last row after calculate_indicators)."""

    def test_buy_when_all_four_conditions_met(self, strategy, sample_data):
        """Price > SMA_200, MACD_cross_above_zero, RSI < 70, (SMA_50_up or Price > SMA_50) -> BUY."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 105.0
        df.loc[last, 'SMA_200'] = 100.0
        df.loc[last, 'SMA_50'] = 102.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = True
        df.loc[last, 'ATR'] = 2.0
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.BUY
        assert "Trend+EMA cross+RSI+SMA50" in sig.reason or "confirm" in sig.reason

    def test_hold_price_below_sma200(self, strategy, sample_data):
        """Price < SMA_200 -> HOLD, reason 'Price below SMA 200'."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 95.0
        df.loc[last, 'SMA_200'] = 100.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = True
        df.loc[last, 'ATR'] = 2.0
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert "Price below SMA 200" in sig.reason

    def test_hold_rsi_overbought(self, strategy, sample_data):
        """RSI >= 70 -> HOLD, reason 'RSI overbought (>= 70)'."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 105.0
        df.loc[last, 'SMA_200'] = 100.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 72.0
        df.loc[last, 'SMA_50_up'] = True
        df.loc[last, 'ATR'] = 2.0
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert "RSI overbought" in sig.reason

    def test_hold_no_macd_cross(self, strategy, sample_data):
        """No MACD cross above zero -> HOLD, reason 'No EMA12/26 bullish cross'."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 105.0
        df.loc[last, 'SMA_200'] = 100.0
        df.loc[last, 'MACD_cross_above_zero'] = False
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = True
        df.loc[last, 'ATR'] = 2.0
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert "MACD" in sig.reason or "EMA" in sig.reason or "cross" in sig.reason.lower()

    def test_hold_sma50_not_rising_and_price_below(self, strategy, sample_data):
        """SMA50 not rising and Price <= SMA_50 -> HOLD."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 99.0
        df.loc[last, 'SMA_200'] = 98.0
        df.loc[last, 'SMA_50'] = 100.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = False
        df.loc[last, 'ATR'] = 2.0
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert "SMA50" in sig.reason


class TestSellTrendBreakdown:
    """SELL when in position and price < SMA_200."""

    def test_sell_when_price_below_sma200_in_position(self, strategy, sample_data):
        """current_position is not None, last close < SMA_200 -> SELL, reason mentions SMA 200 or trend."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 95.0
        df.loc[last, 'SMA_200'] = 100.0
        df.loc[last, 'ATR'] = 2.0
        pos = {'avg_price': 90.0, 'side': 'buy'}
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST', current_position=pos)
        assert sig.signal == Signal.SELL
        assert "SMA 200" in sig.reason or "trend" in sig.reason.lower()


class TestEdgeCases:
    """ATR 0/NaN fallback, get_indicator_summary insufficient data, empty DataFrame."""

    def test_atr_zero_nan_fallback(self, strategy, sample_data):
        """When ATR is 0 or NaN, strategy uses atr=price*0.02; BUY has stop_loss < price, take_profit > price."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        price = 100.0
        df.loc[last, 'Close'] = price
        df.loc[last, 'ATR'] = 0.0
        df.loc[last, 'SMA_200'] = 98.0
        df.loc[last, 'SMA_50'] = 99.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = True
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.stop_loss < sig.price
        assert sig.take_profit > sig.price

    def test_atr_nan_fallback(self, strategy, sample_data):
        """ATR NaN: uses price*0.02; BUY yields stop_loss < price, take_profit > price."""
        df = strategy.calculate_indicators(sample_data).copy()
        last = df.index[-1]
        df.loc[last, 'Close'] = 100.0
        df.loc[last, 'ATR'] = np.nan
        df.loc[last, 'SMA_200'] = 98.0
        df.loc[last, 'SMA_50'] = 99.0
        df.loc[last, 'MACD_cross_above_zero'] = True
        df.loc[last, 'RSI'] = 50.0
        df.loc[last, 'SMA_50_up'] = True
        with patch.object(strategy, 'calculate_indicators', return_value=df):
            sig = strategy.generate_signal(df, 'TEST')
        assert sig.stop_loss < sig.price
        assert sig.take_profit > sig.price

    def test_get_indicator_summary_insufficient_data(self, strategy):
        """len(df) < min_data_points -> summary['error'] == 'Insufficient data'."""
        n = 50
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        df = pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99., 'Close': 100., 'Volume': 1e6}, index=dates)
        summary = strategy.get_indicator_summary(df)
        assert summary.get('error') == 'Insufficient data'

    def test_insufficient_data_hold(self, strategy):
        """Insufficient data (< min_data_points) -> HOLD with reason containing 'Insufficient'."""
        n = 20
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        df = pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99., 'Close': 100., 'Volume': 1e6}, index=dates)
        sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert 'Insufficient' in sig.reason

    def test_empty_dataframe_handling(self, strategy):
        """Empty DataFrame -> HOLD, no crash."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        sig = strategy.generate_signal(df, 'TEST')
        assert sig.signal == Signal.HOLD
        assert 'Insufficient' in sig.reason


class TestStrategyConfig:
    """StrategyConfig and ConfigLoader SMA ordering validation."""

    def test_sma_200_must_be_greater_than_sma_50(self):
        """StrategyConfig(sma_50=100, sma_200=50) -> ValueError('sma_200 must be greater than sma_50')."""
        from modules.config_loader import StrategyConfig
        with pytest.raises(ValueError, match="sma_200 must be greater than sma_50"):
            StrategyConfig(sma_50=100, sma_200=50)

    def test_strategy_config_valid_when_sma_50_lt_sma_200(self):
        """StrategyConfig(sma_50=50, sma_200=200) -> valid."""
        from modules.config_loader import StrategyConfig
        cfg = StrategyConfig(sma_50=50, sma_200=200)
        assert cfg.sma_50 == 50
        assert cfg.sma_200 == 200

    def test_config_loader_validate_sma_order(self):
        """ConfigLoader: when strategy has sma_50 >= sma_200, validate() returns (False, ...)."""
        from modules.config_loader import ConfigLoader, StrategyConfig, TradingConfig
        path = Path(__file__).parent.parent / "usa_stock_trading_config.yaml"
        if not path.exists():
            path = Path(__file__).parent.parent.parent / "BackEnd" / "usa_stock_trading_config.yaml"
        loader = ConfigLoader(str(path))
        loader.load()
        # Inject invalid strategy (bypass Pydantic for this test)
        loader.config.strategy = StrategyConfig.model_construct(sma_50=100, sma_200=50)
        ok, err = loader.validate()
        assert ok is False
        assert any("sma_50" in e and "sma_200" in e for e in err)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


