"""
Trading Strategy - Hybrid strategy using SMA, MACD, RSI, and ATR.
Generates buy/sell signals based on technical indicators.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """Trade signal with metadata"""
    signal: Signal
    symbol: str
    price: float
    atr: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0.0 to 1.0
    reason: str
    timestamp: pd.Timestamp


class USAStrategy:
    """Hybrid trading strategy with SMA, MACD, RSI, and ATR"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # SMA parameters
        self.sma_short = config.get('sma_short', 10)
        self.sma_long = config.get('sma_long', 30)
        
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # ATR parameters for stop-loss/take-profit
        self.atr_period = config.get('atr_period', 14)
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_multiplier = config.get('take_profit_atr_multiplier', 3.0)
        
        # Minimum data points required
        self.min_data_points = max(self.sma_long, self.macd_slow + self.macd_signal, 
                                    self.rsi_period, self.atr_period) + 10
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators including ATR.
        
        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_short'] = df['Close'].rolling(window=self.sma_short).mean()
        df['SMA_long'] = df['Close'].rolling(window=self.sma_long).mean()
        
        # SMA crossover signal
        df['SMA_cross'] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)
        df['SMA_cross_change'] = df['SMA_cross'].diff()
        
        # MACD
        exp_fast = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp_slow = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = exp_fast - exp_slow
        df['MACD_signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # MACD crossover
        df['MACD_cross'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
        df['MACD_cross_change'] = df['MACD_cross'].diff()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.inf)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR'] = tr.rolling(window=self.atr_period).mean()
        
        # Volatility measure
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        return df
    
    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """
        Calculate stop-loss based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: 'buy' or 'sell'
            
        Returns:
            Stop-loss price
        """
        if side == 'buy':
            return entry_price - (atr * self.stop_loss_atr_multiplier)
        else:
            return entry_price + (atr * self.stop_loss_atr_multiplier)
    
    def calculate_take_profit(self, entry_price: float, atr: float, side: str) -> float:
        """
        Calculate take-profit based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: 'buy' or 'sell'
            
        Returns:
            Take-profit price
        """
        if side == 'buy':
            return entry_price + (atr * self.take_profit_atr_multiplier)
        else:
            return entry_price - (atr * self.take_profit_atr_multiplier)
    
    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence score based on indicator agreement.
        
        Returns value between 0.0 and 1.0
        """
        latest = df.iloc[-1]
        score = 0.0
        max_score = 4.0
        
        # SMA trend alignment
        if latest['SMA_short'] > latest['SMA_long']:
            score += 1.0
        
        # MACD positive
        if latest['MACD'] > latest['MACD_signal']:
            score += 1.0
        
        # MACD histogram increasing
        if len(df) >= 2 and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2]:
            score += 0.5
        
        # RSI in favorable range (not overbought)
        if 30 <= latest['RSI'] <= 65:
            score += 1.0
        elif 65 < latest['RSI'] <= 70:
            score += 0.5
        
        # Low volatility bonus
        if latest['Volatility'] < 0.02:
            score += 0.5
        
        return min(score / max_score, 1.0)
    
    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Dict] = None) -> TradeSignal:
        """
        Generate trading signal based on current data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            current_position: Current position info if any
            
        Returns:
            TradeSignal object
        """
        if len(df) < self.min_data_points:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                price=df['Close'].iloc[-1] if len(df) > 0 else 0,
                atr=0,
                stop_loss=0,
                take_profit=0,
                confidence=0,
                reason=f"Insufficient data: {len(df)}/{self.min_data_points} points",
                timestamp=df.index[-1] if len(df) > 0 else pd.Timestamp.now()
            )
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        
        current_price = latest['Close']
        atr = latest['ATR']
        
        # Default to HOLD
        signal = Signal.HOLD
        reason = "No clear signal"
        
        # Check for BUY signals
        buy_conditions = []
        
        # Condition 1: SMA golden cross (short crosses above long)
        if latest['SMA_cross_change'] == 2:  # Changed from -1 to 1
            buy_conditions.append("SMA golden cross")
        
        # Condition 2: MACD bullish crossover
        if latest['MACD_cross_change'] == 2:
            buy_conditions.append("MACD bullish crossover")
        
        # Condition 3: RSI oversold recovery
        if prev['RSI'] < self.rsi_oversold and latest['RSI'] >= self.rsi_oversold:
            buy_conditions.append("RSI oversold recovery")
        
        # Condition 4: Price above both SMAs with positive MACD
        if (latest['Close'] > latest['SMA_short'] > latest['SMA_long'] and 
            latest['MACD'] > 0 and latest['RSI'] < 65):
            buy_conditions.append("Trend aligned bullish")
        
        # Check for SELL signals
        sell_conditions = []
        
        # Condition 1: SMA death cross (short crosses below long)
        if latest['SMA_cross_change'] == -2:  # Changed from 1 to -1
            sell_conditions.append("SMA death cross")
        
        # Condition 2: MACD bearish crossover
        if latest['MACD_cross_change'] == -2:
            sell_conditions.append("MACD bearish crossover")
        
        # Condition 3: RSI overbought reversal
        if prev['RSI'] > self.rsi_overbought and latest['RSI'] <= self.rsi_overbought:
            sell_conditions.append("RSI overbought reversal")
        
        # Condition 4: Price below both SMAs with negative MACD
        if (latest['Close'] < latest['SMA_short'] < latest['SMA_long'] and 
            latest['MACD'] < 0 and latest['RSI'] > 35):
            sell_conditions.append("Trend aligned bearish")
        
        # Determine final signal
        # Need at least 2 conditions for a signal
        if len(buy_conditions) >= 2 and current_position is None:
            signal = Signal.BUY
            reason = " + ".join(buy_conditions)
        elif len(sell_conditions) >= 2 and current_position is not None:
            signal = Signal.SELL
            reason = " + ".join(sell_conditions)
        elif len(buy_conditions) == 1:
            reason = f"Weak buy: {buy_conditions[0]}"
        elif len(sell_conditions) == 1:
            reason = f"Weak sell: {sell_conditions[0]}"
        
        # Calculate stop-loss and take-profit
        if signal == Signal.BUY:
            stop_loss = self.calculate_stop_loss(current_price, atr, 'buy')
            take_profit = self.calculate_take_profit(current_price, atr, 'buy')
        elif signal == Signal.SELL:
            stop_loss = self.calculate_stop_loss(current_price, atr, 'sell')
            take_profit = self.calculate_take_profit(current_price, atr, 'sell')
        else:
            stop_loss = 0
            take_profit = 0
        
        confidence = self._calculate_confidence(df)
        
        return TradeSignal(
            signal=signal,
            symbol=symbol,
            price=current_price,
            atr=atr,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
            timestamp=df.index[-1]
        )
    
    def check_exit_conditions(self, df: pd.DataFrame, position: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            df: Current price data
            position: Current position info
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if len(df) < 2:
            return False, ""
        
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        current_price = latest['Close']
        entry_price = position.get('avg_price', position.get('entry_price', current_price))
        side = position.get('side', 'buy')
        
        # Check stop-loss
        stop_loss = position.get('stop_loss')
        if stop_loss:
            if side == 'buy' and current_price <= stop_loss:
                return True, "Stop-loss triggered"
            elif side == 'sell' and current_price >= stop_loss:
                return True, "Stop-loss triggered"
        
        # Check take-profit
        take_profit = position.get('take_profit')
        if take_profit:
            if side == 'buy' and current_price >= take_profit:
                return True, "Take-profit triggered"
            elif side == 'sell' and current_price <= take_profit:
                return True, "Take-profit triggered"
        
        # Check for reversal signals
        if side == 'buy':
            # Exit long if bearish signals
            if (latest['SMA_cross_change'] == -2 or  # Death cross
                latest['MACD_cross_change'] == -2 or  # MACD bearish
                latest['RSI'] > 80):  # Extremely overbought
                return True, "Bearish reversal signal"
        else:
            # Exit short if bullish signals
            if (latest['SMA_cross_change'] == 2 or  # Golden cross
                latest['MACD_cross_change'] == 2 or  # MACD bullish
                latest['RSI'] < 20):  # Extremely oversold
                return True, "Bullish reversal signal"
        
        return False, ""
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of current indicator values.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            Dictionary with indicator values
        """
        if len(df) < self.min_data_points:
            return {"error": "Insufficient data"}
        
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        return {
            "price": latest['Close'],
            "sma_short": latest['SMA_short'],
            "sma_long": latest['SMA_long'],
            "sma_trend": "bullish" if latest['SMA_short'] > latest['SMA_long'] else "bearish",
            "macd": latest['MACD'],
            "macd_signal": latest['MACD_signal'],
            "macd_histogram": latest['MACD_hist'],
            "macd_trend": "bullish" if latest['MACD'] > latest['MACD_signal'] else "bearish",
            "rsi": latest['RSI'],
            "rsi_status": "oversold" if latest['RSI'] < 30 else "overbought" if latest['RSI'] > 70 else "neutral",
            "atr": latest['ATR'],
            "volatility": latest['Volatility']
        }


