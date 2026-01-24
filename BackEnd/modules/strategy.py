"""
Trading Strategy - Hybrid MACD/RSI/ATR strategy (long-only).
Uses SMA 50/200, EMA 12/26, MACD, RSI, and ATR for entries and trailing exits.
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
    """
    Hybrid trading strategy: MACD/RSI/ATR with SMA 50/200 and EMA 12/26.
    Long positions only. Entry: trend filter + EMA crossover + RSI + SMA50 confirmation.
    Exit: trailing ATR stop-loss and risk-reward take-profit.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # SMA (medium- and long-term trend)
        self.sma_50 = config.get('sma_50', 50)
        self.sma_200 = config.get('sma_200', 200)

        # EMA (for MACD line and crossover signal)
        self.ema_fast = config.get('ema_fast', config.get('macd_fast', 12))
        self.ema_slow = config.get('ema_slow', config.get('macd_slow', 26))

        # MACD (Fast: 12, Slow: 26, Signal: 9). MACD line = EMA12 - EMA26.
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)

        # RSI (Period: 14, Overbought: 70, Oversold: 30)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)

        # ATR (Period: 14) for stop-loss / take-profit
        self.atr_period = config.get('atr_period', 14)
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_multiplier = config.get('take_profit_atr_multiplier', 3.0)
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)

        # Min data: SMA 200 is the longest
        self.min_data_points = max(
            self.sma_200,
            self.macd_slow + self.macd_signal,
            self.rsi_period,
            self.atr_period,
        ) + 10

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators using pandas (no 'ta' required).

        Formulas:
        - EMA: exponential weighted mean, span=period, adjust=False
        - MACD: EMA_fast - EMA_slow; Signal = EMA(9) of MACD; Hist = MACD - Signal
        - RSI: 100 - 100/(1 + RS), RS = avg_gain / avg_loss over period
        - ATR: SMA(period) of True Range; TR = max(H-L, |H-C_prev|, |L-C_prev|)
        - SMA: rolling mean of Close
        """
        df = df.copy()

        # --- EMA 12 and EMA 26 (used for MACD and crossover signal) ---
        df['EMA_12'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()

        # --- MACD (Fast: 12, Slow: 26, Signal: 9) ---
        exp_fast = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp_slow = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = exp_fast - exp_slow
        df['MACD_signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # --- RSI (Period: 14, Overbought: 70, Oversold: 30) ---
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.inf)
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- ATR (Period: 14) ---
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR'] = tr.rolling(window=self.atr_period).mean()

        # --- SMA 50 (medium-term trend) and SMA 200 (long-term trend) ---
        df['SMA_50'] = df['Close'].rolling(window=self.sma_50).mean()
        df['SMA_200'] = df['Close'].rolling(window=self.sma_200).mean()

        # SMA 50 trending upward: current > prior
        df['SMA_50_up'] = df['SMA_50'] > df['SMA_50'].shift(1)

        # MACD crosses above 0 (EMA 12 crosses above EMA 26)
        df['MACD_prev'] = df['MACD'].shift(1)
        df['MACD_cross_above_zero'] = (df['MACD_prev'] <= 0) & (df['MACD'] > 0)

        # For ML/backward compatibility: alias short/long to EMA 12/26
        df['SMA_short'] = df['EMA_12']
        df['SMA_long'] = df['EMA_26']

        # Legacy crossover columns used by some consumers
        df['SMA_cross'] = np.where(df['EMA_12'] > df['EMA_26'], 1, -1)
        df['SMA_cross_change'] = df['SMA_cross'].diff()
        df['MACD_cross'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
        df['MACD_cross_change'] = df['MACD_cross'].diff()

        # Volatility (for confidence, optional)
        df['Volatility'] = (
            df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Buy_Signal: 1 where all 4 entry conditions hold (for chart markers, backtest, APIs)
        ok_trend = (df['Close'] > df['SMA_200']).fillna(False)
        ok_cross = df['MACD_cross_above_zero'].fillna(False)
        ok_rsi = (df['RSI'] < self.rsi_overbought).fillna(False)
        ok_confirm = (df['SMA_50_up'] | (df['Close'] > df['SMA_50'])).fillna(False)
        df['Buy_Signal'] = (ok_trend & ok_cross & ok_rsi & ok_confirm).astype(int)

        return df

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """
        ATR-based stop-loss: entry +/- (stop_loss_atr_multiplier * ATR).
        Default 2.0 * ATR.
        """
        if side == 'buy':
            return entry_price - (atr * self.stop_loss_atr_multiplier)
        return entry_price + (atr * self.stop_loss_atr_multiplier)

    def calculate_take_profit(self, entry_price: float, atr: float, side: str) -> float:
        """
        Take-profit from risk-reward vs ATR-based stop:
        reward = (stop_loss_atr_multiplier * ATR) * risk_reward_ratio.
        E.g. 2*ATR stop and 2:1 R:R => 4*ATR target.
        """
        risk_dist = atr * self.stop_loss_atr_multiplier
        reward_dist = risk_dist * self.risk_reward_ratio
        if side == 'buy':
            return entry_price + reward_dist
        return entry_price - reward_dist

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Confidence 0–1 from: price > SMA200, MACD > 0, RSI not overbought, low vol."""
        latest = df.iloc[-1]
        score, max_s = 0.0, 4.0
        if latest['Close'] > latest['SMA_200']:
            score += 1.0
        if latest['MACD'] > 0:
            score += 1.0
        if latest['RSI'] < self.rsi_overbought:
            score += 1.0
        if latest.get('Volatility', 0) < 0.02:
            score += 1.0
        return min(score / max_s, 1.0)

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_position: Optional[Dict] = None,
    ) -> TradeSignal:
        """
        Long-only entry logic:

        1. Trend filter: Price > SMA 200
        2. Signal: EMA 12 crosses above EMA 26 (MACD crosses above 0)
        3. Overbought protection: RSI(14) < 70
        4. Confirmation: SMA 50 trending up OR Price > SMA 50
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
                timestamp=df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
            )

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        price = float(latest['Close'])
        atr = float(latest['ATR'])
        if atr <= 0 or np.isnan(atr):
            atr = price * 0.02

        signal = Signal.HOLD
        reason = "No clear signal"

        # --- BUY (long only when no position) ---
        if current_position is None:
            # 1) Trend: Price > SMA 200
            ok_trend = price > latest['SMA_200']
            # 2) EMA 12 crosses above EMA 26  =>  MACD crosses above 0
            ok_cross = bool(latest.get('MACD_cross_above_zero', False))
            # 3) RSI < 70 (overbought protection)
            ok_rsi = latest['RSI'] < self.rsi_overbought
            # 4) SMA 50 trending up OR Price > SMA 50
            ok_confirm = bool(latest.get('SMA_50_up', False)) or (price > latest['SMA_50'])

            if ok_trend and ok_cross and ok_rsi and ok_confirm:
                signal = Signal.BUY
                reason = "Trend+EMA cross+RSI+SMA50 confirm"
            elif not ok_trend:
                reason = "Price below SMA 200"
            elif not ok_cross:
                reason = "No EMA12/26 bullish cross (MACD not above 0)"
            elif not ok_rsi:
                reason = f"RSI overbought (>= {self.rsi_overbought})"
            elif not ok_confirm:
                reason = "SMA50 not rising and Price <= SMA50"

        # --- SELL: in position, trend breakdown (price < SMA 200) ---
        if current_position is not None and price < latest['SMA_200']:
            signal = Signal.SELL
            reason = "Price below SMA 200 (trend breakdown)"

        # Stop-loss and take-profit
        if signal == Signal.BUY:
            stop_loss = self.calculate_stop_loss(price, atr, 'buy')
            take_profit = self.calculate_take_profit(price, atr, 'buy')
        elif signal == Signal.SELL:
            stop_loss = self.calculate_stop_loss(price, atr, 'sell')
            take_profit = self.calculate_take_profit(price, atr, 'sell')
        else:
            stop_loss = take_profit = 0.0

        confidence = self._calculate_confidence(df)

        return TradeSignal(
            signal=signal,
            symbol=symbol,
            price=price,
            atr=atr,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
            timestamp=df.index[-1],
        )

    def check_exit_conditions(
        self, df: pd.DataFrame, position: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Exit via:
        1. Trailing stop: max(entry - 2*ATR, high_since_entry - 2*ATR). Never lowered.
        2. Take-profit: entry + (2*ATR * risk_reward_ratio) e.g. 1.5:1 or 2:1.
        """
        if len(df) < 2:
            return False, ""

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        price = float(latest['Close'])
        atr = float(latest['ATR'])
        if atr <= 0 or np.isnan(atr):
            atr = price * 0.02

        entry = float(position.get('avg_price', position.get('entry_price', price)))
        side = position.get('side', 'buy')
        entry_time = position.get('entry_time', position.get('entry_date'))

        # Take-profit: 1.5:1 or 2:1 R:R vs ATR-based stop
        risk = atr * self.stop_loss_atr_multiplier
        reward = risk * self.risk_reward_ratio
        if side == 'buy':
            take_profit = entry + reward
        else:
            take_profit = entry - reward

        if side == 'buy' and price >= take_profit:
            return True, "Take-profit (R:R) triggered"
        if side == 'sell' and price <= take_profit:
            return True, "Take-profit (R:R) triggered"

        # Trailing stop: 2 * ATR below highest high since entry
        if side == 'buy':
            if entry_time is not None and hasattr(entry_time, 'tzinfo'):
                try:
                    mask = df.index >= pd.Timestamp(entry_time)
                except Exception:
                    mask = pd.Series(True, index=df.index)
            else:
                # Fallback: use last 60 rows to approximate “since entry”
                mask = df.index >= df.index[-min(60, len(df))]
            high_since = float(df.loc[mask, 'High'].max()) if mask.any() else price
            trailing_stop = max(entry - risk, high_since - risk)
            if price <= trailing_stop:
                return True, "Trailing stop-loss triggered"
        else:
            if entry_time is not None and hasattr(entry_time, 'tzinfo'):
                try:
                    mask = df.index >= pd.Timestamp(entry_time)
                except Exception:
                    mask = pd.Series(True, index=df.index)
            else:
                mask = df.index >= df.index[-min(60, len(df))]
            low_since = float(df.loc[mask, 'Low'].min()) if mask.any() else price
            trailing_stop = min(entry + risk, low_since + risk)
            if price >= trailing_stop:
                return True, "Trailing stop-loss triggered"

        # Fixed stop from position if present (e.g. first bar)
        sl = position.get('stop_loss')
        if sl is not None:
            if side == 'buy' and price <= sl:
                return True, "Stop-loss triggered"
            if side == 'sell' and price >= sl:
                return True, "Stop-loss triggered"

        return False, ""

    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Current indicator snapshot for monitoring."""
        if len(df) < self.min_data_points:
            return {"error": "Insufficient data"}

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]

        return {
            "price": float(latest['Close']),
            "sma_50": float(latest['SMA_50']),
            "sma_200": float(latest['SMA_200']),
            "ema_12": float(latest['EMA_12']),
            "ema_26": float(latest['EMA_26']),
            "sma_short": float(latest['SMA_short']),
            "sma_long": float(latest['SMA_long']),
            "sma_trend": "bullish" if latest['Close'] > latest['SMA_200'] else "bearish",
            "macd": float(latest['MACD']),
            "macd_signal": float(latest['MACD_signal']),
            "macd_histogram": float(latest['MACD_hist']),
            "macd_trend": "bullish" if latest['MACD'] > latest['MACD_signal'] else "bearish",
            "rsi": float(latest['RSI']),
            "rsi_status": (
                "oversold" if latest['RSI'] < self.rsi_oversold
                else "overbought" if latest['RSI'] > self.rsi_overbought
                else "neutral"
            ),
            "atr": float(latest['ATR']),
            "volatility": float(latest.get('Volatility', 0)),
        }
