"""
Feature Extractor - Extracts features for ML model from price data and indicators.
Prepares data in the format required for training and inference.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np


@dataclass
class MLFeatures:
    """
    Complete feature set for ML model.
    Stores all relevant information at signal time.
    """
    # Identifier
    symbol: str
    timestamp: datetime
    
    # Price data
    price: float
    price_change_1d: float  # 1-day price change %
    price_change_5d: float  # 5-day price change %
    price_change_20d: float  # 20-day price change %
    
    # Moving averages (EMA 12/26)
    ema_12: float
    ema_26: float
    ema_ratio: float  # EMA_12/EMA_26 ratio
    price_to_ema_12: float  # price/EMA_12 ratio
    price_to_ema_26: float  # price/EMA_26 ratio
    ema_trend: int  # 1 if EMA_12 > EMA_26, else -1
    
    # MACD
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_hist_change: float  # Change from previous bar
    macd_trend: int  # 1 if MACD > signal, else -1
    
    # RSI
    rsi: float
    rsi_zone: int  # -1 oversold, 0 neutral, 1 overbought
    rsi_change: float  # Change from previous bar
    
    # Volatility
    atr: float
    atr_percent: float  # ATR as % of price
    volatility: float  # 20-day realized volatility
    volatility_change: float  # Change in volatility
    
    # Volume
    volume: float
    volume_sma: float
    volume_ratio: float  # Current volume / average volume
    
    # Time features
    day_of_week: int  # 0-4 (Mon-Fri)
    hour_of_day: int  # 0-23
    days_since_last_signal: int = 0
    
    # Signal context
    signal_type: str = ""  # "buy" or "sell"
    base_confidence: float = 0.0  # Confidence from TA
    
    # For training: outcome (filled after trade closes)
    outcome_pnl_percent: Optional[float] = None
    outcome_won: Optional[bool] = None
    outcome_hold_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_model_input(self) -> Dict[str, float]:
        """
        Convert to model input format (numeric features only).
        Excludes identifiers and outcome fields.
        """
        return {
            'price_change_1d': self.price_change_1d,
            'price_change_5d': self.price_change_5d,
            'price_change_20d': self.price_change_20d,
            'ema_ratio': self.ema_ratio,
            'price_to_ema_12': self.price_to_ema_12,
            'price_to_ema_26': self.price_to_ema_26,
            'ema_trend': self.ema_trend,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'macd_hist_change': self.macd_hist_change,
            'macd_trend': self.macd_trend,
            'rsi': self.rsi,
            'rsi_zone': self.rsi_zone,
            'rsi_change': self.rsi_change,
            'atr_percent': self.atr_percent,
            'volatility': self.volatility,
            'volatility_change': self.volatility_change,
            'volume_ratio': self.volume_ratio,
            'day_of_week': self.day_of_week,
            'hour_of_day': self.hour_of_day,
            'base_confidence': self.base_confidence,
        }


class FeatureExtractor:
    """
    Extracts ML features from price data and technical indicators.
    """
    
    # Feature names in order for model input
    FEATURE_NAMES = [
        'price_change_1d', 'price_change_5d', 'price_change_20d',
        'ema_ratio', 'price_to_ema_12', 'price_to_ema_26', 'ema_trend',
        'macd', 'macd_signal', 'macd_histogram', 'macd_hist_change', 'macd_trend',
        'rsi', 'rsi_zone', 'rsi_change',
        'atr_percent', 'volatility', 'volatility_change',
        'volume_ratio',
        'day_of_week', 'hour_of_day',
        'base_confidence'
    ]
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize feature extractor."""
        self.logger = logger or logging.getLogger(__name__)
    
    def extract(self, df: pd.DataFrame, symbol: str,
                signal_type: str = "", base_confidence: float = 0.0,
                days_since_last_signal: int = 0) -> MLFeatures:
        """
        Extract ML features from indicator-enriched DataFrame.
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            symbol: Stock symbol
            signal_type: 'buy' or 'sell'
            base_confidence: Base confidence from technical analysis
            days_since_last_signal: Days since last trading signal
            
        Returns:
            MLFeatures object with all extracted features
        """
        if len(df) < 2:
            raise ValueError("Need at least 2 data points for feature extraction")
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Get timestamp
        if isinstance(df.index[-1], pd.Timestamp):
            timestamp = df.index[-1].to_pydatetime()
        else:
            timestamp = datetime.now()
        
        # Price features
        price = float(latest['Close'])
        
        # Price changes
        price_change_1d = self._safe_pct_change(df['Close'], 1)
        price_change_5d = self._safe_pct_change(df['Close'], 5)
        price_change_20d = self._safe_pct_change(df['Close'], 20)
        
        # Moving average features (EMA 12/26)
        ema_12 = float(latest.get('EMA_12', price))
        ema_26 = float(latest.get('EMA_26', price))
        ema_ratio = ema_12 / ema_26 if ema_26 != 0 else 1.0
        price_to_ema_12 = price / ema_12 if ema_12 != 0 else 1.0
        price_to_ema_26 = price / ema_26 if ema_26 != 0 else 1.0
        ema_trend = 1 if ema_12 > ema_26 else -1
        
        # MACD features
        macd = float(latest.get('MACD', 0))
        macd_signal = float(latest.get('MACD_signal', 0))
        macd_histogram = float(latest.get('MACD_hist', 0))
        macd_hist_prev = float(prev.get('MACD_hist', 0))
        macd_hist_change = macd_histogram - macd_hist_prev
        macd_trend = 1 if macd > macd_signal else -1
        
        # RSI features
        rsi = float(latest.get('RSI', 50))
        rsi_prev = float(prev.get('RSI', 50))
        rsi_change = rsi - rsi_prev
        rsi_zone = -1 if rsi < 30 else (1 if rsi > 70 else 0)
        
        # Volatility features
        atr = float(latest.get('ATR', 0))
        atr_percent = (atr / price * 100) if price != 0 else 0
        volatility = float(latest.get('Volatility', 0))
        volatility_prev = float(prev.get('Volatility', 0))
        volatility_change = volatility - volatility_prev
        
        # Volume features
        volume = float(latest.get('Volume', 0))
        volume_sma = float(df['Volume'].rolling(window=20).mean().iloc[-1]) if 'Volume' in df else volume
        volume_ratio = volume / volume_sma if volume_sma != 0 else 1.0
        
        # Time features
        if isinstance(timestamp, datetime):
            day_of_week = timestamp.weekday()
            hour_of_day = timestamp.hour
        else:
            day_of_week = 0
            hour_of_day = 12
        
        return MLFeatures(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            price_change_1d=price_change_1d,
            price_change_5d=price_change_5d,
            price_change_20d=price_change_20d,
            ema_12=ema_12,
            ema_26=ema_26,
            ema_ratio=ema_ratio,
            price_to_ema_12=price_to_ema_12,
            price_to_ema_26=price_to_ema_26,
            ema_trend=ema_trend,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            macd_hist_change=macd_hist_change,
            macd_trend=macd_trend,
            rsi=rsi,
            rsi_zone=rsi_zone,
            rsi_change=rsi_change,
            atr=atr,
            atr_percent=atr_percent,
            volatility=volatility,
            volatility_change=volatility_change,
            volume=volume,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            days_since_last_signal=days_since_last_signal,
            signal_type=signal_type,
            base_confidence=base_confidence
        )
    
    def _safe_pct_change(self, series: pd.Series, periods: int) -> float:
        """Calculate percentage change safely"""
        if len(series) <= periods:
            return 0.0
        
        current = series.iloc[-1]
        previous = series.iloc[-periods - 1]
        
        if previous == 0 or pd.isna(previous) or pd.isna(current):
            return 0.0
        
        return ((current - previous) / previous) * 100
    
    def features_to_array(self, features: MLFeatures) -> np.ndarray:
        """
        Convert MLFeatures to numpy array for model input.
        
        Args:
            features: MLFeatures object
            
        Returns:
            1D numpy array of feature values
        """
        model_input = features.to_model_input()
        return np.array([model_input[name] for name in self.FEATURE_NAMES])
    
    def batch_to_dataframe(self, features_list: List[MLFeatures]) -> pd.DataFrame:
        """
        Convert list of MLFeatures to DataFrame for training.
        
        Args:
            features_list: List of MLFeatures objects
            
        Returns:
            DataFrame with features and outcomes
        """
        records = [f.to_dict() for f in features_list]
        return pd.DataFrame(records)

