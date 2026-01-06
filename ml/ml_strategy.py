"""
ML-Enhanced Strategy - Combines technical analysis with ML confidence boosting.
Extends USAStrategy with AI-powered signal enhancement.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.strategy import USAStrategy, TradeSignal, Signal
from .feature_extractor import FeatureExtractor, MLFeatures
from .confidence_booster import MLConfidenceBooster
from .training_data_manager import TrainingDataManager


@dataclass
class EnhancedTradeSignal(TradeSignal):
    """Trade signal enhanced with ML data"""
    ml_confidence: float = 0.0
    ml_enabled: bool = False
    ml_features: Optional[MLFeatures] = None
    ml_metadata: Optional[Dict[str, Any]] = None
    
    @property
    def final_confidence(self) -> float:
        """Get final confidence (ML if available, else base)"""
        return self.ml_confidence if self.ml_enabled else self.confidence


class MLEnhancedStrategy(USAStrategy):
    """
    ML-Enhanced trading strategy.
    Combines technical analysis signals with ML confidence boosting.
    """
    
    def __init__(self, config: Dict[str, Any],
                 ml_config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ML-enhanced strategy.
        
        Args:
            config: Strategy configuration
            ml_config: ML-specific configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        
        self.ml_config = ml_config or {}
        
        # ML components
        self.feature_extractor = FeatureExtractor(self.logger)
        self.confidence_booster = MLConfidenceBooster(
            self.ml_config,
            logger=self.logger
        )
        self.training_data_manager = TrainingDataManager(
            logger=self.logger
        )
        
        # Settings
        self.ml_enabled = self.ml_config.get('enabled', True)
        self.min_confidence = self.ml_config.get('min_confidence', 0.6)
        # Lower threshold for TA-only when ML is not trained (allows trading to collect data)
        self.min_confidence_ta_only = self.ml_config.get('min_confidence_ta_only', 0.4)
        self.use_ab_testing = self.ml_config.get('ab_testing', False)
        self.ab_test_ratio = self.ml_config.get('ab_test_ratio', 0.5)
        
        # Tracking
        self._last_signal_time: Dict[str, pd.Timestamp] = {}
        
        # Log initialization status
        ml_status = "enabled"
        if not self.confidence_booster.is_trained:
            ml_status = "enabled but not trained (using TA-only threshold)"
        elif not self.ml_enabled:
            ml_status = "disabled"
        
        self.logger.info(
            f"ML-Enhanced Strategy initialized. ML: {ml_status}. "
            f"Thresholds: ML={self.min_confidence:.2f}, TA-only={self.min_confidence_ta_only:.2f}"
        )
    
    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Dict] = None) -> EnhancedTradeSignal:
        """
        Generate ML-enhanced trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            current_position: Current position info if any
            
        Returns:
            EnhancedTradeSignal with ML confidence
        """
        # Get base signal from parent class
        base_signal = super().generate_signal(df, symbol, current_position)
        
        # If no actionable signal, return enhanced version without ML
        if base_signal.signal == Signal.HOLD:
            return EnhancedTradeSignal(
                signal=base_signal.signal,
                symbol=base_signal.symbol,
                price=base_signal.price,
                atr=base_signal.atr,
                stop_loss=base_signal.stop_loss,
                take_profit=base_signal.take_profit,
                confidence=base_signal.confidence,
                reason=base_signal.reason,
                timestamp=base_signal.timestamp,
                ml_confidence=base_signal.confidence,
                ml_enabled=False
            )
        
        # Calculate indicators if not already done
        if 'RSI' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Calculate days since last signal for this symbol
        days_since_last = 0
        if symbol in self._last_signal_time:
            days_since_last = (base_signal.timestamp - self._last_signal_time[symbol]).days
        
        # Extract ML features
        try:
            ml_features = self.feature_extractor.extract(
                df=df,
                symbol=symbol,
                signal_type=base_signal.signal.value,
                base_confidence=base_signal.confidence,
                days_since_last_signal=days_since_last
            )
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            ml_features = None
        
        # Boost confidence with ML
        ml_confidence = base_signal.confidence
        ml_metadata = {'ml_enabled': False}
        
        if self.ml_enabled and ml_features and self.confidence_booster.is_trained:
            ml_confidence, ml_metadata = self.confidence_booster.boost_confidence(
                ml_features
            )
        
        # Update last signal time
        self._last_signal_time[symbol] = base_signal.timestamp
        
        # Create enhanced signal
        enhanced_signal = EnhancedTradeSignal(
            signal=base_signal.signal,
            symbol=base_signal.symbol,
            price=base_signal.price,
            atr=base_signal.atr,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit,
            confidence=base_signal.confidence,
            reason=base_signal.reason,
            timestamp=base_signal.timestamp,
            ml_confidence=ml_confidence,
            ml_enabled=ml_metadata.get('ml_enabled', False),
            ml_features=ml_features,
            ml_metadata=ml_metadata
        )
        
        # Log ML enhancement
        if enhanced_signal.ml_enabled:
            delta = ml_confidence - base_signal.confidence
            self.logger.info(
                f"{symbol}: ML adjusted confidence {base_signal.confidence:.2f} -> "
                f"{ml_confidence:.2f} ({delta:+.2f})"
            )
        
        return enhanced_signal
    
    def should_execute(self, signal: EnhancedTradeSignal) -> Tuple[bool, str]:
        """
        Determine if signal should be executed based on ML confidence.
        Uses lower threshold for TA-only when ML is not trained.
        
        Args:
            signal: Enhanced trade signal
            
        Returns:
            Tuple of (should_execute, reason)
        """
        # Non-actionable signals
        if signal.signal == Signal.HOLD:
            return False, "HOLD signal"
        
        # Check confidence threshold
        final_confidence = signal.final_confidence
        
        # Use lower threshold when ML is not trained/available (TA-only mode)
        # This allows trading to collect data for ML training
        if signal.ml_enabled:
            # ML is active - use full threshold
            threshold = self.min_confidence
            threshold_type = "ML"
        else:
            # ML not available - use lower TA-only threshold
            threshold = self.min_confidence_ta_only
            threshold_type = "TA-only"
        
        if final_confidence < threshold:
            return False, f"Confidence {final_confidence:.2f} < {threshold} ({threshold_type})"
        
        # A/B testing mode
        if self.use_ab_testing:
            import random
            if random.random() > self.ab_test_ratio:
                # Use base confidence for this trade (control group)
                if signal.confidence < self.min_confidence:
                    return False, f"A/B control: base confidence {signal.confidence:.2f} < {self.min_confidence}"
        
        return True, f"Confidence {final_confidence:.2f} >= {threshold} ({threshold_type})"
    
    def record_trade_start(self, order_id: str, signal: EnhancedTradeSignal) -> None:
        """
        Record trade start for ML training data.
        
        Args:
            order_id: Unique order identifier
            signal: The signal that triggered the trade
        """
        if signal.ml_features:
            self.training_data_manager.record_signal(order_id, signal.ml_features)
    
    def record_trade_end(self, order_id: str, entry_price: float,
                         exit_price: float, entry_time, exit_time,
                         side: str = 'buy') -> None:
        """
        Record trade outcome for ML training.
        
        Args:
            order_id: Order identifier
            entry_price: Trade entry price
            exit_price: Trade exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            side: 'buy' or 'sell'
        """
        self.training_data_manager.record_outcome(
            order_id=order_id,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            side=side
        )
    
    def train_ml_model(self, min_samples: int = 100,
                       lookback_days: int = 90) -> Dict[str, Any]:
        """
        Train the ML model on collected trade data.
        
        Args:
            min_samples: Minimum required samples
            lookback_days: Days of data to use
            
        Returns:
            Training metrics
        """
        from datetime import datetime, timedelta
        
        min_date = datetime.now() - timedelta(days=lookback_days)
        
        training_data = self.training_data_manager.load_training_data(
            min_date=min_date
        )
        
        if len(training_data) < min_samples:
            return {
                'error': f'Insufficient data: {len(training_data)} < {min_samples}'
            }
        
        return self.confidence_booster.train(training_data)
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get ML system status"""
        return {
            'ml_enabled': self.ml_enabled,
            'min_confidence': self.min_confidence,
            'ab_testing': self.use_ab_testing,
            'booster_status': self.confidence_booster.get_status(),
            'training_stats': self.training_data_manager.get_training_stats()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ML model feature importance"""
        return self.confidence_booster.get_feature_importance()

