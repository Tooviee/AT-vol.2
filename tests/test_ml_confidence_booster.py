"""
Tests for ML Confidence Booster
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_extractor import MLFeatures, FeatureExtractor
from ml.confidence_booster import MLConfidenceBooster


@pytest.fixture
def sample_features():
    """Create sample MLFeatures"""
    return MLFeatures(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=150.0,
        price_change_1d=0.5,
        price_change_5d=1.2,
        price_change_20d=3.5,
        sma_short=148.0,
        sma_long=145.0,
        sma_ratio=1.02,
        price_to_sma_short=1.01,
        price_to_sma_long=1.03,
        sma_trend=1,
        macd=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        macd_hist_change=0.1,
        macd_trend=1,
        rsi=55.0,
        rsi_zone=0,
        rsi_change=2.0,
        atr=2.5,
        atr_percent=1.67,
        volatility=0.015,
        volatility_change=0.001,
        volume=5000000,
        volume_sma=4500000,
        volume_ratio=1.11,
        day_of_week=2,
        hour_of_day=10,
        signal_type="buy",
        base_confidence=0.65
    )


@pytest.fixture
def booster():
    """Create confidence booster instance"""
    return MLConfidenceBooster(
        config={
            'enabled': True,
            'min_training_samples': 10,
            'confidence_threshold': 0.6,
            'ml_weight': 0.5
        },
        model_path="ml/models/test_model.pkl"
    )


class TestMLConfidenceBooster:
    """Tests for MLConfidenceBooster class"""
    
    def test_initialization(self, booster):
        """Test booster initialization"""
        assert booster.enabled is True
        assert booster.min_training_samples == 10
        assert booster.confidence_threshold == 0.6
        assert booster.weight_ml == 0.5
    
    def test_boost_confidence_untrained(self, booster, sample_features):
        """Test boosting with untrained model"""
        # Model is not trained yet
        assert booster.is_trained is False
        
        boosted, metadata = booster.boost_confidence(sample_features)
        
        # Should return base confidence when not trained
        assert boosted == sample_features.base_confidence
        assert metadata['ml_enabled'] is False
    
    def test_should_trade_above_threshold(self, booster):
        """Test trade decision above threshold"""
        should_trade, reason = booster.should_trade(0.7)
        
        assert should_trade is True
        assert "0.70" in reason
    
    def test_should_trade_below_threshold(self, booster):
        """Test trade decision below threshold"""
        should_trade, reason = booster.should_trade(0.5)
        
        assert should_trade is False
        assert "0.50" in reason
    
    def test_get_status(self, booster):
        """Test status retrieval"""
        status = booster.get_status()
        
        assert 'enabled' in status
        assert 'is_trained' in status
        assert 'confidence_threshold' in status
        assert status['enabled'] is True
        assert status['is_trained'] is False
    
    @pytest.mark.skipif(not MLConfidenceBooster.DEFAULT_PARAMS, 
                        reason="LightGBM not installed")
    def test_train_insufficient_data(self, booster, sample_features):
        """Test training with insufficient data"""
        # Only 5 samples, need 10
        training_data = [sample_features] * 5
        for f in training_data:
            f.outcome_won = True
        
        result = booster.train(training_data)
        
        assert 'error' in result
        assert 'Insufficient' in result['error']
    
    def test_disabled_booster(self, sample_features):
        """Test disabled booster"""
        booster = MLConfidenceBooster(
            config={'enabled': False}
        )
        
        assert booster.enabled is False
        
        boosted, metadata = booster.boost_confidence(sample_features)
        
        assert boosted == sample_features.base_confidence
        assert metadata['ml_enabled'] is False


class TestMLConfidenceBoosterWithMockModel:
    """Tests with mocked ML model"""
    
    @patch('ml.confidence_booster.lgb')
    def test_boost_confidence_with_model(self, mock_lgb, sample_features):
        """Test boosting with a mocked trained model"""
        # Create booster and mock the model
        booster = MLConfidenceBooster(
            config={'ml_weight': 0.5},
            model_path="ml/models/test_model.pkl"
        )
        
        # Mock model to return 0.8 probability
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        
        booster.model = mock_model
        booster.is_trained = True
        
        boosted, metadata = booster.boost_confidence(sample_features)
        
        # Expected: 0.5 * 0.65 (base) + 0.5 * 0.8 (ML) = 0.725
        expected = 0.5 * 0.65 + 0.5 * 0.8
        
        assert abs(boosted - expected) < 0.01
        assert metadata['ml_enabled'] is True
        assert metadata['ml_probability'] == 0.8
    
    @patch('ml.confidence_booster.lgb')
    def test_confidence_weighting(self, mock_lgb, sample_features):
        """Test different ML weights"""
        # 80% weight on ML
        booster = MLConfidenceBooster(
            config={'ml_weight': 0.8}
        )
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        
        booster.model = mock_model
        booster.is_trained = True
        
        boosted, metadata = booster.boost_confidence(sample_features)
        
        # Expected: 0.2 * 0.65 (base) + 0.8 * 0.9 (ML) = 0.85
        expected = 0.2 * 0.65 + 0.8 * 0.9
        
        assert abs(boosted - expected) < 0.01

