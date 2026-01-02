"""
Tests for ML Feature Extractor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_extractor import FeatureExtractor, MLFeatures


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame with indicators"""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(50) * 0.5)
    
    df = pd.DataFrame({
        'Open': close - np.random.rand(50) * 0.5,
        'High': close + np.random.rand(50) * 1.0,
        'Low': close - np.random.rand(50) * 1.0,
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    # Add indicators (normally calculated by strategy)
    df['SMA_short'] = df['Close'].rolling(10).mean()
    df['SMA_long'] = df['Close'].rolling(30).mean()
    
    # MACD
    exp_fast = df['Close'].ewm(span=12).mean()
    exp_slow = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp_fast - exp_slow
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
    
    return df


@pytest.fixture
def feature_extractor():
    """Create feature extractor instance"""
    return FeatureExtractor()


class TestFeatureExtractor:
    """Tests for FeatureExtractor class"""
    
    def test_extract_features(self, feature_extractor, sample_df):
        """Test basic feature extraction"""
        features = feature_extractor.extract(
            df=sample_df,
            symbol="AAPL",
            signal_type="buy",
            base_confidence=0.7
        )
        
        assert isinstance(features, MLFeatures)
        assert features.symbol == "AAPL"
        assert features.signal_type == "buy"
        assert features.base_confidence == 0.7
    
    def test_price_features(self, feature_extractor, sample_df):
        """Test price-related features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert features.price > 0
        assert isinstance(features.price_change_1d, float)
        assert isinstance(features.price_change_5d, float)
        assert isinstance(features.price_change_20d, float)
    
    def test_sma_features(self, feature_extractor, sample_df):
        """Test SMA features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert features.sma_short > 0
        assert features.sma_long > 0
        assert features.sma_ratio > 0
        assert features.sma_trend in [-1, 1]
    
    def test_macd_features(self, feature_extractor, sample_df):
        """Test MACD features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert isinstance(features.macd, float)
        assert isinstance(features.macd_signal, float)
        assert isinstance(features.macd_histogram, float)
        assert features.macd_trend in [-1, 1]
    
    def test_rsi_features(self, feature_extractor, sample_df):
        """Test RSI features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert 0 <= features.rsi <= 100
        assert features.rsi_zone in [-1, 0, 1]
    
    def test_volume_features(self, feature_extractor, sample_df):
        """Test volume features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert features.volume > 0
        assert features.volume_ratio > 0
    
    def test_time_features(self, feature_extractor, sample_df):
        """Test time-related features"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        assert 0 <= features.day_of_week <= 6
        assert 0 <= features.hour_of_day <= 23
    
    def test_to_model_input(self, feature_extractor, sample_df):
        """Test conversion to model input format"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        model_input = features.to_model_input()
        
        assert isinstance(model_input, dict)
        assert len(model_input) == len(FeatureExtractor.FEATURE_NAMES)
        
        # Check all values are numeric
        for key, value in model_input.items():
            assert isinstance(value, (int, float))
    
    def test_features_to_array(self, feature_extractor, sample_df):
        """Test conversion to numpy array"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        array = feature_extractor.features_to_array(features)
        
        assert isinstance(array, np.ndarray)
        assert len(array) == len(FeatureExtractor.FEATURE_NAMES)
    
    def test_batch_to_dataframe(self, feature_extractor, sample_df):
        """Test batch conversion to DataFrame"""
        features_list = [
            feature_extractor.extract(sample_df, f"SYM{i}")
            for i in range(5)
        ]
        
        df = feature_extractor.batch_to_dataframe(features_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'symbol' in df.columns
        assert 'rsi' in df.columns
    
    def test_insufficient_data(self, feature_extractor):
        """Test with insufficient data"""
        small_df = pd.DataFrame({
            'Close': [100],
            'Open': [99],
            'High': [101],
            'Low': [98],
            'Volume': [1000000]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            feature_extractor.extract(small_df, "TEST")
    
    def test_outcome_fields(self, feature_extractor, sample_df):
        """Test outcome fields initialization"""
        features = feature_extractor.extract(sample_df, "TEST")
        
        # Outcomes should be None initially
        assert features.outcome_pnl_percent is None
        assert features.outcome_won is None
        assert features.outcome_hold_days is None
        
        # Can be set later
        features.outcome_pnl_percent = 5.5
        features.outcome_won = True
        features.outcome_hold_days = 3
        
        assert features.outcome_pnl_percent == 5.5
        assert features.outcome_won is True

