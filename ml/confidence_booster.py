"""
ML Confidence Booster - Uses LightGBM to enhance trading signal confidence.
Learns from historical trade outcomes to predict signal quality.
"""

import os
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .feature_extractor import MLFeatures, FeatureExtractor


class MLConfidenceBooster:
    """
    ML-based confidence booster for trading signals.
    Uses LightGBM to predict probability of successful trade.
    """
    
    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'early_stopping_rounds': 10,
    }
    
    def __init__(self, config: Dict[str, Any] = None,
                 model_path: str = "ml/models/confidence_model.pkl",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the ML confidence booster.
        
        Args:
            config: ML configuration
            model_path: Path to save/load model
            logger: Optional logger instance
        """
        self.config = config or {}
        self.model_path = Path(model_path)
        self.logger = logger or logging.getLogger(__name__)
        
        self.model: Optional[lgb.Booster] = None
        self.feature_extractor = FeatureExtractor(self.logger)
        self.is_trained = False
        
        # Settings
        self.enabled = self.config.get('enabled', True)
        self.min_training_samples = self.config.get('min_training_samples', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.weight_ml = self.config.get('ml_weight', 0.5)  # Weight for ML vs TA
        
        # Model parameters
        self.params = {**self.DEFAULT_PARAMS, **self.config.get('model_params', {})}
        
        # Try to load existing model
        self._load_model()
        
        if not HAS_LIGHTGBM:
            self.logger.warning("LightGBM not installed. ML boosting disabled.")
            self.enabled = False
    
    def boost_confidence(self, features: MLFeatures) -> Tuple[float, Dict[str, Any]]:
        """
        Boost signal confidence using ML model.
        
        Args:
            features: Extracted ML features
            
        Returns:
            Tuple of (boosted_confidence, metadata)
        """
        base_confidence = features.base_confidence
        
        # If ML not available/trained, return base confidence
        if not self.enabled or not self.is_trained or self.model is None:
            return base_confidence, {
                'ml_enabled': False,
                'reason': 'ML not available' if not self.enabled else 'Model not trained'
            }
        
        try:
            # Convert features to model input
            feature_array = self.feature_extractor.features_to_array(features)
            feature_array = feature_array.reshape(1, -1)
            
            # Get ML prediction (probability of success)
            ml_probability = self.model.predict(feature_array)[0]
            
            # Combine TA confidence with ML probability
            # Weighted average: (1 - weight) * TA + weight * ML
            boosted_confidence = (
                (1 - self.weight_ml) * base_confidence +
                self.weight_ml * ml_probability
            )
            
            # Clamp to [0, 1]
            boosted_confidence = max(0.0, min(1.0, boosted_confidence))
            
            metadata = {
                'ml_enabled': True,
                'base_confidence': base_confidence,
                'ml_probability': ml_probability,
                'boosted_confidence': boosted_confidence,
                'ml_weight': self.weight_ml,
                'confidence_delta': boosted_confidence - base_confidence
            }
            
            self.logger.debug(
                f"ML Boost: {base_confidence:.2f} -> {boosted_confidence:.2f} "
                f"(ML prob: {ml_probability:.2f})"
            )
            
            return boosted_confidence, metadata
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return base_confidence, {
                'ml_enabled': False,
                'reason': f'Prediction error: {str(e)}'
            }
    
    def should_trade(self, boosted_confidence: float) -> Tuple[bool, str]:
        """
        Determine if trade should be executed based on boosted confidence.
        
        Args:
            boosted_confidence: Confidence after ML boosting
            
        Returns:
            Tuple of (should_trade, reason)
        """
        if boosted_confidence >= self.confidence_threshold:
            return True, f"Confidence {boosted_confidence:.2f} >= {self.confidence_threshold}"
        else:
            return False, f"Confidence {boosted_confidence:.2f} < {self.confidence_threshold}"
    
    def train(self, training_data: List[MLFeatures], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML model on historical trade data.
        
        Args:
            training_data: List of MLFeatures with outcome data
            validation_split: Fraction for validation set
            
        Returns:
            Training metrics
        """
        if not HAS_LIGHTGBM or not HAS_SKLEARN:
            return {'error': 'LightGBM or sklearn not installed'}
        
        if len(training_data) < self.min_training_samples:
            return {
                'error': f'Insufficient data: {len(training_data)} < {self.min_training_samples}'
            }
        
        # Filter data with outcomes
        valid_data = [f for f in training_data if f.outcome_won is not None]
        
        if len(valid_data) < self.min_training_samples:
            return {
                'error': f'Insufficient labeled data: {len(valid_data)} < {self.min_training_samples}'
            }
        
        self.logger.info(f"Training ML model with {len(valid_data)} samples...")
        
        try:
            # Prepare features and labels
            X = np.array([
                self.feature_extractor.features_to_array(f) 
                for f in valid_data
            ])
            y = np.array([1 if f.outcome_won else 0 for f in valid_data])
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train, label=y_train,
                feature_name=FeatureExtractor.FEATURE_NAMES
            )
            val_data = lgb.Dataset(
                X_val, label=y_val,
                reference=train_data
            )
            
            # Train model
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
            )
            
            self.is_trained = True
            
            # Evaluate
            y_pred_proba = self.model.predict(X_val)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            metrics = {
                'samples_total': len(valid_data),
                'samples_train': len(X_train),
                'samples_val': len(X_val),
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'win_rate_actual': y_val.mean(),
                'win_rate_predicted': y_pred.mean(),
                'trained_at': datetime.now().isoformat()
            }
            
            # Save model
            self._save_model()
            
            self.logger.info(
                f"Model trained. Accuracy: {metrics['accuracy']:.3f}, "
                f"F1: {metrics['f1']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(FeatureExtractor.FEATURE_NAMES, importance))
    
    def _save_model(self) -> bool:
        """Save model to disk"""
        if self.model is None:
            return False
        
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': FeatureExtractor.FEATURE_NAMES,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def _load_model(self) -> bool:
        """Load model from disk"""
        if not self.model_path.exists():
            self.logger.info("No saved model found")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.params = model_data.get('params', self.params)
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get ML booster status"""
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'has_lightgbm': HAS_LIGHTGBM,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'confidence_threshold': self.confidence_threshold,
            'ml_weight': self.weight_ml,
            'min_training_samples': self.min_training_samples
        }

