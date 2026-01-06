"""
Training Data Manager - Manages ML training data collection and storage.
Stores features at signal time and updates with trade outcomes.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from .feature_extractor import MLFeatures, FeatureExtractor


class TrainingDataManager:
    """
    Manages ML training data collection and storage.
    Stores signal features and updates them with trade outcomes.
    """
    
    def __init__(self, data_path: str = "ml/training_data",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize training data manager.
        
        Args:
            data_path: Directory for training data files
            logger: Optional logger instance
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory storage for pending trades (waiting for outcome)
        self.pending_trades: Dict[str, MLFeatures] = {}  # order_id -> features
        
        # Training data file
        self.data_file = self.data_path / "training_data.jsonl"
        
        # Load any pending trades from previous session
        self._load_pending_trades()
    
    def record_signal(self, order_id: str, features: MLFeatures) -> None:
        """
        Record features at signal time.
        
        Args:
            order_id: Unique order identifier
            features: Extracted ML features
        """
        self.pending_trades[order_id] = features
        self._save_pending_trades()
        
        self.logger.debug(f"Recorded signal features for order {order_id}")
    
    def record_outcome(self, order_id: str, 
                       entry_price: float, exit_price: float,
                       entry_time: datetime, exit_time: datetime,
                       side: str = 'buy') -> Optional[MLFeatures]:
        """
        Record trade outcome and save to training data.
        
        Args:
            order_id: Order identifier
            entry_price: Trade entry price
            exit_price: Trade exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            side: 'buy' or 'sell'
            
        Returns:
            Updated MLFeatures or None if order not found
        """
        if order_id not in self.pending_trades:
            self.logger.warning(f"No pending features for order {order_id}")
            return None
        
        features = self.pending_trades.pop(order_id)
        
        # Calculate outcome
        if side == 'buy':
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
        
        won = pnl_percent > 0
        hold_days = (exit_time - entry_time).days
        
        # Update features with outcome
        features.outcome_pnl_percent = pnl_percent
        features.outcome_won = won
        features.outcome_hold_days = hold_days
        
        # Save to training data file
        self._append_training_data(features)
        self._save_pending_trades()
        
        self.logger.info(
            f"Recorded outcome for {order_id}: "
            f"{'WIN' if won else 'LOSS'} ({pnl_percent:+.2f}%)"
        )
        
        return features
    
    def _append_training_data(self, features: MLFeatures) -> None:
        """Append features to training data file"""
        try:
            data = features.to_dict()
            
            # Convert datetime to string
            if isinstance(data['timestamp'], datetime):
                data['timestamp'] = data['timestamp'].isoformat()
            
            with open(self.data_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
    
    def load_training_data(self, min_date: Optional[datetime] = None,
                           max_date: Optional[datetime] = None) -> List[MLFeatures]:
        """
        Load training data from file.
        
        Args:
            min_date: Optional minimum date filter
            max_date: Optional maximum date filter
            
        Returns:
            List of MLFeatures with outcomes
        """
        if not self.data_file.exists():
            return []
        
        features_list = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    # Parse timestamp
                    if isinstance(data['timestamp'], str):
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    # Apply date filters
                    if min_date and data['timestamp'] < min_date:
                        continue
                    if max_date and data['timestamp'] > max_date:
                        continue
                    
                    # Create MLFeatures object
                    features = MLFeatures(**data)
                    
                    # Only include data with outcomes
                    if features.outcome_won is not None:
                        features_list.append(features)
            
            self.logger.info(f"Loaded {len(features_list)} training samples")
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
        
        return features_list
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about training data"""
        data = self.load_training_data()
        
        if not data:
            return {
                'total_samples': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'avg_pnl_percent': 0,
                'avg_win_pnl': 0,
                'avg_loss_pnl': 0,
                'pending_trades': len(self.pending_trades),
                'date_range': {
                    'first': None,
                    'last': None
                }
            }
        
        wins = [f for f in data if f.outcome_won]
        losses = [f for f in data if not f.outcome_won]
        
        return {
            'total_samples': len(data),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / len(data) * 100 if data else 0,
            'avg_pnl_percent': sum(f.outcome_pnl_percent or 0 for f in data) / len(data),
            'avg_win_pnl': sum(f.outcome_pnl_percent or 0 for f in wins) / len(wins) if wins else 0,
            'avg_loss_pnl': sum(f.outcome_pnl_percent or 0 for f in losses) / len(losses) if losses else 0,
            'pending_trades': len(self.pending_trades),
            'date_range': {
                'first': min(f.timestamp for f in data).isoformat() if data else None,
                'last': max(f.timestamp for f in data).isoformat() if data else None
            }
        }
    
    def _save_pending_trades(self) -> None:
        """Save pending trades to file"""
        pending_file = self.data_path / "pending_trades.json"
        
        try:
            pending_data = {}
            for order_id, features in self.pending_trades.items():
                data = features.to_dict()
                if isinstance(data['timestamp'], datetime):
                    data['timestamp'] = data['timestamp'].isoformat()
                pending_data[order_id] = data
            
            with open(pending_file, 'w', encoding='utf-8') as f:
                json.dump(pending_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save pending trades: {e}")
    
    def _load_pending_trades(self) -> None:
        """Load pending trades from file"""
        pending_file = self.data_path / "pending_trades.json"
        
        if not pending_file.exists():
            return
        
        try:
            with open(pending_file, 'r', encoding='utf-8') as f:
                pending_data = json.load(f)
            
            for order_id, data in pending_data.items():
                if isinstance(data['timestamp'], str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                self.pending_trades[order_id] = MLFeatures(**data)
            
            self.logger.info(f"Loaded {len(self.pending_trades)} pending trades")
            
        except Exception as e:
            self.logger.error(f"Failed to load pending trades: {e}")
    
    def clear_old_pending(self, max_age_days: int = 30) -> int:
        """
        Clear pending trades older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of cleared trades
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        old_orders = [
            order_id for order_id, features in self.pending_trades.items()
            if features.timestamp < cutoff
        ]
        
        for order_id in old_orders:
            del self.pending_trades[order_id]
        
        if old_orders:
            self._save_pending_trades()
            self.logger.info(f"Cleared {len(old_orders)} old pending trades")
        
        return len(old_orders)
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export training data to CSV for analysis.
        
        Args:
            output_path: Output CSV file path
            
        Returns:
            True if successful
        """
        data = self.load_training_data()
        
        if not data:
            self.logger.warning("No training data to export")
            return False
        
        try:
            df = pd.DataFrame([f.to_dict() for f in data])
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported {len(df)} samples to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export: {e}")
            return False

