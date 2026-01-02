"""
A/B Testing Module - Compares ML-enhanced vs standard signal performance.
Tracks trades in both groups and provides statistical comparison.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class TestGroup(Enum):
    """A/B test groups"""
    ML_ENHANCED = "ml_enhanced"
    CONTROL = "control"


@dataclass
class ABTradeRecord:
    """Record of a trade in A/B test"""
    order_id: str
    symbol: str
    group: TestGroup
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    won: Optional[bool] = None
    
    # Signal data
    base_confidence: float = 0.0
    ml_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['group'] = self.group.value
        if isinstance(data['entry_time'], datetime):
            data['entry_time'] = data['entry_time'].isoformat()
        if isinstance(data['exit_time'], datetime):
            data['exit_time'] = data['exit_time'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABTradeRecord":
        data['group'] = TestGroup(data['group'])
        if isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data['exit_time'] and isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)


@dataclass
class GroupStats:
    """Statistics for a test group"""
    group: str
    total_trades: int = 0
    completed_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_percent: float = 0.0
    avg_pnl_percent: float = 0.0
    win_rate: float = 0.0
    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0
    profit_factor: float = 0.0


@dataclass
class ABTestResults:
    """A/B test comparison results"""
    ml_group: GroupStats
    control_group: GroupStats
    ml_advantage_win_rate: float = 0.0
    ml_advantage_pnl: float = 0.0
    is_ml_better: bool = False
    recommendation: str = ""
    statistical_significance: Optional[float] = None


class ABTestingManager:
    """
    Manages A/B testing for ML vs standard signals.
    Randomly assigns trades to ML or control group and tracks outcomes.
    """
    
    def __init__(self, ml_ratio: float = 0.5,
                 data_path: str = "ml/ab_test_data",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize A/B testing manager.
        
        Args:
            ml_ratio: Ratio of trades assigned to ML group (0.5 = 50%)
            data_path: Directory for A/B test data
            logger: Optional logger instance
        """
        self.ml_ratio = ml_ratio
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Active trades
        self.active_trades: Dict[str, ABTradeRecord] = {}
        
        # Data file
        self.data_file = self.data_path / "ab_test_results.jsonl"
        
        # Load active trades
        self._load_active_trades()
    
    def assign_group(self) -> TestGroup:
        """
        Randomly assign a trade to ML or control group.
        
        Returns:
            TestGroup assignment
        """
        if random.random() < self.ml_ratio:
            return TestGroup.ML_ENHANCED
        return TestGroup.CONTROL
    
    def should_use_ml(self, order_id: str, symbol: str,
                      base_confidence: float, ml_confidence: float) -> Tuple[bool, TestGroup]:
        """
        Decide whether to use ML confidence for this trade.
        
        Args:
            order_id: Unique order identifier
            symbol: Stock symbol
            base_confidence: Base TA confidence
            ml_confidence: ML-boosted confidence
            
        Returns:
            Tuple of (use_ml, assigned_group)
        """
        group = self.assign_group()
        
        # Record the trade
        record = ABTradeRecord(
            order_id=order_id,
            symbol=symbol,
            group=group,
            entry_time=datetime.now(),
            entry_price=0,  # Will be updated when trade executes
            base_confidence=base_confidence,
            ml_confidence=ml_confidence
        )
        
        self.active_trades[order_id] = record
        self._save_active_trades()
        
        use_ml = group == TestGroup.ML_ENHANCED
        
        self.logger.info(
            f"A/B Test: {symbol} assigned to {group.value} "
            f"(use_ml={use_ml})"
        )
        
        return use_ml, group
    
    def update_entry(self, order_id: str, entry_price: float) -> None:
        """Update trade entry price"""
        if order_id in self.active_trades:
            self.active_trades[order_id].entry_price = entry_price
            self._save_active_trades()
    
    def record_outcome(self, order_id: str, exit_price: float) -> Optional[ABTradeRecord]:
        """
        Record trade outcome.
        
        Args:
            order_id: Order identifier
            exit_price: Exit price
            
        Returns:
            Completed trade record or None
        """
        if order_id not in self.active_trades:
            return None
        
        record = self.active_trades.pop(order_id)
        record.exit_time = datetime.now()
        record.exit_price = exit_price
        
        # Calculate P&L
        if record.entry_price > 0:
            record.pnl_percent = ((exit_price - record.entry_price) / record.entry_price) * 100
            record.won = record.pnl_percent > 0
        
        # Save to results file
        self._append_result(record)
        self._save_active_trades()
        
        self.logger.info(
            f"A/B Test outcome: {record.symbol} [{record.group.value}] "
            f"{'WIN' if record.won else 'LOSS'} ({record.pnl_percent:+.2f}%)"
        )
        
        return record
    
    def get_results(self, lookback_days: int = 30) -> ABTestResults:
        """
        Get A/B test comparison results.
        
        Args:
            lookback_days: Days of data to analyze
            
        Returns:
            ABTestResults with comparison
        """
        min_date = datetime.now() - timedelta(days=lookback_days)
        
        # Load all results
        records = self._load_results(min_date)
        
        # Separate by group
        ml_records = [r for r in records if r.group == TestGroup.ML_ENHANCED and r.won is not None]
        control_records = [r for r in records if r.group == TestGroup.CONTROL and r.won is not None]
        
        # Calculate stats for each group
        ml_stats = self._calculate_stats(TestGroup.ML_ENHANCED.value, ml_records)
        control_stats = self._calculate_stats(TestGroup.CONTROL.value, control_records)
        
        # Compare
        ml_advantage_win_rate = ml_stats.win_rate - control_stats.win_rate
        ml_advantage_pnl = ml_stats.avg_pnl_percent - control_stats.avg_pnl_percent
        
        is_ml_better = ml_advantage_pnl > 0 and ml_advantage_win_rate >= 0
        
        # Generate recommendation
        if len(ml_records) < 30 or len(control_records) < 30:
            recommendation = "Need more data (min 30 trades per group)"
        elif ml_advantage_pnl > 1.0 and ml_advantage_win_rate > 5:
            recommendation = "ML shows strong improvement - consider enabling full ML mode"
        elif ml_advantage_pnl > 0.5:
            recommendation = "ML shows moderate improvement - continue testing"
        elif ml_advantage_pnl < -1.0:
            recommendation = "Control group outperforming - review ML model"
        else:
            recommendation = "No significant difference - continue testing"
        
        return ABTestResults(
            ml_group=ml_stats,
            control_group=control_stats,
            ml_advantage_win_rate=ml_advantage_win_rate,
            ml_advantage_pnl=ml_advantage_pnl,
            is_ml_better=is_ml_better,
            recommendation=recommendation
        )
    
    def _calculate_stats(self, group_name: str, records: List[ABTradeRecord]) -> GroupStats:
        """Calculate statistics for a group"""
        stats = GroupStats(group=group_name)
        
        if not records:
            return stats
        
        stats.total_trades = len(records)
        stats.completed_trades = len([r for r in records if r.won is not None])
        
        completed = [r for r in records if r.won is not None]
        if not completed:
            return stats
        
        wins = [r for r in completed if r.won]
        losses = [r for r in completed if not r.won]
        
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.win_rate = (len(wins) / len(completed)) * 100 if completed else 0
        
        stats.total_pnl_percent = sum(r.pnl_percent for r in completed if r.pnl_percent)
        stats.avg_pnl_percent = stats.total_pnl_percent / len(completed) if completed else 0
        
        if wins:
            stats.avg_win_pnl = sum(r.pnl_percent for r in wins if r.pnl_percent) / len(wins)
        if losses:
            stats.avg_loss_pnl = sum(r.pnl_percent for r in losses if r.pnl_percent) / len(losses)
        
        # Profit factor
        total_wins = sum(r.pnl_percent for r in wins if r.pnl_percent) if wins else 0
        total_losses = abs(sum(r.pnl_percent for r in losses if r.pnl_percent)) if losses else 0
        stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return stats
    
    def _append_result(self, record: ABTradeRecord) -> None:
        """Append result to file"""
        try:
            with open(self.data_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save A/B result: {e}")
    
    def _load_results(self, min_date: Optional[datetime] = None) -> List[ABTradeRecord]:
        """Load results from file"""
        if not self.data_file.exists():
            return []
        
        records = []
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    record = ABTradeRecord.from_dict(data)
                    
                    if min_date and record.entry_time < min_date:
                        continue
                    
                    records.append(record)
        except Exception as e:
            self.logger.error(f"Failed to load A/B results: {e}")
        
        return records
    
    def _save_active_trades(self) -> None:
        """Save active trades"""
        active_file = self.data_path / "active_ab_trades.json"
        try:
            data = {k: v.to_dict() for k, v in self.active_trades.items()}
            with open(active_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save active trades: {e}")
    
    def _load_active_trades(self) -> None:
        """Load active trades"""
        active_file = self.data_path / "active_ab_trades.json"
        if not active_file.exists():
            return
        
        try:
            with open(active_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for order_id, record_data in data.items():
                self.active_trades[order_id] = ABTradeRecord.from_dict(record_data)
        except Exception as e:
            self.logger.error(f"Failed to load active trades: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current A/B test summary"""
        results = self.get_results()
        
        return {
            'active_trades': len(self.active_trades),
            'ml_ratio': self.ml_ratio,
            'ml_group': {
                'trades': results.ml_group.completed_trades,
                'win_rate': f"{results.ml_group.win_rate:.1f}%",
                'avg_pnl': f"{results.ml_group.avg_pnl_percent:.2f}%"
            },
            'control_group': {
                'trades': results.control_group.completed_trades,
                'win_rate': f"{results.control_group.win_rate:.1f}%",
                'avg_pnl': f"{results.control_group.avg_pnl_percent:.2f}%"
            },
            'ml_advantage': {
                'win_rate': f"{results.ml_advantage_win_rate:+.1f}%",
                'pnl': f"{results.ml_advantage_pnl:+.2f}%"
            },
            'recommendation': results.recommendation
        }

