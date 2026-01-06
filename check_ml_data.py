"""
Quick script to check ML training data status.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ml.training_data_manager import TrainingDataManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Check ML training data status"""
    print("=" * 60)
    print("ML Training Data Status Check")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = TrainingDataManager(logger=logger)
    
    # Get statistics
    stats = data_manager.get_training_stats()
    
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"Wins: {stats['win_count']}")
    print(f"Losses: {stats['loss_count']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Average P&L: {stats['avg_pnl_percent']:.2f}%")
    print(f"Average Win: {stats['avg_win_pnl']:.2f}%")
    print(f"Average Loss: {stats['avg_loss_pnl']:.2f}%")
    print(f"Pending Trades (waiting for outcome): {stats['pending_trades']}")
    
    if stats.get('date_range', {}).get('first'):
        print(f"\nDate Range:")
        print(f"  First: {stats['date_range']['first']}")
        print(f"  Last: {stats['date_range']['last']}")
    
    # Check minimum requirement
    min_samples = 100
    print(f"\n{'=' * 60}")
    print(f"Minimum Required: {min_samples} samples")
    
    if stats['total_samples'] >= min_samples:
        print(f"[OK] SUFFICIENT DATA: {stats['total_samples']} >= {min_samples}")
    else:
        print(f"[X] INSUFFICIENT DATA: {stats['total_samples']} < {min_samples}")
        print(f"  Need {min_samples - stats['total_samples']} more samples")
    
    # Check file existence
    data_file = data_manager.data_file
    print(f"\n{'=' * 60}")
    print(f"Data File: {data_file}")
    if data_file.exists():
        file_size = data_file.stat().st_size
        print(f"  Exists: Yes ({file_size:,} bytes)")
        
        # Count lines manually
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f if line.strip())
            print(f"  Lines in file: {line_count}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print(f"  Exists: No")
    
    pending_file = data_manager.data_path / "pending_trades.json"
    print(f"\nPending Trades File: {pending_file}")
    if pending_file.exists():
        file_size = pending_file.stat().st_size
        print(f"  Exists: Yes ({file_size:,} bytes)")
    else:
        print(f"  Exists: No")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

