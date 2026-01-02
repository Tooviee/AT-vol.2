"""
ML Trainer - Command-line tool for training the ML confidence model.
Can be run standalone or scheduled periodically.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.training_data_manager import TrainingDataManager
from ml.confidence_booster import MLConfidenceBooster
from ml.feature_extractor import FeatureExtractor


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def train_model(lookback_days: int = 90,
                min_samples: int = 100,
                model_path: str = "ml/models/confidence_model.pkl",
                data_path: str = "ml/training_data",
                logger: logging.Logger = None) -> dict:
    """
    Train the ML confidence model.
    
    Args:
        lookback_days: Number of days of data to use
        min_samples: Minimum samples required
        model_path: Path to save model
        data_path: Path to training data
        logger: Logger instance
        
    Returns:
        Training results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("ML Model Training")
    logger.info("=" * 50)
    
    # Load training data
    data_manager = TrainingDataManager(data_path, logger)
    
    min_date = datetime.now() - timedelta(days=lookback_days)
    training_data = data_manager.load_training_data(min_date=min_date)
    
    logger.info(f"Loaded {len(training_data)} samples from last {lookback_days} days")
    
    # Get training stats
    stats = data_manager.get_training_stats()
    logger.info(f"Win rate: {stats['win_rate']:.1f}%")
    logger.info(f"Avg P&L: {stats['avg_pnl_percent']:.2f}%")
    
    if len(training_data) < min_samples:
        logger.warning(f"Insufficient data: {len(training_data)} < {min_samples}")
        return {
            'success': False,
            'error': f'Need at least {min_samples} samples',
            'samples': len(training_data)
        }
    
    # Train model
    booster = MLConfidenceBooster(
        config={'min_training_samples': min_samples},
        model_path=model_path,
        logger=logger
    )
    
    results = booster.train(training_data)
    
    if 'error' in results:
        logger.error(f"Training failed: {results['error']}")
        return {'success': False, **results}
    
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Precision: {results['precision']:.3f}")
    logger.info(f"Recall: {results['recall']:.3f}")
    logger.info(f"F1 Score: {results['f1']:.3f}")
    logger.info("=" * 50)
    
    # Show feature importance
    importance = booster.get_feature_importance()
    if importance:
        logger.info("\nTop 10 Feature Importance:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_features[:10]:
            logger.info(f"  {name}: {score:.2f}")
    
    return {'success': True, **results}


def analyze_data(data_path: str = "ml/training_data",
                 output_csv: str = None,
                 logger: logging.Logger = None) -> dict:
    """
    Analyze training data.
    
    Args:
        data_path: Path to training data
        output_csv: Optional CSV export path
        logger: Logger instance
        
    Returns:
        Analysis results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    data_manager = TrainingDataManager(data_path, logger)
    
    stats = data_manager.get_training_stats()
    
    logger.info("=" * 50)
    logger.info("Training Data Analysis")
    logger.info("=" * 50)
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Wins: {stats['win_count']}")
    logger.info(f"Losses: {stats['loss_count']}")
    logger.info(f"Win rate: {stats['win_rate']:.1f}%")
    logger.info(f"Avg P&L: {stats['avg_pnl_percent']:.2f}%")
    logger.info(f"Avg Win: {stats['avg_win_pnl']:.2f}%")
    logger.info(f"Avg Loss: {stats['avg_loss_pnl']:.2f}%")
    logger.info(f"Pending trades: {stats['pending_trades']}")
    
    if stats['date_range']['first']:
        logger.info(f"Date range: {stats['date_range']['first']} to {stats['date_range']['last']}")
    
    if output_csv:
        if data_manager.export_to_csv(output_csv):
            logger.info(f"Data exported to {output_csv}")
    
    return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train ML confidence model for trading signals"
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'analyze', 'export'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--lookback-days', '-d',
        type=int,
        default=90,
        help='Days of data to use for training (default: 90)'
    )
    
    parser.add_argument(
        '--min-samples', '-n',
        type=int,
        default=100,
        help='Minimum samples required (default: 100)'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        default='ml/models/confidence_model.pkl',
        help='Path to save/load model'
    )
    
    parser.add_argument(
        '--data-path',
        default='ml/training_data',
        help='Path to training data directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (for export command)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    if args.command == 'train':
        train_model(
            lookback_days=args.lookback_days,
            min_samples=args.min_samples,
            model_path=args.model_path,
            data_path=args.data_path,
            logger=logger
        )
    
    elif args.command == 'analyze':
        analyze_data(
            data_path=args.data_path,
            output_csv=args.output,
            logger=logger
        )
    
    elif args.command == 'export':
        if not args.output:
            args.output = 'ml/training_data/export.csv'
        
        data_manager = TrainingDataManager(args.data_path, logger)
        data_manager.export_to_csv(args.output)


if __name__ == '__main__':
    main()

