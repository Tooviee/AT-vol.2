"""
Database Adapter - Connects Django to existing SQLAlchemy trading database.
This allows Django to read from the same database without conflicts.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta

# Add BackEnd to path to import existing modules
BACKEND_PATH = Path(__file__).resolve().parent.parent.parent / "BackEnd"
sys.path.insert(0, str(BACKEND_PATH))

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, scoped_session

# Import existing models
from data_persistence.models import (
    Order, Trade, Position, DailyPnL, CircuitBreakerEvent,
    OrderStatus, TradeSide, OrderType
)


class TradingDatabaseAdapter:
    """
    Adapter to read from existing trading database.
    Uses SQLAlchemy to access the same database the trading system uses.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database adapter.
        
        Args:
            db_path: Path to trading database. If None, uses default BackEnd/data/trading.db
        """
        if db_path is None:
            # Default to BackEnd/data/trading.db
            db_path = BACKEND_PATH / "data" / "trading.db"
        
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        # Create SQLAlchemy engine (read-only mode recommended)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 5.0
            }
        )
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)
    
    def get_session(self):
        """Get a database session"""
        return self.Session()
    
    def close_session(self):
        """Close the current session"""
        self.Session.remove()
    
    # ===== Trade Operations =====
    
    def get_recent_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent trades"""
        session = self.get_session()
        try:
            query = session.query(Trade).order_by(Trade.executed_at.desc())
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            trades = query.limit(limit).all()
            
            return [{
                'id': trade.id,
                'order_id': trade.order_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price_usd': trade.price_usd,
                'price_krw': trade.price_krw,
                'total_usd': trade.total_usd,
                'total_krw': trade.total_krw,
                'executed_at': trade.executed_at,
                'exchange_rate': trade.exchange_rate,
                'is_paper': trade.is_paper,
                'slippage': trade.slippage,
                'spread': trade.spread,
            } for trade in trades]
        finally:
            session.close()
    
    def get_trades_by_date_range(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get trades within date range"""
        session = self.get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.executed_at >= datetime.combine(start_date, datetime.min.time()),
                Trade.executed_at <= datetime.combine(end_date, datetime.max.time())
            ).order_by(Trade.executed_at.desc()).all()
            
            return [{
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price_usd': trade.price_usd,
                'price_krw': trade.price_krw,
                'total_usd': trade.total_usd,
                'total_krw': trade.total_krw,
                'executed_at': trade.executed_at,
                'is_paper': trade.is_paper,
            } for trade in trades]
        finally:
            session.close()
    
    # ===== Order Operations =====
    
    def get_recent_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent orders"""
        session = self.get_session()
        try:
            orders = session.query(Order).order_by(Order.created_at.desc()).limit(limit).all()
            
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'limit_price': order.limit_price,
                'avg_fill_price': order.avg_fill_price,
                'status': order.status.value,
                'stop_loss': order.stop_loss,
                'take_profit': order.take_profit,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'is_paper': order.is_paper,
                'reason': order.reason,
            } for order in orders]
        finally:
            session.close()
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get active (pending/submitted) orders"""
        session = self.get_session()
        try:
            orders = session.query(Order).filter(
                Order.status.in_([
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                    OrderStatus.PARTIAL_FILL
                ])
            ).order_by(Order.created_at.desc()).all()
            
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'status': order.status.value,
                'created_at': order.created_at,
            } for order in orders]
        finally:
            session.close()
    
    # ===== Position Operations =====
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        session = self.get_session()
        try:
            positions = session.query(Position).filter(Position.quantity > 0).all()
            
            return [{
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'avg_price_usd': pos.avg_price_usd,
                'avg_price_krw': pos.avg_price_krw,
                'current_price_usd': pos.current_price_usd,
                'unrealized_pnl_usd': pos.unrealized_pnl_usd,
                'unrealized_pnl_krw': pos.unrealized_pnl_krw,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'opened_at': pos.opened_at,
                'updated_at': pos.updated_at,
                'is_paper': pos.is_paper,
            } for pos in positions]
        finally:
            session.close()
    
    # ===== P&L Operations =====
    
    def get_daily_pnl(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """Get daily P&L for a specific date (defaults to today)"""
        if target_date is None:
            target_date = date.today()
        
        session = self.get_session()
        try:
            daily_pnl = session.query(DailyPnL).filter(DailyPnL.date == target_date).first()
            
            if not daily_pnl:
                return None
            
            return {
                'date': daily_pnl.date,
                'realized_pnl_usd': daily_pnl.realized_pnl_usd,
                'realized_pnl_krw': daily_pnl.realized_pnl_krw,
                'unrealized_pnl_usd': daily_pnl.unrealized_pnl_usd,
                'unrealized_pnl_krw': daily_pnl.unrealized_pnl_krw,
                'total_trades': daily_pnl.total_trades,
                'win_count': daily_pnl.win_count,
                'loss_count': daily_pnl.loss_count,
                'win_rate': daily_pnl.win_rate,
                'buy_volume_usd': daily_pnl.buy_volume_usd,
                'sell_volume_usd': daily_pnl.sell_volume_usd,
                'starting_balance_krw': daily_pnl.starting_balance_krw,
                'ending_balance_krw': daily_pnl.ending_balance_krw,
                'peak_balance_krw': daily_pnl.peak_balance_krw,
                'is_paper': daily_pnl.is_paper,
            }
        finally:
            session.close()
    
    def get_pnl_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get P&L history for last N days"""
        session = self.get_session()
        try:
            start_date = date.today() - timedelta(days=days)
            daily_pnls = session.query(DailyPnL).filter(
                DailyPnL.date >= start_date
            ).order_by(DailyPnL.date.desc()).all()
            
            return [{
                'date': pnl.date,
                'realized_pnl_krw': pnl.realized_pnl_krw,
                'unrealized_pnl_krw': pnl.unrealized_pnl_krw,
                'total_trades': pnl.total_trades,
                'win_rate': pnl.win_rate,
                'ending_balance_krw': pnl.ending_balance_krw,
            } for pnl in daily_pnls]
        finally:
            session.close()
    
    # ===== Circuit Breaker Operations =====
    
    def get_circuit_breaker_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent circuit breaker events"""
        session = self.get_session()
        try:
            events = session.query(CircuitBreakerEvent).order_by(
                CircuitBreakerEvent.triggered_at.desc()
            ).limit(limit).all()
            
            return [{
                'id': event.id,
                'event_type': event.event_type,
                'reason': event.reason,
                'triggered_at': event.triggered_at,
                'resolved_at': event.resolved_at,
                'consecutive_losses': event.consecutive_losses,
                'daily_loss_percent': event.daily_loss_percent,
                'daily_loss_krw': event.daily_loss_krw,
            } for event in events]
        finally:
            session.close()
    
    # ===== Statistics =====
    
    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for last N days"""
        session = self.get_session()
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Total trades
            total_trades = session.query(Trade).filter(
                Trade.executed_at >= start_date
            ).count()
            
            # Buy vs Sell
            buy_trades = session.query(Trade).filter(
                Trade.executed_at >= start_date,
                Trade.side == TradeSide.BUY
            ).count()
            
            sell_trades = session.query(Trade).filter(
                Trade.executed_at >= start_date,
                Trade.side == TradeSide.SELL
            ).count()
            
            # Total volume
            buy_result = session.query(func.sum(Trade.total_usd)).filter(
                Trade.executed_at >= start_date,
                Trade.side == TradeSide.BUY
            ).scalar()
            buy_volume = buy_result or 0
            
            sell_result = session.query(func.sum(Trade.total_usd)).filter(
                Trade.executed_at >= start_date,
                Trade.side == TradeSide.SELL
            ).scalar()
            sell_volume = sell_result or 0
            
            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'buy_volume_usd': buy_volume,
                'sell_volume_usd': sell_volume,
            }
        except Exception as e:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'buy_volume_usd': 0,
                'sell_volume_usd': 0,
                'error': str(e),
            }
        finally:
            session.close()


# Global instance (singleton pattern)
_db_adapter: Optional[TradingDatabaseAdapter] = None


def get_db_adapter() -> TradingDatabaseAdapter:
    """Get or create database adapter instance"""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = TradingDatabaseAdapter()
    return _db_adapter
