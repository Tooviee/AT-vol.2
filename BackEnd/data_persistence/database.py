"""
Database Manager - SQLite database with WAL mode for concurrency.
Handles database connection, migrations, and backup.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session

from .models import (
    Base, Order, Trade, Position, DailyPnL, 
    SystemState, CircuitBreakerEvent, OrderStatus, TradeSide, OrderType
)


class Database:
    """SQLite database with WAL mode for concurrency"""
    
    def __init__(self, db_path: str, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize database.
        
        Args:
            db_path: Path to SQLite database file
            config: Database configuration
            logger: Optional logger instance
        """
        self.db_path = db_path
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite settings from config
        busy_timeout = config.get('busy_timeout_ms', 5000)
        
        # Create engine with optimizations
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "timeout": busy_timeout / 1000,
                "check_same_thread": False
            }
        )
        
        # Enable WAL mode and other optimizations
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout}")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        self.logger.info(f"Database initialized with WAL mode: {db_path}")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def ping(self) -> bool:
        """Check database connectivity"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    # === Order Operations ===
    
    def save_order(self, order: Any) -> None:
        """Save or update an order"""
        with self.session_scope() as session:
            db_order = session.query(Order).filter(Order.id == order.id).first()
            
            if db_order:
                # Update existing
                db_order.status = OrderStatus(order.status.value) if hasattr(order.status, 'value') else OrderStatus(order.status)
                db_order.filled_quantity = getattr(order, 'filled_quantity', 0)
                db_order.avg_fill_price = getattr(order, 'avg_fill_price', None)
                db_order.submitted_at = getattr(order, 'submitted_at', None)
                db_order.filled_at = getattr(order, 'filled_at', None)
                db_order.reason = getattr(order, 'reason', None)
                # Update order_type if changed
                order_type_str = getattr(order, 'order_type', None)
                if order_type_str:
                    if isinstance(order_type_str, str):
                        order_type_enum = OrderType.MARKET if order_type_str.lower() == 'market' else OrderType.LIMIT
                        db_order.order_type = order_type_enum
                    else:
                        db_order.order_type = order_type_str
            else:
                # Create new
                # Convert order_type string to enum
                order_type_str = getattr(order, 'order_type', 'market')
                if isinstance(order_type_str, str):
                    # Convert lowercase string to enum
                    order_type_enum = OrderType.MARKET if order_type_str.lower() == 'market' else OrderType.LIMIT
                else:
                    order_type_enum = order_type_str  # Already an enum
                
                db_order = Order(
                    id=order.id,
                    symbol=order.symbol,
                    side=TradeSide(order.side),
                    order_type=order_type_enum,
                    quantity=order.quantity,
                    limit_price=getattr(order, 'limit_price', None),
                    stop_loss=getattr(order, 'stop_loss', None),
                    take_profit=getattr(order, 'take_profit', None),
                    is_paper=getattr(order, 'is_paper', True)
                )
                session.add(db_order)
    
    def update_order(self, order: Any) -> None:
        """Update an order"""
        self.save_order(order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID"""
        with self.session_scope() as session:
            return session.query(Order).filter(Order.id == order_id).first()
    
    def get_orders_by_status(self, statuses: List[str]) -> List[Order]:
        """Get orders by status"""
        with self.session_scope() as session:
            status_enums = [OrderStatus(s) if isinstance(s, str) else s for s in statuses]
            return session.query(Order).filter(Order.status.in_(status_enums)).all()
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending/submitted orders"""
        return self.get_orders_by_status([
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILL
        ])
    
    # === Trade Operations ===
    
    def save_trade(self, order_id: str, symbol: str, side: str,
                   quantity: int, price: float, is_paper: bool = True,
                   exchange_rate: float = 1450.0,
                   slippage: float = 0, spread: float = 0) -> Trade:
        """Save a trade execution"""
        with self.session_scope() as session:
            trade = Trade(
                order_id=order_id,
                symbol=symbol,
                side=TradeSide(side),
                quantity=quantity,
                price_usd=price,
                price_krw=price * exchange_rate,
                exchange_rate=exchange_rate,
                total_usd=quantity * price,
                total_krw=quantity * price * exchange_rate,
                is_paper=is_paper,
                slippage=slippage,
                spread=spread
            )
            session.add(trade)
            session.flush()
            
            # Update daily P&L
            self._update_daily_pnl(session, trade)
            
            return trade
    
    def get_trades_today(self) -> List[Trade]:
        """Get today's trades"""
        with self.session_scope() as session:
            today = date.today()
            return session.query(Trade).filter(
                Trade.executed_at >= datetime.combine(today, datetime.min.time())
            ).all()
    
    def get_trades_by_symbol(self, symbol: str, days: int = 30) -> List[Trade]:
        """Get trades for a symbol"""
        with self.session_scope() as session:
            since = datetime.now() - timedelta(days=days)
            return session.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.executed_at >= since
            ).order_by(Trade.executed_at.desc()).all()
    
    # === Position Operations ===
    
    def save_position(self, symbol: str, quantity: int, avg_price: float,
                      exchange_rate: float = 1450.0, **kwargs) -> None:
        """Save or update a position"""
        with self.session_scope() as session:
            position = session.query(Position).filter(Position.symbol == symbol).first()
            
            if position:
                # Update existing position
                position.quantity = quantity
                position.avg_price_usd = avg_price
                position.avg_price_krw = avg_price * exchange_rate
                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price_usd=avg_price,
                    avg_price_krw=avg_price * exchange_rate,
                    **{k: v for k, v in kwargs.items() if hasattr(Position, k)}
                )
                session.add(position)
            
            # Explicitly flush to ensure it's written
            session.flush()
    
    def update_position(self, symbol: str, quantity: int) -> None:
        """Update position quantity"""
        with self.session_scope() as session:
            position = session.query(Position).filter(Position.symbol == symbol).first()
            if position:
                position.quantity = quantity
                if quantity <= 0:
                    session.delete(position)
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions"""
        with self.session_scope() as session:
            positions = session.query(Position).filter(Position.quantity > 0).all()
            return {
                pos.symbol: {
                    'quantity': pos.quantity,
                    'avg_price_usd': pos.avg_price_usd,
                    'avg_price': pos.avg_price_usd,  # Alias for compatibility
                    'current_price_usd': pos.current_price_usd,
                    'current_price': pos.current_price_usd,  # Alias for compatibility
                    'unrealized_pnl_usd': pos.unrealized_pnl_usd,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for pos in positions
            }
    
    def delete_position(self, symbol: str) -> None:
        """Delete a position"""
        with self.session_scope() as session:
            session.query(Position).filter(Position.symbol == symbol).delete()
    
    # === Daily P&L Operations ===
    
    def _update_daily_pnl(self, session: Session, trade: Trade) -> None:
        """Update daily P&L with a new trade"""
        today = date.today()
        daily = session.query(DailyPnL).filter(DailyPnL.date == today).first()
        
        if not daily:
            daily = DailyPnL(date=today, is_paper=trade.is_paper)
            session.add(daily)
        
        # Handle None values (from old database records)
        if daily.total_trades is None:
            daily.total_trades = 0
        daily.total_trades += 1
        
        if trade.side == TradeSide.BUY:
            if daily.buy_volume_usd is None:
                daily.buy_volume_usd = 0
            daily.buy_volume_usd += trade.total_usd
        else:
            if daily.sell_volume_usd is None:
                daily.sell_volume_usd = 0
            daily.sell_volume_usd += trade.total_usd
    
    def get_daily_pnl(self, target_date: Optional[date] = None) -> Optional[DailyPnL]:
        """Get daily P&L for a date"""
        with self.session_scope() as session:
            target = target_date or date.today()
            daily = session.query(DailyPnL).filter(DailyPnL.date == target).first()
            if daily:
                # Access all properties within session to avoid lazy loading issues
                # Force evaluation of win_rate property
                _ = daily.win_rate
                # Expunge from session so it can be used outside
                session.expunge(daily)
            return daily
    
    def get_realized_pnl_today(self) -> float:
        """Get today's realized P&L in KRW"""
        with self.session_scope() as session:
            today = date.today()
            daily = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            return daily.realized_pnl_krw if daily else 0.0
    
    def update_daily_pnl(self, realized_pnl_krw: float = None,
                         unrealized_pnl_krw: float = None,
                         win: bool = None, **kwargs) -> None:
        """Update daily P&L"""
        with self.session_scope() as session:
            today = date.today()
            daily = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            
            if not daily:
                daily = DailyPnL(date=today)
                session.add(daily)
            
            if realized_pnl_krw is not None:
                daily.realized_pnl_krw += realized_pnl_krw
            if unrealized_pnl_krw is not None:
                daily.unrealized_pnl_krw = unrealized_pnl_krw
            if win is not None:
                if win:
                    daily.win_count += 1
                else:
                    daily.loss_count += 1
            
            for key, value in kwargs.items():
                if hasattr(daily, key):
                    setattr(daily, key, value)
    
    # === Peak Balance ===
    
    def get_peak_balance(self) -> float:
        """Get peak balance from system state"""
        with self.session_scope() as session:
            state = session.query(SystemState).filter(
                SystemState.key == 'peak_balance_krw'
            ).first()
            return float(state.value) if state and state.value else 0.0
    
    def update_peak_balance(self, balance: float) -> None:
        """Update peak balance if higher"""
        current_peak = self.get_peak_balance()
        if balance > current_peak:
            self.set_system_state('peak_balance_krw', str(balance))
    
    # === System State ===
    
    def get_system_state(self, key: str) -> Optional[str]:
        """Get a system state value"""
        with self.session_scope() as session:
            state = session.query(SystemState).filter(SystemState.key == key).first()
            return state.value if state else None
    
    def set_system_state(self, key: str, value: str) -> None:
        """Set a system state value"""
        with self.session_scope() as session:
            state = session.query(SystemState).filter(SystemState.key == key).first()
            if state:
                state.value = value
            else:
                state = SystemState(key=key, value=value)
                session.add(state)
    
    # === Circuit Breaker ===
    
    def log_circuit_breaker_event(self, event_type: str, reason: str,
                                   consecutive_losses: int = 0,
                                   daily_loss_percent: float = 0,
                                   daily_loss_krw: float = 0) -> None:
        """Log a circuit breaker event"""
        with self.session_scope() as session:
            event = CircuitBreakerEvent(
                event_type=event_type,
                reason=reason,
                consecutive_losses=consecutive_losses,
                daily_loss_percent=daily_loss_percent,
                daily_loss_krw=daily_loss_krw
            )
            session.add(event)
    
    # === Backup ===
    
    def backup(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the database"""
        if backup_path is None:
            backup_dir = self.config.get('backup_path', 'data/backups/')
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"trading_{timestamp}.db")
        
        try:
            # Checkpoint WAL before backup
            with self.session_scope() as session:
                session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups()
            
            return True
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Remove backups older than keep_backups days"""
        backup_dir = self.config.get('backup_path', 'data/backups/')
        keep_count = self.config.get('keep_backups', 7)
        
        if not os.path.exists(backup_dir):
            return
        
        backups = sorted(
            [f for f in os.listdir(backup_dir) if f.endswith('.db')],
            reverse=True
        )
        
        for old_backup in backups[keep_count:]:
            try:
                os.remove(os.path.join(backup_dir, old_backup))
                self.logger.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old backup: {e}")
    
    def close(self) -> None:
        """Close database connection"""
        self.Session.remove()
        self.engine.dispose()
        self.logger.info("Database connection closed")


