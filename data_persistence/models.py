"""
SQLAlchemy Models - Database models for the trading system.
Defines tables for trades, orders, positions, and daily P&L.
"""

from datetime import datetime, date
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, 
    Boolean, Enum as SQLEnum, ForeignKey, Index, Text
)
from sqlalchemy.orm import declarative_base, relationship
import enum


Base = declarative_base()


class OrderStatus(enum.Enum):
    """Order status enumeration"""
    CREATED = "created"
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class TradeSide(enum.Enum):
    """Trade side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"


class Order(Base):
    """Orders table"""
    __tablename__ = 'orders'
    
    id = Column(String(50), primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(SQLEnum(TradeSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False, default=OrderType.MARKET)
    quantity = Column(Integer, nullable=False)
    filled_quantity = Column(Integer, default=0)
    limit_price = Column(Float, nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.CREATED)
    
    # Risk levels
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    is_paper = Column(Boolean, default=True)
    reason = Column(String(255), nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Relationship to trades
    trades = relationship("Trade", back_populates="order")
    
    def __repr__(self):
        return f"<Order {self.id}: {self.side.value} {self.quantity} {self.symbol} @ {self.status.value}>"


class Trade(Base):
    """Trades table - executed trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), ForeignKey('orders.id'), nullable=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(SQLEnum(TradeSide), nullable=False)
    quantity = Column(Integer, nullable=False)
    price_usd = Column(Float, nullable=False)
    price_krw = Column(Float, nullable=False)
    exchange_rate = Column(Float, nullable=False, default=1350.0)
    
    # Calculated fields
    total_usd = Column(Float, nullable=False)
    total_krw = Column(Float, nullable=False)
    
    # Timestamps
    executed_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Metadata
    is_paper = Column(Boolean, default=True)
    slippage = Column(Float, default=0)
    spread = Column(Float, default=0)
    
    # Relationship
    order = relationship("Order", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('idx_trade_symbol_date', 'symbol', 'executed_at'),
    )
    
    def __repr__(self):
        return f"<Trade {self.id}: {self.side.value} {self.quantity} {self.symbol} @ {self.price_usd:.2f}>"


class Position(Base):
    """Positions table - current holdings"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    quantity = Column(Integer, default=0)
    avg_price_usd = Column(Float, nullable=False)
    avg_price_krw = Column(Float, nullable=False)
    
    # Current values (updated regularly)
    current_price_usd = Column(Float, nullable=True)
    unrealized_pnl_usd = Column(Float, default=0)
    unrealized_pnl_krw = Column(Float, default=0)
    
    # Risk levels
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    is_paper = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Position {self.symbol}: {self.quantity} @ {self.avg_price_usd:.2f}>"


class DailyPnL(Base):
    """Daily P&L table - daily trading summary"""
    __tablename__ = 'daily_pnl'
    
    date = Column(Date, primary_key=True, default=date.today)
    
    # P&L
    realized_pnl_usd = Column(Float, default=0)
    realized_pnl_krw = Column(Float, default=0)
    unrealized_pnl_usd = Column(Float, default=0)
    unrealized_pnl_krw = Column(Float, default=0)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    
    # Volume
    buy_volume_usd = Column(Float, default=0)
    sell_volume_usd = Column(Float, default=0)
    
    # Balance snapshots
    starting_balance_krw = Column(Float, nullable=True)
    ending_balance_krw = Column(Float, nullable=True)
    peak_balance_krw = Column(Float, nullable=True)
    
    # Metadata
    is_paper = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<DailyPnL {self.date}: {self.realized_pnl_krw:+,.0f} KRW>"
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        total = self.win_count + self.loss_count
        return (self.win_count / total * 100) if total > 0 else 0


class SystemState(Base):
    """System state table - for recovery"""
    __tablename__ = 'system_state'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemState {self.key}>"


class CircuitBreakerEvent(Base):
    """Circuit breaker events table"""
    __tablename__ = 'circuit_breaker_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)  # triggered, reset, etc.
    reason = Column(String(255), nullable=False)
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # State at trigger
    consecutive_losses = Column(Integer, default=0)
    daily_loss_percent = Column(Float, default=0)
    daily_loss_krw = Column(Float, default=0)
    
    def __repr__(self):
        return f"<CircuitBreakerEvent {self.event_type} at {self.triggered_at}>"


