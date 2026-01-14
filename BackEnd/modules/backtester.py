"""
Backtester - Historical backtesting engine for strategy validation.
Tests strategy performance on historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .strategy import USAStrategy, Signal


@dataclass
class BacktestTrade:
    """Record of a backtest trade"""
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    reason: str


@dataclass
class BacktestResult:
    """Result of a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_percent: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    trades: List[BacktestTrade]
    equity_curve: pd.Series


class Backtester:
    """Historical backtesting engine"""
    
    def __init__(self, strategy: USAStrategy,
                 initial_balance: float = 10000000,  # 10M KRW
                 risk_per_trade_percent: float = 1.0,
                 max_position_size_percent: float = 10.0,
                 exchange_rate: float = 1450.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize backtester.
        
        Args:
            strategy: Trading strategy to test
            initial_balance: Starting balance in KRW
            risk_per_trade_percent: Risk per trade as percentage
            max_position_size_percent: Max position size as percentage
            exchange_rate: USD/KRW exchange rate
            logger: Optional logger instance
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.risk_per_trade_percent = risk_per_trade_percent
        self.max_position_size_percent = max_position_size_percent
        self.exchange_rate = exchange_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_history: List[tuple] = []
    
    def run(self, data: Dict[str, pd.DataFrame], 
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            BacktestResult with performance metrics
        """
        self.logger.info("Starting backtest...")
        
        # Reset state
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_history = []
        
        # Get all unique dates across all symbols
        all_dates = set()
        for symbol, df in data.items():
            all_dates.update(df.index.tolist())
        
        sorted_dates = sorted(all_dates)
        
        # Filter dates
        if start_date:
            sorted_dates = [d for d in sorted_dates if d >= start_date]
        if end_date:
            sorted_dates = [d for d in sorted_dates if d <= end_date]
        
        if not sorted_dates:
            raise ValueError("No data within date range")
        
        self.logger.info(f"Backtesting from {sorted_dates[0]} to {sorted_dates[-1]}")
        self.logger.info(f"Symbols: {list(data.keys())}")
        
        # Iterate through each date
        for current_date in sorted_dates:
            # Update positions with current prices
            self._update_positions(data, current_date)
            
            # Record equity
            total_equity = self._calculate_equity()
            self.equity_history.append((current_date, total_equity))
            
            # Process each symbol
            for symbol, df in data.items():
                if current_date not in df.index:
                    continue
                
                # Get historical data up to current date
                hist_data = df.loc[:current_date]
                
                if len(hist_data) < self.strategy.min_data_points:
                    continue
                
                # Check for exit signals on existing positions
                if symbol in self.positions:
                    self._check_exits(symbol, hist_data, current_date)
                
                # Generate signal
                current_position = self.positions.get(symbol)
                position_dict = None
                if current_position:
                    position_dict = {
                        'symbol': symbol,
                        'quantity': current_position['quantity'],
                        'avg_price': current_position['entry_price'],
                        'stop_loss': current_position.get('stop_loss'),
                        'take_profit': current_position.get('take_profit')
                    }
                
                signal = self.strategy.generate_signal(hist_data, symbol, position_dict)
                
                # Execute signals
                current_price = hist_data['Close'].iloc[-1]
                
                if signal.signal == Signal.BUY and symbol not in self.positions:
                    self._enter_position(symbol, current_price, signal, current_date)
                elif signal.signal == Signal.SELL and symbol in self.positions:
                    self._exit_position(symbol, current_price, current_date, "Signal")
        
        # Close any remaining positions at end
        for symbol in list(self.positions.keys()):
            if symbol in data:
                final_price = data[symbol]['Close'].iloc[-1]
                self._exit_position(symbol, final_price, sorted_dates[-1], "End of backtest")
        
        # Calculate results
        return self._calculate_results(sorted_dates[0], sorted_dates[-1])
    
    def _calculate_equity(self) -> float:
        """Calculate total equity"""
        positions_value = sum(
            pos['quantity'] * pos['current_price'] * self.exchange_rate
            for pos in self.positions.values()
        )
        return self.balance + positions_value
    
    def _update_positions(self, data: Dict[str, pd.DataFrame], 
                          current_date: datetime) -> None:
        """Update positions with current prices"""
        for symbol, pos in self.positions.items():
            if symbol in data and current_date in data[symbol].index:
                pos['current_price'] = data[symbol].loc[current_date, 'Close']
    
    def _enter_position(self, symbol: str, price: float, 
                        signal: Any, date: datetime) -> None:
        """Enter a new position"""
        # Calculate position size
        risk_amount = self.balance * (self.risk_per_trade_percent / 100)
        max_position = self.balance * (self.max_position_size_percent / 100)
        
        risk_per_share = abs(price - signal.stop_loss) if signal.stop_loss else price * 0.02
        
        # Calculate shares
        shares_by_risk = int((risk_amount / self.exchange_rate) / risk_per_share)
        shares_by_max = int((max_position / self.exchange_rate) / price)
        shares = min(shares_by_risk, shares_by_max)
        
        if shares <= 0:
            return
        
        # Check if we have enough cash
        cost = shares * price * self.exchange_rate
        if cost > self.balance:
            shares = int(self.balance / (price * self.exchange_rate))
            cost = shares * price * self.exchange_rate
        
        if shares <= 0:
            return
        
        self.balance -= cost
        self.positions[symbol] = {
            'quantity': shares,
            'entry_price': price,
            'current_price': price,
            'entry_time': date,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }
        
        self.logger.debug(f"BUY {shares} {symbol} @ ${price:.2f}")
    
    def _exit_position(self, symbol: str, price: float, 
                       date: datetime, reason: str) -> None:
        """Exit a position"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        quantity = pos['quantity']
        entry_price = pos['entry_price']
        
        # Calculate P&L
        pnl_usd = (price - entry_price) * quantity
        pnl_krw = pnl_usd * self.exchange_rate
        pnl_percent = ((price - entry_price) / entry_price) * 100
        
        # Update balance
        proceeds = quantity * price * self.exchange_rate
        self.balance += proceeds
        
        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            side='sell',
            quantity=quantity,
            entry_price=entry_price,
            exit_price=price,
            entry_time=pos['entry_time'],
            exit_time=date,
            pnl=pnl_krw,
            pnl_percent=pnl_percent,
            reason=reason
        )
        self.trades.append(trade)
        
        del self.positions[symbol]
        
        self.logger.debug(f"SELL {quantity} {symbol} @ ${price:.2f} | P&L: {pnl_krw:+,.0f} KRW")
    
    def _check_exits(self, symbol: str, hist_data: pd.DataFrame, 
                     date: datetime) -> None:
        """Check for exit conditions"""
        pos = self.positions[symbol]
        current_price = hist_data['Close'].iloc[-1]
        
        # Check stop-loss
        if pos.get('stop_loss') and current_price <= pos['stop_loss']:
            self._exit_position(symbol, current_price, date, "Stop-loss")
            return
        
        # Check take-profit
        if pos.get('take_profit') and current_price >= pos['take_profit']:
            self._exit_position(symbol, current_price, date, "Take-profit")
            return
    
    def _calculate_results(self, start_date: datetime, 
                           end_date: datetime) -> BacktestResult:
        """Calculate backtest results"""
        final_balance = self._calculate_equity()
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_trades = win_count + loss_count
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # P&L statistics
        avg_trade_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        max_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        max_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Equity curve
        equity_series = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())
        
        # Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return_percent=total_return,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            max_drawdown_percent=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            trades=self.trades,
            equity_curve=equity_series
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results"""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Balance: {result.initial_balance:,.0f} KRW")
        print(f"Final Balance: {result.final_balance:,.0f} KRW")
        print(f"Total Return: {result.total_return_percent:+.2f}%")
        print("-" * 50)
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
        print("-" * 50)
        print(f"Avg Trade P&L: {result.avg_trade_pnl:+,.0f} KRW")
        print(f"Avg Win: {result.avg_win:+,.0f} KRW")
        print(f"Avg Loss: {result.avg_loss:+,.0f} KRW")
        print(f"Max Win: {result.max_win:+,.0f} KRW")
        print(f"Max Loss: {result.max_loss:+,.0f} KRW")
        print("=" * 50)


