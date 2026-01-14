"""
USA Auto Trader - Main Entry Point
Coordinates all modules and manages the trading loop.
"""

import time
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
from modules.config_loader import load_config, TradingConfig
from modules.logger import init_logging, get_logger
from modules.timezone_utils import get_timezone_manager, is_market_open, get_market_status
from modules.kis_api_manager import KISAPIManager
from modules.balance_tracker import BalanceTracker
from modules.exchange_rate import init_exchange_rate_tracker
from modules.strategy import USAStrategy, Signal
from ml.ml_strategy import MLEnhancedStrategy, EnhancedTradeSignal
from modules.risk_management import RiskManager
from modules.market_hours import MarketHoursManager
from modules.order_manager import OrderManager, Order
from modules.paper_trading import PaperTradingExecutor
from modules.circuit_breaker import CircuitBreaker
from modules.health_monitor import HealthMonitor
from modules.network_monitor import NetworkMonitor
from modules.notifier import Notifier
from modules.shutdown_handler import init_shutdown_handler, should_stop
from modules.data_validator import DataValidator
from modules.rate_limiter import init_rate_limiters
from modules.position_reconciler import PositionReconciler
from modules.startup_recovery import perform_startup_recovery

from data_persistence.database import Database


class USAAutoTrader:
    """Main trading system controller"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading system.
        
        Args:
            config_path: Optional path to config file
        """
        self.config: Optional[TradingConfig] = None
        self.logger: Optional[logging.Logger] = None
        self.is_running = False
        
        # Components (initialized in setup)
        self.database: Optional[Database] = None
        self.kis_api: Optional[KISAPIManager] = None
        self.balance_tracker: Optional[BalanceTracker] = None
        self.strategy: Optional[USAStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.market_hours: Optional[MarketHoursManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.executor: Optional[PaperTradingExecutor] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.network_monitor: Optional[NetworkMonitor] = None
        self.notifier: Optional[Notifier] = None
        self.data_validator: Optional[DataValidator] = None
        self.position_reconciler: Optional[PositionReconciler] = None
        self.exchange_rate_tracker = None
        
        # Trading loop settings
        self.loop_interval = 10  # seconds
        
        # Load configuration
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load and validate configuration"""
        try:
            self.config = load_config(config_path)
            print(f"Configuration loaded: mode={self.config.mode}")
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def setup(self) -> None:
        """Initialize all components"""
        print("Initializing trading system...")
        
        # Initialize logging first
        trading_logger = init_logging(
            self.config.logging.model_dump(),
            log_dir="BackEnd/logs"
        )
        self.logger = get_logger("trading.main")
        self.logger.info("Logger initialized")
        
        # Initialize rate limiters
        init_rate_limiters(self.config.rate_limits.model_dump(), self.logger)
        
        # Initialize exchange rate tracker
        self.exchange_rate_tracker = init_exchange_rate_tracker(
            self.config.exchange_rate.model_dump(),
            self.logger
        )
        
        # Initialize database
        self.database = Database(
            self.config.database.path,
            self.config.database.model_dump(),
            self.logger
        )
        
        # Initialize KIS API
        self.kis_api = KISAPIManager(
            {
                'kis': self.config.kis.model_dump(),
                'mode': self.config.mode
            },
            self.logger
        )
        
        # Initialize balance tracker
        self.balance_tracker = BalanceTracker(
            {
                'mode': self.config.mode,
                'paper_trading': self.config.paper_trading.model_dump(),
                'exchange_rate': self.config.exchange_rate.model_dump()
            },
            self.exchange_rate_tracker,
            self.logger
        )
        
        # Initialize notifier
        self.notifier = Notifier(
            self.config.notifications.model_dump(),
            self.logger
        )
        
        # Initialize executor (paper or live)
        if self.config.mode == 'paper':
            self.executor = PaperTradingExecutor(
                self.config.paper_trading.model_dump(),
                self.balance_tracker,
                self.database,
                self.kis_api,  # For price fetching
                self.logger
            )
        else:
            # Live mode uses KIS API directly
            self.executor = self.kis_api
        
        # Initialize order manager
        self.order_manager = OrderManager(
            self.database,
            self.executor,
            self.notifier,
            self.logger,
            self.config.order_management.model_dump()
        )
        
        # Initialize strategy (ML-enhanced if enabled)
        strategy_config = self.config.strategy.model_dump()
        strategy_config.update({
            'stop_loss_atr_multiplier': self.config.risk.stop_loss_atr_multiplier,
            'take_profit_atr_multiplier': self.config.risk.take_profit_atr_multiplier
        })
        
        ml_config = self.config.ml.model_dump()
        
        if self.config.ml.enabled:
            self.strategy = MLEnhancedStrategy(
                strategy_config,
                ml_config=ml_config,
                logger=self.logger
            )
            self.logger.info("ML-Enhanced Strategy initialized")
        else:
            self.strategy = USAStrategy(strategy_config, self.logger)
            self.logger.info("Standard Strategy initialized (ML disabled)")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            self.config.risk.model_dump(),
            self.balance_tracker,
            self.database,
            self.logger
        )
        
        # Initialize market hours manager
        self.market_hours = MarketHoursManager(
            self.config.market_hours.model_dump(),
            self.logger
        )
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker.model_dump(),
            self.balance_tracker,
            self.database,
            self.notifier,
            self.logger
        )
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(
            self.config.health_monitor.model_dump(),
            self.notifier,
            self.logger
        )
        self.health_monitor.set_components(self.kis_api, self.database)
        
        # Initialize network monitor
        self.network_monitor = NetworkMonitor(
            self.config.network.model_dump(),
            self.logger
        )
        
        # Initialize data validator
        self.data_validator = DataValidator(
            self.config.data_validation.model_dump(),
            self.logger
        )
        
        # Initialize position reconciler
        self.position_reconciler = PositionReconciler(
            self.database,
            self.kis_api,
            self.notifier,
            self.logger,
            self.config.reconciliation.model_dump()
        )
        
        # Initialize shutdown handler
        init_shutdown_handler(self, self.notifier, self.logger)
        
        self.logger.info("All components initialized")
    
    def startup_recovery(self) -> None:
        """Perform startup recovery"""
        if not self.config.reconciliation.run_on_startup:
            self.logger.info("Startup recovery disabled in config")
            return
        
        self.logger.info("Performing startup recovery...")
        
        # Load positions from database FIRST, before recovery
        self._load_positions_from_database()
        
        result = perform_startup_recovery(
            self.order_manager,
            self.position_reconciler,
            self.database,
            self.kis_api,
            self.config.reconciliation.model_dump(),
            self.logger
        )
        
        if result.success:
            self.logger.info(f"Recovery completed: {result.orders_recovered} orders, "
                           f"{result.discrepancies_found} discrepancies")
        else:
            self.logger.warning(f"Recovery had errors: {result.errors}")
    
    def process_symbol(self, symbol: str) -> None:
        """
        Process a single symbol - fetch data, analyze, and trade if signal.
        
        Args:
            symbol: Stock symbol to process
        """
        try:
            # Check circuit breaker
            can_trade, reason = self.circuit_breaker.can_trade()
            if not can_trade:
                self.logger.debug(f"Circuit breaker: {reason}")
                return
            
            # Get historical data for analysis
            hist_data = self.kis_api.get_historical_data(symbol, period='1y', interval='1d')
            
            if hist_data is None or hist_data.empty:
                self.logger.warning(f"No data for {symbol}")
                return
            
            # Validate data
            is_valid, validation_msg = self.data_validator.validate_price_data(hist_data, symbol)
            if not is_valid:
                self.logger.warning(f"Data validation failed for {symbol}: {validation_msg}")
                return
            
            # Check if we have a position
            current_position = self.balance_tracker.get_position(symbol)
            position_dict = None
            if current_position:
                position_dict = {
                    'symbol': symbol,
                    'quantity': current_position.quantity,
                    'avg_price': current_position.avg_price,
                    'stop_loss': current_position.stop_loss,
                    'take_profit': current_position.take_profit
                }
                self.logger.debug(f"{symbol}: Existing position detected ({current_position.quantity} shares @ ${current_position.avg_price:.2f})")
            
            # Generate signal (ML-enhanced if enabled)
            signal = self.strategy.generate_signal(hist_data, symbol, position_dict)
            
            # Log signal with ML info if available
            if hasattr(signal, 'ml_enabled') and signal.ml_enabled:
                self.logger.debug(
                    f"{symbol}: {signal.signal.value} - {signal.reason} "
                    f"[ML: {signal.ml_confidence:.2f}]"
                )
            else:
                self.logger.debug(f"{symbol}: {signal.signal.value} - {signal.reason}")
            
            # Check if signal should be executed (includes ML confidence check)
            should_execute = True
            if hasattr(self.strategy, 'should_execute'):
                should_execute, exec_reason = self.strategy.should_execute(signal)
                if not should_execute:
                    self.logger.debug(f"{symbol}: Signal filtered - {exec_reason}")
            
            # Execute if we have a trading signal and ML approves
            # IMPORTANT: Only buy if we don't already have a position
            if signal.signal == Signal.BUY and current_position is None and should_execute:
                self._execute_buy(symbol, signal)
            elif signal.signal == Signal.SELL and current_position is not None:
                self._execute_sell(symbol, signal, current_position)
            elif current_position is not None:
                # Check exit conditions for existing positions
                should_exit, exit_reason = self.strategy.check_exit_conditions(
                    hist_data, position_dict
                )
                if should_exit:
                    self.logger.info(f"{symbol}: Exit signal - {exit_reason}")
                    self._execute_sell(symbol, signal, current_position)
        
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            self.circuit_breaker.record_api_error()
    
    def _execute_buy(self, symbol: str, signal) -> None:
        """Execute a buy order"""
        try:
            # Calculate position size
            exchange_rate = self.exchange_rate_tracker.get_rate()
            
            position_result = self.risk_manager.calculate_position_size(
                symbol,
                signal.price,
                signal.stop_loss,
                exchange_rate
            )
            
            if not position_result.allowed or position_result.shares <= 0:
                self.logger.info(f"{symbol}: Position not allowed - {position_result.reason}")
                return
            
            # Log ML confidence if available
            if hasattr(signal, 'ml_confidence'):
                self.logger.info(
                    f"{symbol}: ML confidence {signal.ml_confidence:.2f} "
                    f"(base: {signal.confidence:.2f})"
                )
            
            # Create and submit order
            order = self.order_manager.create_order(
                symbol=symbol,
                side='buy',
                quantity=position_result.shares,
                order_type='market',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                is_paper=(self.config.mode == 'paper')
            )
            
            self.order_manager.submit_order(order)
            
            self.logger.info(
                f"BUY {position_result.shares} {symbol} @ ~${signal.price:.2f} "
                f"(SL: ${signal.stop_loss:.2f}, TP: ${signal.take_profit:.2f})"
            )
            
            # Save position to database if order was filled
            # Check order status (OrderState.FILLED enum has value 'filled')
            order_filled = (hasattr(order.status, 'value') and order.status.value == 'filled') or (str(order.status) == "OrderState.FILLED")
            if order_filled and order.avg_fill_price and self.database:
                try:
                    position = self.balance_tracker.get_position(symbol)
                    if position:
                        self.database.save_position(
                            symbol=symbol,
                            quantity=position.quantity,
                            avg_price=position.avg_price,
                            exchange_rate=exchange_rate,
                            stop_loss=position.stop_loss,
                            take_profit=position.take_profit,
                            current_price_usd=position.current_price,
                            is_paper=(self.config.mode == 'paper')
                        )
                    else:
                        self.logger.warning(f"Position {symbol} not found in balance_tracker after order fill")
                except Exception as e:
                    self.logger.error(f"Failed to save position {symbol} to database: {e}", exc_info=True)
            
            # Record trade for ML training (if ML strategy)
            if hasattr(self.strategy, 'record_trade_start') and hasattr(signal, 'ml_features'):
                self.strategy.record_trade_start(order.id, signal)
            
            # Record trade in health monitor
            self.health_monitor.record_trade()
            
        except Exception as e:
            self.logger.error(f"Failed to execute buy for {symbol}: {e}")
    
    def _execute_sell(self, symbol: str, signal, position) -> None:
        """Execute a sell order"""
        try:
            # Create and submit order
            order = self.order_manager.create_order(
                symbol=symbol,
                side='sell',
                quantity=position.quantity,
                order_type='market',
                is_paper=(self.config.mode == 'paper')
            )
            
            self.order_manager.submit_order(order)
            
            # Update/delete position in database if order was filled
            if order.status.value == 'filled' and order.avg_fill_price and self.database:
                try:
                    # Check if position still exists (might be closed)
                    remaining_position = self.balance_tracker.get_position(symbol)
                    if remaining_position and remaining_position.quantity > 0:
                        # Position partially closed, update it
                        exchange_rate = self.exchange_rate_tracker.get_rate()
                        self.database.save_position(
                            symbol=symbol,
                            quantity=remaining_position.quantity,
                            avg_price=remaining_position.avg_price,
                            exchange_rate=exchange_rate,
                            stop_loss=remaining_position.stop_loss,
                            take_profit=remaining_position.take_profit,
                            current_price_usd=remaining_position.current_price,
                            is_paper=(self.config.mode == 'paper')
                        )
                    else:
                        # Position fully closed, delete it
                        self.database.delete_position(symbol)
                except Exception as e:
                    self.logger.warning(f"Failed to update position in database: {e}")
            
            # Calculate P&L
            if order.status.value == 'filled' and order.avg_fill_price:
                pnl = (order.avg_fill_price - position.avg_price) * position.quantity
                self.circuit_breaker.record_trade_result(pnl > 0, pnl)
                
                # Record trade outcome for ML training
                if hasattr(self.strategy, 'record_trade_end') and hasattr(position, 'entry_order_id'):
                    self.strategy.record_trade_end(
                        order_id=position.entry_order_id,
                        entry_price=position.avg_price,
                        exit_price=order.avg_fill_price,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(),
                        side='buy'  # Assuming long positions
                    )
            
            self.logger.info(f"SELL {position.quantity} {symbol}")
            
            self.health_monitor.record_trade()
            
        except Exception as e:
            self.logger.error(f"Failed to execute sell for {symbol}: {e}")
    
    def _load_positions_from_database(self) -> None:
        """Load positions from database into balance_tracker on startup"""
        if not self.database:
            return
        
        try:
            db_positions = self.database.get_positions()
            if not db_positions:
                self.logger.debug("No positions found in database to load")
                return
            
            self.logger.info(f"Loading {len(db_positions)} positions from database...")
            for symbol, pos_data in db_positions.items():
                try:
                    quantity = pos_data.get('quantity', 0)
                    avg_price = pos_data.get('avg_price_usd', 0) or pos_data.get('avg_price', 0)
                    
                    if quantity > 0 and avg_price > 0:
                        # Check if position already exists (avoid duplicates from multiple loads)
                        existing = self.balance_tracker.get_position(symbol)
                        if existing:
                            self.logger.debug(f"Position {symbol} already in balance_tracker ({existing.quantity} shares), skipping duplicate load")
                        else:
                            # Add position to balance_tracker
                            self.balance_tracker.add_position(
                                symbol=symbol,
                                quantity=quantity,
                                price=avg_price,
                                stop_loss=pos_data.get('stop_loss'),
                                take_profit=pos_data.get('take_profit')
                            )
                            self.logger.info(f"Loaded position: {symbol} {quantity} @ ${avg_price:.2f}")
                except Exception as e:
                    self.logger.warning(f"Failed to load position {symbol}: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to load positions from database: {e}")
    
    def _update_position_prices(self) -> None:
        """Update current prices for all positions"""
        all_positions = self.balance_tracker.get_all_positions()
        if not all_positions:
            return
        
        symbols = list(all_positions.keys())
        self.logger.debug(f"Updating prices for {len(symbols)} positions: {', '.join(symbols)}")
        
        # Fetch prices for all positions
        price_updates = {}
        for symbol in symbols:
            try:
                price_data = self.kis_api.get_price(symbol)
                if price_data and 'close' in price_data:
                    price_updates[symbol] = price_data['close']
                else:
                    self.logger.debug(f"No price data for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch price for {symbol}: {e}")
        
        # Update positions with new prices
        if price_updates:
            self.balance_tracker.update_prices(price_updates)
            self.logger.debug(f"Updated prices for {len(price_updates)} positions")
    
    def _sync_positions_to_database(self) -> None:
        """Sync all positions from balance_tracker to database"""
        if not self.database:
            return
        
        try:
            exchange_rate = self.exchange_rate_tracker.get_rate()
            all_positions = self.balance_tracker.get_all_positions()
            
            for symbol, position in all_positions.items():
                try:
                    self.database.save_position(
                        symbol=symbol,
                        quantity=position.quantity,
                        avg_price=position.avg_price,
                        exchange_rate=exchange_rate,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        current_price_usd=position.current_price,
                        is_paper=(self.config.mode == 'paper')
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to sync position {symbol}: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to sync positions to database: {e}")
    
    def run_trading_loop(self) -> None:
        """Main trading loop"""
        self.logger.info(f"Starting trading loop in {self.config.mode} mode")
        self.logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        
        # Send startup notification
        self.notifier.send_startup_alert(self.config.mode)
        
        # Note: Positions are already loaded in startup_recovery(), no need to load again
        # Sync any in-memory positions to database (in case of discrepancies)
        self._sync_positions_to_database()
        
        self.is_running = True
        last_position_sync = time.time()
        last_price_update = time.time()
        position_sync_interval = 60  # Sync every 60 seconds
        price_update_interval = 60  # Update prices every 60 seconds
        
        while self.is_running and not should_stop():
            try:
                # Update heartbeat
                self.health_monitor.pulse()
                
                # Periodically update position prices
                if time.time() - last_price_update >= price_update_interval:
                    self._update_position_prices()
                    last_price_update = time.time()
                
                # Periodically sync positions to database
                if time.time() - last_position_sync >= position_sync_interval:
                    self._sync_positions_to_database()
                    last_position_sync = time.time()
                
                # Check network
                if self.network_monitor.should_pause_trading():
                    self.logger.warning("Network issues detected, waiting...")
                    self.network_monitor.wait_for_connection()
                    continue
                
                # Check market hours
                if not self.market_hours.can_trade_now():
                    window = self.market_hours.get_trading_window()
                    self.logger.info(f"Market closed: {window.message}")
                    
                    # Sleep until market opens or check periodically
                    sleep_time = min(60, window.time_until_open.total_seconds() if window.time_until_open else 60)
                    time.sleep(max(10, sleep_time))
                    continue
                
                # Process each symbol
                for symbol in self.config.symbols:
                    if should_stop():
                        break
                    
                    self.process_symbol(symbol)
                    
                    # Small delay between symbols to respect rate limits
                    time.sleep(0.5)
                
                # Check for order timeouts
                self.order_manager.check_order_timeouts()
                
                # Periodic health check
                if self.health_monitor.should_check():
                    self.health_monitor.alert_if_unhealthy()
                
                # Update peak balance
                self.balance_tracker.update_peak_balance()
                
                # Log status periodically
                self._log_status()
                
                # Wait for next iteration
                time.sleep(self.loop_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                self.circuit_breaker.record_api_error()
                time.sleep(5)  # Brief pause before retry
        
        self.logger.info("Trading loop ended")
    
    def _log_status(self) -> None:
        """Log current status"""
        try:
            summary = self.balance_tracker.get_summary()
            self.logger.info(
                f"Balance: {summary['total_balance_krw']:,.0f} KRW | "
                f"Positions: {summary['position_count']} | "
                f"P&L: {summary['unrealized_pnl_krw']:+,.0f} KRW"
            )
        except Exception:
            pass
    
    def save_state(self) -> None:
        """Save current state (called on shutdown)"""
        self.logger.info("Saving state...")
        
        try:
            if self.database:
                # Backup database
                self.database.backup()
                
                # Save system state
                self.database.set_system_state('last_shutdown', datetime.now().isoformat())
                self.database.set_system_state('shutdown_clean', 'true')
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def close_connections(self) -> None:
        """Close all connections (called on shutdown)"""
        self.logger.info("Closing connections...")
        
        try:
            if self.database:
                self.database.close()
            
            if self.kis_api:
                self.kis_api.close()
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    def run(self) -> None:
        """Main entry point"""
        try:
            # Setup
            self.setup()
            
            # Startup recovery
            self.startup_recovery()
            
            # Run trading loop
            self.run_trading_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.save_state()
            self.close_connections()


def main():
    """Main entry point"""
    print("=" * 50)
    print("USA Auto Trader v1.0")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("Warning: Python 3.11+ recommended for best performance")
    
    # Create and run trader
    trader = USAAutoTrader()
    trader.run()


if __name__ == "__main__":
    main()


