"""
Startup Recovery - Recovers system state on startup.
Syncs with broker and reconciles orders/positions.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class RecoveryResult:
    """Result of startup recovery"""
    success: bool
    orders_recovered: int
    positions_synced: int
    discrepancies_found: int
    errors: List[str]
    duration_seconds: float


class StartupRecovery:
    """Recovers system state on startup"""
    
    def __init__(self, order_manager: Any, position_reconciler: Any,
                 database: Any, kis_api: Any,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize startup recovery.
        
        Args:
            order_manager: Order manager instance
            position_reconciler: Position reconciler instance
            database: Database instance
            kis_api: KIS API manager instance
            logger: Optional logger instance
            config: Reconciliation configuration
        """
        self.order_manager = order_manager
        self.position_reconciler = position_reconciler
        self.database = database
        self.kis_api = kis_api
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        self.run_on_startup = self.config.get('run_on_startup', True)
        self.last_recovery: Optional[RecoveryResult] = None
    
    def recover(self) -> RecoveryResult:
        """
        Perform startup recovery.
        
        Returns:
            RecoveryResult with recovery details
        """
        start_time = datetime.now()
        errors = []
        orders_recovered = 0
        positions_synced = 0
        discrepancies_found = 0
        
        self.logger.info("Starting system recovery...")
        
        # Step 1: Recover pending orders from database
        try:
            if self.order_manager:
                orders_recovered = self.order_manager.recover_orders()
                self.logger.info(f"Recovered {orders_recovered} pending orders")
        except Exception as e:
            error_msg = f"Failed to recover orders: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        # Step 2: Check order status with broker
        try:
            self._sync_order_status()
        except Exception as e:
            error_msg = f"Failed to sync order status: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        # Step 3: Reconcile positions with broker
        try:
            if self.position_reconciler:
                discrepancies = self.position_reconciler.reconcile()
                discrepancies_found = len(discrepancies)
                
                if discrepancies_found > 0:
                    self.logger.warning(f"Found {discrepancies_found} position discrepancies")
                else:
                    self.logger.info("Positions match broker state")
        except Exception as e:
            error_msg = f"Failed to reconcile positions: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        # Step 4: Update system state
        try:
            self._update_system_state()
        except Exception as e:
            error_msg = f"Failed to update system state: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = RecoveryResult(
            success=len(errors) == 0,
            orders_recovered=orders_recovered,
            positions_synced=positions_synced,
            discrepancies_found=discrepancies_found,
            errors=errors,
            duration_seconds=duration
        )
        
        self.last_recovery = result
        
        if result.success:
            self.logger.info(f"Recovery completed successfully in {duration:.2f}s")
        else:
            self.logger.warning(f"Recovery completed with {len(errors)} errors in {duration:.2f}s")
        
        return result
    
    def _sync_order_status(self) -> None:
        """Sync pending order status with broker"""
        if not self.order_manager or not self.kis_api:
            return
        
        active_orders = self.order_manager.get_active_orders()
        
        for order in active_orders:
            try:
                broker_status = self.kis_api.get_order_status(order.id)
                
                if broker_status:
                    # Update order status based on broker response
                    # This is a simplified version - actual implementation
                    # would need proper status mapping
                    self.logger.info(f"Order {order.id} status: {broker_status.get('status')}")
            except Exception as e:
                self.logger.warning(f"Failed to get status for order {order.id}: {e}")
    
    def _update_system_state(self) -> None:
        """Update system state after recovery"""
        if not self.database:
            return
        
        try:
            self.database.set_system_state('last_startup', datetime.now().isoformat())
            self.database.set_system_state('recovery_status', 'completed')
        except Exception as e:
            self.logger.warning(f"Failed to update system state: {e}")
    
    def get_last_recovery(self) -> Optional[RecoveryResult]:
        """Get last recovery result"""
        return self.last_recovery
    
    def should_run(self) -> bool:
        """Check if recovery should run on startup"""
        return self.run_on_startup


def perform_startup_recovery(order_manager: Any, position_reconciler: Any,
                             database: Any, kis_api: Any,
                             config: Optional[Dict[str, Any]] = None,
                             logger: Optional[logging.Logger] = None) -> RecoveryResult:
    """
    Convenience function to perform startup recovery.
    
    Args:
        order_manager: Order manager instance
        position_reconciler: Position reconciler instance
        database: Database instance
        kis_api: KIS API manager instance
        config: Reconciliation configuration
        logger: Optional logger instance
        
    Returns:
        RecoveryResult
    """
    recovery = StartupRecovery(
        order_manager=order_manager,
        position_reconciler=position_reconciler,
        database=database,
        kis_api=kis_api,
        logger=logger,
        config=config
    )
    
    return recovery.recover()


