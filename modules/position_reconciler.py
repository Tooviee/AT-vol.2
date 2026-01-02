"""
Position Reconciler - Reconciles local positions with broker state.
Ensures consistency between local tracking and actual broker positions.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Discrepancy:
    """Position discrepancy between local and broker"""
    symbol: str
    local_qty: int
    broker_qty: int
    local_price: float
    broker_price: float
    timestamp: datetime
    
    @property
    def difference(self) -> int:
        return self.broker_qty - self.local_qty
    
    @property
    def is_missing_locally(self) -> bool:
        return self.local_qty == 0 and self.broker_qty > 0
    
    @property
    def is_orphaned(self) -> bool:
        return self.local_qty > 0 and self.broker_qty == 0


class PositionReconciler:
    """Reconciles local positions with broker state"""
    
    def __init__(self, database: Any, kis_api: Any, notifier: Any = None,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize position reconciler.
        
        Args:
            database: Database instance
            kis_api: KIS API manager instance
            notifier: Optional notifier instance
            logger: Optional logger instance
            config: Reconciliation configuration
        """
        self.database = database
        self.kis_api = kis_api
        self.notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        self.auto_sync = self.config.get('auto_sync_to_broker', False)
        self.last_reconciliation: Optional[datetime] = None
        self.discrepancies: List[Discrepancy] = []
    
    def reconcile(self) -> List[Discrepancy]:
        """
        Compare local positions with broker positions.
        
        Returns:
            List of discrepancies found
        """
        self.discrepancies = []
        self.last_reconciliation = datetime.now()
        
        try:
            # Get local positions from database
            local_positions = self.database.get_positions() if self.database else {}
            
            # Get broker positions
            broker_positions = self.kis_api.get_positions() if self.kis_api else {}
            
        except Exception as e:
            self.logger.error(f"Failed to fetch positions for reconciliation: {e}")
            return []
        
        # Find all symbols
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        
        for symbol in all_symbols:
            local = local_positions.get(symbol, {})
            broker = broker_positions.get(symbol, {})
            
            local_qty = local.get('quantity', 0)
            broker_qty = broker.get('quantity', 0)
            local_price = local.get('avg_price', 0)
            broker_price = broker.get('avg_price', 0)
            
            if local_qty != broker_qty:
                discrepancy = Discrepancy(
                    symbol=symbol,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    local_price=local_price,
                    broker_price=broker_price,
                    timestamp=datetime.now()
                )
                self.discrepancies.append(discrepancy)
                
                self.logger.warning(
                    f"Position mismatch: {symbol} - Local: {local_qty}, Broker: {broker_qty}"
                )
        
        # Handle discrepancies
        if self.discrepancies:
            self._handle_discrepancies()
        else:
            self.logger.info("Position reconciliation: All positions match")
        
        return self.discrepancies
    
    def _handle_discrepancies(self) -> None:
        """Handle found discrepancies"""
        # Send notification
        if self.notifier:
            try:
                self._send_reconciliation_alert()
            except Exception as e:
                self.logger.warning(f"Failed to send reconciliation alert: {e}")
        
        # Auto-sync if enabled
        if self.auto_sync:
            self._sync_to_broker()
    
    def _send_reconciliation_alert(self) -> None:
        """Send alert about discrepancies"""
        if not self.notifier or not self.discrepancies:
            return
        
        # Format discrepancies
        details = []
        for d in self.discrepancies[:5]:  # Limit to 5
            details.append(f"{d.symbol}: Local={d.local_qty}, Broker={d.broker_qty}")
        
        try:
            # Use notifier's _send method if available
            if hasattr(self.notifier, '_send'):
                from .notifier import NotificationMessage
                message = NotificationMessage(
                    title="⚠️ Position Reconciliation Alert",
                    description=f"{len(self.discrepancies)} discrepancies found",
                    color=0xFFA500,
                    fields=[{"name": "Discrepancies", "value": "\n".join(details)}],
                    timestamp=datetime.now()
                )
                self.notifier._send(message)
        except Exception as e:
            self.logger.warning(f"Failed to send reconciliation alert: {e}")
    
    def _sync_to_broker(self) -> None:
        """Sync local state to match broker"""
        for d in self.discrepancies:
            try:
                if self.database:
                    self.database.update_position(d.symbol, d.broker_qty)
                    self.logger.info(f"Synced {d.symbol} to broker quantity: {d.broker_qty}")
            except Exception as e:
                self.logger.error(f"Failed to sync {d.symbol}: {e}")
    
    def sync_position(self, symbol: str) -> bool:
        """
        Sync a single position with broker.
        
        Args:
            symbol: Symbol to sync
            
        Returns:
            True if synced successfully
        """
        try:
            broker_positions = self.kis_api.get_positions() if self.kis_api else {}
            broker_pos = broker_positions.get(symbol, {})
            broker_qty = broker_pos.get('quantity', 0)
            
            if self.database:
                self.database.update_position(symbol, broker_qty)
            
            self.logger.info(f"Synced {symbol} to broker quantity: {broker_qty}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync {symbol}: {e}")
            return False
    
    def get_orphaned_positions(self) -> List[Discrepancy]:
        """Get positions tracked locally but not in broker"""
        return [d for d in self.discrepancies if d.is_orphaned]
    
    def get_missing_positions(self) -> List[Discrepancy]:
        """Get positions in broker but not tracked locally"""
        return [d for d in self.discrepancies if d.is_missing_locally]
    
    def get_status(self) -> Dict[str, Any]:
        """Get reconciliation status"""
        return {
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "discrepancy_count": len(self.discrepancies),
            "auto_sync": self.auto_sync,
            "discrepancies": [
                {
                    "symbol": d.symbol,
                    "local_qty": d.local_qty,
                    "broker_qty": d.broker_qty,
                    "difference": d.difference
                }
                for d in self.discrepancies
            ]
        }


