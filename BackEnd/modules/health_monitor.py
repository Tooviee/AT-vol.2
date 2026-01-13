"""
Health Monitor - Monitors system health and sends alerts.
Includes heartbeat tracking and subsystem checks.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class SystemHealth:
    """System health information"""
    status: HealthStatus
    last_heartbeat: datetime
    api_connected: bool
    database_connected: bool
    last_trade_time: Optional[datetime]
    issues: List[str]
    uptime_seconds: float


class HealthMonitor:
    """Monitors system health and sends alerts"""
    
    def __init__(self, config: Dict[str, Any],
                 notifier: Any = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize health monitor.
        
        Args:
            config: Health monitor configuration
            notifier: Optional notifier for alerts
            logger: Optional logger instance
        """
        self.config = config
        self.notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        
        self.heartbeat_interval = config.get('heartbeat_interval_seconds', 60)
        self.alert_threshold = config.get('alert_after_missed_heartbeats', 3)
        self.check_interval = config.get('check_interval_seconds', 300)
        
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.last_trade_time: Optional[datetime] = None
        self.missed_heartbeats = 0
        self.last_check_time: Optional[datetime] = None
        self.last_health: Optional[SystemHealth] = None
        
        # Component references
        self.kis_api: Any = None
        self.database: Any = None
    
    def set_components(self, kis_api: Any = None, database: Any = None) -> None:
        """
        Set component references for health checks.
        
        Args:
            kis_api: KIS API manager instance
            database: Database instance
        """
        self.kis_api = kis_api
        self.database = database
    
    def pulse(self) -> None:
        """Update heartbeat timestamp - call this every loop iteration"""
        self.last_heartbeat = datetime.now()
        self.missed_heartbeats = 0
    
    def record_trade(self) -> None:
        """Record that a trade was executed"""
        self.last_trade_time = datetime.now()
    
    def check_heartbeat(self) -> bool:
        """
        Check if heartbeat is healthy.
        
        Returns:
            True if healthy
        """
        since_heartbeat = datetime.now() - self.last_heartbeat
        expected_interval = timedelta(seconds=self.heartbeat_interval * 2)
        
        if since_heartbeat > expected_interval:
            self.missed_heartbeats += 1
            return False
        
        return True
    
    def check_health(self) -> SystemHealth:
        """Comprehensive health check"""
        issues = []
        self.last_check_time = datetime.now()
        
        # Check heartbeat
        since_heartbeat = datetime.now() - self.last_heartbeat
        if since_heartbeat > timedelta(seconds=self.heartbeat_interval * 2):
            issues.append(f"Heartbeat delayed: {since_heartbeat.seconds}s")
        
        # Check API connectivity
        api_ok = True
        if self.kis_api:
            try:
                api_ok = self.kis_api.ping()
            except Exception:
                api_ok = False
            
            if not api_ok:
                issues.append("KIS API not responding")
        
        # Check database
        db_ok = True
        if self.database:
            try:
                db_ok = self.database.ping()
            except Exception:
                db_ok = False
            
            if not db_ok:
                issues.append("Database connection lost")
        
        # Check trade activity
        if self.last_trade_time:
            since_trade = datetime.now() - self.last_trade_time
            if since_trade > timedelta(hours=24):
                issues.append(f"No trades for {since_trade.days}d {since_trade.seconds // 3600}h")
        
        # Determine overall status
        if len(issues) == 0:
            status = HealthStatus.HEALTHY
        elif len(issues) <= 1:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        health = SystemHealth(
            status=status,
            last_heartbeat=self.last_heartbeat,
            api_connected=api_ok,
            database_connected=db_ok,
            last_trade_time=self.last_trade_time,
            issues=issues,
            uptime_seconds=uptime
        )
        
        self.last_health = health
        return health
    
    def alert_if_unhealthy(self) -> None:
        """Send alert if system is unhealthy"""
        health = self.check_health()
        
        if health.status == HealthStatus.UNHEALTHY:
            if self.notifier:
                try:
                    self.notifier.send_health_alert(health)
                except Exception as e:
                    self.logger.error(f"Failed to send health alert: {e}")
            
            self.logger.warning(f"System unhealthy: {health.issues}")
    
    def should_check(self) -> bool:
        """Check if it's time for a health check"""
        if self.last_check_time is None:
            return True
        
        since_check = datetime.now() - self.last_check_time
        return since_check >= timedelta(seconds=self.check_interval)
    
    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self.start_time
    
    def get_uptime_string(self) -> str:
        """Get formatted uptime string"""
        uptime = self.get_uptime()
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def get_status(self) -> Dict[str, Any]:
        """Get health status as dictionary"""
        health = self.last_health or self.check_health()
        
        return {
            "status": health.status.value,
            "last_heartbeat": health.last_heartbeat.isoformat(),
            "api_connected": health.api_connected,
            "database_connected": health.database_connected,
            "last_trade_time": health.last_trade_time.isoformat() if health.last_trade_time else None,
            "issues": health.issues,
            "uptime": self.get_uptime_string(),
            "uptime_seconds": health.uptime_seconds,
            "missed_heartbeats": self.missed_heartbeats
        }


