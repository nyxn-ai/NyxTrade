"""
Alert management system for monitoring agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    agent_name: str
    alert_type: str
    severity: AlertSeverity
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


class AlertManager:
    """
    Manages alerts from monitoring agents
    Handles alert routing, deduplication, and notifications
    """
    
    def __init__(self):
        self.logger = logging.getLogger("alert_manager")
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Configuration
        self.max_history_size = 10000
        self.deduplication_window = 300  # 5 minutes
        
        self.logger.info("Initialized AlertManager")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__name__}")
    
    async def send_alerts(self, alerts_data: List[Dict[str, Any]]):
        """Process and send alerts"""
        for alert_data in alerts_data:
            try:
                alert = self._create_alert(alert_data)
                
                # Check for deduplication
                if self._should_deduplicate(alert):
                    self.logger.debug(f"Deduplicated alert: {alert.id}")
                    continue
                
                # Store alert
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                
                # Trim history if needed
                if len(self.alert_history) > self.max_history_size:
                    self.alert_history = self.alert_history[-self.max_history_size:]
                
                # Send to handlers
                await self._notify_handlers(alert)
                
                self.logger.info(f"Sent alert: {alert.id} - {alert.message}")
                
            except Exception as e:
                self.logger.error(f"Failed to process alert: {e}")
    
    def _create_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Create Alert object from data"""
        import uuid
        
        # Determine severity
        severity = self._determine_severity(alert_data)
        
        # Generate alert ID
        alert_id = str(uuid.uuid4())
        
        return Alert(
            id=alert_id,
            agent_name=alert_data.get("agent", "unknown"),
            alert_type=alert_data.get("type", "general"),
            severity=severity,
            message=self._format_alert_message(alert_data),
            data=alert_data,
            timestamp=alert_data.get("timestamp", datetime.now())
        )
    
    def _determine_severity(self, alert_data: Dict[str, Any]) -> AlertSeverity:
        """Determine alert severity based on data"""
        alert_type = alert_data.get("type", "")
        
        # Critical alerts
        if "critical" in alert_type.lower():
            return AlertSeverity.CRITICAL
        
        # High severity alerts
        if any(keyword in alert_type.lower() for keyword in ["error", "failure", "crash"]):
            return AlertSeverity.HIGH
        
        # Medium severity alerts
        if any(keyword in alert_type.lower() for keyword in ["threshold", "warning", "anomaly"]):
            return AlertSeverity.MEDIUM
        
        # Default to low
        return AlertSeverity.LOW
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert message"""
        alert_type = alert_data.get("type", "Alert")
        agent = alert_data.get("agent", "Unknown Agent")
        
        if alert_type == "threshold_exceeded":
            metric = alert_data.get("metric", "unknown")
            value = alert_data.get("value", "N/A")
            threshold = alert_data.get("threshold", "N/A")
            return f"{agent}: {metric} threshold exceeded - Value: {value}, Threshold: {threshold}"
        
        elif alert_type == "anomaly_detected":
            description = alert_data.get("description", "Anomaly detected")
            return f"{agent}: {description}"
        
        elif alert_type == "system_error":
            error = alert_data.get("error", "System error occurred")
            return f"{agent}: {error}"
        
        else:
            # Generic message
            message = alert_data.get("message", f"{alert_type} alert")
            return f"{agent}: {message}"
    
    def _should_deduplicate(self, alert: Alert) -> bool:
        """Check if alert should be deduplicated"""
        # Look for similar alerts in recent history
        cutoff_time = alert.timestamp.timestamp() - self.deduplication_window
        
        for existing_alert in reversed(self.alert_history):
            if existing_alert.timestamp.timestamp() < cutoff_time:
                break
            
            # Check for duplicate
            if (existing_alert.agent_name == alert.agent_name and
                existing_alert.alert_type == alert.alert_type and
                not existing_alert.resolved):
                
                # Update existing alert data
                existing_alert.data.update(alert.data)
                existing_alert.timestamp = alert.timestamp
                
                return True
        
        return False
    
    async def _notify_handlers(self, alert: Alert):
        """Notify all alert handlers"""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved by {user}")
            return True
        return False
    
    def get_active_alerts(self, agent_name: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by agent"""
        alerts = list(self.active_alerts.values())
        
        if agent_name:
            alerts = [a for a in alerts if a.agent_name == agent_name]
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.timestamp), reverse=True)
        return alerts
    
    def get_alert_history(self, hours_back: int = 24, agent_name: Optional[str] = None) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        
        alerts = [
            a for a in self.alert_history
            if a.timestamp.timestamp() >= cutoff_time
        ]
        
        if agent_name:
            alerts = [a for a in alerts if a.agent_name == agent_name]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_alerts = list(self.active_alerts.values())
        
        stats = {
            "active_alerts_count": len(active_alerts),
            "total_alerts_24h": len(self.get_alert_history(24)),
            "alerts_by_severity": {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            },
            "alerts_by_agent": {}
        }
        
        # Count alerts by agent
        for alert in active_alerts:
            agent = alert.agent_name
            if agent not in stats["alerts_by_agent"]:
                stats["alerts_by_agent"][agent] = 0
            stats["alerts_by_agent"][agent] += 1
        
        return stats


# Default alert handlers
async def console_alert_handler(alert: Alert):
    """Simple console alert handler"""
    severity_symbols = {
        AlertSeverity.CRITICAL: "🚨",
        AlertSeverity.HIGH: "⚠️",
        AlertSeverity.MEDIUM: "⚡",
        AlertSeverity.LOW: "ℹ️"
    }
    
    symbol = severity_symbols.get(alert.severity, "📢")
    timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{symbol} [{timestamp}] {alert.severity.value.upper()}: {alert.message}")


def log_alert_handler(alert: Alert):
    """Log alert handler"""
    logger = logging.getLogger("alerts")
    
    log_methods = {
        AlertSeverity.CRITICAL: logger.critical,
        AlertSeverity.HIGH: logger.error,
        AlertSeverity.MEDIUM: logger.warning,
        AlertSeverity.LOW: logger.info
    }
    
    log_method = log_methods.get(alert.severity, logger.info)
    log_method(f"[{alert.agent_name}] {alert.message}")
