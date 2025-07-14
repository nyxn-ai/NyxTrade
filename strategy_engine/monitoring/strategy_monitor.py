"""
Strategy Performance Monitoring and Analytics
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from collections import defaultdict, deque

from ..core.strategy_base import StrategyBase
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    strategy_id: str
    strategy_name: str
    pnl: float
    position_size: float
    win_rate: float
    total_trades: int
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class AlertConfig:
    """Configuration for performance alerts"""
    max_drawdown_threshold: float = -10.0  # -10%
    min_win_rate_threshold: float = 40.0    # 40%
    max_consecutive_losses: int = 5
    pnl_threshold: float = -1000.0          # -$1000


class StrategyMonitor:
    """
    Real-time strategy performance monitoring and alerting system
    Designed for AI agent oversight and risk management
    """
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        self.monitored_strategies: Dict[str, StrategyBase] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_callbacks: List[Callable] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 60  # seconds
        
        logger.info("Initialized StrategyMonitor")
    
    def add_strategy(self, strategy_id: str, strategy: StrategyBase):
        """Add strategy to monitoring"""
        self.monitored_strategies[strategy_id] = strategy
        logger.info(f"Added strategy to monitoring: {strategy_id} ({strategy.name})")
    
    def remove_strategy(self, strategy_id: str):
        """Remove strategy from monitoring"""
        if strategy_id in self.monitored_strategies:
            del self.monitored_strategies[strategy_id]
            if strategy_id in self.performance_history:
                del self.performance_history[strategy_id]
            logger.info(f"Removed strategy from monitoring: {strategy_id}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started strategy monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped strategy monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_performance_snapshots()
                self._check_alerts()
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_performance_snapshots(self):
        """Collect performance snapshots for all monitored strategies"""
        current_time = datetime.now()
        
        for strategy_id, strategy in self.monitored_strategies.items():
            try:
                performance_metrics = strategy.get_performance_metrics()
                
                snapshot = PerformanceSnapshot(
                    timestamp=current_time,
                    strategy_id=strategy_id,
                    strategy_name=strategy.name,
                    pnl=strategy.pnl,
                    position_size=strategy.position_size,
                    win_rate=performance_metrics.get('win_rate', 0.0),
                    total_trades=performance_metrics.get('total_trades', 0)
                )
                
                self.performance_history[strategy_id].append(snapshot)
                
            except Exception as e:
                logger.error(f"Error collecting snapshot for {strategy_id}: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions"""
        for strategy_id, strategy in self.monitored_strategies.items():
            try:
                self._check_strategy_alerts(strategy_id, strategy)
            except Exception as e:
                logger.error(f"Error checking alerts for {strategy_id}: {e}")
    
    def _check_strategy_alerts(self, strategy_id: str, strategy: StrategyBase):
        """Check alerts for a specific strategy"""
        performance_metrics = strategy.get_performance_metrics()
        history = self.performance_history[strategy_id]
        
        # Check drawdown
        if len(history) > 1:
            max_pnl = max(snapshot.pnl for snapshot in history)
            current_pnl = strategy.pnl
            drawdown = ((current_pnl - max_pnl) / max_pnl * 100) if max_pnl > 0 else 0
            
            if drawdown < self.alert_config.max_drawdown_threshold:
                self._trigger_alert(strategy_id, "MAX_DRAWDOWN", {
                    "current_drawdown": drawdown,
                    "threshold": self.alert_config.max_drawdown_threshold,
                    "current_pnl": current_pnl,
                    "max_pnl": max_pnl
                })
        
        # Check win rate
        win_rate = performance_metrics.get('win_rate', 0.0)
        if win_rate < self.alert_config.min_win_rate_threshold and performance_metrics.get('total_trades', 0) >= 10:
            self._trigger_alert(strategy_id, "LOW_WIN_RATE", {
                "current_win_rate": win_rate,
                "threshold": self.alert_config.min_win_rate_threshold,
                "total_trades": performance_metrics.get('total_trades', 0)
            })
        
        # Check consecutive losses
        consecutive_losses = self._count_consecutive_losses(strategy)
        if consecutive_losses >= self.alert_config.max_consecutive_losses:
            self._trigger_alert(strategy_id, "CONSECUTIVE_LOSSES", {
                "consecutive_losses": consecutive_losses,
                "threshold": self.alert_config.max_consecutive_losses
            })
        
        # Check PnL threshold
        if strategy.pnl < self.alert_config.pnl_threshold:
            self._trigger_alert(strategy_id, "PNL_THRESHOLD", {
                "current_pnl": strategy.pnl,
                "threshold": self.alert_config.pnl_threshold
            })
    
    def _count_consecutive_losses(self, strategy: StrategyBase) -> int:
        """Count consecutive losing trades"""
        if not strategy.trade_history:
            return 0
        
        consecutive_losses = 0
        for trade in reversed(strategy.trade_history):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        return consecutive_losses
    
    def _trigger_alert(self, strategy_id: str, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger alert and notify callbacks"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "strategy_name": self.monitored_strategies[strategy_id].name,
            "alert_type": alert_type,
            "data": alert_data
        }
        
        logger.warning(f"ALERT [{alert_type}] for strategy {strategy_id}: {alert_data}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(strategy_id, alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_strategy_performance_summary(self, strategy_id: str, 
                                       hours_back: int = 24) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        if strategy_id not in self.monitored_strategies:
            raise ValueError(f"Strategy not monitored: {strategy_id}")
        
        strategy = self.monitored_strategies[strategy_id]
        history = self.performance_history[strategy_id]
        
        # Filter history by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_history = [s for s in history if s.timestamp >= cutoff_time]
        
        if not recent_history:
            return {"error": "No recent performance data available"}
        
        # Calculate summary statistics
        pnl_values = [s.pnl for s in recent_history]
        
        summary = {
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "current_state": strategy.get_strategy_state(),
            "time_period_hours": hours_back,
            "data_points": len(recent_history),
            "pnl_statistics": {
                "current": pnl_values[-1] if pnl_values else 0,
                "min": min(pnl_values) if pnl_values else 0,
                "max": max(pnl_values) if pnl_values else 0,
                "change": pnl_values[-1] - pnl_values[0] if len(pnl_values) > 1 else 0
            },
            "performance_metrics": strategy.get_performance_metrics(),
            "recent_snapshots": [asdict(s) for s in recent_history[-10:]]  # Last 10 snapshots
        }
        
        return summary
    
    def get_all_strategies_summary(self) -> Dict[str, Any]:
        """Get summary for all monitored strategies"""
        summary = {
            "monitoring_status": self.is_monitoring,
            "total_strategies": len(self.monitored_strategies),
            "strategies": {}
        }
        
        for strategy_id in self.monitored_strategies:
            try:
                summary["strategies"][strategy_id] = self.get_strategy_performance_summary(strategy_id)
            except Exception as e:
                summary["strategies"][strategy_id] = {"error": str(e)}
        
        return summary
    
    def export_performance_data(self, strategy_id: str, 
                              format: str = "json") -> str:
        """Export performance data for analysis"""
        if strategy_id not in self.performance_history:
            raise ValueError(f"No performance data for strategy: {strategy_id}")
        
        history = list(self.performance_history[strategy_id])
        
        if format.lower() == "json":
            return json.dumps([asdict(snapshot) for snapshot in history], 
                            default=str, indent=2)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if history:
                fieldnames = asdict(history[0]).keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for snapshot in history:
                    writer.writerow(asdict(snapshot))
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def set_monitor_interval(self, seconds: int):
        """Set monitoring interval"""
        if seconds < 10:
            raise ValueError("Monitor interval must be at least 10 seconds")
        
        self.monitor_interval = seconds
        logger.info(f"Set monitor interval to {seconds} seconds")
