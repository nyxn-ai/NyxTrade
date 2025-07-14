"""
Base monitoring agent class providing common functionality
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .gemini_client import GeminiClient
from .alert_manager import AlertManager
from .config_manager import ConfigManager


@dataclass
class MonitoringResult:
    """Result from monitoring agent analysis"""
    agent_name: str
    timestamp: datetime
    data: Dict[str, Any]
    analysis: Dict[str, Any]
    ai_insights: Optional[Dict[str, Any]] = None
    alerts: List[Dict[str, Any]] = None
    confidence: float = 0.0
    recommendations: List[str] = None


@dataclass
class AgentConfig:
    """Configuration for monitoring agent"""
    name: str
    enabled: bool = True
    update_interval: int = 60  # seconds
    alert_thresholds: Dict[str, float] = None
    data_sources: List[str] = None
    gemini_enabled: bool = True
    gemini_model: str = "gemini-pro"


class BaseMonitoringAgent(ABC):
    """
    Base class for all monitoring agents
    Provides common functionality for data collection, AI analysis, and alerting
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.is_running = False
        self.last_update = None
        self.results_history: List[MonitoringResult] = []
        
        # Initialize components
        self.gemini_client = GeminiClient() if config.gemini_enabled else None
        self.alert_manager = AlertManager()
        self.config_manager = ConfigManager()
        
        # Setup logging
        self.logger = logging.getLogger(f"monitoring_agent.{self.name}")
        
        self.logger.info(f"Initialized monitoring agent: {self.name}")
    
    @abstractmethod
    async def collect_data(self) -> Dict[str, Any]:
        """
        Collect data from various sources
        Must be implemented by concrete agent classes
        """
        pass
    
    @abstractmethod
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze collected data and generate insights
        Must be implemented by concrete agent classes
        """
        pass
    
    @abstractmethod
    def get_gemini_prompt(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """
        Generate Gemini AI prompt for this specific agent
        Must be implemented by concrete agent classes
        """
        pass
    
    async def run_analysis_cycle(self) -> MonitoringResult:
        """Run a complete analysis cycle"""
        try:
            # Collect data
            self.logger.debug(f"Collecting data for {self.name}")
            data = await self.collect_data()
            
            # Analyze data
            self.logger.debug(f"Analyzing data for {self.name}")
            analysis = await self.analyze_data(data)
            
            # Get AI insights if enabled
            ai_insights = None
            if self.gemini_client and self.config.gemini_enabled:
                try:
                    prompt = self.get_gemini_prompt(data, analysis)
                    ai_insights = await self.gemini_client.analyze(prompt)
                except Exception as e:
                    self.logger.error(f"Gemini analysis failed: {e}")
            
            # Create result
            result = MonitoringResult(
                agent_name=self.name,
                timestamp=datetime.now(),
                data=data,
                analysis=analysis,
                ai_insights=ai_insights,
                alerts=[],
                recommendations=[]
            )
            
            # Check for alerts
            alerts = self.check_alerts(result)
            result.alerts = alerts
            
            # Generate recommendations
            recommendations = self.generate_recommendations(result)
            result.recommendations = recommendations
            
            # Store result
            self.results_history.append(result)
            if len(self.results_history) > 1000:  # Keep last 1000 results
                self.results_history.pop(0)
            
            self.last_update = datetime.now()
            
            # Send alerts if any
            if alerts:
                await self.alert_manager.send_alerts(alerts)
            
            self.logger.info(f"Analysis cycle completed for {self.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis cycle failed for {self.name}: {e}")
            raise
    
    def check_alerts(self, result: MonitoringResult) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met"""
        alerts = []
        
        if not self.config.alert_thresholds:
            return alerts
        
        # Check thresholds against analysis results
        for metric, threshold in self.config.alert_thresholds.items():
            if metric in result.analysis:
                value = result.analysis[metric]
                if isinstance(value, (int, float)) and value > threshold:
                    alerts.append({
                        "type": "threshold_exceeded",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": result.timestamp,
                        "agent": self.name
                    })
        
        return alerts
    
    def generate_recommendations(self, result: MonitoringResult) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Extract recommendations from AI insights
        if result.ai_insights and "recommendations" in result.ai_insights:
            ai_recommendations = result.ai_insights["recommendations"]
            if isinstance(ai_recommendations, list):
                recommendations.extend(ai_recommendations)
            elif isinstance(ai_recommendations, str):
                recommendations.append(ai_recommendations)
        
        # Add agent-specific recommendations
        agent_recommendations = self.get_agent_recommendations(result)
        recommendations.extend(agent_recommendations)
        
        return recommendations
    
    def get_agent_recommendations(self, result: MonitoringResult) -> List[str]:
        """
        Generate agent-specific recommendations
        Can be overridden by concrete agent classes
        """
        return []
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_running:
            self.logger.warning(f"Agent {self.name} is already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting monitoring agent: {self.name}")
        
        while self.is_running:
            try:
                await self.run_analysis_cycle()
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        self.logger.info(f"Stopped monitoring agent: {self.name}")
    
    def get_latest_result(self) -> Optional[MonitoringResult]:
        """Get the latest monitoring result"""
        return self.results_history[-1] if self.results_history else None
    
    def get_results_history(self, hours_back: int = 24) -> List[MonitoringResult]:
        """Get results history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [r for r in self.results_history if r.timestamp >= cutoff_time]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "results_count": len(self.results_history),
            "config": {
                "enabled": self.config.enabled,
                "update_interval": self.config.update_interval,
                "gemini_enabled": self.config.gemini_enabled
            }
        }
