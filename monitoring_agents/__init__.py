"""
NyxTrade Monitoring Agents System

A comprehensive monitoring system with AI-powered analysis using Google Gemini AI.
Includes specialized agents for market regression, trend tracking, fund flow,
indicator collection, and hotspot tracking.
"""

from .core.base_agent import BaseMonitoringAgent
from .core.gemini_client import GeminiClient
from .core.alert_manager import AlertManager

__version__ = "1.0.0"
__author__ = "NyxTrade Team"

__all__ = [
    "BaseMonitoringAgent",
    "GeminiClient", 
    "AlertManager"
]
