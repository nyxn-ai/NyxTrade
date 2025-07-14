"""
NyxTrade Strategy Engine

A comprehensive trading strategy development framework that integrates
Jesse, pandas-ta, FinTA, and other technical analysis libraries.
Designed for AI agent automation and advanced strategy development.
"""

from .core.strategy_base import StrategyBase
from .core.indicator_service import IndicatorService
from .core.backtest_engine import BacktestEngine
from .ai_interface.agent_strategy_manager import AgentStrategyManager
from .monitoring.strategy_monitor import StrategyMonitor

__version__ = "1.0.0"
__author__ = "NyxTrade Team"

__all__ = [
    "StrategyBase",
    "IndicatorService", 
    "BacktestEngine",
    "AgentStrategyManager",
    "StrategyMonitor"
]
