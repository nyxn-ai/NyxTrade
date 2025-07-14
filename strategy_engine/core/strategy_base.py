"""
Base strategy class for NyxTrade strategy engine
Provides common functionality for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    confidence: float  # 0.0 to 1.0
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


class StrategyBase(ABC):
    """
    Base class for all trading strategies
    Provides common functionality and interface for AI agent integration
    """
    
    def __init__(self, 
                 name: str,
                 symbol: str,
                 timeframe: str = "1h",
                 parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.parameters = parameters or {}
        self.is_active = False
        self.position_size = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.trade_history: List[Dict] = []
        self.signals_history: List[TradingSignal] = []
        
        logger.info(f"Initialized strategy: {name} for {symbol} on {timeframe}")
    
    @abstractmethod
    def generate_signal(self, market_data: MarketData, 
                       historical_data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on market data
        Must be implemented by concrete strategy classes
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, 
                              account_balance: float) -> float:
        """
        Calculate position size based on risk management rules
        Must be implemented by concrete strategy classes
        """
        pass
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update strategy parameters (useful for AI agent optimization)"""
        self.parameters.update(new_parameters)
        logger.info(f"Updated parameters for {self.name}: {new_parameters}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        if not self.trade_history:
            return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0}
        
        total_pnl = sum(trade.get("pnl", 0) for trade in self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0.0
        }
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal before execution"""
        if signal.confidence < 0.5:  # Minimum confidence threshold
            logger.warning(f"Signal confidence too low: {signal.confidence}")
            return False
        
        if signal.price <= 0:
            logger.error(f"Invalid price in signal: {signal.price}")
            return False
        
        return True
    
    def record_trade(self, signal: TradingSignal, executed_price: float, 
                    quantity: float, pnl: float) -> None:
        """Record executed trade for performance tracking"""
        trade_record = {
            "timestamp": signal.timestamp,
            "signal_type": signal.signal_type.value,
            "symbol": signal.symbol,
            "price": executed_price,
            "quantity": quantity,
            "pnl": pnl,
            "confidence": signal.confidence
        }
        
        self.trade_history.append(trade_record)
        self.pnl += pnl
        
        logger.info(f"Recorded trade: {trade_record}")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for AI agent monitoring"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "is_active": self.is_active,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "current_pnl": self.pnl,
            "parameters": self.parameters,
            "performance": self.get_performance_metrics()
        }
    
    def reset_strategy(self) -> None:
        """Reset strategy state (useful for backtesting)"""
        self.position_size = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.trade_history.clear()
        self.signals_history.clear()
        self.is_active = False
        
        logger.info(f"Reset strategy: {self.name}")
