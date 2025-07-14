"""
Moving Average Crossover Strategy Example
Demonstrates basic strategy implementation using the NyxTrade Strategy Engine
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from ..core.strategy_base import StrategyBase, TradingSignal, MarketData, SignalType
from ..core.indicator_service import IndicatorService, IndicatorLibrary
from utils.logger import get_logger

logger = get_logger(__name__)


class MovingAverageStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    Strategy Logic:
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA
    - Uses position sizing based on volatility
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str = "1h", 
                 parameters: Dict[str, Any] = None):
        
        # Default parameters
        default_params = {
            "fast_period": 10,
            "slow_period": 30,
            "risk_per_trade": 0.02,  # 2% risk per trade
            "min_confidence": 0.6,
            "use_stop_loss": True,
            "stop_loss_pct": 0.02,   # 2% stop loss
            "take_profit_pct": 0.04  # 4% take profit
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, symbol, timeframe, default_params)
        
        self.indicator_service = IndicatorService()
        self.last_signal_time = None
        self.min_signal_interval = 3600  # 1 hour minimum between signals
        
        logger.info(f"Initialized MovingAverageStrategy: {name}")
    
    def generate_signal(self, market_data: MarketData, 
                       historical_data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal based on MA crossover"""
        
        # Need at least slow_period + 1 data points
        min_data_points = self.parameters["slow_period"] + 1
        if len(historical_data) < min_data_points:
            return self._create_hold_signal(market_data)
        
        # Calculate moving averages
        try:
            fast_ma = self.indicator_service.calculate_indicator(
                historical_data,
                "sma",
                IndicatorLibrary.SIMPLE,
                period=self.parameters["fast_period"]
            )

            slow_ma = self.indicator_service.calculate_indicator(
                historical_data,
                "sma",
                IndicatorLibrary.SIMPLE,
                period=self.parameters["slow_period"]
            )
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return self._create_hold_signal(market_data)
        
        # Get current and previous values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow
        
        # Check for crossover
        signal_type = SignalType.HOLD
        confidence = 0.5
        
        # Bullish crossover: fast MA crosses above slow MA
        if prev_fast <= prev_slow and current_fast > current_slow:
            if self.position_size <= 0:  # Only buy if not already long
                signal_type = SignalType.BUY
                confidence = self._calculate_confidence(historical_data, current_fast, current_slow)
        
        # Bearish crossover: fast MA crosses below slow MA  
        elif prev_fast >= prev_slow and current_fast < current_slow:
            if self.position_size > 0:  # Only sell if currently long
                signal_type = SignalType.SELL
                confidence = self._calculate_confidence(historical_data, current_fast, current_slow)
        
        # Check minimum confidence threshold
        if confidence < self.parameters["min_confidence"]:
            signal_type = SignalType.HOLD
        
        # Check minimum time between signals
        if self._should_skip_signal_timing(market_data.timestamp):
            signal_type = SignalType.HOLD
        
        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            price=market_data.close,
            confidence=confidence
        )
        
        # Add stop loss and take profit if enabled
        if signal_type == SignalType.BUY and self.parameters["use_stop_loss"]:
            signal.stop_loss = market_data.close * (1 - self.parameters["stop_loss_pct"])
            signal.take_profit = market_data.close * (1 + self.parameters["take_profit_pct"])
        
        # Update last signal time
        if signal_type != SignalType.HOLD:
            self.last_signal_time = market_data.timestamp
            self.signals_history.append(signal)
        
        return signal
    
    def calculate_position_size(self, signal: TradingSignal, 
                              account_balance: float) -> float:
        """Calculate position size based on risk management"""
        
        if signal.signal_type not in [SignalType.BUY]:
            return 0.0
        
        # Calculate risk amount
        risk_amount = account_balance * self.parameters["risk_per_trade"]
        
        # Calculate position size based on stop loss
        if signal.stop_loss and self.parameters["use_stop_loss"]:
            price_risk = abs(signal.price - signal.stop_loss)
            if price_risk > 0:
                position_size = risk_amount / price_risk
            else:
                position_size = 0.0
        else:
            # Default to 10% of balance if no stop loss
            position_size = (account_balance * 0.1) / signal.price
        
        # Ensure position size is reasonable
        max_position_value = account_balance * 0.2  # Max 20% of balance per trade
        max_position_size = max_position_value / signal.price
        
        position_size = min(position_size, max_position_size)
        
        return max(0.0, position_size)
    
    def _calculate_confidence(self, historical_data: pd.DataFrame, 
                            fast_ma: float, slow_ma: float) -> float:
        """Calculate signal confidence based on various factors"""
        
        base_confidence = 0.6
        
        # Factor 1: MA separation (larger separation = higher confidence)
        ma_separation = abs(fast_ma - slow_ma) / slow_ma
        separation_bonus = min(ma_separation * 10, 0.2)  # Max 0.2 bonus
        
        # Factor 2: Volume confirmation (if available)
        volume_bonus = 0.0
        if 'volume' in historical_data.columns and len(historical_data) >= 5:
            recent_volume = historical_data['volume'].iloc[-5:].mean()
            avg_volume = historical_data['volume'].mean()
            if recent_volume > avg_volume * 1.2:  # 20% above average
                volume_bonus = 0.1
        
        # Factor 3: Trend strength
        trend_bonus = 0.0
        if len(historical_data) >= 20:
            price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[-20]) / historical_data['close'].iloc[-20]
            if abs(price_change) > 0.05:  # 5% move in 20 periods
                trend_bonus = 0.1
        
        confidence = base_confidence + separation_bonus + volume_bonus + trend_bonus
        return min(confidence, 1.0)
    
    def _should_skip_signal_timing(self, current_time) -> bool:
        """Check if we should skip signal due to timing constraints"""
        if self.last_signal_time is None:
            return False
        
        time_diff = (current_time - self.last_signal_time).total_seconds()
        return time_diff < self.min_signal_interval
    
    def _create_hold_signal(self, market_data: MarketData) -> TradingSignal:
        """Create a HOLD signal"""
        return TradingSignal(
            signal_type=SignalType.HOLD,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            price=market_data.close,
            confidence=0.5
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {
            "strategy_type": "MovingAverageStrategy",
            "description": "Simple Moving Average Crossover Strategy",
            "parameters": self.parameters,
            "indicators_used": ["SMA"],
            "signal_count": len(self.signals_history),
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None
        }
