"""
RSI Mean Reversion Strategy Example
Demonstrates RSI-based trading strategy implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from ..core.strategy_base import StrategyBase, TradingSignal, MarketData, SignalType
from ..core.indicator_service import IndicatorService, IndicatorLibrary
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(StrategyBase):
    """
    RSI Mean Reversion Strategy
    
    Strategy Logic:
    - Buy when RSI < oversold_threshold (default 30)
    - Sell when RSI > overbought_threshold (default 70)
    - Uses additional filters for confirmation
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str = "1h",
                 parameters: Dict[str, Any] = None):
        
        # Default parameters
        default_params = {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "risk_per_trade": 0.02,
            "min_confidence": 0.65,
            "use_volume_filter": True,
            "use_trend_filter": True,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, symbol, timeframe, default_params)
        
        self.indicator_service = IndicatorService()
        self.last_rsi_value = None
        
        logger.info(f"Initialized RSIStrategy: {name}")
    
    def generate_signal(self, market_data: MarketData,
                       historical_data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal based on RSI levels"""
        
        # Need at least RSI period + 1 data points
        min_data_points = self.parameters["rsi_period"] + 1
        if len(historical_data) < min_data_points:
            return self._create_hold_signal(market_data)
        
        # Calculate RSI
        try:
            rsi = self.indicator_service.calculate_indicator(
                historical_data,
                "rsi",
                IndicatorLibrary.SIMPLE,
                period=self.parameters["rsi_period"]
            )
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return self._create_hold_signal(market_data)
        
        current_rsi = rsi.iloc[-1]
        self.last_rsi_value = current_rsi
        
        # Determine signal type based on RSI levels
        signal_type = SignalType.HOLD
        confidence = 0.5
        
        # Oversold condition - potential buy signal
        if current_rsi < self.parameters["oversold_threshold"]:
            if self.position_size <= 0:  # Only buy if not already long
                signal_type = SignalType.BUY
                confidence = self._calculate_buy_confidence(historical_data, current_rsi)
        
        # Overbought condition - potential sell signal
        elif current_rsi > self.parameters["overbought_threshold"]:
            if self.position_size > 0:  # Only sell if currently long
                signal_type = SignalType.SELL
                confidence = self._calculate_sell_confidence(historical_data, current_rsi)
        
        # Check minimum confidence threshold
        if confidence < self.parameters["min_confidence"]:
            signal_type = SignalType.HOLD
        
        # Apply additional filters
        if signal_type != SignalType.HOLD:
            if not self._passes_filters(historical_data, signal_type):
                signal_type = SignalType.HOLD
        
        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            price=market_data.close,
            confidence=confidence,
            metadata={"rsi_value": current_rsi}
        )
        
        # Add stop loss and take profit
        if signal_type == SignalType.BUY:
            signal.stop_loss = market_data.close * (1 - self.parameters["stop_loss_pct"])
            signal.take_profit = market_data.close * (1 + self.parameters["take_profit_pct"])
        elif signal_type == SignalType.SELL:
            # For sell signals, we're closing a long position
            pass
        
        # Record signal
        if signal_type != SignalType.HOLD:
            self.signals_history.append(signal)
        
        return signal
    
    def calculate_position_size(self, signal: TradingSignal,
                              account_balance: float) -> float:
        """Calculate position size based on RSI-specific risk management"""
        
        if signal.signal_type not in [SignalType.BUY]:
            return 0.0
        
        # Base risk amount
        risk_amount = account_balance * self.parameters["risk_per_trade"]
        
        # Adjust risk based on RSI extremity
        rsi_value = signal.metadata.get("rsi_value", 50) if signal.metadata else 50
        
        # More extreme RSI = higher confidence = larger position
        if rsi_value < 20:  # Very oversold
            risk_multiplier = 1.5
        elif rsi_value < 25:  # Oversold
            risk_multiplier = 1.2
        else:
            risk_multiplier = 1.0
        
        adjusted_risk = risk_amount * risk_multiplier
        
        # Calculate position size based on stop loss
        if signal.stop_loss:
            price_risk = abs(signal.price - signal.stop_loss)
            if price_risk > 0:
                position_size = adjusted_risk / price_risk
            else:
                position_size = 0.0
        else:
            # Default to percentage of balance
            position_size = (adjusted_risk * 5) / signal.price  # 5x leverage on risk amount
        
        # Apply maximum position limits
        max_position_value = account_balance * 0.25  # Max 25% of balance
        max_position_size = max_position_value / signal.price
        
        position_size = min(position_size, max_position_size)
        
        return max(0.0, position_size)
    
    def _calculate_buy_confidence(self, historical_data: pd.DataFrame,
                                current_rsi: float) -> float:
        """Calculate confidence for buy signals"""
        
        base_confidence = 0.6
        
        # Factor 1: RSI extremity (lower RSI = higher confidence)
        rsi_bonus = 0.0
        if current_rsi < 20:
            rsi_bonus = 0.3
        elif current_rsi < 25:
            rsi_bonus = 0.2
        elif current_rsi < 30:
            rsi_bonus = 0.1
        
        # Factor 2: RSI divergence (price falling but RSI rising)
        divergence_bonus = self._check_bullish_divergence(historical_data)
        
        # Factor 3: Recent price decline
        decline_bonus = 0.0
        if len(historical_data) >= 5:
            recent_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[-5]) / historical_data['close'].iloc[-5]
            if recent_change < -0.05:  # 5% decline
                decline_bonus = 0.1
        
        confidence = base_confidence + rsi_bonus + divergence_bonus + decline_bonus
        return min(confidence, 1.0)
    
    def _calculate_sell_confidence(self, historical_data: pd.DataFrame,
                                 current_rsi: float) -> float:
        """Calculate confidence for sell signals"""
        
        base_confidence = 0.6
        
        # Factor 1: RSI extremity (higher RSI = higher confidence)
        rsi_bonus = 0.0
        if current_rsi > 80:
            rsi_bonus = 0.3
        elif current_rsi > 75:
            rsi_bonus = 0.2
        elif current_rsi > 70:
            rsi_bonus = 0.1
        
        # Factor 2: RSI divergence (price rising but RSI falling)
        divergence_bonus = self._check_bearish_divergence(historical_data)
        
        # Factor 3: Recent price rally
        rally_bonus = 0.0
        if len(historical_data) >= 5:
            recent_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[-5]) / historical_data['close'].iloc[-5]
            if recent_change > 0.05:  # 5% rally
                rally_bonus = 0.1
        
        confidence = base_confidence + rsi_bonus + divergence_bonus + rally_bonus
        return min(confidence, 1.0)
    
    def _check_bullish_divergence(self, historical_data: pd.DataFrame) -> float:
        """Check for bullish RSI divergence"""
        if len(historical_data) < 20:
            return 0.0
        
        try:
            # Calculate RSI for divergence check
            rsi = self.indicator_service.calculate_indicator(
                historical_data,
                "rsi",
                IndicatorLibrary.SIMPLE,
                period=self.parameters["rsi_period"]
            )
            
            # Simple divergence check: price making lower lows, RSI making higher lows
            recent_prices = historical_data['close'].iloc[-10:]
            recent_rsi = rsi.iloc[-10:]
            
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            # Bullish divergence: price declining but RSI rising
            if price_trend < 0 and rsi_trend > 0:
                return 0.15
            
        except Exception as e:
            logger.error(f"Error checking bullish divergence: {e}")
        
        return 0.0
    
    def _check_bearish_divergence(self, historical_data: pd.DataFrame) -> float:
        """Check for bearish RSI divergence"""
        if len(historical_data) < 20:
            return 0.0
        
        try:
            # Calculate RSI for divergence check
            rsi = self.indicator_service.calculate_indicator(
                historical_data,
                "rsi",
                IndicatorLibrary.SIMPLE,
                period=self.parameters["rsi_period"]
            )
            
            # Simple divergence check: price making higher highs, RSI making lower highs
            recent_prices = historical_data['close'].iloc[-10:]
            recent_rsi = rsi.iloc[-10:]
            
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            # Bearish divergence: price rising but RSI falling
            if price_trend > 0 and rsi_trend < 0:
                return 0.15
            
        except Exception as e:
            logger.error(f"Error checking bearish divergence: {e}")
        
        return 0.0
    
    def _passes_filters(self, historical_data: pd.DataFrame,
                       signal_type: SignalType) -> bool:
        """Apply additional filters to signals"""
        
        # Volume filter
        if self.parameters["use_volume_filter"]:
            if not self._volume_filter(historical_data):
                return False
        
        # Trend filter
        if self.parameters["use_trend_filter"]:
            if not self._trend_filter(historical_data, signal_type):
                return False
        
        return True
    
    def _volume_filter(self, historical_data: pd.DataFrame) -> bool:
        """Check if volume supports the signal"""
        if 'volume' not in historical_data.columns or len(historical_data) < 10:
            return True  # Skip filter if no volume data
        
        recent_volume = historical_data['volume'].iloc[-3:].mean()
        avg_volume = historical_data['volume'].iloc[-20:].mean()
        
        # Require above-average volume
        return recent_volume > avg_volume * 0.8
    
    def _trend_filter(self, historical_data: pd.DataFrame,
                     signal_type: SignalType) -> bool:
        """Check if signal aligns with longer-term trend"""
        if len(historical_data) < 50:
            return True  # Skip filter if insufficient data
        
        try:
            # Calculate longer-term moving average
            long_ma = self.indicator_service.calculate_indicator(
                historical_data,
                "sma",
                IndicatorLibrary.SIMPLE,
                period=50
            )
            
            current_price = historical_data['close'].iloc[-1]
            current_ma = long_ma.iloc[-1]
            
            # For buy signals, prefer when price is near or below long-term MA
            if signal_type == SignalType.BUY:
                return current_price <= current_ma * 1.05  # Within 5% of MA
            
            # For sell signals, prefer when price is above long-term MA
            elif signal_type == SignalType.SELL:
                return current_price >= current_ma * 0.95  # Within 5% of MA
            
        except Exception as e:
            logger.error(f"Error in trend filter: {e}")
        
        return True
    
    def _create_hold_signal(self, market_data: MarketData) -> TradingSignal:
        """Create a HOLD signal"""
        return TradingSignal(
            signal_type=SignalType.HOLD,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            price=market_data.close,
            confidence=0.5,
            metadata={"rsi_value": self.last_rsi_value}
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {
            "strategy_type": "RSIStrategy",
            "description": "RSI Mean Reversion Strategy",
            "parameters": self.parameters,
            "indicators_used": ["RSI", "SMA"],
            "signal_count": len(self.signals_history),
            "last_rsi_value": self.last_rsi_value
        }
