"""
Signal generation for mean reversion strategies
"""

from typing import Dict, List, Any
from enum import Enum
import logging


class SignalType(Enum):
    """Types of regression signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class RegressionSignalGenerator:
    """
    Generates trading signals based on regression analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger("regression_signals")
    
    def generate_signals(self, symbol: str, current_price: float, 
                        analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate regression signals based on analysis
        
        Args:
            symbol: Trading symbol (BTC, ETH)
            current_price: Current price
            analysis: Analysis results from regression calculator
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        try:
            # Get overall regression score
            regression_score = analysis.get("regression_score", 0)
            
            # Generate signals based on different periods
            for period_key, period_analysis in analysis.items():
                if not period_key.startswith("period_") or not isinstance(period_analysis, dict):
                    continue
                
                period = period_analysis.get("period", 0)
                signal = self._generate_period_signal(symbol, current_price, period_analysis, period)
                
                if signal:
                    signals.append(signal)
            
            # Generate overall signal
            overall_signal = self._generate_overall_signal(symbol, current_price, regression_score, analysis)
            if overall_signal:
                signals.append(overall_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _generate_period_signal(self, symbol: str, current_price: float,
                               period_analysis: Dict[str, Any], period: int) -> Dict[str, Any]:
        """Generate signal for specific period analysis"""
        
        z_score = period_analysis.get("z_score", 0)
        confidence = period_analysis.get("confidence", 0)
        rsi = period_analysis.get("rsi", 50)
        is_oversold = period_analysis.get("is_oversold", False)
        is_overbought = period_analysis.get("is_overbought", False)
        
        # Determine signal type
        signal_type = SignalType.HOLD
        strength = 0.0
        
        # Strong signals (high confidence + extreme conditions)
        if confidence > 0.7:
            if z_score < -2.0 and rsi < 30:
                signal_type = SignalType.STRONG_BUY
                strength = min(abs(z_score) / 3.0, 1.0)
            elif z_score > 2.0 and rsi > 70:
                signal_type = SignalType.STRONG_SELL
                strength = min(abs(z_score) / 3.0, 1.0)
        
        # Regular signals (moderate confidence + oversold/overbought)
        elif confidence > 0.5:
            if is_oversold and rsi < 40:
                signal_type = SignalType.BUY
                strength = confidence * 0.7
            elif is_overbought and rsi > 60:
                signal_type = SignalType.SELL
                strength = confidence * 0.7
        
        # Only return signal if there's a clear direction
        if signal_type == SignalType.HOLD:
            return None
        
        return {
            "symbol": symbol,
            "signal_type": signal_type.value,
            "period": period,
            "strength": strength,
            "confidence": confidence,
            "current_price": current_price,
            "z_score": z_score,
            "rsi": rsi,
            "reasoning": self._get_signal_reasoning(signal_type, z_score, rsi, confidence),
            "timestamp": period_analysis.get("timestamp")
        }
    
    def _generate_overall_signal(self, symbol: str, current_price: float,
                                regression_score: float, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall signal combining all periods"""
        
        # Calculate average confidence across periods
        confidences = []
        z_scores = []
        
        for key, period_analysis in analysis.items():
            if key.startswith("period_") and isinstance(period_analysis, dict):
                confidences.append(period_analysis.get("confidence", 0))
                z_scores.append(period_analysis.get("z_score", 0))
        
        if not confidences:
            return None
        
        avg_confidence = sum(confidences) / len(confidences)
        avg_z_score = sum(z_scores) / len(z_scores)
        
        # Determine overall signal
        signal_type = SignalType.HOLD
        strength = abs(regression_score)
        
        if avg_confidence > 0.6:
            if regression_score < -0.7:
                signal_type = SignalType.STRONG_BUY if regression_score < -0.8 else SignalType.BUY
            elif regression_score > 0.7:
                signal_type = SignalType.STRONG_SELL if regression_score > 0.8 else SignalType.SELL
        
        if signal_type == SignalType.HOLD:
            return None
        
        return {
            "symbol": symbol,
            "signal_type": signal_type.value,
            "period": "overall",
            "strength": strength,
            "confidence": avg_confidence,
            "current_price": current_price,
            "regression_score": regression_score,
            "avg_z_score": avg_z_score,
            "reasoning": self._get_overall_reasoning(signal_type, regression_score, avg_confidence),
            "risk_level": self._assess_risk_level(regression_score, avg_confidence)
        }
    
    def _get_signal_reasoning(self, signal_type: SignalType, z_score: float,
                             rsi: float, confidence: float) -> str:
        """Generate reasoning text for signal"""
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return (f"Oversold conditions detected: Z-score {z_score:.2f}, "
                   f"RSI {rsi:.1f}, confidence {confidence:.2f}")
        
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return (f"Overbought conditions detected: Z-score {z_score:.2f}, "
                   f"RSI {rsi:.1f}, confidence {confidence:.2f}")
        
        else:
            return "No clear signal - market in neutral range"
    
    def _get_overall_reasoning(self, signal_type: SignalType, regression_score: float,
                              confidence: float) -> str:
        """Generate reasoning for overall signal"""
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return (f"Mean reversion opportunity: Regression score {regression_score:.2f} "
                   f"indicates oversold conditions with {confidence:.2f} confidence")
        
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return (f"Mean reversion opportunity: Regression score {regression_score:.2f} "
                   f"indicates overbought conditions with {confidence:.2f} confidence")
        
        else:
            return "No significant mean reversion opportunity detected"
    
    def _assess_risk_level(self, regression_score: float, confidence: float) -> str:
        """Assess risk level for the signal"""
        
        # High confidence + extreme score = lower risk
        if confidence > 0.8 and abs(regression_score) > 0.8:
            return "low"
        
        # Moderate confidence or score = medium risk
        elif confidence > 0.6 or abs(regression_score) > 0.6:
            return "medium"
        
        # Low confidence or weak signal = high risk
        else:
            return "high"
    
    def filter_signals_by_quality(self, signals: List[Dict[str, Any]], 
                                 min_confidence: float = 0.6,
                                 min_strength: float = 0.5) -> List[Dict[str, Any]]:
        """Filter signals by quality thresholds"""
        
        filtered_signals = []
        
        for signal in signals:
            confidence = signal.get("confidence", 0)
            strength = signal.get("strength", 0)
            
            if confidence >= min_confidence and strength >= min_strength:
                filtered_signals.append(signal)
        
        # Sort by strength (highest first)
        filtered_signals.sort(key=lambda s: s.get("strength", 0), reverse=True)
        
        return filtered_signals
    
    def get_signal_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of generated signals"""
        
        if not signals:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "strong_signals": 0,
                "avg_confidence": 0.0,
                "max_strength": 0.0
            }
        
        buy_count = len([s for s in signals if s.get("signal_type") in ["buy", "strong_buy"]])
        sell_count = len([s for s in signals if s.get("signal_type") in ["sell", "strong_sell"]])
        strong_count = len([s for s in signals if s.get("signal_type") in ["strong_buy", "strong_sell"]])
        
        confidences = [s.get("confidence", 0) for s in signals]
        strengths = [s.get("strength", 0) for s in signals]
        
        return {
            "total_signals": len(signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "strong_signals": strong_count,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "max_strength": max(strengths) if strengths else 0.0
        }
