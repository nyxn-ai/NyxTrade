"""
Regression analysis calculator for mean reversion strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
import logging


class RegressionCalculator:
    """
    Calculates various regression and mean reversion metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger("regression_calculator")
    
    def calculate_regression_metrics(self, historical_data: pd.DataFrame, 
                                   current_price: float, period: int) -> Dict[str, Any]:
        """
        Calculate regression metrics for a specific period
        
        Args:
            historical_data: DataFrame with OHLCV data
            current_price: Current price to compare against
            period: Lookback period for calculations
            
        Returns:
            Dictionary with regression metrics
        """
        try:
            if len(historical_data) < period:
                return self._get_empty_metrics()
            
            # Get price data for the period
            prices = historical_data['close'].tail(period)
            
            # Calculate moving average
            moving_average = prices.mean()
            
            # Calculate standard deviation
            std_dev = prices.std()
            
            # Calculate Z-score
            z_score = (current_price - moving_average) / std_dev if std_dev > 0 else 0
            
            # Calculate percentage deviation
            deviation_percent = ((current_price - moving_average) / moving_average) * 100
            
            # Calculate Bollinger Bands
            bb_upper = moving_average + (2 * std_dev)
            bb_lower = moving_average - (2 * std_dev)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Calculate RSI for the period
            rsi = self._calculate_rsi(prices, min(14, period // 2))
            
            # Calculate price momentum
            momentum = self._calculate_momentum(prices, min(10, period // 3))
            
            # Calculate volatility metrics
            volatility = self._calculate_volatility_metrics(prices)
            
            # Calculate regression strength
            regression_strength = self._calculate_regression_strength(prices, current_price)
            
            # Calculate statistical significance
            p_value = self._calculate_statistical_significance(prices, current_price)
            
            return {
                "period": period,
                "moving_average": moving_average,
                "standard_deviation": std_dev,
                "z_score": z_score,
                "deviation_percent": deviation_percent,
                "bollinger_upper": bb_upper,
                "bollinger_lower": bb_lower,
                "bollinger_position": bb_position,
                "rsi": rsi,
                "momentum": momentum,
                "volatility": volatility,
                "regression_strength": regression_strength,
                "p_value": p_value,
                "is_oversold": z_score < -2.0,
                "is_overbought": z_score > 2.0,
                "confidence": self._calculate_confidence(z_score, p_value, regression_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {e}")
            return self._get_empty_metrics()
    
    def calculate_overall_score(self, symbol_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall regression score combining multiple periods
        
        Returns:
            Score between -1 (strong oversold) and +1 (strong overbought)
        """
        try:
            scores = []
            weights = []
            
            # Weight longer periods more heavily
            period_weights = {20: 1.0, 50: 1.5, 200: 2.0}
            
            for key, analysis in symbol_analysis.items():
                if key.startswith("period_") and isinstance(analysis, dict):
                    period = analysis.get("period", 0)
                    z_score = analysis.get("z_score", 0)
                    confidence = analysis.get("confidence", 0)
                    
                    if period in period_weights:
                        # Normalize z_score to -1 to +1 range
                        normalized_score = np.tanh(z_score / 2.0)
                        
                        # Weight by confidence and period importance
                        weight = period_weights[period] * confidence
                        
                        scores.append(normalized_score)
                        weights.append(weight)
            
            if not scores:
                return 0.0
            
            # Calculate weighted average
            weighted_score = np.average(scores, weights=weights)
            
            # Ensure score is in [-1, 1] range
            return np.clip(weighted_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-period-1]
            
            momentum = (current_price - past_price) / past_price
            return momentum
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate volatility metrics"""
        try:
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return {"daily_vol": 0.0, "vol_percentile": 0.5}
            
            # Daily volatility (annualized)
            daily_vol = returns.std() * np.sqrt(365)
            
            # Volatility percentile (current vs historical)
            rolling_vol = returns.rolling(window=min(20, len(returns))).std()
            current_vol = rolling_vol.iloc[-1]
            vol_percentile = (rolling_vol < current_vol).mean()
            
            return {
                "daily_vol": daily_vol,
                "vol_percentile": vol_percentile
            }
            
        except Exception:
            return {"daily_vol": 0.0, "vol_percentile": 0.5}
    
    def _calculate_regression_strength(self, prices: pd.Series, current_price: float) -> float:
        """Calculate strength of mean reversion tendency"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Calculate how often price reverts after extreme moves
            mean_price = prices.mean()
            std_price = prices.std()
            
            if std_price == 0:
                return 0.0
            
            # Find extreme deviations
            z_scores = (prices - mean_price) / std_price
            extreme_threshold = 1.5
            
            extreme_points = np.abs(z_scores) > extreme_threshold
            
            if extreme_points.sum() < 3:
                return 0.0
            
            # Calculate reversion rate
            reversion_count = 0
            total_extreme = 0
            
            for i in range(1, len(prices)):
                if extreme_points.iloc[i-1]:
                    total_extreme += 1
                    
                    # Check if price moved toward mean in next few periods
                    future_periods = min(5, len(prices) - i)
                    if future_periods > 0:
                        future_prices = prices.iloc[i:i+future_periods]
                        
                        # Check if price moved closer to mean
                        initial_distance = abs(prices.iloc[i-1] - mean_price)
                        final_distance = abs(future_prices.mean() - mean_price)
                        
                        if final_distance < initial_distance:
                            reversion_count += 1
            
            reversion_rate = reversion_count / total_extreme if total_extreme > 0 else 0.0
            
            # Normalize to 0-1 range
            return min(reversion_rate, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_statistical_significance(self, prices: pd.Series, current_price: float) -> float:
        """Calculate statistical significance of current deviation"""
        try:
            if len(prices) < 10:
                return 1.0
            
            # Perform t-test against historical mean
            mean_price = prices.mean()
            
            # Create sample with current price
            sample = list(prices) + [current_price]
            
            # Test if current price is significantly different from historical mean
            t_stat, p_value = stats.ttest_1samp(sample, mean_price)
            
            return p_value
            
        except Exception:
            return 1.0
    
    def _calculate_confidence(self, z_score: float, p_value: float, regression_strength: float) -> float:
        """Calculate confidence in regression signal"""
        try:
            # Base confidence on Z-score magnitude
            z_confidence = min(abs(z_score) / 3.0, 1.0)  # Normalize to 0-1
            
            # Statistical significance confidence
            sig_confidence = 1.0 - p_value
            
            # Regression strength confidence
            strength_confidence = regression_strength
            
            # Combine confidences with weights
            overall_confidence = (
                z_confidence * 0.4 +
                sig_confidence * 0.3 +
                strength_confidence * 0.3
            )
            
            return min(overall_confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "period": 0,
            "moving_average": 0.0,
            "standard_deviation": 0.0,
            "z_score": 0.0,
            "deviation_percent": 0.0,
            "bollinger_upper": 0.0,
            "bollinger_lower": 0.0,
            "bollinger_position": 0.5,
            "rsi": 50.0,
            "momentum": 0.0,
            "volatility": {"daily_vol": 0.0, "vol_percentile": 0.5},
            "regression_strength": 0.0,
            "p_value": 1.0,
            "is_oversold": False,
            "is_overbought": False,
            "confidence": 0.0
        }
