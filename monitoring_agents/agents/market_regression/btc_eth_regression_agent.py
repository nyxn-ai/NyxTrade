"""
BTC and ETH mean reversion monitoring agent
Monitors price deviations from historical means and identifies regression opportunities
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

from ...core.base_agent import BaseMonitoringAgent, AgentConfig
from ...core.data_collector import DataCollector
from .regression_calculator import RegressionCalculator
from .regression_signals import RegressionSignalGenerator


class BTCETHRegressionAgent(BaseMonitoringAgent):
    """
    Monitors BTC and ETH for mean reversion opportunities
    
    Key Features:
    - Tracks price deviations from multiple moving averages
    - Calculates Z-scores and statistical significance
    - Identifies overbought/oversold conditions
    - Generates regression trading signals
    - Uses Gemini AI for market context analysis
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="btc_eth_regression",
                update_interval=300,  # 5 minutes
                alert_thresholds={
                    "z_score_btc": 2.0,
                    "z_score_eth": 2.0,
                    "deviation_percent": 10.0
                }
            )
        
        super().__init__(config)
        
        # Initialize components
        self.data_collector = DataCollector()
        self.regression_calculator = RegressionCalculator()
        self.signal_generator = RegressionSignalGenerator()
        
        # Configuration
        agent_config = self.config_manager.get_agent_config("market_regression")
        self.symbols = agent_config.get("symbols", ["BTC", "ETH"])
        self.lookback_periods = agent_config.get("lookback_periods", [20, 50, 200])
        
        self.logger.info(f"Initialized BTC ETH Regression Agent for symbols: {self.symbols}")
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect price data for BTC and ETH"""
        try:
            data = {}
            
            for symbol in self.symbols:
                # Get current price
                current_price = await self.data_collector.get_current_price(f"{symbol}USDT")
                
                # Get historical price data
                historical_data = await self.data_collector.get_historical_prices(
                    f"{symbol}USDT",
                    interval="1h",
                    limit=max(self.lookback_periods) + 50  # Extra buffer for calculations
                )
                
                data[symbol] = {
                    "current_price": current_price,
                    "historical_data": historical_data,
                    "timestamp": datetime.now()
                }
            
            # Get market context data
            market_data = await self.data_collector.get_market_overview()
            data["market_context"] = market_data
            
            self.logger.debug(f"Collected data for {len(self.symbols)} symbols")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to collect data: {e}")
            raise
    
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price data for mean reversion opportunities"""
        try:
            analysis = {
                "timestamp": datetime.now(),
                "symbols_analysis": {},
                "market_summary": {},
                "regression_opportunities": []
            }
            
            for symbol in self.symbols:
                if symbol not in data:
                    continue
                
                symbol_data = data[symbol]
                current_price = symbol_data["current_price"]
                historical_data = symbol_data["historical_data"]
                
                # Calculate regression metrics for different periods
                symbol_analysis = {}
                
                for period in self.lookback_periods:
                    period_analysis = self.regression_calculator.calculate_regression_metrics(
                        historical_data,
                        current_price,
                        period
                    )
                    symbol_analysis[f"period_{period}"] = period_analysis
                
                # Calculate overall regression score
                regression_score = self.regression_calculator.calculate_overall_score(symbol_analysis)
                symbol_analysis["regression_score"] = regression_score
                
                # Generate signals
                signals = self.signal_generator.generate_signals(
                    symbol,
                    current_price,
                    symbol_analysis
                )
                symbol_analysis["signals"] = signals
                
                analysis["symbols_analysis"][symbol] = symbol_analysis
                
                # Check for regression opportunities
                if abs(regression_score) > 0.7:  # Strong regression signal
                    opportunity = {
                        "symbol": symbol,
                        "type": "mean_reversion",
                        "direction": "buy" if regression_score < -0.7 else "sell",
                        "strength": abs(regression_score),
                        "current_price": current_price,
                        "signals": signals
                    }
                    analysis["regression_opportunities"].append(opportunity)
            
            # Market summary
            analysis["market_summary"] = self._generate_market_summary(analysis["symbols_analysis"])
            
            self.logger.info(f"Analysis completed. Found {len(analysis['regression_opportunities'])} opportunities")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data: {e}")
            raise
    
    def get_gemini_prompt(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate Gemini AI prompt for regression analysis"""
        
        # Extract key metrics for prompt
        symbols_summary = []
        for symbol, symbol_analysis in analysis["symbols_analysis"].items():
            current_price = data[symbol]["current_price"]
            regression_score = symbol_analysis.get("regression_score", 0)
            
            # Get key metrics from different periods
            period_20 = symbol_analysis.get("period_20", {})
            period_50 = symbol_analysis.get("period_50", {})
            
            symbols_summary.append({
                "symbol": symbol,
                "current_price": current_price,
                "regression_score": regression_score,
                "z_score_20d": period_20.get("z_score", 0),
                "z_score_50d": period_50.get("z_score", 0),
                "deviation_20d": period_20.get("deviation_percent", 0),
                "deviation_50d": period_50.get("deviation_percent", 0),
                "signals": symbol_analysis.get("signals", [])
            })
        
        opportunities = analysis.get("regression_opportunities", [])
        market_context = data.get("market_context", {})
        
        prompt = f"""
Analyze the following BTC and ETH mean reversion data and provide insights:

CURRENT MARKET DATA:
{symbols_summary}

REGRESSION OPPORTUNITIES DETECTED:
{opportunities}

MARKET CONTEXT:
{market_context}

Please analyze:
1. The statistical significance of the current price deviations
2. Historical context - how often do these deviation levels reverse?
3. Market conditions that might support or hinder mean reversion
4. Risk factors that could prevent normal regression patterns
5. Optimal entry/exit strategies for identified opportunities
6. Timeline expectations for potential mean reversion

Focus on:
- Statistical probability of reversion based on historical patterns
- Current market sentiment and its impact on regression timing
- Risk management considerations for mean reversion trades
- Correlation between BTC and ETH regression patterns
"""
        
        return prompt
    
    def _generate_market_summary(self, symbols_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market summary from symbols analysis"""
        summary = {
            "total_symbols": len(symbols_analysis),
            "symbols_oversold": 0,
            "symbols_overbought": 0,
            "average_regression_score": 0,
            "market_regime": "neutral"
        }
        
        regression_scores = []
        
        for symbol, analysis in symbols_analysis.items():
            regression_score = analysis.get("regression_score", 0)
            regression_scores.append(regression_score)
            
            if regression_score < -0.5:
                summary["symbols_oversold"] += 1
            elif regression_score > 0.5:
                summary["symbols_overbought"] += 1
        
        if regression_scores:
            summary["average_regression_score"] = np.mean(regression_scores)
            
            # Determine market regime
            avg_score = summary["average_regression_score"]
            if avg_score < -0.3:
                summary["market_regime"] = "oversold"
            elif avg_score > 0.3:
                summary["market_regime"] = "overbought"
            else:
                summary["market_regime"] = "neutral"
        
        return summary
    
    def get_agent_recommendations(self, result) -> List[str]:
        """Generate agent-specific recommendations"""
        recommendations = []
        
        opportunities = result.analysis.get("regression_opportunities", [])
        market_summary = result.analysis.get("market_summary", {})
        
        # Recommendations based on opportunities
        for opp in opportunities:
            symbol = opp["symbol"]
            direction = opp["direction"]
            strength = opp["strength"]
            
            if strength > 0.8:
                recommendations.append(
                    f"Strong {direction} signal for {symbol} (strength: {strength:.2f}) - "
                    f"Consider position sizing based on statistical confidence"
                )
            elif strength > 0.6:
                recommendations.append(
                    f"Moderate {direction} opportunity for {symbol} - "
                    f"Wait for additional confirmation or use smaller position size"
                )
        
        # Market regime recommendations
        regime = market_summary.get("market_regime", "neutral")
        if regime == "oversold":
            recommendations.append(
                "Market showing oversold conditions - Consider gradual accumulation strategy"
            )
        elif regime == "overbought":
            recommendations.append(
                "Market showing overbought conditions - Consider profit-taking or hedging"
            )
        
        # Risk management recommendations
        if len(opportunities) > 1:
            recommendations.append(
                "Multiple regression opportunities detected - Diversify entries and manage correlation risk"
            )
        
        return recommendations
