#!/usr/bin/env python3
"""
Simple demonstration of NyxTrade Monitoring Agents System
This demo works without external API dependencies
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SimpleMonitoringAgent:
    """Simplified monitoring agent for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.results = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"agent.{name}")
    
    async def collect_mock_data(self) -> Dict[str, Any]:
        """Generate mock market data"""
        # Simulate BTC and ETH prices
        btc_price = 45000 + np.random.normal(0, 2000)
        eth_price = 3000 + np.random.normal(0, 200)
        
        # Generate historical data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        btc_prices = 45000 + np.cumsum(np.random.normal(0, 500, 100))
        eth_prices = 3000 + np.cumsum(np.random.normal(0, 50, 100))
        
        return {
            "timestamp": datetime.now(),
            "btc": {
                "current_price": btc_price,
                "historical_prices": btc_prices.tolist(),
                "volume_24h": np.random.uniform(20000, 50000)
            },
            "eth": {
                "current_price": eth_price,
                "historical_prices": eth_prices.tolist(),
                "volume_24h": np.random.uniform(10000, 30000)
            },
            "market": {
                "fear_greed_index": np.random.randint(20, 80),
                "total_market_cap": 2.5e12,
                "btc_dominance": 45.0
            }
        }
    
    async def analyze_regression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mean reversion opportunities"""
        analysis = {
            "timestamp": datetime.now(),
            "opportunities": [],
            "market_regime": "neutral"
        }
        
        for symbol in ["btc", "eth"]:
            if symbol not in data:
                continue
            
            current_price = data[symbol]["current_price"]
            historical = np.array(data[symbol]["historical_prices"])
            
            # Calculate regression metrics
            mean_price = np.mean(historical)
            std_price = np.std(historical)
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            # Calculate moving averages
            ma_20 = np.mean(historical[-20:])
            ma_50 = np.mean(historical[-50:])
            
            # Determine signal
            signal = "hold"
            confidence = 0.5
            
            if z_score < -2.0:
                signal = "buy"
                confidence = min(abs(z_score) / 3.0, 1.0)
            elif z_score > 2.0:
                signal = "sell"
                confidence = min(abs(z_score) / 3.0, 1.0)
            
            symbol_analysis = {
                "symbol": symbol.upper(),
                "current_price": current_price,
                "mean_price": mean_price,
                "z_score": z_score,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "signal": signal,
                "confidence": confidence,
                "deviation_percent": ((current_price - mean_price) / mean_price) * 100
            }
            
            analysis[symbol] = symbol_analysis
            
            # Add to opportunities if significant
            if abs(z_score) > 1.5:
                analysis["opportunities"].append({
                    "symbol": symbol.upper(),
                    "type": "mean_reversion",
                    "signal": signal,
                    "strength": abs(z_score),
                    "confidence": confidence
                })
        
        # Determine market regime
        avg_z_score = np.mean([analysis.get("btc", {}).get("z_score", 0),
                              analysis.get("eth", {}).get("z_score", 0)])
        
        if avg_z_score < -1.0:
            analysis["market_regime"] = "oversold"
        elif avg_z_score > 1.0:
            analysis["market_regime"] = "overbought"
        
        return analysis
    
    async def generate_ai_insights(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate AI analysis (mock Gemini response)"""
        opportunities = analysis.get("opportunities", [])
        market_regime = analysis.get("market_regime", "neutral")
        
        # Simulate AI analysis
        if opportunities:
            main_opportunity = opportunities[0]
            analysis_text = f"Detected {main_opportunity['signal']} opportunity for {main_opportunity['symbol']} with {main_opportunity['confidence']:.2f} confidence."
        else:
            analysis_text = f"Market is in {market_regime} regime with no significant mean reversion opportunities."
        
        recommendations = []
        
        if market_regime == "oversold":
            recommendations.extend([
                "Consider gradual accumulation strategy",
                "Monitor for trend reversal confirmation",
                "Use smaller position sizes due to volatility"
            ])
        elif market_regime == "overbought":
            recommendations.extend([
                "Consider profit-taking opportunities",
                "Monitor for distribution patterns",
                "Prepare for potential correction"
            ])
        else:
            recommendations.extend([
                "Maintain current positions",
                "Wait for clearer signals",
                "Focus on risk management"
            ])
        
        return {
            "analysis": analysis_text,
            "market_sentiment": "bearish" if market_regime == "oversold" else "bullish" if market_regime == "overbought" else "neutral",
            "confidence": 0.75,
            "recommendations": recommendations,
            "risk_level": "medium",
            "timeframe": "short-term"
        }
    
    async def run_analysis_cycle(self) -> Dict[str, Any]:
        """Run complete analysis cycle"""
        try:
            self.logger.info(f"Starting analysis cycle for {self.name}")
            
            # Collect data
            data = await self.collect_mock_data()
            
            # Analyze data
            analysis = await self.analyze_regression(data)
            
            # Get AI insights
            ai_insights = await self.generate_ai_insights(data, analysis)
            
            # Create result
            result = {
                "agent": self.name,
                "timestamp": datetime.now(),
                "data_summary": {
                    "btc_price": data["btc"]["current_price"],
                    "eth_price": data["eth"]["current_price"],
                    "fear_greed": data["market"]["fear_greed_index"]
                },
                "analysis": analysis,
                "ai_insights": ai_insights,
                "alerts": self._check_alerts(analysis)
            }
            
            self.results.append(result)
            self.logger.info(f"Analysis cycle completed for {self.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in analysis cycle: {e}")
            raise
    
    def _check_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        for symbol in ["btc", "eth"]:
            if symbol in analysis:
                symbol_data = analysis[symbol]
                z_score = symbol_data.get("z_score", 0)
                
                if abs(z_score) > 2.0:
                    alerts.append({
                        "type": "extreme_deviation",
                        "symbol": symbol.upper(),
                        "z_score": z_score,
                        "severity": "high" if abs(z_score) > 2.5 else "medium",
                        "message": f"{symbol.upper()} showing extreme deviation: Z-score {z_score:.2f}"
                    })
        
        return alerts


async def demo_simple_monitoring():
    """Demonstrate simple monitoring system"""
    print("🚀 NyxTrade Monitoring Agents - Simple Demo")
    print("=" * 50)
    
    # Create agent
    agent = SimpleMonitoringAgent("btc_eth_regression_demo")
    
    print(f"✅ Created agent: {agent.name}")
    print("🔄 Running analysis cycles...\n")
    
    # Run multiple cycles
    for cycle in range(3):
        print(f"--- Cycle {cycle + 1} ---")
        
        try:
            result = await agent.run_analysis_cycle()
            
            # Display results
            print(f"⏰ Time: {result['timestamp'].strftime('%H:%M:%S')}")
            
            # Data summary
            data_summary = result["data_summary"]
            print(f"📊 BTC: ${data_summary['btc_price']:,.0f}")
            print(f"📊 ETH: ${data_summary['eth_price']:,.0f}")
            print(f"😨 Fear & Greed: {data_summary['fear_greed']}")
            
            # Analysis results
            analysis = result["analysis"]
            opportunities = analysis.get("opportunities", [])
            
            if opportunities:
                print(f"🎯 Opportunities found: {len(opportunities)}")
                for opp in opportunities:
                    print(f"   • {opp['symbol']}: {opp['signal']} (confidence: {opp['confidence']:.2f})")
            else:
                print("📈 No significant opportunities")
            
            print(f"🌍 Market regime: {analysis['market_regime']}")
            
            # AI insights
            ai_insights = result["ai_insights"]
            print(f"🤖 AI Analysis: {ai_insights['analysis']}")
            print(f"💡 Top recommendation: {ai_insights['recommendations'][0]}")
            
            # Alerts
            alerts = result["alerts"]
            if alerts:
                print(f"🚨 Alerts: {len(alerts)}")
                for alert in alerts:
                    print(f"   • {alert['message']}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error in cycle {cycle + 1}: {e}")
        
        # Wait between cycles
        if cycle < 2:
            await asyncio.sleep(2)
    
    print("📊 Demo Summary:")
    print(f"   • Total cycles: {len(agent.results)}")
    print(f"   • Agent: {agent.name}")
    print(f"   • Status: ✅ Working")
    
    return agent


async def demo_configuration():
    """Demonstrate configuration system"""
    print("\n🔧 Configuration Demo")
    print("=" * 30)
    
    # Mock configuration
    config = {
        "agents": {
            "market_regression": {
                "enabled": True,
                "update_interval": 300,
                "symbols": ["BTC", "ETH"],
                "alert_thresholds": {
                    "z_score": 2.0,
                    "deviation_percent": 10.0
                }
            }
        },
        "gemini": {
            "model": "gemini-pro",
            "temperature": 0.7,
            "api_key_configured": False
        }
    }
    
    print("⚙️  Agent Configuration:")
    agent_config = config["agents"]["market_regression"]
    print(f"   • Enabled: {'✅' if agent_config['enabled'] else '❌'}")
    print(f"   • Update interval: {agent_config['update_interval']}s")
    print(f"   • Symbols: {', '.join(agent_config['symbols'])}")
    print(f"   • Z-score threshold: {agent_config['alert_thresholds']['z_score']}")
    
    print("\n🤖 Gemini Configuration:")
    gemini_config = config["gemini"]
    print(f"   • Model: {gemini_config['model']}")
    print(f"   • Temperature: {gemini_config['temperature']}")
    print(f"   • API Key: {'✅' if gemini_config['api_key_configured'] else '❌ Not configured'}")


async def main():
    """Main demo function"""
    try:
        # Run simple monitoring demo
        agent = await demo_simple_monitoring()
        
        # Show configuration
        await demo_configuration()
        
        print("\n" + "=" * 50)
        print("✅ Simple demo completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Set up virtual environment: chmod +x setup.sh && ./setup.sh")
        print("2. Configure API keys in .env file")
        print("3. Run full demo: python examples/agent_demo.py")
        print("4. Start production: python run_agents.py")
        
        print("\n🔧 Virtual Environment Setup:")
        print("   cd monitoring_agents")
        print("   chmod +x setup.sh")
        print("   ./setup.sh")
        print("   ./activate.sh")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
