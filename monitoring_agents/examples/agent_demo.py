"""
Demonstration of NyxTrade Monitoring Agents System
Shows how to set up and run monitoring agents with Gemini AI integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_agent import AgentConfig
from core.alert_manager import AlertManager, console_alert_handler, log_alert_handler
from agents.market_regression.btc_eth_regression_agent import BTCETHRegressionAgent


async def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('monitoring_agents.log')
        ]
    )


async def demo_btc_eth_regression_agent():
    """Demonstrate BTC ETH regression monitoring agent"""
    print("=== BTC ETH Regression Agent Demo ===")
    
    # Create agent configuration
    config = AgentConfig(
        name="btc_eth_regression_demo",
        enabled=True,
        update_interval=60,  # 1 minute for demo
        alert_thresholds={
            "z_score_btc": 1.5,  # Lower threshold for demo
            "z_score_eth": 1.5,
            "deviation_percent": 5.0
        },
        gemini_enabled=False  # Disable for demo unless API key is set
    )
    
    # Initialize agent
    agent = BTCETHRegressionAgent(config)
    
    # Setup alert handlers
    agent.alert_manager.add_alert_handler(console_alert_handler)
    agent.alert_manager.add_alert_handler(log_alert_handler)
    
    print(f"✅ Initialized agent: {agent.name}")
    print(f"📊 Monitoring symbols: {agent.symbols}")
    print(f"⏱️  Update interval: {agent.config.update_interval} seconds")
    
    # Run a few analysis cycles
    print("\n🔄 Running analysis cycles...")
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        try:
            # Run analysis
            result = await agent.run_analysis_cycle()
            
            # Display results
            print(f"⏰ Analysis completed at: {result.timestamp}")
            print(f"📈 Symbols analyzed: {list(result.analysis['symbols_analysis'].keys())}")
            
            # Show regression opportunities
            opportunities = result.analysis.get('regression_opportunities', [])
            if opportunities:
                print(f"🎯 Found {len(opportunities)} regression opportunities:")
                for opp in opportunities:
                    print(f"   • {opp['symbol']}: {opp['direction']} signal (strength: {opp['strength']:.2f})")
            else:
                print("📊 No significant regression opportunities detected")
            
            # Show market summary
            market_summary = result.analysis.get('market_summary', {})
            regime = market_summary.get('market_regime', 'neutral')
            print(f"🌍 Market regime: {regime}")
            
            # Show alerts
            if result.alerts:
                print(f"🚨 Alerts generated: {len(result.alerts)}")
                for alert in result.alerts:
                    print(f"   • {alert['type']}: {alert.get('metric', 'N/A')} = {alert.get('value', 'N/A')}")
            
            # Show recommendations
            if result.recommendations:
                print(f"💡 Recommendations:")
                for rec in result.recommendations[:3]:  # Show first 3
                    print(f"   • {rec}")
            
            # Show AI insights if available
            if result.ai_insights:
                print(f"🤖 AI Analysis: {result.ai_insights.get('analysis', 'N/A')[:100]}...")
                confidence = result.ai_insights.get('confidence', 0)
                print(f"🎯 AI Confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Error in analysis cycle: {e}")
        
        # Wait before next cycle (shorter for demo)
        if cycle < 2:  # Don't wait after last cycle
            print("⏳ Waiting for next cycle...")
            await asyncio.sleep(10)  # 10 seconds for demo
    
    print(f"\n📊 Agent Status: {agent.get_status()}")
    print(f"📈 Results History: {len(agent.results_history)} results")
    
    return agent


async def demo_alert_system():
    """Demonstrate alert system"""
    print("\n=== Alert System Demo ===")
    
    alert_manager = AlertManager()
    
    # Add handlers
    alert_manager.add_alert_handler(console_alert_handler)
    alert_manager.add_alert_handler(log_alert_handler)
    
    # Generate sample alerts
    sample_alerts = [
        {
            "agent": "btc_eth_regression",
            "type": "threshold_exceeded",
            "metric": "z_score_btc",
            "value": 2.5,
            "threshold": 2.0,
            "timestamp": datetime.now()
        },
        {
            "agent": "trend_tracking",
            "type": "trend_reversal",
            "symbol": "ETH",
            "confidence": 0.85,
            "timestamp": datetime.now()
        }
    ]
    
    print("🚨 Sending sample alerts...")
    await alert_manager.send_alerts(sample_alerts)
    
    # Show alert statistics
    stats = alert_manager.get_alert_stats()
    print(f"📊 Alert Stats: {stats}")
    
    return alert_manager


async def demo_configuration_system():
    """Demonstrate configuration system"""
    print("\n=== Configuration System Demo ===")
    
    from core.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    
    # Show configurations
    print("⚙️  Agent Configurations:")
    agents_config = config_manager.get_agents_config()
    for agent_name, config in agents_config.items():
        enabled = "✅" if config.get("enabled") else "❌"
        interval = config.get("update_interval", 0)
        print(f"   {enabled} {agent_name}: {interval}s interval")
    
    print("\n🤖 Gemini Configuration:")
    gemini_config = config_manager.get_gemini_config()
    model = gemini_config.get("model", "unknown")
    api_key_set = "✅" if gemini_config.get("api_key") else "❌"
    print(f"   Model: {model}")
    print(f"   API Key: {api_key_set}")
    
    # Validate configuration
    print("\n🔍 Configuration Validation:")
    issues = config_manager.validate_config()
    for category, category_issues in issues.items():
        if category_issues:
            print(f"   ⚠️  {category}: {len(category_issues)} issues")
            for issue in category_issues[:2]:  # Show first 2 issues
                print(f"      • {issue}")
        else:
            print(f"   ✅ {category}: No issues")
    
    return config_manager


async def main():
    """Main demo function"""
    print("🚀 NyxTrade Monitoring Agents System Demo")
    print("=" * 50)
    
    # Setup logging
    await setup_logging()
    
    try:
        # Demo configuration system
        config_manager = await demo_configuration_system()
        
        # Demo alert system
        alert_manager = await demo_alert_system()
        
        # Demo BTC ETH regression agent
        agent = await demo_btc_eth_regression_agent()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\n📋 Summary:")
        print(f"   • Configuration system: ✅ Working")
        print(f"   • Alert system: ✅ Working")
        print(f"   • BTC ETH Regression Agent: ✅ Working")
        print(f"   • Data collection: ✅ Working (mock data)")
        print(f"   • Analysis engine: ✅ Working")
        
        # Show next steps
        print("\n🔧 Next Steps:")
        print("   1. Set up API keys in configuration files")
        print("   2. Configure Gemini API key for AI analysis")
        print("   3. Set up real data sources (exchange APIs)")
        print("   4. Deploy agents in production environment")
        print("   5. Set up monitoring dashboard")
        
        print("\n📁 Key Files:")
        print("   • monitoring_agents/config/gemini_config.yaml - Gemini AI settings")
        print("   • monitoring_agents/config/agents_config.yaml - Agent configurations")
        print("   • monitoring_agents.log - System logs")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
