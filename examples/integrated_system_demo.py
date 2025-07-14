"""
NyxTrade Integrated System Demo
Demonstrates how all modules work together in a coordinated trading system
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strategy Engine imports
from strategy_engine.ai_interface.agent_strategy_manager import AgentStrategyManager
from strategy_engine.examples.moving_average_strategy import MovingAverageStrategy
from strategy_engine.examples.rsi_strategy import RSIStrategy
from strategy_engine.monitoring.strategy_monitor import StrategyMonitor, AlertConfig

# Monitoring Agents imports
from monitoring_agents.agents.market_regression.btc_eth_regression_agent import BTCETHRegressionAgent
from monitoring_agents.core.base_agent import AgentConfig


class IntegratedTradingSystem:
    """
    Integrated trading system that combines all NyxTrade modules
    """
    
    def __init__(self):
        # Initialize strategy management
        self.strategy_manager = AgentStrategyManager()
        
        # Initialize monitoring agents
        self.market_agent = BTCETHRegressionAgent()
        
        # Initialize strategy monitoring
        self.strategy_monitor = StrategyMonitor(AlertConfig(
            max_drawdown_threshold=-5.0,
            min_win_rate_threshold=45.0
        ))
        
        # Active strategies
        self.active_strategies = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("integrated_system")
        
        self.logger.info("Initialized Integrated Trading System")
    
    async def initialize_system(self):
        """Initialize the complete trading system"""
        self.logger.info("🚀 Initializing Integrated Trading System")
        
        # 1. Register strategy classes
        self.strategy_manager.register_strategy_class(MovingAverageStrategy, "MovingAverage")
        self.strategy_manager.register_strategy_class(RSIStrategy, "RSI")
        
        # 2. Setup alert handlers for coordination
        self.strategy_monitor.add_alert_callback(self._handle_strategy_alert)
        
        # 3. Create initial strategies
        await self._create_initial_strategies()
        
        self.logger.info("✅ System initialization completed")
    
    async def _create_initial_strategies(self):
        """Create initial trading strategies"""
        # Create BTC strategy
        btc_strategy_id = self.strategy_manager.create_strategy_instance(
            "MovingAverage",
            "BTC_MA_Strategy",
            "BTCUSDT",
            parameters={
                "fast_period": 10,
                "slow_period": 30,
                "risk_per_trade": 0.02
            }
        )
        
        # Create ETH strategy  
        eth_strategy_id = self.strategy_manager.create_strategy_instance(
            "RSI",
            "ETH_RSI_Strategy", 
            "ETHUSDT",
            parameters={
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            }
        )
        
        self.active_strategies["BTC"] = btc_strategy_id
        self.active_strategies["ETH"] = eth_strategy_id
        
        # Add strategies to monitoring
        for symbol, strategy_id in self.active_strategies.items():
            strategy = self.strategy_manager.active_strategies[strategy_id]
            self.strategy_monitor.add_strategy(strategy_id, strategy)
        
        self.logger.info(f"Created {len(self.active_strategies)} initial strategies")
    
    async def run_market_analysis_cycle(self):
        """Run market analysis and update strategies accordingly"""
        self.logger.info("📊 Running market analysis cycle")
        
        # 1. Get market insights from monitoring agents
        market_result = await self.market_agent.run_analysis_cycle()
        
        # 2. Process market signals
        await self._process_market_signals(market_result)
        
        # 3. Update strategy parameters based on market conditions
        await self._update_strategies_based_on_market(market_result)
        
        # 4. Run strategy performance analysis
        await self._analyze_strategy_performance()
        
        return market_result
    
    async def _process_market_signals(self, market_result):
        """Process signals from market monitoring agents"""
        opportunities = market_result.analysis.get('regression_opportunities', [])
        market_regime = market_result.analysis.get('market_summary', {}).get('market_regime', 'neutral')
        
        self.logger.info(f"Market regime: {market_regime}, Opportunities: {len(opportunities)}")
        
        # Create new strategies based on strong signals
        for opp in opportunities:
            if opp['strength'] > 0.8:  # High confidence opportunity
                await self._create_opportunity_strategy(opp)
        
        # Adjust existing strategies based on market regime
        await self._adjust_strategies_for_regime(market_regime)
    
    async def _create_opportunity_strategy(self, opportunity):
        """Create new strategy based on market opportunity"""
        symbol = opportunity['symbol']
        direction = opportunity['direction']
        strength = opportunity['strength']
        
        strategy_name = f"{symbol}_Opportunity_{direction.upper()}"
        
        # Choose strategy type based on opportunity
        if direction == "buy":
            strategy_id = self.strategy_manager.create_strategy_instance(
                "RSI",
                strategy_name,
                f"{symbol}USDT",
                parameters={
                    "rsi_period": 14,
                    "oversold_threshold": 25,  # More aggressive for opportunities
                    "risk_per_trade": min(0.03, strength * 0.04)  # Risk based on strength
                }
            )
        else:
            strategy_id = self.strategy_manager.create_strategy_instance(
                "MovingAverage", 
                strategy_name,
                f"{symbol}USDT",
                parameters={
                    "fast_period": 8,
                    "slow_period": 20,
                    "risk_per_trade": min(0.02, strength * 0.03)
                }
            )
        
        # Add to monitoring
        strategy = self.strategy_manager.active_strategies[strategy_id]
        self.strategy_monitor.add_strategy(strategy_id, strategy)
        
        self.logger.info(f"Created opportunity strategy: {strategy_name} (strength: {strength:.2f})")
    
    async def _adjust_strategies_for_regime(self, market_regime):
        """Adjust existing strategies based on market regime"""
        if market_regime == "oversold":
            # More aggressive buying in oversold market
            adjustments = {
                "risk_per_trade": 0.025,
                "min_confidence": 0.5
            }
        elif market_regime == "overbought":
            # More conservative in overbought market
            adjustments = {
                "risk_per_trade": 0.015,
                "min_confidence": 0.7
            }
        else:
            # Neutral market - standard parameters
            adjustments = {
                "risk_per_trade": 0.02,
                "min_confidence": 0.6
            }
        
        # Apply adjustments to all active strategies
        for strategy_id in self.active_strategies.values():
            self.strategy_manager.update_strategy_parameters(strategy_id, adjustments)
        
        self.logger.info(f"Adjusted strategies for {market_regime} market regime")
    
    async def _analyze_strategy_performance(self):
        """Analyze performance of all strategies"""
        performance_summary = {}
        
        for symbol, strategy_id in self.active_strategies.items():
            performance = self.strategy_manager.get_strategy_performance(strategy_id)
            performance_summary[symbol] = performance
            
            # Log key metrics
            strategy_state = performance['strategy_state']
            metrics = performance['performance_metrics']
            
            self.logger.info(f"{symbol} Strategy - PnL: {strategy_state['current_pnl']:.2f}, "
                           f"Win Rate: {metrics['win_rate']:.1f}%, "
                           f"Total Trades: {metrics['total_trades']}")
        
        return performance_summary
    
    async def _handle_strategy_alert(self, strategy_id: str, alert: dict):
        """Handle alerts from strategy monitoring"""
        alert_type = alert['alert_type']
        
        if alert_type == 'MAX_DRAWDOWN':
            # Reduce risk on high drawdown
            self.strategy_manager.update_strategy_parameters(strategy_id, {
                "risk_per_trade": 0.01
            })
            self.logger.warning(f"Reduced risk for {strategy_id} due to drawdown")
            
        elif alert_type == 'LOW_WIN_RATE':
            # Increase confidence threshold for low win rate
            self.strategy_manager.update_strategy_parameters(strategy_id, {
                "min_confidence": 0.8
            })
            self.logger.warning(f"Increased confidence threshold for {strategy_id}")
        
        elif alert_type == 'CONSECUTIVE_LOSSES':
            # Temporarily pause strategy
            strategy = self.strategy_manager.active_strategies[strategy_id]
            strategy.is_active = False
            self.logger.warning(f"Paused {strategy_id} due to consecutive losses")
    
    async def run_integrated_backtest(self, market_data: pd.DataFrame):
        """Run integrated backtest using market analysis"""
        self.logger.info("🔄 Running integrated backtest")
        
        results = {}
        
        # Run backtest for each strategy
        for symbol, strategy_id in self.active_strategies.items():
            try:
                result = self.strategy_manager.run_strategy_backtest(
                    strategy_id,
                    market_data,
                    market_data.index[50],  # Skip first 50 periods
                    market_data.index[-1],
                    initial_capital=100000
                )
                
                results[symbol] = {
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades
                }
                
                self.logger.info(f"{symbol} Backtest - Return: {result.total_return:.2f}%, "
                               f"Sharpe: {result.sharpe_ratio:.2f}")
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    async def get_system_status(self):
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now(),
            "active_strategies": len(self.active_strategies),
            "monitoring_agents": {
                "market_regression": self.market_agent.get_status()
            },
            "strategy_monitor": self.strategy_monitor.get_all_strategies_summary(),
            "recent_performance": await self._analyze_strategy_performance()
        }


async def demo_integrated_system():
    """Demonstrate the integrated trading system"""
    print("🚀 NyxTrade Integrated System Demo")
    print("=" * 50)
    
    # Create and initialize system
    system = IntegratedTradingSystem()
    await system.initialize_system()
    
    # Generate sample market data
    print("📊 Generating sample market data...")
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    btc_prices = 45000 + np.cumsum(np.random.normal(0, 500, 200))
    eth_prices = 3000 + np.cumsum(np.random.normal(0, 50, 200))
    
    market_data = pd.DataFrame({
        'open': btc_prices,
        'high': btc_prices * 1.02,
        'low': btc_prices * 0.98,
        'close': btc_prices,
        'volume': np.random.uniform(1000, 10000, 200)
    }, index=dates)
    
    # Run market analysis cycles
    print("\n🔄 Running market analysis cycles...")
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        market_result = await system.run_market_analysis_cycle()
        
        # Show results
        opportunities = market_result.analysis.get('regression_opportunities', [])
        print(f"📈 Market opportunities: {len(opportunities)}")
        
        if opportunities:
            for opp in opportunities[:2]:  # Show first 2
                print(f"   • {opp['symbol']}: {opp['direction']} (strength: {opp['strength']:.2f})")
        
        await asyncio.sleep(1)  # Brief pause
    
    # Run integrated backtest
    print("\n📊 Running integrated backtest...")
    backtest_results = await system.run_integrated_backtest(market_data)
    
    print("\n📈 Backtest Results:")
    for symbol, result in backtest_results.items():
        if "error" not in result:
            print(f"   {symbol}: {result['total_return']:.2f}% return, "
                  f"{result['win_rate']:.1f}% win rate")
    
    # Show system status
    print("\n📊 System Status:")
    status = await system.get_system_status()
    print(f"   • Active strategies: {status['active_strategies']}")
    print(f"   • Monitoring agents: {len(status['monitoring_agents'])}")
    
    print("\n✅ Integrated system demo completed!")
    return system


if __name__ == "__main__":
    asyncio.run(demo_integrated_system())
