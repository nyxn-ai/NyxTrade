"""
Portfolio Manager Agent
Manages asset allocation, rebalancing, and portfolio optimization
Integrates with Google ADK and multi-agent verification system
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from utils.a2a_coordinator import A2ACoordinator, A2AMessage, MessageType
from utils.database import DatabaseManager


@dataclass
class AssetAllocation:
    """Asset allocation data structure"""
    symbol: str
    target_weight: float
    current_weight: float
    current_value: float
    rebalance_amount: float
    rebalance_action: str  # 'buy', 'sell', 'hold'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "current_value": self.current_value,
            "rebalance_amount": self.rebalance_amount,
            "rebalance_action": self.rebalance_action
        }


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    daily_return: float
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_value": self.total_value,
            "daily_return": self.daily_return,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "beta": self.beta,
            "alpha": self.alpha
        }


class PortfolioManager:
    """
    Portfolio Management Agent with ADK integration and multi-agent verification
    """
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager, coordinator: A2ACoordinator):
        self.config = config
        self.db_manager = db_manager
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        self.agent_id = "portfolio_manager"
        self.rebalance_interval = config.get('rebalance_interval', 3600)  # 1 hour
        self.target_allocations = config.get('target_allocations', {
            'BTC': 0.4,
            'ETH': 0.3,
            'Others': 0.3
        })
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5%
        
        # ADK Agent for enhanced portfolio analysis
        self.adk_agent = None
        self.running = False
        
        # Portfolio state
        self.current_portfolio: Dict[str, float] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.last_rebalance_time = None
        
    async def initialize(self):
        """Initialize the portfolio manager agent with ADK integration"""
        try:
            # Initialize ADK Agent
            self.adk_agent = LlmAgent(
                name="portfolio_optimization_agent",
                model="gemini-2.0-flash",
                instruction="""You are an expert portfolio management specialist for cryptocurrency investments. Your capabilities include:
                
                1. Asset Allocation: Optimize portfolio allocation based on risk-return profiles
                2. Rebalancing Strategy: Determine optimal rebalancing timing and amounts
                3. Risk Management: Assess and manage portfolio-level risks
                4. Performance Analysis: Analyze portfolio performance and attribution
                5. Market Regime Detection: Adapt strategies based on market conditions
                
                Key principles:
                - Diversification and correlation management
                - Risk-adjusted return optimization
                - Dynamic allocation based on market conditions
                - Cost-efficient rebalancing strategies
                - Long-term wealth preservation and growth
                
                Always provide:
                - Specific allocation recommendations with rationale
                - Risk assessment and mitigation strategies
                - Performance attribution analysis
                - Rebalancing cost-benefit analysis
                - Market outlook and strategy adjustments""",
                description="Advanced cryptocurrency portfolio optimization and management",
                tools=[google_search]
            )
            
            # Register with A2A coordinator
            await self.coordinator.register_adk_agent(
                self.agent_id,
                self.adk_agent,
                ['asset_allocation', 'portfolio_optimization', 'rebalancing', 'performance_analysis']
            )
            
            # Register message handler
            self.coordinator.register_message_handler(self.agent_id, self._handle_message)
            
            # Load current portfolio
            await self._load_current_portfolio()
            
            self.logger.info("Portfolio Manager with ADK integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Portfolio Manager: {e}")
            raise
    
    async def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                # Update portfolio analysis based on new market data
                await self._update_portfolio_analysis(message.payload)
            elif message.message_type == MessageType.TRADE_SIGNAL:
                # Consider trade signal for portfolio optimization
                await self._evaluate_trade_signal(message.payload)
            elif message.message_type == MessageType.RISK_ALERT:
                # Adjust portfolio based on risk alerts
                await self._handle_risk_alert(message.payload)
            elif message.message_type == MessageType.SYSTEM_STATUS:
                payload = message.payload
                if payload.get('action') == 'emergency_stop':
                    await self._handle_emergency_stop()
                    
            await self.coordinator.update_heartbeat(self.agent_id)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def analyze_portfolio(self) -> Dict[str, Any]:
        """Comprehensive portfolio analysis with verification"""
        try:
            # Get current portfolio data
            portfolio_data = await self._get_portfolio_data()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(portfolio_data)
            
            # Analyze asset allocations
            allocations = self._analyze_asset_allocations(portfolio_data)
            
            # Enhanced ADK analysis
            adk_analysis = await self._perform_adk_portfolio_analysis(portfolio_data, metrics, allocations)
            
            # Combine analyses
            combined_analysis = {
                'timestamp': datetime.now().isoformat(),
                'type': 'portfolio_analysis',
                'portfolio_data': portfolio_data,
                'metrics': metrics.to_dict(),
                'allocations': [alloc.to_dict() for alloc in allocations],
                'adk_analysis': adk_analysis,
                'rebalancing_needed': self._check_rebalancing_needed(allocations)
            }
            
            # Multi-agent verification
            verified_result = await self.coordinator.verify_and_reflect(combined_analysis)
            
            # Final analysis with verification
            final_analysis = {
                **combined_analysis,
                'verification': verified_result,
                'confidence': verified_result.get('confidence', 0.5)
            }
            
            # Store analysis
            await self._store_portfolio_analysis(final_analysis)
            
            return final_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio: {e}")
            return {'error': str(e)}
    
    async def _perform_adk_portfolio_analysis(self, portfolio_data: Dict[str, Any], 
                                           metrics: PortfolioMetrics, 
                                           allocations: List[AssetAllocation]) -> Dict[str, Any]:
        """Perform enhanced portfolio analysis using ADK agent"""
        try:
            analysis_prompt = f"""
            Analyze this cryptocurrency portfolio:
            
            Portfolio Value: ${metrics.total_value:,.2f}
            Daily Return: {metrics.daily_return:.2%}
            Total Return: {metrics.total_return:.2%}
            Volatility: {metrics.volatility:.2%}
            Sharpe Ratio: {metrics.sharpe_ratio:.2f}
            Max Drawdown: {metrics.max_drawdown:.2%}
            
            Current Allocations:
            {chr(10).join([f"- {alloc.symbol}: {alloc.current_weight:.1%} (Target: {alloc.target_weight:.1%})" for alloc in allocations])}
            
            Please provide:
            1. Portfolio performance assessment
            2. Asset allocation optimization recommendations
            3. Risk analysis and mitigation strategies
            4. Rebalancing recommendations
            5. Market outlook impact on portfolio
            6. Diversification assessment
            
            Search for current crypto market trends that might affect portfolio strategy.
            """
            
            adk_response = await self.adk_agent.run(analysis_prompt)
            
            return {
                'adk_insight': adk_response,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-2.0-flash'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ADK portfolio analysis: {e}")
            return {
                'adk_insight': f"ADK portfolio analysis failed: {str(e)}",
                'analysis_timestamp': datetime.now().isoformat(),
                'error': True
            }
    
    def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any]) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        try:
            total_value = portfolio_data.get('total_value', 0.0)
            historical_values = portfolio_data.get('historical_values', [total_value])
            
            if len(historical_values) < 2:
                return PortfolioMetrics(
                    total_value=total_value,
                    daily_return=0.0,
                    total_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    beta=0.0,
                    alpha=0.0
                )
            
            # Calculate returns
            returns = np.diff(historical_values) / historical_values[:-1]
            daily_return = returns[-1] if len(returns) > 0 else 0.0
            total_return = (historical_values[-1] - historical_values[0]) / historical_values[0]
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(365) if len(returns) > 1 else 0.0
            
            # Calculate Sharpe ratio (assuming 0.01% daily risk-free rate)
            risk_free_rate = 0.0001
            excess_returns = returns - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if np.std(excess_returns) > 0 else 0.0
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(historical_values)
            drawdown = (historical_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Simplified beta and alpha (would need market benchmark data)
            beta = 1.0  # Placeholder
            alpha = 0.0  # Placeholder
            
            return PortfolioMetrics(
                total_value=total_value,
                daily_return=daily_return,
                total_return=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                alpha=alpha
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _analyze_asset_allocations(self, portfolio_data: Dict[str, Any]) -> List[AssetAllocation]:
        """Analyze current vs target asset allocations"""
        try:
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 0.0)
            
            allocations = []
            
            # Analyze each target allocation
            for symbol, target_weight in self.target_allocations.items():
                if symbol == 'Others':
                    # Calculate others as remaining allocation
                    current_value = sum(
                        pos['value'] for sym, pos in positions.items()
                        if sym not in self.target_allocations or sym == 'Others'
                    )
                else:
                    current_value = positions.get(symbol, {}).get('value', 0.0)
                
                current_weight = current_value / total_value if total_value > 0 else 0.0
                weight_diff = current_weight - target_weight
                
                # Calculate rebalancing needed
                if abs(weight_diff) > self.rebalance_threshold:
                    rebalance_amount = abs(weight_diff * total_value)
                    rebalance_action = 'sell' if weight_diff > 0 else 'buy'
                else:
                    rebalance_amount = 0.0
                    rebalance_action = 'hold'
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    current_value=current_value,
                    rebalance_amount=rebalance_amount,
                    rebalance_action=rebalance_action
                )
                
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error analyzing asset allocations: {e}")
            return []
    
    def _check_rebalancing_needed(self, allocations: List[AssetAllocation]) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            # Check if any allocation exceeds threshold
            for allocation in allocations:
                weight_diff = abs(allocation.current_weight - allocation.target_weight)
                if weight_diff > self.rebalance_threshold:
                    return True
            
            # Check time-based rebalancing
            if self.last_rebalance_time:
                time_since_rebalance = datetime.now() - self.last_rebalance_time
                if time_since_rebalance.total_seconds() > self.rebalance_interval:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rebalancing need: {e}")
            return False
    
    async def generate_rebalancing_plan(self) -> Dict[str, Any]:
        """Generate portfolio rebalancing plan"""
        try:
            # Analyze current portfolio
            analysis = await self.analyze_portfolio()
            
            if not analysis.get('rebalancing_needed', False):
                return {
                    'rebalancing_needed': False,
                    'message': 'Portfolio is within target allocations'
                }
            
            allocations = [AssetAllocation(**alloc) for alloc in analysis.get('allocations', [])]
            
            # Generate rebalancing trades
            rebalancing_trades = []
            
            for allocation in allocations:
                if allocation.rebalance_action != 'hold':
                    trade = {
                        'symbol': allocation.symbol,
                        'action': allocation.rebalance_action,
                        'amount': allocation.rebalance_amount,
                        'current_weight': allocation.current_weight,
                        'target_weight': allocation.target_weight,
                        'priority': abs(allocation.current_weight - allocation.target_weight)
                    }
                    rebalancing_trades.append(trade)
            
            # Sort by priority (largest deviations first)
            rebalancing_trades.sort(key=lambda x: x['priority'], reverse=True)
            
            # Enhanced ADK rebalancing analysis
            adk_rebalancing_analysis = await self._analyze_rebalancing_with_adk(rebalancing_trades)
            
            rebalancing_plan = {
                'rebalancing_needed': True,
                'trades': rebalancing_trades,
                'adk_analysis': adk_rebalancing_analysis,
                'estimated_cost': self._estimate_rebalancing_cost(rebalancing_trades),
                'execution_priority': 'high' if len(rebalancing_trades) > 2 else 'medium',
                'timestamp': datetime.now().isoformat()
            }
            
            return rebalancing_plan
            
        except Exception as e:
            self.logger.error(f"Error generating rebalancing plan: {e}")
            return {'error': str(e)}
    
    async def _analyze_rebalancing_with_adk(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze rebalancing plan with ADK agent"""
        try:
            if not trades:
                return {'message': 'No rebalancing trades needed'}
            
            rebalancing_prompt = f"""
            Analyze this portfolio rebalancing plan:
            
            Proposed Trades:
            {chr(10).join([f"- {trade['action'].upper()} {trade['symbol']}: ${trade['amount']:,.2f} (Current: {trade['current_weight']:.1%} → Target: {trade['target_weight']:.1%})" for trade in trades])}
            
            Please provide:
            1. Rebalancing strategy assessment
            2. Optimal execution sequence
            3. Market timing considerations
            4. Cost optimization recommendations
            5. Risk assessment of rebalancing
            6. Alternative rebalancing approaches
            
            Consider current market conditions and volatility.
            """
            
            adk_response = await self.adk_agent.run(rebalancing_prompt)
            
            return {
                'adk_rebalancing_insight': adk_response,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ADK rebalancing analysis: {e}")
            return {'error': str(e)}
    
    def _estimate_rebalancing_cost(self, trades: List[Dict[str, Any]]) -> float:
        """Estimate total cost of rebalancing"""
        try:
            total_cost = 0.0
            
            for trade in trades:
                # Estimate trading fees (simplified)
                trade_amount = trade['amount']
                estimated_fee = trade_amount * 0.001  # 0.1% fee estimate
                total_cost += estimated_fee
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Error estimating rebalancing cost: {e}")
            return 0.0
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data"""
        try:
            # Get portfolio balance from database
            balance = await self.db_manager.get_portfolio_balance()
            
            # Calculate total value
            total_value = sum(balance.values()) if balance else 0.0
            
            # Get historical data (simplified)
            return {
                'positions': {symbol: {'value': amount, 'weight': amount/total_value if total_value > 0 else 0} 
                             for symbol, amount in balance.items()},
                'total_value': total_value,
                'historical_values': [total_value]  # Would be populated from historical data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {'positions': {}, 'total_value': 0.0, 'historical_values': []}
    
    async def _load_current_portfolio(self):
        """Load current portfolio state"""
        try:
            portfolio_data = await self._get_portfolio_data()
            self.current_portfolio = portfolio_data.get('positions', {})
            
        except Exception as e:
            self.logger.error(f"Error loading current portfolio: {e}")
    
    async def _store_portfolio_analysis(self, analysis: Dict[str, Any]):
        """Store portfolio analysis in database"""
        try:
            # Store analysis in database (implementation depends on schema)
            pass
        except Exception as e:
            self.logger.error(f"Error storing portfolio analysis: {e}")
    
    async def run(self):
        """Main run loop for the portfolio manager"""
        self.running = True
        self.logger.info("Portfolio Manager started")
        
        while self.running:
            try:
                # Analyze portfolio
                analysis = await self.analyze_portfolio()
                
                # Check if rebalancing is needed
                if analysis.get('rebalancing_needed', False):
                    rebalancing_plan = await self.generate_rebalancing_plan()
                    
                    # Send rebalancing recommendations
                    await self.coordinator.broadcast_message(
                        self.agent_id,
                        MessageType.PORTFOLIO_UPDATE,
                        rebalancing_plan
                    )
                
                # Send portfolio update
                await self.coordinator.broadcast_message(
                    self.agent_id,
                    MessageType.PORTFOLIO_UPDATE,
                    analysis
                )
                
                # Update heartbeat
                await self.coordinator.update_heartbeat(self.agent_id)
                
                # Wait for next analysis
                await asyncio.sleep(self.rebalance_interval)
                
            except Exception as e:
                self.logger.error(f"Error in portfolio manager run loop: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop the portfolio manager"""
        self.running = False
        await self.coordinator.unregister_agent(self.agent_id)
        self.logger.info("Portfolio Manager stopped")
