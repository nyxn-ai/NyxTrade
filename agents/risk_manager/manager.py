"""
Risk Manager Agent
Manages portfolio risk, position sizing, and risk assessment
Integrates with Google ADK and multi-agent verification system
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from utils.a2a_coordinator import A2ACoordinator, A2AMessage, MessageType
from utils.database import DatabaseManager


class RiskManager:
    """
    Risk Management Agent with ADK integration and multi-agent verification
    """
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager, coordinator: A2ACoordinator):
        self.config = config
        self.db_manager = db_manager
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        self.agent_id = "risk_manager"
        self.check_interval = config.get('check_interval', 30)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.var_confidence = config.get('var_confidence', 0.95)
        
        # ADK Agent for enhanced risk analysis
        self.adk_agent = None
        self.running = False
        
        # Risk metrics
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.current_positions = {}
        
    async def initialize(self):
        """Initialize the risk manager agent with ADK integration"""
        try:
            # Initialize ADK Agent
            self.adk_agent = LlmAgent(
                name="risk_management_agent",
                model="gemini-2.0-flash",
                instruction="""You are an expert risk management specialist for cryptocurrency trading. Your responsibilities include:
                
                1. Portfolio Risk Assessment: Analyze overall portfolio risk and exposure
                2. Position Sizing: Determine optimal position sizes based on risk tolerance
                3. Correlation Analysis: Identify correlations between assets
                4. Stress Testing: Evaluate portfolio performance under adverse scenarios
                5. Risk Monitoring: Continuously monitor risk metrics and alert on breaches
                
                Key principles:
                - Capital preservation is paramount
                - Risk-adjusted returns over absolute returns
                - Diversification and correlation management
                - Dynamic risk adjustment based on market conditions
                - Clear risk limits and stop-loss mechanisms
                
                Always provide:
                - Quantitative risk metrics (VaR, Sharpe ratio, max drawdown)
                - Specific risk recommendations
                - Position sizing suggestions
                - Risk limit breach alerts
                - Scenario analysis results""",
                description="Advanced cryptocurrency portfolio risk management and assessment",
                tools=[google_search]
            )
            
            # Register with A2A coordinator
            await self.coordinator.register_adk_agent(
                self.agent_id,
                self.adk_agent,
                ['risk_assessment', 'position_sizing', 'portfolio_analysis', 'stress_testing']
            )
            
            # Register message handler
            self.coordinator.register_message_handler(self.agent_id, self._handle_message)
            
            self.logger.info("Risk Manager with ADK integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Manager: {e}")
            raise
    
    async def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                # Update risk assessment based on new market data
                await self._update_risk_assessment(message.payload)
            elif message.message_type == MessageType.TRADE_SIGNAL:
                # Evaluate trade signal from risk perspective
                risk_evaluation = await self._evaluate_trade_risk(message.payload)
                await self._send_risk_evaluation(message.sender, risk_evaluation)
            elif message.message_type == MessageType.SYSTEM_STATUS:
                payload = message.payload
                if payload.get('action') == 'emergency_stop':
                    await self._handle_emergency_stop()
                    
            await self.coordinator.update_heartbeat(self.agent_id)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment with verification"""
        try:
            # Get current portfolio data
            portfolio_data = await self._get_portfolio_data()
            
            # Calculate traditional risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_data)
            
            # Enhanced ADK risk analysis
            adk_risk_analysis = await self._perform_adk_risk_analysis(portfolio_data, risk_metrics)
            
            # Combine analyses
            combined_assessment = {
                'timestamp': datetime.now().isoformat(),
                'type': 'risk_assessment',
                'portfolio_data': portfolio_data,
                'risk_metrics': risk_metrics,
                'adk_analysis': adk_risk_analysis,
                'preliminary_risk_level': self._determine_risk_level(risk_metrics)
            }
            
            # Multi-agent verification and reflection
            verified_result = await self.coordinator.verify_and_reflect(combined_assessment)
            
            # Final risk assessment
            final_assessment = {
                **combined_assessment,
                'verification': verified_result,
                'final_risk_level': verified_result.get('final_recommendation', 'MEDIUM'),
                'confidence': verified_result.get('confidence', 0.5),
                'risk_alerts': self._generate_risk_alerts(risk_metrics, verified_result)
            }
            
            # Store assessment
            await self._store_risk_assessment(final_assessment)
            
            return final_assessment
            
        except Exception as e:
            self.logger.error(f"Error in portfolio risk assessment: {e}")
            return {'error': str(e), 'risk_level': 'HIGH'}  # Conservative default
    
    async def _perform_adk_risk_analysis(self, portfolio_data: Dict[str, Any], risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced risk analysis using ADK agent"""
        try:
            risk_prompt = f"""
            Analyze the following cryptocurrency portfolio risk:
            
            Portfolio Data:
            {json.dumps(portfolio_data, indent=2)}
            
            Risk Metrics:
            {json.dumps(risk_metrics, indent=2)}
            
            Please provide comprehensive risk analysis including:
            1. Overall portfolio risk level (LOW/MEDIUM/HIGH/CRITICAL)
            2. Key risk factors and vulnerabilities
            3. Correlation risks between assets
            4. Market risk exposure
            5. Liquidity risks
            6. Recommended position adjustments
            7. Stress test scenarios
            8. Risk-adjusted return optimization suggestions
            
            Also search for current market conditions that might affect portfolio risk.
            """
            
            adk_response = await self.adk_agent.run(risk_prompt)
            
            return {
                'adk_risk_insight': adk_response,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-2.0-flash'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ADK risk analysis: {e}")
            return {
                'adk_risk_insight': f"ADK risk analysis failed: {str(e)}",
                'analysis_timestamp': datetime.now().isoformat(),
                'error': True
            }
    
    def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate traditional risk metrics"""
        try:
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 0)
            
            if not positions or total_value == 0:
                return {
                    'var_95': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'concentration_risk': 0.0,
                    'leverage_ratio': 0.0
                }
            
            # Calculate concentration risk
            position_weights = [pos['value'] / total_value for pos in positions.values()]
            concentration_risk = max(position_weights) if position_weights else 0.0
            
            # Simplified VaR calculation (in production, use historical simulation or Monte Carlo)
            daily_returns = portfolio_data.get('daily_returns', [])
            if len(daily_returns) >= 30:
                var_95 = np.percentile(daily_returns, 5)  # 95% VaR
            else:
                var_95 = -0.05  # Conservative estimate
            
            # Max drawdown calculation
            portfolio_values = portfolio_data.get('historical_values', [total_value])
            if len(portfolio_values) > 1:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0.0
            
            # Simplified Sharpe ratio
            if len(daily_returns) >= 30:
                excess_returns = np.array(daily_returns) - 0.0001  # Assume 0.01% daily risk-free rate
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            return {
                'var_95': float(var_95),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'concentration_risk': float(concentration_risk),
                'leverage_ratio': 0.0,  # Simplified - no leverage in this example
                'portfolio_volatility': float(np.std(daily_returns)) if daily_returns else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {
                'var_95': -0.1,  # Conservative
                'max_drawdown': -0.2,
                'sharpe_ratio': 0.0,
                'concentration_risk': 1.0,  # High risk
                'leverage_ratio': 0.0,
                'portfolio_volatility': 0.1
            }
    
    def _determine_risk_level(self, risk_metrics: Dict[str, Any]) -> str:
        """Determine overall risk level based on metrics"""
        var_95 = abs(risk_metrics.get('var_95', 0))
        max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
        concentration_risk = risk_metrics.get('concentration_risk', 0)
        
        # Risk scoring
        risk_score = 0
        
        if var_95 > 0.1:  # VaR > 10%
            risk_score += 2
        elif var_95 > 0.05:  # VaR > 5%
            risk_score += 1
        
        if max_drawdown > 0.2:  # Max drawdown > 20%
            risk_score += 2
        elif max_drawdown > 0.1:  # Max drawdown > 10%
            risk_score += 1
        
        if concentration_risk > 0.5:  # Single position > 50%
            risk_score += 2
        elif concentration_risk > 0.3:  # Single position > 30%
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            return 'CRITICAL'
        elif risk_score >= 3:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_risk_alerts(self, risk_metrics: Dict[str, Any], verification_result: Dict[str, Any]) -> List[str]:
        """Generate risk alerts based on metrics and verification"""
        alerts = []
        
        # Check risk limits
        if abs(risk_metrics.get('var_95', 0)) > self.max_daily_loss:
            alerts.append(f"VaR exceeds daily loss limit: {abs(risk_metrics.get('var_95', 0)):.2%}")
        
        if risk_metrics.get('concentration_risk', 0) > self.max_position_size:
            alerts.append(f"Position concentration too high: {risk_metrics.get('concentration_risk', 0):.2%}")
        
        # Check verification issues
        verification_results = verification_result.get('verification_results', {})
        for verifier, result in verification_results.items():
            if not result.get('verified', True):
                alerts.append(f"Verification failed: {verifier}")
        
        return alerts
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data"""
        try:
            # Get portfolio balance from database
            balance = await self.db_manager.get_portfolio_balance()
            
            # Calculate total value (simplified)
            total_value = sum(balance.values()) if balance else 0.0
            
            # Get historical data (simplified)
            return {
                'positions': {symbol: {'value': amount, 'weight': amount/total_value if total_value > 0 else 0} 
                             for symbol, amount in balance.items()},
                'total_value': total_value,
                'daily_returns': [],  # Would be populated from historical data
                'historical_values': [total_value]  # Would be populated from historical data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {'positions': {}, 'total_value': 0.0, 'daily_returns': [], 'historical_values': []}
    
    async def _store_risk_assessment(self, assessment: Dict[str, Any]):
        """Store risk assessment in database"""
        try:
            # Store in database (implementation depends on schema)
            pass
        except Exception as e:
            self.logger.error(f"Error storing risk assessment: {e}")
    
    async def run(self):
        """Main run loop for the risk manager"""
        self.running = True
        self.logger.info("Risk Manager started")
        
        while self.running:
            try:
                # Perform risk assessment
                risk_assessment = await self.assess_portfolio_risk()
                
                # Send risk update to other agents
                await self.coordinator.broadcast_message(
                    self.agent_id,
                    MessageType.RISK_ALERT,
                    risk_assessment
                )
                
                # Update heartbeat
                await self.coordinator.update_heartbeat(self.agent_id)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in risk manager run loop: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop the risk manager"""
        self.running = False
        await self.coordinator.unregister_agent(self.agent_id)
        self.logger.info("Risk Manager stopped")
