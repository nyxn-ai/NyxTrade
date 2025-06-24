"""
Arbitrage Hunter Agent
Detects and executes arbitrage opportunities across exchanges and DEX/CEX
Integrates with Google ADK and multi-agent verification system
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from utils.a2a_coordinator import A2ACoordinator, A2AMessage, MessageType
from utils.database import DatabaseManager
from data.collectors.price_collector import PriceCollector


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data structure"""
    symbol: str
    exchange_a: str
    exchange_b: str
    price_a: float
    price_b: float
    profit_percentage: float
    profit_amount: float
    volume_available: float
    execution_time_estimate: float
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "price_a": self.price_a,
            "price_b": self.price_b,
            "profit_percentage": self.profit_percentage,
            "profit_amount": self.profit_amount,
            "volume_available": self.volume_available,
            "execution_time_estimate": self.execution_time_estimate,
            "confidence_score": self.confidence_score
        }


class ArbitrageHunter:
    """
    Arbitrage Hunter Agent with ADK integration and multi-agent verification
    """
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager, coordinator: A2ACoordinator):
        self.config = config
        self.db_manager = db_manager
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        self.agent_id = "arbitrage_hunter"
        self.scan_interval = config.get('scan_interval', 10)
        self.min_profit_threshold = config.get('min_profit_threshold', 0.005)  # 0.5%
        self.max_execution_time = config.get('max_execution_time', 30)  # seconds
        self.max_exposure = config.get('max_exposure', 1000)  # USD
        
        # ADK Agent for enhanced arbitrage analysis
        self.adk_agent = None
        self.price_collector = PriceCollector()
        self.running = False
        
        # Supported exchanges and trading pairs
        self.exchanges = ['binance', 'coinbase', 'kraken']
        self.trading_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
        
        # Arbitrage opportunities cache
        self.opportunities: List[ArbitrageOpportunity] = []
        
    async def initialize(self):
        """Initialize the arbitrage hunter agent with ADK integration"""
        try:
            # Initialize ADK Agent
            self.adk_agent = LlmAgent(
                name="arbitrage_analysis_agent",
                model="gemini-2.0-flash",
                instruction="""You are an expert arbitrage trading analyst for cryptocurrency markets. Your capabilities include:
                
                1. Cross-Exchange Analysis: Identify price differences between exchanges
                2. DEX-CEX Arbitrage: Find opportunities between decentralized and centralized exchanges
                3. Triangular Arbitrage: Detect multi-asset arbitrage opportunities
                4. Risk Assessment: Evaluate execution risks and timing constraints
                5. Profit Optimization: Calculate optimal trade sizes and execution sequences
                
                Key considerations:
                - Transaction fees and gas costs
                - Slippage impact on large orders
                - Execution time and market volatility
                - Liquidity availability across exchanges
                - Regulatory and technical constraints
                
                Always provide:
                - Detailed profit calculations including all costs
                - Risk assessment and mitigation strategies
                - Optimal execution timing and sequence
                - Confidence levels for opportunity viability
                - Alternative execution strategies""",
                description="Advanced cryptocurrency arbitrage opportunity detection and analysis",
                tools=[google_search]
            )
            
            # Register with A2A coordinator
            await self.coordinator.register_adk_agent(
                self.agent_id,
                self.adk_agent,
                ['arbitrage_detection', 'cross_exchange_analysis', 'profit_optimization', 'risk_assessment']
            )
            
            # Register message handler
            self.coordinator.register_message_handler(self.agent_id, self._handle_message)
            
            # Initialize price collector
            # In production, this would be configured with actual exchange APIs
            
            self.logger.info("Arbitrage Hunter with ADK integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Arbitrage Hunter: {e}")
            raise
    
    async def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                # Update arbitrage analysis based on new market data
                await self._update_arbitrage_analysis(message.payload)
            elif message.message_type == MessageType.SYSTEM_STATUS:
                payload = message.payload
                if payload.get('action') == 'emergency_stop':
                    await self._handle_emergency_stop()
                    
            await self.coordinator.update_heartbeat(self.agent_id)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            for symbol in self.trading_pairs:
                # Get prices from multiple exchanges
                exchange_prices = await self._get_multi_exchange_prices(symbol)
                
                if len(exchange_prices) < 2:
                    continue
                
                # Find arbitrage opportunities
                symbol_opportunities = await self._find_arbitrage_pairs(symbol, exchange_prices)
                opportunities.extend(symbol_opportunities)
            
            # Filter by minimum profit threshold
            filtered_opportunities = [
                opp for opp in opportunities 
                if opp.profit_percentage >= self.min_profit_threshold
            ]
            
            # Enhanced ADK analysis for top opportunities
            if filtered_opportunities:
                enhanced_opportunities = await self._enhance_opportunities_with_adk(filtered_opportunities)
                
                # Multi-agent verification for high-value opportunities
                verified_opportunities = []
                for opp in enhanced_opportunities:
                    if opp.profit_amount > 100:  # Verify opportunities > $100 profit
                        verified_opp = await self._verify_opportunity_with_agents(opp)
                        verified_opportunities.append(verified_opp)
                    else:
                        verified_opportunities.append(opp)
                
                self.opportunities = verified_opportunities
            else:
                self.opportunities = []
            
            return self.opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning arbitrage opportunities: {e}")
            return []
    
    async def _get_multi_exchange_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get current prices from multiple exchanges"""
        try:
            exchange_prices = {}
            
            for exchange in self.exchanges:
                try:
                    # Get current price and orderbook data
                    current_price = await self.price_collector.get_current_price(symbol, exchange)
                    orderbook = await self.price_collector.get_orderbook(symbol, exchange, limit=10)
                    
                    if current_price and orderbook:
                        # Calculate bid/ask spread and available volume
                        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else current_price
                        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else current_price
                        bid_volume = orderbook['bids'][0][1] if orderbook['bids'] else 0
                        ask_volume = orderbook['asks'][0][1] if orderbook['asks'] else 0
                        
                        exchange_prices[exchange] = {
                            'price': current_price,
                            'bid': best_bid,
                            'ask': best_ask,
                            'bid_volume': bid_volume,
                            'ask_volume': ask_volume,
                            'spread': (best_ask - best_bid) / best_bid if best_bid > 0 else 0
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get price from {exchange} for {symbol}: {e}")
                    continue
            
            return exchange_prices
            
        except Exception as e:
            self.logger.error(f"Error getting multi-exchange prices: {e}")
            return {}
    
    async def _find_arbitrage_pairs(self, symbol: str, exchange_prices: Dict[str, Dict[str, float]]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between exchange pairs"""
        opportunities = []
        
        try:
            exchanges = list(exchange_prices.keys())
            
            # Compare all exchange pairs
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange_a = exchanges[i]
                    exchange_b = exchanges[j]
                    
                    price_data_a = exchange_prices[exchange_a]
                    price_data_b = exchange_prices[exchange_b]
                    
                    # Calculate potential profit in both directions
                    opportunities.extend(
                        self._calculate_arbitrage_profit(symbol, exchange_a, exchange_b, price_data_a, price_data_b)
                    )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage pairs: {e}")
            return []
    
    def _calculate_arbitrage_profit(self, symbol: str, exchange_a: str, exchange_b: str,
                                  price_data_a: Dict[str, float], price_data_b: Dict[str, float]) -> List[ArbitrageOpportunity]:
        """Calculate arbitrage profit between two exchanges"""
        opportunities = []
        
        try:
            # Direction A -> B (buy on A, sell on B)
            buy_price = price_data_a['ask']  # Buy at ask price on A
            sell_price = price_data_b['bid']  # Sell at bid price on B
            
            if sell_price > buy_price:
                profit_percentage = (sell_price - buy_price) / buy_price
                
                # Calculate available volume (limited by both exchanges)
                available_volume = min(price_data_a['ask_volume'], price_data_b['bid_volume'])
                
                # Estimate execution time (simplified)
                execution_time = self._estimate_execution_time(exchange_a, exchange_b)
                
                # Calculate confidence score
                confidence = self._calculate_confidence_score(
                    profit_percentage, available_volume, execution_time,
                    price_data_a['spread'], price_data_b['spread']
                )
                
                opportunity = ArbitrageOpportunity(
                    symbol=symbol,
                    exchange_a=exchange_a,
                    exchange_b=exchange_b,
                    price_a=buy_price,
                    price_b=sell_price,
                    profit_percentage=profit_percentage,
                    profit_amount=profit_percentage * buy_price * min(available_volume, self.max_exposure / buy_price),
                    volume_available=available_volume,
                    execution_time_estimate=execution_time,
                    confidence_score=confidence
                )
                
                opportunities.append(opportunity)
            
            # Direction B -> A (buy on B, sell on A)
            buy_price = price_data_b['ask']
            sell_price = price_data_a['bid']
            
            if sell_price > buy_price:
                profit_percentage = (sell_price - buy_price) / buy_price
                available_volume = min(price_data_b['ask_volume'], price_data_a['bid_volume'])
                execution_time = self._estimate_execution_time(exchange_b, exchange_a)
                
                confidence = self._calculate_confidence_score(
                    profit_percentage, available_volume, execution_time,
                    price_data_b['spread'], price_data_a['spread']
                )
                
                opportunity = ArbitrageOpportunity(
                    symbol=symbol,
                    exchange_a=exchange_b,
                    exchange_b=exchange_a,
                    price_a=buy_price,
                    price_b=sell_price,
                    profit_percentage=profit_percentage,
                    profit_amount=profit_percentage * buy_price * min(available_volume, self.max_exposure / buy_price),
                    volume_available=available_volume,
                    execution_time_estimate=execution_time,
                    confidence_score=confidence
                )
                
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage profit: {e}")
            return []
    
    def _estimate_execution_time(self, exchange_a: str, exchange_b: str) -> float:
        """Estimate execution time for arbitrage between exchanges"""
        # Simplified execution time estimation
        base_time = 5.0  # Base execution time in seconds
        
        # Add delays based on exchange characteristics
        exchange_delays = {
            'binance': 1.0,
            'coinbase': 2.0,
            'kraken': 3.0
        }
        
        delay_a = exchange_delays.get(exchange_a, 2.0)
        delay_b = exchange_delays.get(exchange_b, 2.0)
        
        return base_time + delay_a + delay_b
    
    def _calculate_confidence_score(self, profit_percentage: float, volume: float, 
                                  execution_time: float, spread_a: float, spread_b: float) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        try:
            # Base confidence from profit margin
            profit_confidence = min(profit_percentage / 0.02, 1.0)  # Max confidence at 2% profit
            
            # Volume confidence (higher volume = higher confidence)
            volume_confidence = min(volume / 10.0, 1.0)  # Max confidence at 10 units
            
            # Time confidence (faster execution = higher confidence)
            time_confidence = max(0.1, 1.0 - (execution_time - 5.0) / 25.0)  # Decreases after 5 seconds
            
            # Spread confidence (tighter spreads = higher confidence)
            avg_spread = (spread_a + spread_b) / 2
            spread_confidence = max(0.1, 1.0 - avg_spread / 0.01)  # Decreases with wider spreads
            
            # Weighted average
            confidence = (
                profit_confidence * 0.4 +
                volume_confidence * 0.2 +
                time_confidence * 0.2 +
                spread_confidence * 0.2
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    async def _enhance_opportunities_with_adk(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Enhance opportunities with ADK agent analysis"""
        try:
            enhanced_opportunities = []
            
            for opp in opportunities[:5]:  # Analyze top 5 opportunities
                analysis_prompt = f"""
                Analyze this arbitrage opportunity:
                
                Symbol: {opp.symbol}
                Buy Exchange: {opp.exchange_a} at ${opp.price_a:.4f}
                Sell Exchange: {opp.exchange_b} at ${opp.price_b:.4f}
                Profit: {opp.profit_percentage:.2%} (${opp.profit_amount:.2f})
                Volume Available: {opp.volume_available:.4f}
                Execution Time: {opp.execution_time_estimate:.1f}s
                
                Please provide:
                1. Risk assessment and potential issues
                2. Optimal execution strategy
                3. Fee impact analysis
                4. Market timing considerations
                5. Confidence adjustment (0-100%)
                
                Search for current market conditions that might affect this arbitrage.
                """
                
                try:
                    adk_response = await self.adk_agent.run(analysis_prompt)
                    
                    # Parse confidence adjustment from response (simplified)
                    # In production, you'd use structured output
                    confidence_adjustment = 1.0  # Default no adjustment
                    
                    # Adjust opportunity confidence based on ADK analysis
                    enhanced_opp = ArbitrageOpportunity(
                        symbol=opp.symbol,
                        exchange_a=opp.exchange_a,
                        exchange_b=opp.exchange_b,
                        price_a=opp.price_a,
                        price_b=opp.price_b,
                        profit_percentage=opp.profit_percentage,
                        profit_amount=opp.profit_amount,
                        volume_available=opp.volume_available,
                        execution_time_estimate=opp.execution_time_estimate,
                        confidence_score=min(opp.confidence_score * confidence_adjustment, 1.0)
                    )
                    
                    enhanced_opportunities.append(enhanced_opp)
                    
                except Exception as e:
                    self.logger.warning(f"ADK analysis failed for opportunity: {e}")
                    enhanced_opportunities.append(opp)  # Use original if analysis fails
            
            # Add remaining opportunities without ADK analysis
            enhanced_opportunities.extend(opportunities[5:])
            
            return enhanced_opportunities
            
        except Exception as e:
            self.logger.error(f"Error enhancing opportunities with ADK: {e}")
            return opportunities
    
    async def _verify_opportunity_with_agents(self, opportunity: ArbitrageOpportunity) -> ArbitrageOpportunity:
        """Verify opportunity with multi-agent system"""
        try:
            verification_data = {
                'type': 'arbitrage_opportunity',
                'opportunity': opportunity.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Multi-agent verification
            verified_result = await self.coordinator.verify_and_reflect(verification_data)
            
            # Adjust confidence based on verification
            verification_confidence = verified_result.get('confidence', 0.5)
            adjusted_confidence = opportunity.confidence_score * verification_confidence
            
            # Create verified opportunity
            verified_opportunity = ArbitrageOpportunity(
                symbol=opportunity.symbol,
                exchange_a=opportunity.exchange_a,
                exchange_b=opportunity.exchange_b,
                price_a=opportunity.price_a,
                price_b=opportunity.price_b,
                profit_percentage=opportunity.profit_percentage,
                profit_amount=opportunity.profit_amount,
                volume_available=opportunity.volume_available,
                execution_time_estimate=opportunity.execution_time_estimate,
                confidence_score=adjusted_confidence
            )
            
            return verified_opportunity
            
        except Exception as e:
            self.logger.error(f"Error verifying opportunity: {e}")
            return opportunity
    
    async def get_best_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Get best arbitrage opportunities sorted by profit potential"""
        try:
            # Sort by profit amount and confidence score
            sorted_opportunities = sorted(
                self.opportunities,
                key=lambda x: x.profit_amount * x.confidence_score,
                reverse=True
            )
            
            return sorted_opportunities[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting best opportunities: {e}")
            return []
    
    async def run(self):
        """Main run loop for the arbitrage hunter"""
        self.running = True
        self.logger.info("Arbitrage Hunter started")
        
        while self.running:
            try:
                # Scan for arbitrage opportunities
                opportunities = await self.scan_arbitrage_opportunities()
                
                if opportunities:
                    self.logger.info(f"Found {len(opportunities)} arbitrage opportunities")
                    
                    # Send best opportunities to other agents
                    best_opportunities = await self.get_best_opportunities(5)
                    
                    for opp in best_opportunities:
                        await self.coordinator.broadcast_message(
                            self.agent_id,
                            MessageType.ARBITRAGE_OPPORTUNITY,
                            opp.to_dict()
                        )
                
                # Update heartbeat
                await self.coordinator.update_heartbeat(self.agent_id)
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in arbitrage hunter run loop: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop the arbitrage hunter"""
        self.running = False
        await self.coordinator.unregister_agent(self.agent_id)
        self.logger.info("Arbitrage Hunter stopped")
