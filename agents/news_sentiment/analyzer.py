"""
News Sentiment Analyzer Agent
Analyzes news, social media, and market sentiment for trading insights
Integrates with Google ADK and multi-agent verification system
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from utils.a2a_coordinator import A2ACoordinator, A2AMessage, MessageType
from utils.database import DatabaseManager


@dataclass
class SentimentData:
    """Sentiment analysis data structure"""
    source: str
    symbol: str
    sentiment_score: float  # -1 (very negative) to 1 (very positive)
    confidence: float
    content: str
    timestamp: datetime
    impact_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source": self.source,
            "symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "timestamp": self.timestamp.isoformat(),
            "impact_level": self.impact_level
        }


@dataclass
class MarketSentiment:
    """Aggregated market sentiment"""
    symbol: str
    overall_sentiment: float
    confidence: float
    bullish_signals: int
    bearish_signals: int
    neutral_signals: int
    trending_topics: List[str]
    sentiment_sources: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "overall_sentiment": self.overall_sentiment,
            "confidence": self.confidence,
            "bullish_signals": self.bullish_signals,
            "bearish_signals": self.bearish_signals,
            "neutral_signals": self.neutral_signals,
            "trending_topics": self.trending_topics,
            "sentiment_sources": self.sentiment_sources
        }


class NewsSentimentAnalyzer:
    """
    News and Sentiment Analysis Agent with ADK integration
    """
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager, coordinator: A2ACoordinator):
        self.config = config
        self.db_manager = db_manager
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        self.agent_id = "news_sentiment"
        self.update_interval = config.get('update_interval', 300)  # 5 minutes
        self.sources = config.get('sources', ['twitter', 'reddit', 'news_api'])
        self.symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC']
        
        # ADK Agent for enhanced sentiment analysis
        self.adk_agent = None
        self.running = False
        
        # Sentiment data cache
        self.sentiment_cache: Dict[str, List[SentimentData]] = {}
        self.market_sentiment_cache: Dict[str, MarketSentiment] = {}
        
    async def initialize(self):
        """Initialize the news sentiment analyzer agent with ADK integration"""
        try:
            # Initialize ADK Agent
            self.adk_agent = LlmAgent(
                name="sentiment_analysis_agent",
                model="gemini-2.0-flash",
                instruction="""You are an expert cryptocurrency market sentiment analyst. Your capabilities include:
                
                1. News Analysis: Analyze cryptocurrency news for market impact and sentiment
                2. Social Media Sentiment: Interpret social media trends and sentiment
                3. Market Psychology: Understand crowd psychology and market emotions
                4. Event Impact Assessment: Evaluate how events affect market sentiment
                5. Sentiment Aggregation: Combine multiple sentiment sources for overall assessment
                
                Key focus areas:
                - Regulatory news and government announcements
                - Institutional adoption and investment flows
                - Technical developments and protocol updates
                - Market manipulation and whale activity
                - Macroeconomic factors affecting crypto markets
                
                Always provide:
                - Sentiment scores with confidence levels
                - Impact assessment (low/medium/high)
                - Key themes and trending topics
                - Potential market implications
                - Contrarian indicators and sentiment extremes""",
                description="Advanced cryptocurrency news and sentiment analysis",
                tools=[google_search]
            )
            
            # Register with A2A coordinator
            await self.coordinator.register_adk_agent(
                self.agent_id,
                self.adk_agent,
                ['sentiment_analysis', 'news_analysis', 'social_media_monitoring', 'market_psychology']
            )
            
            # Register message handler
            self.coordinator.register_message_handler(self.agent_id, self._handle_message)
            
            self.logger.info("News Sentiment Analyzer with ADK integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize News Sentiment Analyzer: {e}")
            raise
    
    async def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                # Update sentiment analysis based on market data
                await self._correlate_sentiment_with_market(message.payload)
            elif message.message_type == MessageType.SYSTEM_STATUS:
                payload = message.payload
                if payload.get('action') == 'emergency_stop':
                    await self._handle_emergency_stop()
                    
            await self.coordinator.update_heartbeat(self.agent_id)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def analyze_sentiment(self, symbol: str) -> MarketSentiment:
        """Analyze sentiment for a specific cryptocurrency"""
        try:
            # Collect sentiment data from various sources
            sentiment_data = await self._collect_sentiment_data(symbol)
            
            # Enhanced ADK sentiment analysis
            adk_sentiment = await self._perform_adk_sentiment_analysis(symbol, sentiment_data)
            
            # Aggregate sentiment scores
            aggregated_sentiment = self._aggregate_sentiment_scores(sentiment_data, adk_sentiment)
            
            # Create market sentiment object
            market_sentiment = MarketSentiment(
                symbol=symbol,
                overall_sentiment=aggregated_sentiment['overall_score'],
                confidence=aggregated_sentiment['confidence'],
                bullish_signals=aggregated_sentiment['bullish_count'],
                bearish_signals=aggregated_sentiment['bearish_count'],
                neutral_signals=aggregated_sentiment['neutral_count'],
                trending_topics=aggregated_sentiment['trending_topics'],
                sentiment_sources=self.sources
            )
            
            # Store in cache
            self.market_sentiment_cache[symbol] = market_sentiment
            
            # Multi-agent verification for significant sentiment changes
            if abs(market_sentiment.overall_sentiment) > 0.7:  # Strong sentiment
                verified_sentiment = await self._verify_sentiment_with_agents(market_sentiment)
                return verified_sentiment
            
            return market_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return MarketSentiment(symbol, 0.0, 0.0, 0, 0, 0, [], [])
    
    async def _collect_sentiment_data(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment data from various sources"""
        sentiment_data = []
        
        try:
            # News API sentiment (simulated)
            if 'news_api' in self.sources:
                news_sentiment = await self._analyze_news_sentiment(symbol)
                sentiment_data.extend(news_sentiment)
            
            # Twitter sentiment (simulated)
            if 'twitter' in self.sources:
                twitter_sentiment = await self._analyze_twitter_sentiment(symbol)
                sentiment_data.extend(twitter_sentiment)
            
            # Reddit sentiment (simulated)
            if 'reddit' in self.sources:
                reddit_sentiment = await self._analyze_reddit_sentiment(symbol)
                sentiment_data.extend(reddit_sentiment)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error collecting sentiment data: {e}")
            return []
    
    async def _analyze_news_sentiment(self, symbol: str) -> List[SentimentData]:
        """Analyze news sentiment (simulated implementation)"""
        try:
            # In production, this would integrate with actual news APIs
            # For demo, we'll simulate news sentiment data
            
            news_items = [
                {
                    "content": f"{symbol} shows strong institutional adoption with major investment announcements",
                    "sentiment": 0.7,
                    "confidence": 0.8,
                    "impact": "high"
                },
                {
                    "content": f"Regulatory clarity improves for {symbol} in major markets",
                    "sentiment": 0.5,
                    "confidence": 0.7,
                    "impact": "medium"
                },
                {
                    "content": f"Technical analysis suggests {symbol} consolidation phase",
                    "sentiment": 0.1,
                    "confidence": 0.6,
                    "impact": "low"
                }
            ]
            
            sentiment_data = []
            for item in news_items:
                sentiment = SentimentData(
                    source="news_api",
                    symbol=symbol,
                    sentiment_score=item["sentiment"],
                    confidence=item["confidence"],
                    content=item["content"],
                    timestamp=datetime.now(),
                    impact_level=item["impact"]
                )
                sentiment_data.append(sentiment)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return []
    
    async def _analyze_twitter_sentiment(self, symbol: str) -> List[SentimentData]:
        """Analyze Twitter sentiment (simulated implementation)"""
        try:
            # Simulated Twitter sentiment data
            twitter_data = [
                {
                    "content": f"#{symbol} to the moon! Great fundamentals and strong community support",
                    "sentiment": 0.8,
                    "confidence": 0.6,
                    "impact": "medium"
                },
                {
                    "content": f"Concerned about {symbol} recent price action, might be time to take profits",
                    "sentiment": -0.4,
                    "confidence": 0.5,
                    "impact": "low"
                },
                {
                    "content": f"{symbol} technical indicators looking bullish for next week",
                    "sentiment": 0.6,
                    "confidence": 0.7,
                    "impact": "medium"
                }
            ]
            
            sentiment_data = []
            for item in twitter_data:
                sentiment = SentimentData(
                    source="twitter",
                    symbol=symbol,
                    sentiment_score=item["sentiment"],
                    confidence=item["confidence"],
                    content=item["content"],
                    timestamp=datetime.now(),
                    impact_level=item["impact"]
                )
                sentiment_data.append(sentiment)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing Twitter sentiment: {e}")
            return []
    
    async def _analyze_reddit_sentiment(self, symbol: str) -> List[SentimentData]:
        """Analyze Reddit sentiment (simulated implementation)"""
        try:
            # Simulated Reddit sentiment data
            reddit_data = [
                {
                    "content": f"Deep dive analysis on {symbol} fundamentals - very bullish long term",
                    "sentiment": 0.7,
                    "confidence": 0.8,
                    "impact": "high"
                },
                {
                    "content": f"Warning: {symbol} showing signs of whale manipulation, be careful",
                    "sentiment": -0.6,
                    "confidence": 0.7,
                    "impact": "medium"
                }
            ]
            
            sentiment_data = []
            for item in reddit_data:
                sentiment = SentimentData(
                    source="reddit",
                    symbol=symbol,
                    sentiment_score=item["sentiment"],
                    confidence=item["confidence"],
                    content=item["content"],
                    timestamp=datetime.now(),
                    impact_level=item["impact"]
                )
                sentiment_data.append(sentiment)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing Reddit sentiment: {e}")
            return []
    
    async def _perform_adk_sentiment_analysis(self, symbol: str, sentiment_data: List[SentimentData]) -> Dict[str, Any]:
        """Perform enhanced sentiment analysis using ADK agent"""
        try:
            if not sentiment_data:
                return {'overall_sentiment': 0.0, 'confidence': 0.0, 'insights': 'No sentiment data available'}
            
            # Prepare sentiment summary for ADK analysis
            sentiment_summary = {
                'symbol': symbol,
                'total_signals': len(sentiment_data),
                'sources': list(set([s.source for s in sentiment_data])),
                'average_sentiment': sum([s.sentiment_score for s in sentiment_data]) / len(sentiment_data),
                'high_impact_signals': [s.to_dict() for s in sentiment_data if s.impact_level == 'high']
            }
            
            analysis_prompt = f"""
            Analyze the sentiment data for {symbol}:
            
            Summary:
            - Total signals: {sentiment_summary['total_signals']}
            - Sources: {', '.join(sentiment_summary['sources'])}
            - Average sentiment: {sentiment_summary['average_sentiment']:.2f}
            - High impact signals: {len(sentiment_summary['high_impact_signals'])}
            
            High Impact Content:
            {chr(10).join([f"- {signal['content'][:100]}..." for signal in sentiment_summary['high_impact_signals']])}
            
            Please provide:
            1. Overall sentiment assessment (-1 to 1 scale)
            2. Confidence level (0-1)
            3. Key themes and trending topics
            4. Potential market impact
            5. Contrarian indicators to watch
            6. Sentiment-based trading implications
            
            Search for recent {symbol} news that might affect sentiment.
            """
            
            adk_response = await self.adk_agent.run(analysis_prompt)
            
            return {
                'adk_sentiment_analysis': adk_response,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-2.0-flash'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ADK sentiment analysis: {e}")
            return {
                'adk_sentiment_analysis': f"ADK sentiment analysis failed: {str(e)}",
                'analysis_timestamp': datetime.now().isoformat(),
                'error': True
            }
    
    def _aggregate_sentiment_scores(self, sentiment_data: List[SentimentData], adk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple sources"""
        try:
            if not sentiment_data:
                return {
                    'overall_score': 0.0,
                    'confidence': 0.0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'trending_topics': []
                }
            
            # Weight sentiment scores by confidence and impact
            weighted_scores = []
            weights = []
            
            for sentiment in sentiment_data:
                # Impact weight multiplier
                impact_weight = {'low': 1.0, 'medium': 1.5, 'high': 2.0}.get(sentiment.impact_level, 1.0)
                
                # Combined weight
                weight = sentiment.confidence * impact_weight
                
                weighted_scores.append(sentiment.sentiment_score * weight)
                weights.append(weight)
            
            # Calculate weighted average
            if sum(weights) > 0:
                overall_score = sum(weighted_scores) / sum(weights)
            else:
                overall_score = 0.0
            
            # Calculate confidence as average of individual confidences
            overall_confidence = sum([s.confidence for s in sentiment_data]) / len(sentiment_data)
            
            # Count sentiment categories
            bullish_count = len([s for s in sentiment_data if s.sentiment_score > 0.2])
            bearish_count = len([s for s in sentiment_data if s.sentiment_score < -0.2])
            neutral_count = len(sentiment_data) - bullish_count - bearish_count
            
            # Extract trending topics (simplified)
            trending_topics = ['institutional_adoption', 'regulatory_clarity', 'technical_analysis']
            
            return {
                'overall_score': max(-1.0, min(1.0, overall_score)),  # Clamp to [-1, 1]
                'confidence': overall_confidence,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'trending_topics': trending_topics
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment scores: {e}")
            return {
                'overall_score': 0.0,
                'confidence': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'trending_topics': []
            }
    
    async def _verify_sentiment_with_agents(self, sentiment: MarketSentiment) -> MarketSentiment:
        """Verify sentiment analysis with multi-agent system"""
        try:
            verification_data = {
                'type': 'sentiment_analysis',
                'sentiment': sentiment.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Multi-agent verification
            verified_result = await self.coordinator.verify_and_reflect(verification_data)
            
            # Adjust confidence based on verification
            verification_confidence = verified_result.get('confidence', 0.5)
            adjusted_confidence = sentiment.confidence * verification_confidence
            
            # Create verified sentiment
            verified_sentiment = MarketSentiment(
                symbol=sentiment.symbol,
                overall_sentiment=sentiment.overall_sentiment,
                confidence=adjusted_confidence,
                bullish_signals=sentiment.bullish_signals,
                bearish_signals=sentiment.bearish_signals,
                neutral_signals=sentiment.neutral_signals,
                trending_topics=sentiment.trending_topics,
                sentiment_sources=sentiment.sentiment_sources
            )
            
            return verified_sentiment
            
        except Exception as e:
            self.logger.error(f"Error verifying sentiment: {e}")
            return sentiment
    
    async def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        try:
            market_summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(self.market_sentiment_cache),
                'overall_market_sentiment': 0.0,
                'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'top_trending_topics': [],
                'sentiment_by_symbol': {}
            }
            
            if self.market_sentiment_cache:
                # Calculate overall market sentiment
                sentiments = [s.overall_sentiment for s in self.market_sentiment_cache.values()]
                market_summary['overall_market_sentiment'] = sum(sentiments) / len(sentiments)
                
                # Count sentiment distribution
                for sentiment in self.market_sentiment_cache.values():
                    if sentiment.overall_sentiment > 0.2:
                        market_summary['sentiment_distribution']['bullish'] += 1
                    elif sentiment.overall_sentiment < -0.2:
                        market_summary['sentiment_distribution']['bearish'] += 1
                    else:
                        market_summary['sentiment_distribution']['neutral'] += 1
                
                # Aggregate trending topics
                all_topics = []
                for sentiment in self.market_sentiment_cache.values():
                    all_topics.extend(sentiment.trending_topics)
                
                # Count topic frequency
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Get top trending topics
                market_summary['top_trending_topics'] = sorted(
                    topic_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
                
                # Add individual symbol sentiments
                market_summary['sentiment_by_symbol'] = {
                    symbol: sentiment.to_dict() 
                    for symbol, sentiment in self.market_sentiment_cache.items()
                }
            
            return market_summary
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment summary: {e}")
            return {'error': str(e)}
    
    async def run(self):
        """Main run loop for the news sentiment analyzer"""
        self.running = True
        self.logger.info("News Sentiment Analyzer started")
        
        while self.running:
            try:
                # Analyze sentiment for all symbols
                for symbol in self.symbols:
                    sentiment = await self.analyze_sentiment(symbol)
                    
                    # Send sentiment update to other agents
                    await self.coordinator.broadcast_message(
                        self.agent_id,
                        MessageType.NEWS_SENTIMENT,
                        sentiment.to_dict()
                    )
                
                # Send market sentiment summary
                market_summary = await self.get_market_sentiment_summary()
                await self.coordinator.broadcast_message(
                    self.agent_id,
                    MessageType.NEWS_SENTIMENT,
                    market_summary
                )
                
                # Update heartbeat
                await self.coordinator.update_heartbeat(self.agent_id)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in news sentiment analyzer run loop: {e}")
                await asyncio.sleep(30)
    
    async def stop(self):
        """Stop the news sentiment analyzer"""
        self.running = False
        await self.coordinator.unregister_agent(self.agent_id)
        self.logger.info("News Sentiment Analyzer stopped")
