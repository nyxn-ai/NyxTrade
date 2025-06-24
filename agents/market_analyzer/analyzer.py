"""
Market Analyzer Agent
Performs technical analysis, market sentiment analysis, and price prediction
Integrates with Google ADK and A2A protocol for multi-agent verification
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import talib
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from utils.a2a_coordinator import A2ACoordinator, A2AMessage, MessageType
from utils.database import DatabaseManager
from data.collectors.price_collector import PriceCollector


class MarketAnalyzer:
    """
    Market Analysis Agent using technical indicators, ML predictions, and ADK integration
    Features multi-agent verification and reflection capabilities
    """

    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager, coordinator: A2ACoordinator):
        self.config = config
        self.db_manager = db_manager
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)

        self.agent_id = "market_analyzer"
        self.update_interval = config.get('update_interval', 60)
        self.indicators = config.get('indicators', ['RSI', 'MACD', 'Bollinger_Bands'])

        # ADK Agent for enhanced analysis
        self.adk_agent = None
        self.price_collector = PriceCollector()
        self.ml_models = {}
        self.running = False
        
    async def initialize(self):
        """Initialize the market analyzer agent with ADK integration"""
        try:
            # Initialize ADK Agent
            self.adk_agent = LlmAgent(
                name="market_analysis_agent",
                model="gemini-2.0-flash",
                instruction="""You are an expert cryptocurrency market analyst. Your capabilities include:

                1. Technical Analysis: Analyze price charts, indicators (RSI, MACD, Bollinger Bands, etc.)
                2. Market Sentiment: Interpret market sentiment from various data sources
                3. Price Prediction: Provide short-term and medium-term price forecasts
                4. Risk Assessment: Identify potential market risks and opportunities

                Always provide:
                - Clear reasoning for your analysis
                - Confidence levels for your predictions
                - Risk factors to consider
                - Specific entry/exit points when applicable

                Be objective, data-driven, and acknowledge uncertainty when present.""",
                description="Advanced cryptocurrency market analysis with technical indicators and sentiment analysis",
                tools=[google_search]  # Add search capability for market news
            )

            # Register with A2A coordinator
            await self.coordinator.register_adk_agent(
                self.agent_id,
                self.adk_agent,
                ['technical_analysis', 'price_prediction', 'market_sentiment', 'risk_assessment']
            )

            # Register message handler
            self.coordinator.register_message_handler(self.agent_id, self._handle_message)

            # Initialize ML models
            await self._initialize_ml_models()

            self.logger.info("Market Analyzer with ADK integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Market Analyzer: {e}")
            raise
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models for price prediction"""
        # This is a simplified LSTM model setup
        # In production, you would load pre-trained models or train them properly
        
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            self.ml_models[symbol] = {
                'model': model,
                'scaler': MinMaxScaler(feature_range=(0, 1))
            }
        
        self.logger.info("ML models initialized")
    
    async def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        try:
            if message.message_type == MessageType.SYSTEM_STATUS:
                payload = message.payload
                if payload.get('action') == 'emergency_stop':
                    self.logger.warning("Emergency stop received")
                    # Handle emergency stop logic
                    
            await self.coordinator.update_heartbeat(self.agent_id)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def analyze_market(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Perform comprehensive market analysis with ADK enhancement and verification"""
        try:
            # Get price data
            price_data = await self.price_collector.get_price_data(symbol, timeframe, limit=200)

            if price_data.empty:
                return {'error': 'No price data available'}

            # Traditional technical analysis
            technical_analysis = self._perform_technical_analysis(price_data)

            # Price prediction
            price_prediction = await self._predict_price(symbol, price_data)

            # Market sentiment
            market_sentiment = self._analyze_market_sentiment(price_data)

            # Enhanced ADK analysis
            adk_analysis = await self._perform_adk_analysis(symbol, price_data, technical_analysis)

            # Combine all analyses
            combined_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'type': 'market_analysis',
                'technical_analysis': technical_analysis,
                'price_prediction': price_prediction,
                'market_sentiment': market_sentiment,
                'adk_analysis': adk_analysis,
                'preliminary_recommendation': self._generate_recommendation(technical_analysis, price_prediction, market_sentiment)
            }

            # Multi-agent verification and reflection
            verified_result = await self.coordinator.verify_and_reflect(combined_analysis)

            # Final analysis with verification
            final_analysis = {
                **combined_analysis,
                'verification': verified_result,
                'final_recommendation': verified_result.get('final_recommendation', 'HOLD'),
                'confidence': verified_result.get('confidence', 0.5)
            }

            # Store analysis in database
            await self._store_analysis(final_analysis)

            return final_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {e}")
            return {'error': str(e)}

    async def _perform_adk_analysis(self, symbol: str, price_data: pd.DataFrame, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced analysis using ADK agent"""
        try:
            # Prepare data summary for ADK agent
            current_price = float(price_data['close'].iloc[-1])
            price_change_24h = ((current_price - float(price_data['close'].iloc[-24])) / float(price_data['close'].iloc[-24])) * 100
            volume_avg = float(price_data['volume'].tail(24).mean())

            analysis_prompt = f"""
            Analyze {symbol} cryptocurrency:

            Current Price: ${current_price:.2f}
            24h Change: {price_change_24h:.2f}%
            Average Volume (24h): {volume_avg:,.0f}

            Technical Indicators:
            {json.dumps(technical_analysis, indent=2)}

            Please provide:
            1. Market trend analysis
            2. Key support/resistance levels
            3. Volume analysis
            4. Risk factors
            5. Trading opportunities
            6. Confidence level (0-100%)

            Also search for recent news about {symbol} that might impact price.
            """

            # Get ADK agent analysis
            adk_response = await self.adk_agent.run(analysis_prompt)

            return {
                'adk_insight': adk_response,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-2.0-flash'
            }

        except Exception as e:
            self.logger.error(f"Error in ADK analysis: {e}")
            return {
                'adk_insight': f"ADK analysis failed: {str(e)}",
                'analysis_timestamp': datetime.now().isoformat(),
                'error': True
            }
    
    def _perform_technical_analysis(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis using various indicators"""
        analysis = {}
        
        close_prices = price_data['close'].values
        high_prices = price_data['high'].values
        low_prices = price_data['low'].values
        volume = price_data['volume'].values
        
        try:
            # RSI
            if 'RSI' in self.indicators:
                rsi = talib.RSI(close_prices, timeperiod=14)
                analysis['rsi'] = {
                    'current': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                    'signal': 'oversold' if rsi[-1] < 30 else 'overbought' if rsi[-1] > 70 else 'neutral'
                }
            
            # MACD
            if 'MACD' in self.indicators:
                macd, macd_signal, macd_hist = talib.MACD(close_prices)
                analysis['macd'] = {
                    'macd': float(macd[-1]) if not np.isnan(macd[-1]) else None,
                    'signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
                    'histogram': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None,
                    'trend': 'bullish' if macd[-1] > macd_signal[-1] else 'bearish'
                }
            
            # Bollinger Bands
            if 'Bollinger_Bands' in self.indicators:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
                current_price = close_prices[-1]
                analysis['bollinger_bands'] = {
                    'upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
                    'middle': float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None,
                    'lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
                    'position': 'above_upper' if current_price > bb_upper[-1] else 'below_lower' if current_price < bb_lower[-1] else 'middle'
                }
            
            # Moving Averages
            if 'EMA' in self.indicators:
                ema_20 = talib.EMA(close_prices, timeperiod=20)
                ema_50 = talib.EMA(close_prices, timeperiod=50)
                analysis['ema'] = {
                    'ema_20': float(ema_20[-1]) if not np.isnan(ema_20[-1]) else None,
                    'ema_50': float(ema_50[-1]) if not np.isnan(ema_50[-1]) else None,
                    'trend': 'bullish' if ema_20[-1] > ema_50[-1] else 'bearish'
                }
            
            if 'SMA' in self.indicators:
                sma_20 = talib.SMA(close_prices, timeperiod=20)
                sma_50 = talib.SMA(close_prices, timeperiod=50)
                analysis['sma'] = {
                    'sma_20': float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
                    'sma_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
                    'trend': 'bullish' if sma_20[-1] > sma_50[-1] else 'bearish'
                }
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    async def _predict_price(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict future price using ML model"""
        try:
            if symbol not in self.ml_models:
                return {'error': 'No ML model available for this symbol'}
            
            model_info = self.ml_models[symbol]
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Prepare data for prediction
            close_prices = price_data['close'].values.reshape(-1, 1)
            scaled_data = scaler.fit_transform(close_prices)
            
            # Create sequences for LSTM
            if len(scaled_data) < 60:
                return {'error': 'Insufficient data for prediction'}
            
            X = []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
            
            if len(X) == 0:
                return {'error': 'No sequences available for prediction'}
            
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Make prediction (this is simplified - in production you'd have trained models)
            # For now, we'll use a simple trend-based prediction
            recent_prices = close_prices[-10:]
            trend = np.polyfit(range(len(recent_prices)), recent_prices.flatten(), 1)[0]
            
            current_price = float(close_prices[-1])
            predicted_price_1h = current_price + (trend * 1)
            predicted_price_24h = current_price + (trend * 24)
            
            return {
                'current_price': current_price,
                'predicted_1h': predicted_price_1h,
                'predicted_24h': predicted_price_24h,
                'trend': 'upward' if trend > 0 else 'downward',
                'confidence': 0.7  # Simplified confidence score
            }
            
        except Exception as e:
            self.logger.error(f"Error in price prediction: {e}")
            return {'error': str(e)}
    
    def _analyze_market_sentiment(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment based on price action"""
        try:
            close_prices = price_data['close'].values
            volume = price_data['volume'].values
            
            # Price momentum
            price_change_24h = (close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100
            
            # Volume analysis
            avg_volume = np.mean(volume[-24:])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume
            
            # Volatility
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-24:]) * 100
            
            # Sentiment score (simplified)
            sentiment_score = 0
            if price_change_24h > 2:
                sentiment_score += 1
            elif price_change_24h < -2:
                sentiment_score -= 1
            
            if volume_ratio > 1.5:
                sentiment_score += 0.5
            
            if volatility > 5:
                sentiment_score -= 0.5
            
            sentiment = 'bullish' if sentiment_score > 0.5 else 'bearish' if sentiment_score < -0.5 else 'neutral'
            
            return {
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'price_change_24h': price_change_24h,
                'volume_ratio': volume_ratio,
                'volatility': volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {'error': str(e)}
    
    def _generate_recommendation(self, technical: Dict, prediction: Dict, sentiment: Dict) -> str:
        """Generate trading recommendation based on all analyses"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Technical analysis signals
            if technical.get('rsi', {}).get('signal') == 'oversold':
                bullish_signals += 1
            elif technical.get('rsi', {}).get('signal') == 'overbought':
                bearish_signals += 1
            
            if technical.get('macd', {}).get('trend') == 'bullish':
                bullish_signals += 1
            elif technical.get('macd', {}).get('trend') == 'bearish':
                bearish_signals += 1
            
            # Prediction signals
            if prediction.get('trend') == 'upward':
                bullish_signals += 1
            elif prediction.get('trend') == 'downward':
                bearish_signals += 1
            
            # Sentiment signals
            if sentiment.get('sentiment') == 'bullish':
                bullish_signals += 1
            elif sentiment.get('sentiment') == 'bearish':
                bearish_signals += 1
            
            # Generate recommendation
            if bullish_signals > bearish_signals + 1:
                return 'BUY'
            elif bearish_signals > bullish_signals + 1:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return 'HOLD'
    
    async def _store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis results in database"""
        try:
            # This would store the analysis in your database
            # Implementation depends on your database schema
            pass
        except Exception as e:
            self.logger.error(f"Error storing analysis: {e}")
    
    async def run(self):
        """Main run loop for the market analyzer"""
        self.running = True
        self.logger.info("Market Analyzer started")
        
        while self.running:
            try:
                # Analyze major trading pairs
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                
                for symbol in symbols:
                    analysis = await self.analyze_market(symbol)
                    
                    # Send analysis to other agents
                    message = A2AMessage(
                        message_id=f"analysis_{symbol}_{datetime.now().timestamp()}",
                        sender=self.agent_id,
                        recipient="all",
                        message_type=MessageType.MARKET_DATA,
                        payload=analysis,
                        timestamp=datetime.now().timestamp()
                    )
                    
                    await self.coordinator.broadcast_message(
                        self.agent_id,
                        MessageType.MARKET_DATA,
                        analysis
                    )
                
                # Update heartbeat
                await self.coordinator.update_heartbeat(self.agent_id)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in market analyzer run loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def stop(self):
        """Stop the market analyzer"""
        self.running = False
        await self.coordinator.unregister_agent(self.agent_id)
        self.logger.info("Market Analyzer stopped")
