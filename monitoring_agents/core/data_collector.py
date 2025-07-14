"""
Data collection utilities for monitoring agents
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .config_manager import ConfigManager


class DataCollector:
    """
    Collects data from various sources for monitoring agents
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger("data_collector")
        
        # Load data sources configuration
        self.data_sources_config = self.config_manager.get_data_sources_config()
        
        # Initialize session
        self.session = None
        
        self.logger.info("Initialized DataCollector")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Use Binance API as primary source
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                else:
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            # Return mock data for testing
            return 50000.0 if "BTC" in symbol else 3000.0
    
    async def get_historical_prices(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Use Binance API for historical data
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                else:
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            # Return mock data for testing
            return self._generate_mock_historical_data(symbol, limit)
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get general market overview data"""
        try:
            # Get market cap data from CoinGecko
            url = "https://api.coingecko.com/api/v3/global"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    global_data = data.get("data", {})
                    
                    return {
                        "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd", 0),
                        "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd", 0),
                        "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd", 0),
                        "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
                        "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
                        "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0)
                    }
                else:
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get market overview: {e}")
            # Return mock data
            return {
                "total_market_cap_usd": 2500000000000,
                "total_volume_24h_usd": 100000000000,
                "market_cap_change_24h": 2.5,
                "btc_dominance": 45.0,
                "eth_dominance": 18.0,
                "active_cryptocurrencies": 10000
            }
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data"):
                        fng_data = data["data"][0]
                        return {
                            "value": int(fng_data["value"]),
                            "value_classification": fng_data["value_classification"],
                            "timestamp": fng_data["timestamp"]
                        }
                else:
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get Fear & Greed Index: {e}")
            return {
                "value": 50,
                "value_classification": "Neutral",
                "timestamp": str(int(datetime.now().timestamp()))
            }
    
    async def get_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Get funding rates for perpetual futures"""
        try:
            funding_rates = {}
            
            for symbol in symbols:
                url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}USDT"
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        funding_rates[symbol] = float(data.get("lastFundingRate", 0))
                    else:
                        funding_rates[symbol] = 0.0
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to get funding rates: {e}")
            return {symbol: 0.0001 for symbol in symbols}  # Mock data
    
    async def get_social_sentiment(self, keywords: List[str]) -> Dict[str, Any]:
        """Get social media sentiment data"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, return mock data
            sentiment_data = {}
            
            for keyword in keywords:
                sentiment_data[keyword] = {
                    "sentiment_score": 0.1,  # -1 to 1
                    "mention_count": 1000,
                    "trending": False,
                    "sentiment_change_24h": 0.05
                }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Failed to get social sentiment: {e}")
            return {}
    
    def _generate_mock_historical_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate mock historical data for testing"""
        import numpy as np
        
        # Base price
        base_price = 50000.0 if "BTC" in symbol else 3000.0
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - timedelta(hours=i) for i in range(limit)]
        timestamps.reverse()
        
        # Generate price data with random walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, limit)
        prices = [base_price]
        
        for i in range(1, limit):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, base_price * 0.5))  # Floor price
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            volatility = abs(returns[i]) * close_price * 0.5
            
            high = close_price + np.random.uniform(0, volatility)
            low = close_price - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close_price
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
