"""
Price Data Collector
Collects price data from various exchanges and sources
"""

import asyncio
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import ccxt
import yfinance as yf


class PriceCollector:
    """
    Collects price data from multiple sources including CEX and DEX
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.initialized = False
    
    async def initialize(self, exchange_configs: Dict[str, Dict[str, str]]):
        """Initialize exchange connections"""
        try:
            for exchange_name, config in exchange_configs.items():
                if exchange_name == 'binance':
                    exchange = ccxt.binance({
                        'apiKey': config.get('api_key'),
                        'secret': config.get('secret_key'),
                        'sandbox': config.get('sandbox', True),
                        'enableRateLimit': True,
                    })
                elif exchange_name == 'coinbase':
                    exchange = ccxt.coinbasepro({
                        'apiKey': config.get('api_key'),
                        'secret': config.get('secret_key'),
                        'passphrase': config.get('passphrase'),
                        'sandbox': config.get('sandbox', True),
                        'enableRateLimit': True,
                    })
                elif exchange_name == 'kraken':
                    exchange = ccxt.kraken({
                        'apiKey': config.get('api_key'),
                        'secret': config.get('secret_key'),
                        'enableRateLimit': True,
                    })
                else:
                    continue
                
                self.exchanges[exchange_name] = exchange
                self.logger.info(f"Initialized {exchange_name} exchange")
            
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")
            raise
    
    async def get_price_data(self, symbol: str, timeframe: str = '1h', limit: int = 100, exchange: str = 'binance') -> pd.DataFrame:
        """Get OHLCV price data for a symbol"""
        try:
            if not self.initialized:
                # Use a simple fallback method for demo purposes
                return await self._get_demo_price_data(symbol, timeframe, limit)
            
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not available")
            
            exchange_obj = self.exchanges[exchange]
            
            # Fetch OHLCV data
            ohlcv = await exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching price data for {symbol}: {e}")
            # Return demo data as fallback
            return await self._get_demo_price_data(symbol, timeframe, limit)
    
    async def _get_demo_price_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Generate demo price data for testing purposes"""
        try:
            # Create synthetic price data for demo
            import numpy as np
            
            # Base price for different symbols
            base_prices = {
                'BTC/USDT': 45000,
                'ETH/USDT': 3000,
                'BNB/USDT': 300,
                'ADA/USDT': 0.5,
                'SOL/USDT': 100,
                'MATIC/USDT': 1.0
            }
            
            base_price = base_prices.get(symbol, 1000)
            
            # Generate timestamps
            now = datetime.now()
            if timeframe == '1m':
                delta = timedelta(minutes=1)
            elif timeframe == '5m':
                delta = timedelta(minutes=5)
            elif timeframe == '15m':
                delta = timedelta(minutes=15)
            elif timeframe == '1h':
                delta = timedelta(hours=1)
            elif timeframe == '4h':
                delta = timedelta(hours=4)
            elif timeframe == '1d':
                delta = timedelta(days=1)
            else:
                delta = timedelta(hours=1)
            
            timestamps = [now - delta * i for i in range(limit, 0, -1)]
            
            # Generate price data with some randomness
            np.random.seed(42)  # For reproducible demo data
            
            prices = []
            current_price = base_price
            
            for i in range(limit):
                # Random walk with slight upward bias
                change = np.random.normal(0, 0.02) + 0.001  # 0.1% upward bias
                current_price *= (1 + change)
                
                # Generate OHLCV
                open_price = current_price
                high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
                close_price = open_price + np.random.normal(0, open_price * 0.005)
                volume = np.random.uniform(1000, 10000)
                
                prices.append({
                    'timestamp': timestamps[i],
                    'open': open_price,
                    'high': max(open_price, high_price, close_price),
                    'low': min(open_price, low_price, close_price),
                    'close': close_price,
                    'volume': volume
                })
                
                current_price = close_price
            
            df = pd.DataFrame(prices)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating demo price data: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str, exchange: str = 'binance') -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if not self.initialized:
                # Return demo price
                base_prices = {
                    'BTC/USDT': 45000,
                    'ETH/USDT': 3000,
                    'BNB/USDT': 300,
                    'ADA/USDT': 0.5,
                    'SOL/USDT': 100,
                    'MATIC/USDT': 1.0
                }
                return base_prices.get(symbol, 1000)
            
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not available")
            
            exchange_obj = self.exchanges[exchange]
            ticker = await exchange_obj.fetch_ticker(symbol)
            
            return ticker['last']
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    async def get_orderbook(self, symbol: str, exchange: str = 'binance', limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get orderbook for a symbol"""
        try:
            if not self.initialized:
                return None
            
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not available")
            
            exchange_obj = self.exchanges[exchange]
            orderbook = await exchange_obj.fetch_order_book(symbol, limit)
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    async def get_24h_stats(self, symbol: str, exchange: str = 'binance') -> Optional[Dict[str, Any]]:
        """Get 24h statistics for a symbol"""
        try:
            if not self.initialized:
                return None
            
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not available")
            
            exchange_obj = self.exchanges[exchange]
            ticker = await exchange_obj.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'quote_volume': ticker['quoteVolume']
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching 24h stats for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(self, symbols: List[str], exchange: str = 'binance') -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}
        
        for symbol in symbols:
            price = await self.get_current_price(symbol, exchange)
            if price:
                prices[symbol] = price
        
        return prices
    
    def close_connections(self):
        """Close all exchange connections"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    exchange.close()
                self.logger.info(f"Closed {exchange_name} connection")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_name} connection: {e}")
