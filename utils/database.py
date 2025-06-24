"""
Database Manager
Handles database connections and operations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from utils.config import DatabaseConfig, RedisConfig


class DatabaseManager:
    """
    Manages database connections and operations
    Supports both PostgreSQL and Redis
    """
    
    def __init__(self, db_config: DatabaseConfig, redis_config: Optional[RedisConfig] = None):
        self.db_config = db_config
        self.redis_config = redis_config
        self.logger = logging.getLogger(__name__)
        
        # Database connections
        self.engine = None
        self.session_factory = None
        self.redis_client = None
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize PostgreSQL
            await self._initialize_postgres()
            
            # Initialize Redis if configured
            if self.redis_config:
                await self._initialize_redis()
            
            self.initialized = True
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.db_config.url.replace('postgresql://', 'postgresql+asyncpg://'),
                pool_size=self.db_config.pool_size,
                max_overflow=self.db_config.max_overflow,
                echo=self.db_config.echo
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.logger.info("PostgreSQL connection initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing PostgreSQL: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_config.url,
                db=self.redis_config.db,
                max_connections=self.redis_config.max_connections,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info("Redis connection initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Redis: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.initialized:
            raise RuntimeError("Database manager not initialized")
        
        return self.session_factory()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query"""
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row) for row in result.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    async def store_market_data(self, symbol: str, timeframe: str, data: Dict[str, Any]):
        """Store market analysis data"""
        try:
            query = """
            INSERT INTO market_analysis (symbol, timeframe, timestamp, data)
            VALUES (:symbol, :timeframe, :timestamp, :data)
            ON CONFLICT (symbol, timeframe, timestamp) 
            DO UPDATE SET data = :data
            """
            
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'data': data
            }
            
            await self.execute_query(query, params)
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def store_trade_signal(self, signal_data: Dict[str, Any]):
        """Store trading signal"""
        try:
            query = """
            INSERT INTO trade_signals (
                symbol, signal_type, direction, strength, 
                price, timestamp, metadata
            ) VALUES (
                :symbol, :signal_type, :direction, :strength,
                :price, :timestamp, :metadata
            )
            """
            
            await self.execute_query(query, signal_data)
            
        except Exception as e:
            self.logger.error(f"Error storing trade signal: {e}")
    
    async def store_trade_execution(self, trade_data: Dict[str, Any]):
        """Store trade execution data"""
        try:
            query = """
            INSERT INTO trade_executions (
                symbol, side, amount, price, exchange,
                order_id, status, timestamp, metadata
            ) VALUES (
                :symbol, :side, :amount, :price, :exchange,
                :order_id, :status, :timestamp, :metadata
            )
            """
            
            await self.execute_query(query, trade_data)
            
        except Exception as e:
            self.logger.error(f"Error storing trade execution: {e}")
    
    async def get_recent_signals(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals for a symbol"""
        try:
            query = """
            SELECT * FROM trade_signals 
            WHERE symbol = :symbol 
            ORDER BY timestamp DESC 
            LIMIT :limit
            """
            
            return await self.execute_query(query, {'symbol': symbol, 'limit': limit})
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {e}")
            return []
    
    async def get_portfolio_balance(self) -> Dict[str, Any]:
        """Get current portfolio balance"""
        try:
            query = """
            SELECT symbol, SUM(amount) as balance
            FROM trade_executions 
            WHERE status = 'filled'
            GROUP BY symbol
            """
            
            results = await self.execute_query(query)
            return {row['symbol']: row['balance'] for row in results}
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio balance: {e}")
            return {}
    
    # Redis operations
    async def cache_set(self, key: str, value: Any, expire: int = 3600):
        """Set value in Redis cache"""
        if not self.redis_client:
            return
        
        try:
            import json
            await self.redis_client.setex(key, expire, json.dumps(value))
            
        except Exception as e:
            self.logger.error(f"Error setting cache: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            import json
            value = await self.redis_client.get(key)
            return json.loads(value) if value else None
            
        except Exception as e:
            self.logger.error(f"Error getting cache: {e}")
            return None
    
    async def cache_delete(self, key: str):
        """Delete value from Redis cache"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
            
        except Exception as e:
            self.logger.error(f"Error deleting cache: {e}")
    
    async def publish_message(self, channel: str, message: Dict[str, Any]):
        """Publish message to Redis channel"""
        if not self.redis_client:
            return
        
        try:
            import json
            await self.redis_client.publish(channel, json.dumps(message))
            
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
    
    async def subscribe_channel(self, channel: str, callback):
        """Subscribe to Redis channel"""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    import json
                    data = json.loads(message['data'])
                    await callback(data)
                    
        except Exception as e:
            self.logger.error(f"Error subscribing to channel: {e}")
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                self.logger.info("PostgreSQL connection closed")
            
            if self.redis_client:
                await self.redis_client.close()
                self.logger.info("Redis connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Database schema creation (simplified)
CREATE_TABLES_SQL = """
-- Market analysis data
CREATE TABLE IF NOT EXISTS market_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    data JSONB NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

-- Trading signals
CREATE TABLE IF NOT EXISTS trade_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    strength FLOAT NOT NULL,
    price FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB
);

-- Trade executions
CREATE TABLE IF NOT EXISTS trade_executions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    amount FLOAT NOT NULL,
    price FLOAT NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    order_id VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB
);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_value FLOAT NOT NULL,
    assets JSONB NOT NULL,
    performance JSONB
);

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    portfolio_var FLOAT,
    max_drawdown FLOAT,
    sharpe_ratio FLOAT,
    metrics JSONB
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_market_analysis_symbol_time ON market_analysis(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol_time ON trade_signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_executions_symbol_time ON trade_executions(symbol, timestamp);
"""
