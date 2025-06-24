#!/usr/bin/env python3
"""
Database Initialization Script
Creates database tables and initial data for NyxTrade
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from utils.database import DatabaseManager, CREATE_TABLES_SQL
from utils.logger import setup_logging


async def create_tables(db_manager: DatabaseManager):
    """Create database tables"""
    try:
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in CREATE_TABLES_SQL.split(';') if stmt.strip()]
        
        for statement in statements:
            await db_manager.execute_query(statement)
            
        logging.info("Database tables created successfully")
        
    except Exception as e:
        logging.error(f"Error creating tables: {e}")
        raise


async def insert_initial_data(db_manager: DatabaseManager):
    """Insert initial configuration data"""
    try:
        # Insert default trading pairs
        trading_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
            'SOL/USDT', 'MATIC/USDT', 'DOT/USDT', 'LINK/USDT'
        ]
        
        # This would insert initial configuration data
        # For now, we'll just log that it's done
        logging.info(f"Initial data prepared for {len(trading_pairs)} trading pairs")
        
    except Exception as e:
        logging.error(f"Error inserting initial data: {e}")
        raise


async def verify_database(db_manager: DatabaseManager):
    """Verify database setup"""
    try:
        # Check if tables exist
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        
        tables = await db_manager.execute_query(tables_query)
        table_names = [table['table_name'] for table in tables]
        
        expected_tables = [
            'market_analysis', 'trade_signals', 'trade_executions',
            'portfolio_snapshots', 'risk_metrics'
        ]
        
        missing_tables = [table for table in expected_tables if table not in table_names]
        
        if missing_tables:
            logging.error(f"Missing tables: {missing_tables}")
            return False
        
        logging.info("Database verification completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying database: {e}")
        return False


async def main():
    """Main initialization function"""
    try:
        # Load configuration
        config = Config()
        
        # Setup logging
        setup_logging(config.logging)
        
        logging.info("Starting database initialization...")
        
        # Initialize database manager
        db_manager = DatabaseManager(config.database, config.redis)
        await db_manager.initialize()
        
        # Create tables
        await create_tables(db_manager)
        
        # Insert initial data
        await insert_initial_data(db_manager)
        
        # Verify setup
        if await verify_database(db_manager):
            logging.info("Database initialization completed successfully")
        else:
            logging.error("Database verification failed")
            sys.exit(1)
        
        # Close connections
        await db_manager.close()
        
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
