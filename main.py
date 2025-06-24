#!/usr/bin/env python3
"""
NyxTrade - Multi-Agent Cryptocurrency Trading AI
Main entry point for the trading system
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List

import click
import yaml
from dotenv import load_dotenv

from agents.market_analyzer.analyzer import MarketAnalyzer
from agents.risk_manager.manager import RiskManager
from agents.arbitrage_hunter.hunter import ArbitrageHunter
from agents.portfolio_manager.manager import PortfolioManager
from agents.news_sentiment.analyzer import NewsSentimentAnalyzer
from utils.config import Config
from utils.logger import setup_logging
from utils.database import DatabaseManager
from utils.a2a_coordinator import A2ACoordinator
from utils.secure_wallet import SecureWalletManager
from utils.langchain_mcp_integration import LangChainMCPIntegration
from utils.exceptions import (
    NyxTradeException, ConfigurationException, SecurityException,
    handle_exceptions, ExceptionContext
)
from trading.secure_executor import SecureTradingExecutor


class NyxTrade:
    """Main NyxTrade application class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize NyxTrade system with security components"""
        self.config = Config(config_path)
        self.db_manager = DatabaseManager(self.config.database, self.config.redis)
        self.a2a_coordinator = A2ACoordinator(self.config.google_a2a)

        # Security components
        self.wallet_manager = None
        self.langchain_integration = None
        self.secure_executor = None

        self.agents: Dict[str, object] = {}
        self.running = False

        # Setup logging
        setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all system components with security"""
        self.logger.info("Initializing NyxTrade system with security components...")

        try:
            # Initialize database
            await self.db_manager.initialize()

            # Initialize security components
            await self._initialize_security_components()

            # Initialize A2A coordinator
            await self.a2a_coordinator.initialize()

            # Initialize agents
            await self._initialize_agents()

            self.logger.info("NyxTrade system initialized successfully with security")

        except Exception as e:
            self.logger.error(f"Failed to initialize NyxTrade: {e}")
            raise

    async def _initialize_security_components(self):
        """Initialize security components"""
        try:
            # Get master password from environment (never hardcoded)
            master_password = os.getenv('NYXTRADE_MASTER_PASSWORD')
            if not master_password:
                self.logger.warning("No master password set - wallet encryption disabled")

            # Initialize secure wallet manager
            self.wallet_manager = SecureWalletManager(master_password)

            # Initialize LangChain/MCP integration
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable required")

            self.langchain_integration = LangChainMCPIntegration(
                self.wallet_manager, google_api_key
            )

            # Initialize secure trading executor
            self.secure_executor = SecureTradingExecutor(
                self.wallet_manager,
                self.db_manager,
                self.langchain_integration
            )

            # Setup exchange connections (API keys from config)
            exchanges = self.config.exchanges
            for exchange_name, exchange_config in exchanges.items():
                if hasattr(exchange_config, 'api_key') and exchange_config.api_key:
                    self.secure_executor.add_exchange(
                        exchange_name,
                        exchange_config.api_key,
                        exchange_config.secret_key,
                        exchange_config.sandbox
                    )

            # Setup blockchain connections
            blockchains = self.config.blockchain
            for network_name, blockchain_config in blockchains.items():
                if hasattr(blockchain_config, 'rpc_url') and blockchain_config.rpc_url:
                    self.secure_executor.add_web3_connection(
                        network_name,
                        blockchain_config.rpc_url
                    )

            self.logger.info("Security components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize security components: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize all trading agents"""
        agent_configs = self.config.agents
        
        # Market Analyzer Agent
        if agent_configs.market_analyzer.enabled:
            self.agents['market_analyzer'] = MarketAnalyzer(
                config=agent_configs.market_analyzer,
                db_manager=self.db_manager,
                coordinator=self.a2a_coordinator
            )
            
        # Risk Manager Agent
        if agent_configs.risk_manager.enabled:
            self.agents['risk_manager'] = RiskManager(
                config=agent_configs.risk_manager,
                db_manager=self.db_manager,
                coordinator=self.a2a_coordinator
            )
            
        # Arbitrage Hunter Agent
        if agent_configs.arbitrage_hunter.enabled:
            self.agents['arbitrage_hunter'] = ArbitrageHunter(
                config=agent_configs.arbitrage_hunter,
                db_manager=self.db_manager,
                coordinator=self.a2a_coordinator
            )
            
        # Portfolio Manager Agent
        if agent_configs.portfolio_manager.enabled:
            self.agents['portfolio_manager'] = PortfolioManager(
                config=agent_configs.portfolio_manager,
                db_manager=self.db_manager,
                coordinator=self.a2a_coordinator
            )
            
        # News Sentiment Agent
        if agent_configs.news_sentiment.enabled:
            self.agents['news_sentiment'] = NewsSentimentAnalyzer(
                config=agent_configs.news_sentiment,
                db_manager=self.db_manager,
                coordinator=self.a2a_coordinator
            )
        
        # Initialize all agents
        for agent_name, agent in self.agents.items():
            await agent.initialize()
            self.logger.info(f"Initialized {agent_name} agent")
    
    async def start(self):
        """Start the NyxTrade system"""
        self.logger.info("Starting NyxTrade system...")
        self.running = True
        
        try:
            # Start all agents
            agent_tasks = []
            for agent_name, agent in self.agents.items():
                task = asyncio.create_task(agent.run())
                agent_tasks.append(task)
                self.logger.info(f"Started {agent_name} agent")
            
            # Start A2A coordinator
            coordinator_task = asyncio.create_task(self.a2a_coordinator.run())
            
            # Wait for all tasks
            await asyncio.gather(*agent_tasks, coordinator_task)
            
        except Exception as e:
            self.logger.error(f"Error running NyxTrade: {e}")
            raise
    
    async def stop(self):
        """Stop the NyxTrade system"""
        self.logger.info("Stopping NyxTrade system...")
        self.running = False
        
        # Stop all agents
        for agent_name, agent in self.agents.items():
            await agent.stop()
            self.logger.info(f"Stopped {agent_name} agent")
        
        # Stop A2A coordinator
        await self.a2a_coordinator.stop()
        
        # Close database connections
        await self.db_manager.close()
        
        self.logger.info("NyxTrade system stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


@click.group()
def cli():
    """NyxTrade - Multi-Agent Cryptocurrency Trading AI"""
    pass


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Configuration file path')
@click.option('--env', '-e', default='.env', help='Environment file path')
def run(config: str, env: str):
    """Run the NyxTrade trading system"""
    
    # Load environment variables
    if Path(env).exists():
        load_dotenv(env)
    
    # Create and run NyxTrade
    nyxtrade = NyxTrade(config)
    
    async def main():
        try:
            await nyxtrade.initialize()
            nyxtrade.setup_signal_handlers()
            await nyxtrade.start()
        except KeyboardInterrupt:
            await nyxtrade.stop()
        except Exception as e:
            logging.error(f"Fatal error: {e}")
            await nyxtrade.stop()
            sys.exit(1)
    
    # Run the main coroutine
    asyncio.run(main())


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Configuration file path')
def validate_config(config: str):
    """Validate configuration file"""
    try:
        config_obj = Config(config)
        click.echo("✅ Configuration is valid")
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}")
        sys.exit(1)


@cli.command()
def init_db():
    """Initialize database schema"""
    from scripts.init_db import main as init_db_main
    init_db_main()


@cli.command()
@click.option('--strategy', '-s', help='Strategy name to backtest')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
def backtest(strategy: str, start_date: str, end_date: str):
    """Run strategy backtesting"""
    from scripts.backtest import run_backtest
    run_backtest(strategy, start_date, end_date)


if __name__ == '__main__':
    cli()
