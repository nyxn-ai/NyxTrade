"""
Configuration Management
Handles loading and validation of configuration files
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from utils.exceptions import ConfigurationException, SecurityException


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str
    db: int = 0
    max_connections: int = 10


@dataclass
class GoogleA2AConfig:
    """Google A2A configuration"""
    project_id: str
    credentials_path: str
    region: str = "us-central1"
    agent_network_id: str = "nyxtrade-network"


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    api_key: str
    secret_key: str
    sandbox: bool = True
    passphrase: Optional[str] = None


@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    rpc_url: str
    chain_id: int


@dataclass
class WalletConfig:
    """Wallet configuration (public data only)"""
    address: str
    network: str = "ethereum"
    wallet_type: str = "local_encrypted"
    # Note: Private keys are NEVER stored in configuration
    # They are managed securely by SecureWalletManager


@dataclass
class TradingConfig:
    """Trading configuration"""
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    stop_loss_percentage: float = 0.02
    take_profit_percentage: float = 0.05
    max_slippage: float = 0.005
    gas_price_multiplier: float = 1.2
    max_gas_price: int = 100
    supported_pairs: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])


@dataclass
class AgentConfig:
    """Agent configuration"""
    enabled: bool = True
    update_interval: int = 60
    indicators: list = field(default_factory=list)
    check_interval: int = 30
    scan_interval: int = 10
    min_profit_threshold: float = 0.005
    max_execution_time: int = 30
    rebalance_interval: int = 3600
    target_allocations: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    enabled: bool = True
    interval: str = "daily"
    amount: float = 100
    assets: list = field(default_factory=list)
    grid_levels: int = 10
    price_range: float = 0.1
    min_profit: float = 0.005
    max_exposure: float = 1000
    gas_limit: int = 500000
    priority_fee: int = 2


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/nyxtrade.log"
    max_size: str = "100MB"
    backup_count: int = 5


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
        self._load_environment_variables()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                self._config_data = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_environment_variables(self):
        """Override configuration with environment variables"""
        env_mappings = {
            'GOOGLE_CLOUD_PROJECT': ['google_a2a', 'project_id'],
            'GOOGLE_APPLICATION_CREDENTIALS': ['google_a2a', 'credentials_path'],
            'DATABASE_URL': ['database', 'url'],
            'REDIS_URL': ['redis', 'url'],
            'BINANCE_API_KEY': ['exchanges', 'binance', 'api_key'],
            'BINANCE_SECRET_KEY': ['exchanges', 'binance', 'secret_key'],
            'COINBASE_API_KEY': ['exchanges', 'coinbase', 'api_key'],
            'COINBASE_SECRET_KEY': ['exchanges', 'coinbase', 'secret_key'],
            'COINBASE_PASSPHRASE': ['exchanges', 'coinbase', 'passphrase'],
            'KRAKEN_API_KEY': ['exchanges', 'kraken', 'api_key'],
            'KRAKEN_SECRET_KEY': ['exchanges', 'kraken', 'secret_key'],
            'ETHEREUM_RPC_URL': ['blockchain', 'ethereum', 'rpc_url'],
            'BSC_RPC_URL': ['blockchain', 'bsc', 'rpc_url'],
            'POLYGON_RPC_URL': ['blockchain', 'polygon', 'rpc_url'],
            'TRADING_WALLET_ADDRESS': ['wallets', 'trading_wallet', 'address'],
            'BACKUP_WALLET_ADDRESS': ['wallets', 'backup_wallet', 'address'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_config(config_path, value)
    
    def _set_nested_config(self, path: list, value: str):
        """Set nested configuration value"""
        current = self._config_data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self):
        """Validate configuration"""
        required_sections = ['google_a2a', 'database', 'redis']
        
        for section in required_sections:
            if section not in self._config_data:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate Google A2A config
        google_config = self._config_data.get('google_a2a', {})
        if not google_config.get('project_id'):
            raise ValueError("Google Cloud project_id is required")
    
    @property
    def google_a2a(self) -> GoogleA2AConfig:
        """Get Google A2A configuration"""
        config = self._config_data.get('google_a2a', {})
        return GoogleA2AConfig(**config)
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration"""
        config = self._config_data.get('database', {})
        return DatabaseConfig(**config)
    
    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration"""
        config = self._config_data.get('redis', {})
        return RedisConfig(**config)
    
    @property
    def exchanges(self) -> Dict[str, ExchangeConfig]:
        """Get exchange configurations"""
        exchanges = {}
        exchange_configs = self._config_data.get('exchanges', {})
        
        for name, config in exchange_configs.items():
            exchanges[name] = ExchangeConfig(**config)
        
        return exchanges
    
    @property
    def blockchain(self) -> Dict[str, BlockchainConfig]:
        """Get blockchain configurations"""
        blockchains = {}
        blockchain_configs = self._config_data.get('blockchain', {})
        
        for name, config in blockchain_configs.items():
            blockchains[name] = BlockchainConfig(**config)
        
        return blockchains
    
    @property
    def wallets(self) -> Dict[str, WalletConfig]:
        """Get wallet configurations"""
        wallets = {}
        wallet_configs = self._config_data.get('wallets', {})
        
        for name, config in wallet_configs.items():
            wallets[name] = WalletConfig(**config)
        
        return wallets
    
    @property
    def trading(self) -> TradingConfig:
        """Get trading configuration"""
        config = self._config_data.get('trading', {})
        return TradingConfig(**config)
    
    @property
    def agents(self) -> Dict[str, AgentConfig]:
        """Get agent configurations"""
        agents = {}
        agent_configs = self._config_data.get('agents', {})
        
        for name, config in agent_configs.items():
            agents[name] = AgentConfig(**config)
        
        return agents
    
    @property
    def strategies(self) -> Dict[str, StrategyConfig]:
        """Get strategy configurations"""
        strategies = {}
        strategy_configs = self._config_data.get('strategies', {})
        
        for name, config in strategy_configs.items():
            strategies[name] = StrategyConfig(**config)
        
        return strategies
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        config = self._config_data.get('logging', {})
        return LoggingConfig(**config)
    
    @property
    def external_apis(self) -> Dict[str, Any]:
        """Get external API configurations"""
        return self._config_data.get('external_apis', {})
    
    @property
    def monitoring(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self._config_data.get('monitoring', {})
    
    @property
    def development(self) -> Dict[str, Any]:
        """Get development configuration"""
        return self._config_data.get('development', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        current = self._config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        current = self._config_data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
