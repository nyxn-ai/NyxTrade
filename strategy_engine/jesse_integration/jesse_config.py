"""
Jesse Framework Configuration
"""

import os
from typing import Dict, Any

class JesseConfig:
    """Configuration manager for Jesse framework integration"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default Jesse configuration"""
        return {
            # Environment
            "APP_ENV": "development",
            
            # Database
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "POSTGRES_NAME": os.getenv("POSTGRES_NAME", "nyxtrade_jesse"),
            "POSTGRES_PORT": int(os.getenv("POSTGRES_PORT", "5432")),
            "POSTGRES_USERNAME": os.getenv("POSTGRES_USERNAME", "postgres"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "password"),
            
            # Redis
            "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
            "REDIS_PORT": int(os.getenv("REDIS_PORT", "6379")),
            "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", ""),
            
            # Exchanges
            "EXCHANGES": {
                "Binance Spot": {
                    "fee": 0.1,
                    "type": "spot",
                    "futures_leverage_mode": "cross",
                    "futures_leverage": 1,
                    "balance": 10000,
                    "assets": [
                        {"asset": "USDT", "balance": 10000},
                        {"asset": "BTC", "balance": 0}
                    ]
                }
            },
            
            # Logging
            "LOGGING": {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO",
                        "formatter": "default",
                        "stream": "ext://sys.stdout"
                    }
                },
                "root": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            
            # Metrics
            "METRICS": {
                "enabled": True,
                "port": 8000
            },
            
            # Notifications
            "NOTIFICATIONS": {
                "enabled": False
            }
        }
    
    @staticmethod
    def create_jesse_project_structure():
        """Create Jesse project directory structure"""
        directories = [
            "jesse_strategies",
            "jesse_strategies/strategies",
            "jesse_data",
            "jesse_logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def generate_jesse_config_file(config_path: str = "jesse_config.py"):
        """Generate Jesse configuration file"""
        config = JesseConfig.get_default_config()
        
        config_content = f'''"""
Jesse Configuration for NyxTrade
Auto-generated configuration file
"""

import os

# Environment
APP_ENV = "{config['APP_ENV']}"

# Database
POSTGRES_HOST = "{config['POSTGRES_HOST']}"
POSTGRES_NAME = "{config['POSTGRES_NAME']}"
POSTGRES_PORT = {config['POSTGRES_PORT']}
POSTGRES_USERNAME = "{config['POSTGRES_USERNAME']}"
POSTGRES_PASSWORD = "{config['POSTGRES_PASSWORD']}"

# Redis
REDIS_HOST = "{config['REDIS_HOST']}"
REDIS_PORT = {config['REDIS_PORT']}
REDIS_PASSWORD = "{config['REDIS_PASSWORD']}"

# Exchanges configuration
EXCHANGES = {config['EXCHANGES']}

# Logging configuration
LOGGING = {config['LOGGING']}

# Metrics
METRICS = {config['METRICS']}

# Notifications
NOTIFICATIONS = {config['NOTIFICATIONS']}
'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
