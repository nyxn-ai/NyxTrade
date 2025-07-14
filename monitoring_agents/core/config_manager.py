"""
Configuration management for monitoring agents
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration for monitoring agents and Gemini integration
    """
    
    def __init__(self, config_dir: str = "monitoring_agents/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self._agents_config = None
        self._gemini_config = None
        self._data_sources_config = None
    
    def get_agents_config(self) -> Dict[str, Any]:
        """Get agents configuration"""
        if self._agents_config is None:
            self._agents_config = self._load_config("agents_config.yaml", self._get_default_agents_config())
        return self._agents_config
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini AI configuration"""
        if self._gemini_config is None:
            self._gemini_config = self._load_config("gemini_config.yaml", self._get_default_gemini_config())
        return self._gemini_config
    
    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration"""
        if self._data_sources_config is None:
            self._data_sources_config = self._load_config("data_sources.yaml", self._get_default_data_sources_config())
        return self._data_sources_config
    
    def _load_config(self, filename: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file or create with defaults"""
        config_path = self.config_dir / filename
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config if config else default_config
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                return default_config
        else:
            # Create default config file
            self._save_config(filename, default_config)
            return default_config
    
    def _save_config(self, filename: str, config: Dict[str, Any]):
        """Save configuration to file"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    def _get_default_agents_config(self) -> Dict[str, Any]:
        """Get default agents configuration"""
        return {
            "market_regression": {
                "enabled": True,
                "update_interval": 300,  # 5 minutes
                "symbols": ["BTC", "ETH"],
                "lookback_periods": [20, 50, 200],
                "alert_thresholds": {
                    "z_score": 2.0,
                    "deviation_percent": 10.0
                },
                "gemini_enabled": True
            },
            "trend_tracking": {
                "enabled": True,
                "update_interval": 180,  # 3 minutes
                "symbols": ["BTC", "ETH", "BNB", "ADA", "SOL"],
                "timeframes": ["1h", "4h", "1d"],
                "alert_thresholds": {
                    "trend_strength": 0.7,
                    "reversal_probability": 0.6
                },
                "gemini_enabled": True
            },
            "fund_flow": {
                "enabled": True,
                "update_interval": 600,  # 10 minutes
                "exchanges": ["binance", "coinbase", "kraken"],
                "whale_threshold": 1000000,  # $1M
                "alert_thresholds": {
                    "large_flow": 10000000,  # $10M
                    "flow_change_percent": 20.0
                },
                "gemini_enabled": True
            },
            "indicator_collector": {
                "enabled": True,
                "update_interval": 240,  # 4 minutes
                "indicators": [
                    "fear_greed_index",
                    "mvrv_ratio",
                    "nvt_ratio",
                    "social_sentiment",
                    "funding_rates"
                ],
                "alert_thresholds": {
                    "extreme_fear": 20,
                    "extreme_greed": 80
                },
                "gemini_enabled": True
            },
            "hotspot_tracking": {
                "enabled": True,
                "update_interval": 120,  # 2 minutes
                "sources": ["twitter", "reddit", "news"],
                "keywords": ["bitcoin", "ethereum", "defi", "nft", "regulation"],
                "alert_thresholds": {
                    "viral_threshold": 1000,
                    "sentiment_change": 0.3
                },
                "gemini_enabled": True
            }
        }
    
    def _get_default_gemini_config(self) -> Dict[str, Any]:
        """Get default Gemini configuration"""
        return {
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": "gemini-pro",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_tokens": 2048,
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 1
        }
    
    def _get_default_data_sources_config(self) -> Dict[str, Any]:
        """Get default data sources configuration"""
        return {
            "exchanges": {
                "binance": {
                    "api_key": os.getenv("BINANCE_API_KEY", ""),
                    "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                    "base_url": "https://api.binance.com",
                    "rate_limit": 1200  # requests per minute
                },
                "coinbase": {
                    "api_key": os.getenv("COINBASE_API_KEY", ""),
                    "api_secret": os.getenv("COINBASE_API_SECRET", ""),
                    "base_url": "https://api.exchange.coinbase.com",
                    "rate_limit": 10  # requests per second
                }
            },
            "data_providers": {
                "coingecko": {
                    "api_key": os.getenv("COINGECKO_API_KEY", ""),
                    "base_url": "https://api.coingecko.com/api/v3",
                    "rate_limit": 50  # requests per minute for free tier
                },
                "glassnode": {
                    "api_key": os.getenv("GLASSNODE_API_KEY", ""),
                    "base_url": "https://api.glassnode.com/v1",
                    "rate_limit": 100
                },
                "alternative_me": {
                    "base_url": "https://api.alternative.me",
                    "rate_limit": 100
                }
            },
            "social_sources": {
                "twitter": {
                    "bearer_token": os.getenv("TWITTER_BEARER_TOKEN", ""),
                    "api_key": os.getenv("TWITTER_API_KEY", ""),
                    "api_secret": os.getenv("TWITTER_API_SECRET", ""),
                    "rate_limit": 300
                },
                "reddit": {
                    "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
                    "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
                    "user_agent": "NyxTrade Monitoring Bot 1.0",
                    "rate_limit": 60
                }
            },
            "news_sources": {
                "newsapi": {
                    "api_key": os.getenv("NEWSAPI_KEY", ""),
                    "base_url": "https://newsapi.org/v2",
                    "rate_limit": 1000
                },
                "cryptonews": {
                    "base_url": "https://cryptonews-api.com/api/v1",
                    "rate_limit": 100
                }
            }
        }
    
    def update_agent_config(self, agent_name: str, config_updates: Dict[str, Any]):
        """Update configuration for specific agent"""
        agents_config = self.get_agents_config()
        if agent_name in agents_config:
            agents_config[agent_name].update(config_updates)
            self._save_config("agents_config.yaml", agents_config)
            self._agents_config = agents_config
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific agent"""
        agents_config = self.get_agents_config()
        return agents_config.get(agent_name)
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            "gemini": [],
            "data_sources": [],
            "agents": []
        }
        
        # Validate Gemini config
        gemini_config = self.get_gemini_config()
        if not gemini_config.get("api_key"):
            issues["gemini"].append("Gemini API key not configured")
        
        # Validate data sources
        data_sources = self.get_data_sources_config()
        
        # Check exchange API keys
        for exchange, config in data_sources.get("exchanges", {}).items():
            if not config.get("api_key") or not config.get("api_secret"):
                issues["data_sources"].append(f"{exchange} API credentials not configured")
        
        # Check data provider API keys
        for provider, config in data_sources.get("data_providers", {}).items():
            if provider != "alternative_me" and not config.get("api_key"):
                issues["data_sources"].append(f"{provider} API key not configured")
        
        # Validate agent configs
        agents_config = self.get_agents_config()
        for agent_name, config in agents_config.items():
            if config.get("enabled") and config.get("update_interval", 0) < 60:
                issues["agents"].append(f"{agent_name} update interval too short (minimum 60 seconds)")
        
        return issues
