"""
Logging Configuration
Sets up structured logging for the NyxTrade system
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import structlog
from utils.config import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(config: LoggingConfig):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_file = Path(config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper()))
    
    if config.format.lower() == 'json':
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        config.file,
        maxBytes=_parse_size(config.max_size),
        backupCount=config.backup_count
    )
    file_handler.setLevel(getattr(logging, config.level.upper()))
    file_handler.setFormatter(console_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set specific logger levels
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def _parse_size(size_str: str) -> int:
    """Parse size string like '100MB' to bytes"""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_trade_signal(self, symbol: str, signal_type: str, direction: str, 
                        strength: float, price: float, metadata: Dict[str, Any] = None):
        """Log trading signal"""
        self.logger.info(
            "Trade signal generated",
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            price=price,
            metadata=metadata or {}
        )
    
    def log_trade_execution(self, symbol: str, side: str, amount: float, 
                           price: float, exchange: str, order_id: str = None,
                           status: str = "pending"):
        """Log trade execution"""
        self.logger.info(
            "Trade executed",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            exchange=exchange,
            order_id=order_id,
            status=status
        )
    
    def log_risk_alert(self, alert_type: str, severity: str, message: str,
                      portfolio_value: float = None, risk_metrics: Dict[str, Any] = None):
        """Log risk management alert"""
        self.logger.warning(
            "Risk alert",
            alert_type=alert_type,
            severity=severity,
            message=message,
            portfolio_value=portfolio_value,
            risk_metrics=risk_metrics or {}
        )
    
    def log_arbitrage_opportunity(self, symbol: str, exchange_a: str, exchange_b: str,
                                 price_a: float, price_b: float, profit_percentage: float):
        """Log arbitrage opportunity"""
        self.logger.info(
            "Arbitrage opportunity detected",
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            price_a=price_a,
            price_b=price_b,
            profit_percentage=profit_percentage
        )
    
    def log_portfolio_update(self, total_value: float, assets: Dict[str, float],
                           performance: Dict[str, Any] = None):
        """Log portfolio update"""
        self.logger.info(
            "Portfolio updated",
            total_value=total_value,
            assets=assets,
            performance=performance or {}
        )
    
    def log_agent_status(self, agent_id: str, status: str, message: str = None,
                        metrics: Dict[str, Any] = None):
        """Log agent status"""
        self.logger.info(
            "Agent status update",
            agent_id=agent_id,
            status=status,
            message=message,
            metrics=metrics or {}
        )
    
    def log_error(self, error_type: str, message: str, exception: Exception = None,
                 context: Dict[str, Any] = None):
        """Log error with context"""
        if exception:
            self.logger.error(
                message,
                error_type=error_type,
                exception=str(exception),
                context=context or {},
                exc_info=True
            )
        else:
            self.logger.error(
                message,
                error_type=error_type,
                context=context or {}
            )


def get_trading_logger(name: str) -> TradingLogger:
    """Get a trading logger instance"""
    return TradingLogger(name)


def get_logger(name: str) -> logging.Logger:
    """Get a standard logger instance"""
    return logging.getLogger(name)
