"""
Custom Exception Classes for NyxTrade
Provides specific exception types for better error handling and debugging
"""

from typing import Optional, Dict, Any


class NyxTradeException(Exception):
    """Base exception class for NyxTrade"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }


class SecurityException(NyxTradeException):
    """Security-related exceptions"""
    pass


class PrivateKeyExposureException(SecurityException):
    """Raised when private key exposure is detected"""
    
    def __init__(self, message: str = "Private key exposure detected", 
                 data_source: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="PRIVATE_KEY_EXPOSURE",
            context={'data_source': data_source}
        )


class WalletSecurityException(SecurityException):
    """Wallet security-related exceptions"""
    
    def __init__(self, message: str, wallet_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="WALLET_SECURITY_ERROR",
            context={'wallet_id': wallet_id}
        )


class EncryptionException(SecurityException):
    """Encryption/decryption related exceptions"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="ENCRYPTION_ERROR",
            context={'operation': operation}
        )


class ConfigurationException(NyxTradeException):
    """Configuration-related exceptions"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context={'config_key': config_key}
        )


class DatabaseException(NyxTradeException):
    """Database-related exceptions"""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            context={'operation': operation, 'table': table}
        )


class ExchangeException(NyxTradeException):
    """Exchange API related exceptions"""
    
    def __init__(self, message: str, exchange: Optional[str] = None, 
                 api_error_code: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="EXCHANGE_ERROR",
            context={'exchange': exchange, 'api_error_code': api_error_code}
        )


class TradingException(NyxTradeException):
    """Trading execution related exceptions"""
    
    def __init__(self, message: str, symbol: Optional[str] = None, 
                 order_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="TRADING_ERROR",
            context={'symbol': symbol, 'order_id': order_id}
        )


class InsufficientFundsException(TradingException):
    """Insufficient funds for trading"""
    
    def __init__(self, required_amount: float, available_amount: float, 
                 symbol: Optional[str] = None):
        message = f"Insufficient funds: required {required_amount}, available {available_amount}"
        super().__init__(
            message=message,
            symbol=symbol
        )
        self.error_code = "INSUFFICIENT_FUNDS"
        self.context.update({
            'required_amount': required_amount,
            'available_amount': available_amount
        })


class RiskLimitExceededException(TradingException):
    """Risk limits exceeded"""
    
    def __init__(self, limit_type: str, limit_value: float, current_value: float):
        message = f"Risk limit exceeded: {limit_type} limit {limit_value}, current {current_value}"
        super().__init__(message=message)
        self.error_code = "RISK_LIMIT_EXCEEDED"
        self.context.update({
            'limit_type': limit_type,
            'limit_value': limit_value,
            'current_value': current_value
        })


class AgentException(NyxTradeException):
    """AI Agent related exceptions"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, 
                 agent_type: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            context={'agent_id': agent_id, 'agent_type': agent_type}
        )


class VerificationException(AgentException):
    """Multi-agent verification exceptions"""
    
    def __init__(self, message: str, verification_type: Optional[str] = None, 
                 consensus_score: Optional[float] = None):
        super().__init__(
            message=message,
            agent_type="verification"
        )
        self.error_code = "VERIFICATION_ERROR"
        self.context.update({
            'verification_type': verification_type,
            'consensus_score': consensus_score
        })


class LangChainException(AgentException):
    """LangChain integration exceptions"""
    
    def __init__(self, message: str, tool_name: Optional[str] = None):
        super().__init__(
            message=message,
            agent_type="langchain"
        )
        self.error_code = "LANGCHAIN_ERROR"
        self.context.update({'tool_name': tool_name})


class MCPException(AgentException):
    """Model Context Protocol exceptions"""
    
    def __init__(self, message: str, resource_uri: Optional[str] = None):
        super().__init__(
            message=message,
            agent_type="mcp"
        )
        self.error_code = "MCP_ERROR"
        self.context.update({'resource_uri': resource_uri})


class NetworkException(NyxTradeException):
    """Network and connectivity exceptions"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 status_code: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            context={'endpoint': endpoint, 'status_code': status_code}
        )


class ValidationException(NyxTradeException):
    """Data validation exceptions"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context={'field': field, 'value': str(value) if value is not None else None}
        )


class ArbitrageException(TradingException):
    """Arbitrage-specific exceptions"""
    
    def __init__(self, message: str, exchange_a: Optional[str] = None, 
                 exchange_b: Optional[str] = None):
        super().__init__(message=message)
        self.error_code = "ARBITRAGE_ERROR"
        self.context.update({
            'exchange_a': exchange_a,
            'exchange_b': exchange_b
        })


class PortfolioException(NyxTradeException):
    """Portfolio management exceptions"""
    
    def __init__(self, message: str, portfolio_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="PORTFOLIO_ERROR",
            context={'portfolio_id': portfolio_id}
        )


class RebalancingException(PortfolioException):
    """Portfolio rebalancing exceptions"""
    
    def __init__(self, message: str, rebalancing_plan: Optional[Dict[str, Any]] = None):
        super().__init__(message=message)
        self.error_code = "REBALANCING_ERROR"
        self.context.update({'rebalancing_plan': rebalancing_plan})


class SentimentAnalysisException(AgentException):
    """Sentiment analysis exceptions"""
    
    def __init__(self, message: str, source: Optional[str] = None, 
                 symbol: Optional[str] = None):
        super().__init__(
            message=message,
            agent_type="sentiment_analysis"
        )
        self.error_code = "SENTIMENT_ANALYSIS_ERROR"
        self.context.update({'source': source, 'symbol': symbol})


class RateLimitException(NetworkException):
    """API rate limit exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 retry_after: Optional[int] = None, endpoint: Optional[str] = None):
        super().__init__(
            message=message,
            endpoint=endpoint
        )
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.context.update({'retry_after': retry_after})


class EmergencyStopException(NyxTradeException):
    """Emergency stop triggered"""
    
    def __init__(self, reason: str, triggered_by: Optional[str] = None):
        super().__init__(
            message=f"Emergency stop triggered: {reason}",
            error_code="EMERGENCY_STOP",
            context={'reason': reason, 'triggered_by': triggered_by}
        )


# Exception handler decorator
def handle_exceptions(exception_types: tuple = (Exception,), 
                     default_return=None, log_error: bool = True):
    """
    Decorator for handling exceptions in agent methods
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    if isinstance(e, NyxTradeException):
                        logger.error(f"NyxTrade exception in {func.__name__}: {e.to_dict()}")
                    else:
                        logger.error(f"Exception in {func.__name__}: {str(e)}")
                return default_return
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    if isinstance(e, NyxTradeException):
                        logger.error(f"NyxTrade exception in {func.__name__}: {e.to_dict()}")
                    else:
                        logger.error(f"Exception in {func.__name__}: {str(e)}")
                return default_return
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for exception handling
class ExceptionContext:
    """Context manager for handling exceptions with logging"""
    
    def __init__(self, operation_name: str, logger=None, 
                 reraise: bool = True, default_return=None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.reraise = reraise
        self.default_return = default_return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if isinstance(exc_val, NyxTradeException):
                self.logger.error(f"NyxTrade exception in {self.operation_name}: {exc_val.to_dict()}")
            else:
                self.logger.error(f"Exception in {self.operation_name}: {str(exc_val)}")
            
            if not self.reraise:
                return True  # Suppress exception
        
        return False  # Let exception propagate
