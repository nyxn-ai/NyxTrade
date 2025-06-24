"""
Secure Trading Executor
Executes trades while ensuring private keys never leave secure storage
Integrates with LangChain/MCP for AI-driven decisions with security validation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

from web3 import Web3
import ccxt

from utils.secure_wallet import SecureWalletManager, WalletSecurityValidator
from utils.langchain_mcp_integration import LangChainMCPIntegration
from utils.database import DatabaseManager


class TradeType(Enum):
    """Types of trades"""
    BUY = "buy"
    SELL = "sell"
    SWAP = "swap"


class ExecutionStatus(Enum):
    """Trade execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TradeOrder:
    """Trade order with security validation"""
    order_id: str
    trade_type: TradeType
    symbol: str
    amount: Decimal
    price: Optional[Decimal] = None
    wallet_id: str = ""
    exchange: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    ai_confidence: float = 0.0
    verification_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe for LLM)"""
        return {
            "order_id": self.order_id,
            "trade_type": self.trade_type.value,
            "symbol": self.symbol,
            "amount": float(self.amount),
            "price": float(self.price) if self.price else None,
            "exchange": self.exchange,
            "status": self.status.value,
            "ai_confidence": self.ai_confidence,
            "verification_score": self.verification_score
        }


class SecureTradingExecutor:
    """
    Secure trading executor that protects private keys while enabling AI-driven trading
    """
    
    def __init__(self, wallet_manager: SecureWalletManager, db_manager: DatabaseManager,
                 langchain_integration: LangChainMCPIntegration):
        self.wallet_manager = wallet_manager
        self.db_manager = db_manager
        self.langchain_integration = langchain_integration
        self.logger = logging.getLogger(__name__)
        
        # Exchange connections (API keys only, no private keys)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.web3_connections: Dict[str, Web3] = {}
        
        # Security settings
        self.max_trade_amount = Decimal('10000')  # Maximum trade amount in USD
        self.require_ai_verification = True
        self.min_confidence_threshold = 0.7
        
        # Active orders
        self.active_orders: Dict[str, TradeOrder] = {}
    
    def add_exchange(self, exchange_name: str, api_key: str, secret: str, 
                    sandbox: bool = True, **kwargs):
        """Add exchange connection (API keys only)"""
        try:
            if exchange_name.lower() == 'binance':
                exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': secret,
                    'sandbox': sandbox,
                    'enableRateLimit': True,
                    **kwargs
                })
            elif exchange_name.lower() == 'coinbase':
                exchange = ccxt.coinbasepro({
                    'apiKey': api_key,
                    'secret': secret,
                    'passphrase': kwargs.get('passphrase', ''),
                    'sandbox': sandbox,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange_name}")
            
            self.exchanges[exchange_name] = exchange
            self.logger.info(f"Added exchange: {exchange_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add exchange {exchange_name}: {e}")
            raise
    
    def add_web3_connection(self, network: str, rpc_url: str):
        """Add Web3 connection for DEX trading"""
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not w3.is_connected():
                raise ConnectionError(f"Failed to connect to {network}")
            
            self.web3_connections[network] = w3
            self.logger.info(f"Added Web3 connection: {network}")
            
        except Exception as e:
            self.logger.error(f"Failed to add Web3 connection {network}: {e}")
            raise
    
    async def create_trade_order(self, trade_data: Dict[str, Any], 
                               ai_analysis: Optional[Dict[str, Any]] = None) -> TradeOrder:
        """
        Create trade order with AI verification and security validation
        """
        try:
            # Validate input data doesn't contain sensitive information
            if not WalletSecurityValidator.validate_no_private_key_exposure(trade_data):
                raise ValueError("Trade data contains sensitive information")
            
            # Create trade order
            order = TradeOrder(
                order_id=f"order_{asyncio.get_event_loop().time()}",
                trade_type=TradeType(trade_data['trade_type']),
                symbol=trade_data['symbol'],
                amount=Decimal(str(trade_data['amount'])),
                price=Decimal(str(trade_data['price'])) if trade_data.get('price') else None,
                wallet_id=trade_data.get('wallet_id', ''),
                exchange=trade_data.get('exchange', '')
            )
            
            # Security validations
            if order.amount > self.max_trade_amount:
                raise ValueError(f"Trade amount exceeds maximum: {self.max_trade_amount}")
            
            # AI verification if enabled
            if self.require_ai_verification:
                verification_result = await self._verify_trade_with_ai(order, ai_analysis)
                order.ai_confidence = verification_result.get('confidence', 0.0)
                order.verification_score = verification_result.get('verification_score', 0.0)
                
                if order.ai_confidence < self.min_confidence_threshold:
                    raise ValueError(f"AI confidence too low: {order.ai_confidence}")
            
            # Store order
            self.active_orders[order.order_id] = order
            
            self.logger.info(f"Created trade order: {order.order_id}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to create trade order: {e}")
            raise
    
    async def _verify_trade_with_ai(self, order: TradeOrder, 
                                  ai_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify trade with AI agents"""
        try:
            # Prepare safe trade data for AI verification
            safe_trade_data = {
                "trade_type": order.trade_type.value,
                "symbol": order.symbol,
                "amount": float(order.amount),
                "price": float(order.price) if order.price else None,
                "exchange": order.exchange
            }
            
            # Add AI analysis context if available
            context = {}
            if ai_analysis:
                context['ai_analysis'] = WalletSecurityValidator.sanitize_for_llm(ai_analysis)
            
            # Query AI for verification
            verification_query = f"""
            Please verify this trade order:
            {json.dumps(safe_trade_data, indent=2)}
            
            Provide verification in JSON format:
            {{
                "confidence": 0.0-1.0,
                "verification_score": 0.0-1.0,
                "risk_assessment": "LOW/MEDIUM/HIGH",
                "recommendations": ["list of recommendations"],
                "approved": true/false,
                "reasoning": "detailed reasoning"
            }}
            """
            
            response = await self.langchain_integration.process_query(
                verification_query, context
            )
            
            # Parse AI response
            try:
                verification_result = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if parsing fails
                verification_result = {
                    "confidence": 0.5,
                    "verification_score": 0.5,
                    "risk_assessment": "MEDIUM",
                    "recommendations": ["Manual review recommended"],
                    "approved": False,
                    "reasoning": "Failed to parse AI verification response"
                }
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"AI verification failed: {e}")
            return {
                "confidence": 0.3,
                "verification_score": 0.3,
                "risk_assessment": "HIGH",
                "recommendations": ["AI verification failed - manual review required"],
                "approved": False,
                "reasoning": f"Verification error: {str(e)}"
            }
    
    async def execute_trade(self, order_id: str) -> Dict[str, Any]:
        """
        Execute trade securely without exposing private keys
        """
        try:
            order = self.active_orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            if order.status != ExecutionStatus.PENDING:
                raise ValueError(f"Order {order_id} is not pending")
            
            order.status = ExecutionStatus.EXECUTING
            
            # Execute based on exchange type
            if order.exchange in self.exchanges:
                result = await self._execute_cex_trade(order)
            elif order.wallet_id and any(order.exchange.lower() in net.lower() 
                                       for net in self.web3_connections.keys()):
                result = await self._execute_dex_trade(order)
            else:
                raise ValueError(f"No execution method available for order {order_id}")
            
            # Update order status
            if result.get('success', False):
                order.status = ExecutionStatus.COMPLETED
            else:
                order.status = ExecutionStatus.FAILED
            
            # Store execution result
            await self._store_execution_result(order, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution failed for order {order_id}: {e}")
            if order_id in self.active_orders:
                self.active_orders[order_id].status = ExecutionStatus.FAILED
            
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id
            }
    
    async def _execute_cex_trade(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute trade on centralized exchange"""
        try:
            exchange = self.exchanges[order.exchange]
            
            # Prepare order parameters
            symbol = order.symbol
            amount = float(order.amount)
            price = float(order.price) if order.price else None
            
            # Execute trade based on type
            if order.trade_type == TradeType.BUY:
                if price:
                    result = await exchange.create_limit_buy_order(symbol, amount, price)
                else:
                    result = await exchange.create_market_buy_order(symbol, amount)
            elif order.trade_type == TradeType.SELL:
                if price:
                    result = await exchange.create_limit_sell_order(symbol, amount, price)
                else:
                    result = await exchange.create_market_sell_order(symbol, amount)
            else:
                raise ValueError(f"Unsupported trade type for CEX: {order.trade_type}")
            
            return {
                "success": True,
                "exchange_order_id": result.get('id'),
                "filled": result.get('filled', 0),
                "remaining": result.get('remaining', amount),
                "fee": result.get('fee', {}),
                "timestamp": result.get('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"CEX trade execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_dex_trade(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute trade on decentralized exchange"""
        try:
            # Get wallet info (safe data only)
            wallet_info = self.wallet_manager.get_wallet_info(order.wallet_id)
            if not wallet_info:
                raise ValueError(f"Wallet {order.wallet_id} not found")
            
            # Get Web3 connection
            network = wallet_info.network
            if network not in self.web3_connections:
                raise ValueError(f"No Web3 connection for network {network}")
            
            w3 = self.web3_connections[network]
            
            # Prepare transaction data (no private keys)
            transaction_data = {
                "from": wallet_info.address,
                "to": "0x...",  # DEX contract address
                "value": int(order.amount * 10**18),  # Convert to wei
                "gas": 200000,
                "gasPrice": w3.to_wei('20', 'gwei'),
                "nonce": w3.eth.get_transaction_count(wallet_info.address)
            }
            
            # Sign transaction securely (private key never exposed)
            signed_txn_hex = self.wallet_manager.sign_transaction(
                order.wallet_id, transaction_data
            )
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn_hex)
            
            # Wait for confirmation
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": receipt.transactionHash.hex(),
                "gas_used": receipt.gasUsed,
                "block_number": receipt.blockNumber
            }
            
        except Exception as e:
            self.logger.error(f"DEX trade execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_execution_result(self, order: TradeOrder, result: Dict[str, Any]):
        """Store trade execution result in database"""
        try:
            execution_data = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.trade_type.value,
                "amount": float(order.amount),
                "price": float(order.price) if order.price else None,
                "exchange": order.exchange,
                "status": order.status.value,
                "ai_confidence": order.ai_confidence,
                "verification_score": order.verification_score,
                "execution_result": WalletSecurityValidator.sanitize_for_llm(result),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self.db_manager.store_trade_execution(execution_data)
            
        except Exception as e:
            self.logger.error(f"Failed to store execution result: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status (safe data only)"""
        order = self.active_orders.get(order_id)
        if not order:
            return None
        
        return order.to_dict()
    
    def list_active_orders(self) -> List[Dict[str, Any]]:
        """List all active orders (safe data only)"""
        return [order.to_dict() for order in self.active_orders.values()]
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return False
            
            if order.status == ExecutionStatus.PENDING:
                order.status = ExecutionStatus.CANCELLED
                self.logger.info(f"Cancelled order: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status report"""
        return {
            "wallet_manager_active": self.wallet_manager is not None,
            "ai_verification_enabled": self.require_ai_verification,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_trade_amount": float(self.max_trade_amount),
            "active_exchanges": list(self.exchanges.keys()),
            "active_networks": list(self.web3_connections.keys()),
            "active_orders_count": len(self.active_orders),
            "security_validations": [
                "Private key protection",
                "AI verification",
                "Input sanitization",
                "Output validation"
            ]
        }
