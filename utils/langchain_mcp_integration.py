"""
LangChain and MCP (Model Context Protocol) Integration
Provides secure context and tool access for AI agents while protecting sensitive data
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

from mcp import MCPServer, MCPClient
from mcp.types import Tool, Resource, Prompt

from utils.secure_wallet import WalletSecurityValidator, SecureWalletManager


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class SecureCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that ensures no sensitive data is logged or exposed
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running"""
        # Validate that prompts don't contain sensitive data
        for prompt in prompts:
            if not WalletSecurityValidator.validate_no_private_key_exposure(prompt):
                self.logger.error("SECURITY ALERT: Sensitive data detected in LLM prompt!")
                raise ValueError("Sensitive data detected in prompt - operation blocked")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running"""
        # Log safe completion
        self.logger.debug("LLM operation completed safely")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts running"""
        # Validate tool inputs
        if not WalletSecurityValidator.validate_no_private_key_exposure(input_str):
            self.logger.error("SECURITY ALERT: Sensitive data detected in tool input!")
            raise ValueError("Sensitive data detected in tool input - operation blocked")


class NyxTradeMCPServer:
    """
    MCP Server for NyxTrade that provides secure context and tools
    """
    
    def __init__(self, wallet_manager: SecureWalletManager):
        self.wallet_manager = wallet_manager
        self.logger = logging.getLogger(__name__)
        self.server = MCPServer("nyxtrade-mcp")
        self._setup_tools()
        self._setup_resources()
    
    def _setup_tools(self):
        """Setup MCP tools for trading operations"""
        
        @self.server.tool("get_wallet_balance")
        async def get_wallet_balance(wallet_id: str) -> Dict[str, Any]:
            """Get wallet balance (safe operation)"""
            try:
                wallet_info = self.wallet_manager.get_wallet_info(wallet_id)
                if not wallet_info:
                    return {"error": f"Wallet {wallet_id} not found"}
                
                # Return safe wallet information
                return {
                    "wallet_id": wallet_info.wallet_id,
                    "address": wallet_info.address,
                    "network": wallet_info.network,
                    "balance": wallet_info.balance,
                    "is_active": wallet_info.is_active
                }
            except Exception as e:
                self.logger.error(f"Error getting wallet balance: {e}")
                return {"error": str(e)}
        
        @self.server.tool("list_wallets")
        async def list_wallets() -> Dict[str, Any]:
            """List all wallets (safe operation)"""
            try:
                wallets = self.wallet_manager.list_wallets()
                return {
                    "wallets": [
                        {
                            "wallet_id": w.wallet_id,
                            "address": w.address,
                            "network": w.network,
                            "balance": w.balance,
                            "is_active": w.is_active
                        }
                        for w in wallets
                    ]
                }
            except Exception as e:
                self.logger.error(f"Error listing wallets: {e}")
                return {"error": str(e)}
        
        @self.server.tool("get_market_data")
        async def get_market_data(symbol: str) -> Dict[str, Any]:
            """Get market data for a symbol (safe operation)"""
            try:
                # This would integrate with price data collectors
                # Return mock data for now
                return {
                    "symbol": symbol,
                    "price": 45000.0,
                    "24h_change": 2.5,
                    "volume": 1500000000,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            except Exception as e:
                self.logger.error(f"Error getting market data: {e}")
                return {"error": str(e)}
        
        @self.server.tool("validate_transaction")
        async def validate_transaction(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate transaction parameters (safe operation)"""
            try:
                # Sanitize transaction data before validation
                safe_data = WalletSecurityValidator.sanitize_for_llm(transaction_data)
                
                # Perform validation logic
                validation_result = {
                    "is_valid": True,
                    "warnings": [],
                    "errors": [],
                    "gas_estimate": 21000,
                    "fee_estimate": 0.001
                }
                
                # Add validation checks
                if "to" not in safe_data:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Missing recipient address")
                
                if "value" not in safe_data:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Missing transaction value")
                
                return validation_result
                
            except Exception as e:
                self.logger.error(f"Error validating transaction: {e}")
                return {"error": str(e)}
    
    def _setup_resources(self):
        """Setup MCP resources for context"""
        
        @self.server.resource("trading_strategies")
        async def get_trading_strategies() -> str:
            """Get available trading strategies"""
            strategies = {
                "dca": {
                    "name": "Dollar Cost Averaging",
                    "description": "Systematic investment strategy",
                    "risk_level": "low"
                },
                "grid": {
                    "name": "Grid Trading",
                    "description": "Buy low, sell high in ranges",
                    "risk_level": "medium"
                },
                "arbitrage": {
                    "name": "Arbitrage Trading",
                    "description": "Profit from price differences",
                    "risk_level": "low"
                }
            }
            return json.dumps(strategies, indent=2)
        
        @self.server.resource("risk_parameters")
        async def get_risk_parameters() -> str:
            """Get current risk management parameters"""
            risk_params = {
                "max_position_size": 0.1,
                "max_daily_loss": 0.05,
                "stop_loss_percentage": 0.02,
                "take_profit_percentage": 0.05,
                "max_slippage": 0.005
            }
            return json.dumps(risk_params, indent=2)


class LangChainMCPIntegration:
    """
    Integration between LangChain and MCP for secure AI agent operations
    """
    
    def __init__(self, wallet_manager: SecureWalletManager, google_api_key: str):
        self.wallet_manager = wallet_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangChain components
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.1,  # Low temperature for consistent trading decisions
            callbacks=[SecureCallbackHandler()]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize MCP components
        self.mcp_server = NyxTradeMCPServer(wallet_manager)
        self.mcp_client = MCPClient()
        
        # Setup tools
        self.tools = self._create_secure_tools()
        
        # Create agent
        self.agent = self._create_secure_agent()
    
    def _create_secure_tools(self) -> List[BaseTool]:
        """Create secure tools that don't expose sensitive data"""
        
        def get_wallet_info(wallet_id: str) -> str:
            """Get wallet information (safe data only)"""
            try:
                wallet_info = self.wallet_manager.get_wallet_info(wallet_id)
                if not wallet_info:
                    return f"Wallet {wallet_id} not found"
                
                safe_info = {
                    "wallet_id": wallet_info.wallet_id,
                    "address": wallet_info.address,
                    "network": wallet_info.network,
                    "balance": wallet_info.balance
                }
                
                return json.dumps(safe_info, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error in get_wallet_info tool: {e}")
                return f"Error: {str(e)}"
        
        def analyze_market_data(symbol: str) -> str:
            """Analyze market data for a symbol"""
            try:
                # This would integrate with market analysis agents
                analysis = {
                    "symbol": symbol,
                    "trend": "bullish",
                    "support_level": 43000,
                    "resistance_level": 47000,
                    "recommendation": "HOLD",
                    "confidence": 0.75
                }
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error in analyze_market_data tool: {e}")
                return f"Error: {str(e)}"
        
        def calculate_position_size(portfolio_value: float, risk_percentage: float) -> str:
            """Calculate optimal position size"""
            try:
                position_size = portfolio_value * (risk_percentage / 100)
                
                result = {
                    "portfolio_value": portfolio_value,
                    "risk_percentage": risk_percentage,
                    "recommended_position_size": position_size,
                    "max_loss": position_size * 0.02  # 2% stop loss
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error in calculate_position_size tool: {e}")
                return f"Error: {str(e)}"
        
        # Create LangChain tools
        tools = [
            StructuredTool.from_function(
                func=get_wallet_info,
                name="get_wallet_info",
                description="Get wallet information (address, balance, network) - no private keys"
            ),
            StructuredTool.from_function(
                func=analyze_market_data,
                name="analyze_market_data",
                description="Analyze market data and provide trading insights"
            ),
            StructuredTool.from_function(
                func=calculate_position_size,
                name="calculate_position_size",
                description="Calculate optimal position size based on risk parameters"
            )
        ]
        
        return tools
    
    def _create_secure_agent(self) -> AgentExecutor:
        """Create secure LangChain agent with MCP integration"""
        
        system_prompt = """You are NyxTrade, an advanced cryptocurrency trading AI assistant.

SECURITY RULES (CRITICAL):
1. NEVER request, store, or process private keys, mnemonics, or passwords
2. NEVER ask users for sensitive wallet information
3. Only work with public addresses and non-sensitive data
4. Always validate that data doesn't contain private keys before processing
5. If you detect sensitive data, immediately stop and warn the user

CAPABILITIES:
- Market analysis and trading recommendations
- Portfolio management and risk assessment
- Transaction validation and optimization
- Multi-agent verification and consensus

TRADING PRINCIPLES:
- Risk management is paramount
- Use multi-agent verification for important decisions
- Provide clear reasoning for all recommendations
- Include confidence levels and risk assessments
- Suggest position sizing based on risk tolerance

Always be helpful, accurate, and prioritize user security above all else."""
        
        # Create agent with secure prompt
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=system_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5  # Limit iterations for safety
        )
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user query securely with MCP context
        """
        try:
            # Validate query doesn't contain sensitive data
            if not WalletSecurityValidator.validate_no_private_key_exposure(query):
                return "SECURITY WARNING: Your query contains sensitive data. Please remove private keys, mnemonics, or passwords and try again."
            
            # Add MCP context if available
            if context:
                safe_context = WalletSecurityValidator.sanitize_for_llm(context)
                enhanced_query = f"Context: {json.dumps(safe_context, indent=2)}\n\nQuery: {query}"
            else:
                enhanced_query = query
            
            # Process with agent
            response = await self.agent.arun(enhanced_query)
            
            # Validate response doesn't leak sensitive data
            if not WalletSecurityValidator.validate_no_private_key_exposure(response):
                return "SECURITY ERROR: Response contained sensitive data and was blocked."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"Error processing your request: {str(e)}"
    
    async def get_mcp_resources(self) -> List[MCPResource]:
        """Get available MCP resources"""
        try:
            resources = [
                MCPResource(
                    uri="nyxtrade://trading_strategies",
                    name="Trading Strategies",
                    description="Available trading strategies and their parameters"
                ),
                MCPResource(
                    uri="nyxtrade://risk_parameters",
                    name="Risk Parameters",
                    description="Current risk management settings"
                ),
                MCPResource(
                    uri="nyxtrade://market_data",
                    name="Market Data",
                    description="Real-time cryptocurrency market data"
                )
            ]
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error getting MCP resources: {e}")
            return []
    
    def add_custom_tool(self, tool: BaseTool):
        """Add custom tool to the agent"""
        try:
            # Validate tool doesn't expose sensitive operations
            if any(keyword in tool.name.lower() for keyword in ['private', 'secret', 'key', 'mnemonic']):
                raise ValueError("Tool name suggests sensitive operations - not allowed")
            
            self.tools.append(tool)
            
            # Recreate agent with new tools
            self.agent = self._create_secure_agent()
            
            self.logger.info(f"Added custom tool: {tool.name}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom tool: {e}")
            raise
    
    async def start_mcp_server(self, host: str = "localhost", port: int = 8765):
        """Start MCP server for external connections"""
        try:
            await self.mcp_server.start(host, port)
            self.logger.info(f"MCP server started on {host}:{port}")
            
        except Exception as e:
            self.logger.error(f"Error starting MCP server: {e}")
            raise
    
    async def stop_mcp_server(self):
        """Stop MCP server"""
        try:
            await self.mcp_server.stop()
            self.logger.info("MCP server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {e}")
            raise
