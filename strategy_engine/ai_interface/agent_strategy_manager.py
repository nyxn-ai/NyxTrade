"""
AI Agent Strategy Manager
Provides interface for AI agents to create, test, and deploy trading strategies
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from dataclasses import asdict

from ..core.strategy_base import StrategyBase, TradingSignal, MarketData
from ..core.indicator_service import IndicatorService, IndicatorConfig
from ..core.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentStrategyManager:
    """
    Manager class for AI agents to interact with the strategy engine
    Provides high-level interface for strategy development and testing
    """
    
    def __init__(self):
        self.indicator_service = IndicatorService()
        self.active_strategies: Dict[str, StrategyBase] = {}
        self.strategy_registry: Dict[str, Type[StrategyBase]] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        
        logger.info("Initialized AgentStrategyManager")
    
    def register_strategy_class(self, strategy_class: Type[StrategyBase], 
                               name: Optional[str] = None) -> str:
        """
        Register a strategy class for AI agent use
        
        Args:
            strategy_class: Strategy class to register
            name: Optional name for the strategy
            
        Returns:
            Strategy class identifier
        """
        strategy_name = name or strategy_class.__name__
        self.strategy_registry[strategy_name] = strategy_class
        
        logger.info(f"Registered strategy class: {strategy_name}")
        return strategy_name
    
    def create_strategy_instance(self, 
                               strategy_class_name: str,
                               instance_name: str,
                               symbol: str,
                               timeframe: str = "1h",
                               parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create strategy instance for AI agent
        
        Args:
            strategy_class_name: Name of registered strategy class
            instance_name: Unique name for this instance
            symbol: Trading symbol
            timeframe: Trading timeframe
            parameters: Strategy parameters
            
        Returns:
            Strategy instance ID
        """
        if strategy_class_name not in self.strategy_registry:
            raise ValueError(f"Strategy class not registered: {strategy_class_name}")
        
        strategy_class = self.strategy_registry[strategy_class_name]
        strategy_instance = strategy_class(
            name=instance_name,
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters
        )
        
        instance_id = str(uuid.uuid4())
        self.active_strategies[instance_id] = strategy_instance
        
        logger.info(f"Created strategy instance: {instance_name} (ID: {instance_id})")
        return instance_id
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Get performance metrics for a strategy"""
        if strategy_id not in self.active_strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        strategy = self.active_strategies[strategy_id]
        return {
            "strategy_state": strategy.get_strategy_state(),
            "performance_metrics": strategy.get_performance_metrics(),
            "backtest_results": self.backtest_results.get(strategy_id)
        }
    
    def update_strategy_parameters(self, strategy_id: str, 
                                 parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters (useful for AI optimization)
        
        Args:
            strategy_id: Strategy instance ID
            parameters: New parameters to update
            
        Returns:
            Success status
        """
        if strategy_id not in self.active_strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        try:
            strategy = self.active_strategies[strategy_id]
            strategy.update_parameters(parameters)
            
            logger.info(f"Updated parameters for strategy {strategy_id}: {parameters}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy parameters: {e}")
            return False
    
    def run_strategy_backtest(self, 
                            strategy_id: str,
                            market_data: pd.DataFrame,
                            start_date: datetime,
                            end_date: datetime,
                            initial_capital: float = 100000,
                            **backtest_config) -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy_id: Strategy instance ID
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            **backtest_config: Additional backtest configuration
            
        Returns:
            Backtest results
        """
        if strategy_id not in self.active_strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        strategy = self.active_strategies[strategy_id]
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            **backtest_config
        )
        
        # Run backtest
        backtest_engine = BacktestEngine(config)
        result = backtest_engine.run_backtest(strategy, market_data)
        
        # Store results
        self.backtest_results[strategy_id] = result
        
        logger.info(f"Completed backtest for strategy {strategy_id}: "
                   f"Return: {result.total_return:.2f}%, "
                   f"Sharpe: {result.sharpe_ratio:.2f}, "
                   f"Max DD: {result.max_drawdown:.2f}%")
        
        return result
    
    def optimize_strategy_parameters(self, 
                                   strategy_id: str,
                                   market_data: pd.DataFrame,
                                   parameter_ranges: Dict[str, List],
                                   optimization_metric: str = "sharpe_ratio",
                                   max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_id: Strategy instance ID
            market_data: Historical market data
            parameter_ranges: Dictionary of parameter ranges to test
            optimization_metric: Metric to optimize for
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results with best parameters
        """
        if strategy_id not in self.active_strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        strategy = self.active_strategies[strategy_id]
        original_params = strategy.parameters.copy()
        
        best_result = None
        best_params = None
        best_metric_value = float('-inf')
        
        # Simple grid search (can be enhanced with more sophisticated methods)
        import itertools
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = list(itertools.product(*param_values))[:max_iterations]
        
        logger.info(f"Starting parameter optimization for {strategy_id} "
                   f"with {len(combinations)} combinations")
        
        for i, combination in enumerate(combinations):
            # Create parameter dictionary
            test_params = dict(zip(param_names, combination))
            
            try:
                # Update strategy parameters
                strategy.update_parameters(test_params)
                
                # Run backtest
                result = self.run_strategy_backtest(
                    strategy_id,
                    market_data,
                    market_data.index[0],
                    market_data.index[-1]
                )
                
                # Check if this is the best result
                metric_value = getattr(result, optimization_metric)
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = test_params.copy()
                    best_result = result
                
                logger.debug(f"Iteration {i+1}/{len(combinations)}: "
                           f"{optimization_metric}={metric_value:.4f}, "
                           f"params={test_params}")
                
            except Exception as e:
                logger.error(f"Error in optimization iteration {i+1}: {e}")
                continue
        
        # Restore original parameters
        strategy.update_parameters(original_params)
        
        optimization_results = {
            "best_parameters": best_params,
            "best_metric_value": best_metric_value,
            "best_backtest_result": best_result,
            "optimization_metric": optimization_metric,
            "total_iterations": len(combinations)
        }
        
        logger.info(f"Optimization completed for {strategy_id}: "
                   f"Best {optimization_metric}: {best_metric_value:.4f}, "
                   f"Best params: {best_params}")
        
        return optimization_results
    
    def calculate_indicators_for_strategy(self, 
                                        market_data: pd.DataFrame,
                                        indicator_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate multiple indicators for strategy development
        
        Args:
            market_data: OHLCV data
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary of calculated indicators
        """
        configs = []
        for config in indicator_configs:
            indicator_config = self.indicator_service.create_indicator_config(**config)
            configs.append(indicator_config)
        
        return self.indicator_service.calculate_multiple_indicators(market_data, configs)
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available technical indicators"""
        return self.indicator_service.list_supported_indicators()
    
    def generate_strategy_report(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive strategy report for AI agent analysis
        
        Args:
            strategy_id: Strategy instance ID
            
        Returns:
            Comprehensive strategy report
        """
        if strategy_id not in self.active_strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        strategy = self.active_strategies[strategy_id]
        backtest_result = self.backtest_results.get(strategy_id)
        
        report = {
            "strategy_info": {
                "id": strategy_id,
                "name": strategy.name,
                "symbol": strategy.symbol,
                "timeframe": strategy.timeframe,
                "parameters": strategy.parameters
            },
            "current_state": strategy.get_strategy_state(),
            "performance_metrics": strategy.get_performance_metrics(),
            "backtest_summary": None
        }
        
        if backtest_result:
            report["backtest_summary"] = {
                "total_return": backtest_result.total_return,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown,
                "win_rate": backtest_result.win_rate,
                "total_trades": backtest_result.total_trades,
                "profit_factor": backtest_result.profit_factor,
                "performance_metrics": backtest_result.performance_metrics
            }
        
        return report
    
    def list_active_strategies(self) -> List[Dict[str, Any]]:
        """List all active strategy instances"""
        strategies = []
        for strategy_id, strategy in self.active_strategies.items():
            strategies.append({
                "id": strategy_id,
                "name": strategy.name,
                "symbol": strategy.symbol,
                "timeframe": strategy.timeframe,
                "is_active": strategy.is_active,
                "current_pnl": strategy.pnl
            })
        
        return strategies
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove strategy instance"""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            if strategy_id in self.backtest_results:
                del self.backtest_results[strategy_id]
            
            logger.info(f"Removed strategy: {strategy_id}")
            return True
        
        return False
