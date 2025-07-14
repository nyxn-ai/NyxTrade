"""
NyxTrade Strategy Engine Demo
Demonstrates the complete strategy development workflow for AI agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json

from strategy_engine import (
    StrategyBase, IndicatorService, BacktestEngine, 
    AgentStrategyManager, StrategyMonitor
)
from strategy_engine.examples import MovingAverageStrategy, RSIStrategy
from strategy_engine.core.backtest_engine import BacktestConfig
from strategy_engine.monitoring.strategy_monitor import AlertConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_data(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate realistic price data using random walk
    np.random.seed(42)  # For reproducible results
    
    initial_price = 50000  # Starting BTC price
    returns = np.random.normal(0, 0.02, len(date_range))  # 2% hourly volatility
    
    # Create price series
    prices = [initial_price]
    for i in range(1, len(date_range)):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(date_range, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(returns[i]) * price
        
        high = price + np.random.uniform(0, volatility)
        low = price - np.random.uniform(0, volatility)
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.uniform(100, 1000) * (1 + abs(returns[i]) * 10)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(df)} data points for {symbol}")
    return df


def demo_indicator_service():
    """Demonstrate the IndicatorService functionality"""
    logger.info("=== Indicator Service Demo ===")
    
    # Generate sample data
    data = generate_sample_data(days=100)
    
    # Initialize indicator service
    indicator_service = IndicatorService()
    
    # Test individual indicators
    logger.info("Available indicators:", indicator_service.list_supported_indicators())
    
    # Calculate SMA
    sma_20 = indicator_service.calculate_indicator(data, "sma", timeperiod=20)
    logger.info(f"SMA(20) last value: {sma_20.iloc[-1]:.2f}")
    
    # Calculate RSI
    rsi = indicator_service.calculate_indicator(data, "rsi", timeperiod=14)
    logger.info(f"RSI(14) last value: {rsi.iloc[-1]:.2f}")
    
    # Calculate multiple indicators
    indicator_configs = [
        {"name": "sma", "library": "talib", "timeperiod": 10},
        {"name": "sma", "library": "talib", "timeperiod": 30},
        {"name": "rsi", "library": "talib", "timeperiod": 14},
        {"name": "macd", "library": "talib"}
    ]
    
    results = indicator_service.calculate_multiple_indicators(data, [
        indicator_service.create_indicator_config(**config) for config in indicator_configs
    ])
    
    logger.info("Multiple indicators calculated successfully")
    for name, result in results.items():
        if result is not None:
            if isinstance(result, pd.DataFrame):
                logger.info(f"{name}: {len(result.columns)} columns, last values: {result.iloc[-1].to_dict()}")
            else:
                logger.info(f"{name}: last value: {result.iloc[-1]:.2f}")


def demo_strategy_backtesting():
    """Demonstrate strategy backtesting"""
    logger.info("=== Strategy Backtesting Demo ===")
    
    # Generate sample data
    data = generate_sample_data(days=180)
    
    # Create strategy instance
    strategy = MovingAverageStrategy(
        name="MA_Demo",
        symbol="BTCUSDT",
        timeframe="1h",
        parameters={
            "fast_period": 10,
            "slow_period": 30,
            "risk_per_trade": 0.02
        }
    )
    
    # Configure backtest
    backtest_config = BacktestConfig(
        start_date=data.index[50],  # Skip first 50 periods for indicators
        end_date=data.index[-1],
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Run backtest
    backtest_engine = BacktestEngine(backtest_config)
    result = backtest_engine.run_backtest(strategy, data)
    
    # Display results
    logger.info(f"Backtest Results for {strategy.name}:")
    logger.info(f"  Total Return: {result.total_return:.2f}%")
    logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {result.max_drawdown:.2f}%")
    logger.info(f"  Win Rate: {result.win_rate:.2f}%")
    logger.info(f"  Total Trades: {result.total_trades}")
    logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
    
    return result


def demo_ai_agent_interface():
    """Demonstrate AI Agent Strategy Manager"""
    logger.info("=== AI Agent Interface Demo ===")
    
    # Initialize agent manager
    agent_manager = AgentStrategyManager()
    
    # Register strategy classes
    agent_manager.register_strategy_class(MovingAverageStrategy, "MovingAverage")
    agent_manager.register_strategy_class(RSIStrategy, "RSI")
    
    # Create strategy instances
    ma_strategy_id = agent_manager.create_strategy_instance(
        "MovingAverage",
        "MA_Agent_Test",
        "BTCUSDT",
        parameters={"fast_period": 12, "slow_period": 26}
    )
    
    rsi_strategy_id = agent_manager.create_strategy_instance(
        "RSI",
        "RSI_Agent_Test", 
        "BTCUSDT",
        parameters={"rsi_period": 14, "oversold_threshold": 25}
    )
    
    logger.info(f"Created strategies: {ma_strategy_id}, {rsi_strategy_id}")
    
    # Generate test data
    data = generate_sample_data(days=120)
    
    # Run backtests
    ma_result = agent_manager.run_strategy_backtest(
        ma_strategy_id,
        data,
        data.index[50],
        data.index[-1],
        initial_capital=50000
    )
    
    rsi_result = agent_manager.run_strategy_backtest(
        rsi_strategy_id,
        data,
        data.index[50], 
        data.index[-1],
        initial_capital=50000
    )
    
    # Compare results
    logger.info("Strategy Comparison:")
    logger.info(f"  MA Strategy - Return: {ma_result.total_return:.2f}%, Sharpe: {ma_result.sharpe_ratio:.2f}")
    logger.info(f"  RSI Strategy - Return: {rsi_result.total_return:.2f}%, Sharpe: {rsi_result.sharpe_ratio:.2f}")
    
    # Demonstrate parameter optimization
    logger.info("Running parameter optimization for MA strategy...")
    
    optimization_result = agent_manager.optimize_strategy_parameters(
        ma_strategy_id,
        data,
        parameter_ranges={
            "fast_period": [8, 10, 12, 15],
            "slow_period": [20, 25, 30, 35]
        },
        optimization_metric="sharpe_ratio",
        max_iterations=16
    )
    
    logger.info(f"Optimization completed:")
    logger.info(f"  Best parameters: {optimization_result['best_parameters']}")
    logger.info(f"  Best Sharpe ratio: {optimization_result['best_metric_value']:.2f}")
    
    # Generate strategy reports
    ma_report = agent_manager.generate_strategy_report(ma_strategy_id)
    logger.info(f"MA Strategy Report: {json.dumps(ma_report, indent=2, default=str)}")
    
    return agent_manager


def demo_strategy_monitoring():
    """Demonstrate strategy monitoring"""
    logger.info("=== Strategy Monitoring Demo ===")
    
    # Create strategies
    ma_strategy = MovingAverageStrategy("MA_Monitor", "BTCUSDT")
    rsi_strategy = RSIStrategy("RSI_Monitor", "BTCUSDT")
    
    # Configure monitoring
    alert_config = AlertConfig(
        max_drawdown_threshold=-5.0,
        min_win_rate_threshold=45.0,
        max_consecutive_losses=3
    )
    
    monitor = StrategyMonitor(alert_config)
    
    # Add alert callback
    def alert_handler(strategy_id: str, alert: dict):
        logger.warning(f"ALERT for {strategy_id}: {alert['alert_type']} - {alert['data']}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Add strategies to monitoring
    monitor.add_strategy("ma_001", ma_strategy)
    monitor.add_strategy("rsi_001", rsi_strategy)
    
    # Simulate some trading activity
    import time
    
    # Simulate losses to trigger alerts
    ma_strategy.pnl = -600  # Simulate loss
    rsi_strategy.pnl = -300
    
    # Add some losing trades to trigger consecutive loss alert
    for i in range(4):
        ma_strategy.trade_history.append({"pnl": -100, "timestamp": datetime.now()})
    
    # Start monitoring briefly
    monitor.start_monitoring()
    time.sleep(2)  # Let it run for 2 seconds
    
    # Get performance summaries
    ma_summary = monitor.get_strategy_performance_summary("ma_001", hours_back=1)
    logger.info(f"MA Strategy Summary: {json.dumps(ma_summary, indent=2, default=str)}")
    
    all_summary = monitor.get_all_strategies_summary()
    logger.info(f"All Strategies Summary: {json.dumps(all_summary, indent=2, default=str)}")
    
    monitor.stop_monitoring()
    
    return monitor


async def main():
    """Main demo function"""
    logger.info("Starting NyxTrade Strategy Engine Demo")
    
    try:
        # Run all demos
        demo_indicator_service()
        print("\n" + "="*50 + "\n")
        
        backtest_result = demo_strategy_backtesting()
        print("\n" + "="*50 + "\n")
        
        agent_manager = demo_ai_agent_interface()
        print("\n" + "="*50 + "\n")
        
        monitor = demo_strategy_monitoring()
        print("\n" + "="*50 + "\n")
        
        logger.info("Demo completed successfully!")
        
        # Summary
        logger.info("=== Demo Summary ===")
        logger.info("✅ Indicator Service: Multiple technical indicators calculated")
        logger.info("✅ Backtesting Engine: Strategy performance evaluated")
        logger.info("✅ AI Agent Interface: Automated strategy management")
        logger.info("✅ Strategy Monitoring: Real-time performance tracking")
        logger.info("\nThe NyxTrade Strategy Engine is ready for AI agent integration!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
