"""
Tests for NyxTrade Strategy Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_engine.core.indicator_service import IndicatorService, IndicatorLibrary
from strategy_engine.core.backtest_engine import BacktestEngine, BacktestConfig
from strategy_engine.ai_interface.agent_strategy_manager import AgentStrategyManager
from strategy_engine.examples.moving_average_strategy import MovingAverageStrategy
from strategy_engine.examples.rsi_strategy import RSIStrategy


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    initial_price = 50000
    returns = np.random.normal(0, 0.01, len(dates))
    prices = [initial_price]
    
    for i in range(1, len(dates)):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1000))
    
    data = []
    for i, (timestamp, price) in enumerate(zip(dates, prices)):
        volatility = abs(returns[i]) * price * 0.5
        
        high = price + np.random.uniform(0, volatility)
        low = price - np.random.uniform(0, volatility)
        open_price = prices[i-1] if i > 0 else price
        
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


class TestIndicatorService:
    """Test IndicatorService functionality"""
    
    def test_indicator_service_initialization(self):
        """Test IndicatorService initialization"""
        service = IndicatorService()
        assert service is not None
        assert len(service.supported_indicators) > 0
    
    def test_list_supported_indicators(self):
        """Test listing supported indicators"""
        service = IndicatorService()
        indicators = service.list_supported_indicators()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert "sma" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
    
    def test_calculate_sma(self, sample_data):
        """Test SMA calculation"""
        service = IndicatorService()
        
        sma = service.calculate_indicator(
            sample_data, 
            "sma", 
            IndicatorLibrary.TALIB,
            timeperiod=20
        )
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_data)
        assert not sma.iloc[-1] == 0  # Should have valid values
    
    def test_calculate_rsi(self, sample_data):
        """Test RSI calculation"""
        service = IndicatorService()
        
        rsi = service.calculate_indicator(
            sample_data,
            "rsi",
            IndicatorLibrary.TALIB,
            timeperiod=14
        )
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)
    
    def test_calculate_multiple_indicators(self, sample_data):
        """Test calculating multiple indicators"""
        service = IndicatorService()
        
        configs = [
            service.create_indicator_config("sma", "talib", timeperiod=10),
            service.create_indicator_config("rsi", "talib", timeperiod=14),
            service.create_indicator_config("macd", "talib")
        ]
        
        results = service.calculate_multiple_indicators(sample_data, configs)
        
        assert isinstance(results, dict)
        assert "sma" in results
        assert "rsi" in results
        assert "macd" in results
        assert results["sma"] is not None
        assert results["rsi"] is not None


class TestMovingAverageStrategy:
    """Test MovingAverageStrategy"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        strategy = MovingAverageStrategy(
            name="Test_MA",
            symbol="BTCUSDT",
            timeframe="1h"
        )
        
        assert strategy.name == "Test_MA"
        assert strategy.symbol == "BTCUSDT"
        assert strategy.timeframe == "1h"
        assert strategy.parameters["fast_period"] == 10
        assert strategy.parameters["slow_period"] == 30
    
    def test_strategy_with_custom_parameters(self):
        """Test strategy with custom parameters"""
        custom_params = {
            "fast_period": 5,
            "slow_period": 20,
            "risk_per_trade": 0.03
        }
        
        strategy = MovingAverageStrategy(
            name="Custom_MA",
            symbol="ETHUSDT",
            parameters=custom_params
        )
        
        assert strategy.parameters["fast_period"] == 5
        assert strategy.parameters["slow_period"] == 20
        assert strategy.parameters["risk_per_trade"] == 0.03
    
    def test_generate_signal_insufficient_data(self, sample_data):
        """Test signal generation with insufficient data"""
        strategy = MovingAverageStrategy("Test", "BTCUSDT")
        
        # Use only first few rows (insufficient for slow MA)
        limited_data = sample_data.head(20)
        
        from strategy_engine.core.strategy_base import MarketData
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=limited_data.index[-1],
            open=limited_data['open'].iloc[-1],
            high=limited_data['high'].iloc[-1],
            low=limited_data['low'].iloc[-1],
            close=limited_data['close'].iloc[-1],
            volume=limited_data['volume'].iloc[-1],
            timeframe="1h"
        )
        
        signal = strategy.generate_signal(market_data, limited_data)
        
        from strategy_engine.core.strategy_base import SignalType
        assert signal.signal_type == SignalType.HOLD


class TestRSIStrategy:
    """Test RSIStrategy"""
    
    def test_rsi_strategy_initialization(self):
        """Test RSI strategy initialization"""
        strategy = RSIStrategy(
            name="Test_RSI",
            symbol="BTCUSDT"
        )
        
        assert strategy.name == "Test_RSI"
        assert strategy.parameters["rsi_period"] == 14
        assert strategy.parameters["oversold_threshold"] == 30
        assert strategy.parameters["overbought_threshold"] == 70


class TestBacktestEngine:
    """Test BacktestEngine"""
    
    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            initial_capital=100000
        )
        
        engine = BacktestEngine(config)
        
        assert engine.config.initial_capital == 100000
        assert engine.current_capital == 100000
    
    def test_run_backtest(self, sample_data):
        """Test running a backtest"""
        strategy = MovingAverageStrategy(
            name="Backtest_MA",
            symbol="BTCUSDT",
            parameters={"fast_period": 5, "slow_period": 15}
        )
        
        config = BacktestConfig(
            start_date=sample_data.index[20],  # Skip first 20 for indicators
            end_date=sample_data.index[-1],
            initial_capital=50000
        )
        
        engine = BacktestEngine(config)
        result = engine.run_backtest(strategy, sample_data)
        
        assert result is not None
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'total_trades')


class TestAgentStrategyManager:
    """Test AgentStrategyManager"""
    
    def test_agent_manager_initialization(self):
        """Test agent manager initialization"""
        manager = AgentStrategyManager()
        
        assert manager is not None
        assert len(manager.active_strategies) == 0
        assert len(manager.strategy_registry) == 0
    
    def test_register_strategy_class(self):
        """Test registering strategy classes"""
        manager = AgentStrategyManager()
        
        strategy_name = manager.register_strategy_class(MovingAverageStrategy)
        
        assert strategy_name in manager.strategy_registry
        assert manager.strategy_registry[strategy_name] == MovingAverageStrategy
    
    def test_create_strategy_instance(self):
        """Test creating strategy instances"""
        manager = AgentStrategyManager()
        
        # Register strategy class
        manager.register_strategy_class(MovingAverageStrategy, "MA")
        
        # Create instance
        instance_id = manager.create_strategy_instance(
            "MA",
            "Test_Instance",
            "BTCUSDT",
            parameters={"fast_period": 8}
        )
        
        assert instance_id in manager.active_strategies
        strategy = manager.active_strategies[instance_id]
        assert strategy.name == "Test_Instance"
        assert strategy.parameters["fast_period"] == 8
    
    def test_get_available_indicators(self):
        """Test getting available indicators"""
        manager = AgentStrategyManager()
        
        indicators = manager.get_available_indicators()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert "sma" in indicators
        assert "rsi" in indicators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
