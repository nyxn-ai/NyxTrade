# NyxTrade Strategy Engine

A comprehensive trading strategy development framework designed for AI agent automation and advanced cryptocurrency trading strategies.

## Features

- **Multi-Library Technical Indicators**: Integrates TA-Lib, pandas-ta, FinTA, and custom indicators
- **Jesse Framework Integration**: Supports Jesse trading framework for advanced backtesting
- **AI Agent Interface**: Designed for AI agents to create, test, and optimize strategies automatically
- **Comprehensive Backtesting**: Advanced backtesting engine with detailed performance metrics
- **Real-time Monitoring**: Strategy performance monitoring with alerting system
- **Risk Management**: Built-in position sizing and risk management tools

## Architecture

```
strategy_engine/
├── core/                    # Core engine components
│   ├── strategy_base.py     # Base strategy class
│   ├── indicator_service.py # Technical indicators service
│   └── backtest_engine.py   # Backtesting engine
├── ai_interface/            # AI agent interface
│   └── agent_strategy_manager.py
├── monitoring/              # Strategy monitoring
│   └── strategy_monitor.py
├── jesse_integration/       # Jesse framework integration
├── examples/               # Example strategies
│   ├── moving_average_strategy.py
│   └── rsi_strategy.py
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Strategy Development

```python
from strategy_engine import StrategyBase, TradingSignal, SignalType
from strategy_engine.core.indicator_service import IndicatorService

class MyStrategy(StrategyBase):
    def __init__(self, name, symbol, timeframe="1h", parameters=None):
        super().__init__(name, symbol, timeframe, parameters)
        self.indicator_service = IndicatorService()
    
    def generate_signal(self, market_data, historical_data):
        # Calculate RSI
        rsi = self.indicator_service.calculate_indicator(
            historical_data, "rsi", timeperiod=14
        )
        
        # Generate signal based on RSI
        if rsi.iloc[-1] < 30:
            return TradingSignal(
                signal_type=SignalType.BUY,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                price=market_data.close,
                confidence=0.8
            )
        elif rsi.iloc[-1] > 70:
            return TradingSignal(
                signal_type=SignalType.SELL,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                price=market_data.close,
                confidence=0.8
            )
        
        return self._create_hold_signal(market_data)
    
    def calculate_position_size(self, signal, account_balance):
        # Simple 2% risk per trade
        return (account_balance * 0.02) / signal.price
```

### 3. AI Agent Integration

```python
from strategy_engine import AgentStrategyManager

# Initialize agent manager
agent_manager = AgentStrategyManager()

# Register your strategy
agent_manager.register_strategy_class(MyStrategy, "MyStrategy")

# Create strategy instance
strategy_id = agent_manager.create_strategy_instance(
    "MyStrategy",
    "AI_Strategy_001",
    "BTCUSDT",
    parameters={"rsi_period": 14}
)

# Run backtest
result = agent_manager.run_strategy_backtest(
    strategy_id,
    market_data,
    start_date,
    end_date,
    initial_capital=100000
)

# Optimize parameters
optimization_result = agent_manager.optimize_strategy_parameters(
    strategy_id,
    market_data,
    parameter_ranges={"rsi_period": [10, 14, 21, 28]},
    optimization_metric="sharpe_ratio"
)
```

### 4. Strategy Monitoring

```python
from strategy_engine.monitoring import StrategyMonitor, AlertConfig

# Configure monitoring
alert_config = AlertConfig(
    max_drawdown_threshold=-10.0,
    min_win_rate_threshold=40.0,
    max_consecutive_losses=5
)

monitor = StrategyMonitor(alert_config)

# Add alert callback
def handle_alert(strategy_id, alert):
    print(f"ALERT: {alert['alert_type']} for {strategy_id}")

monitor.add_alert_callback(handle_alert)

# Add strategies to monitoring
monitor.add_strategy("strategy_001", my_strategy)

# Start monitoring
monitor.start_monitoring()
```

## Available Technical Indicators

The IndicatorService supports 10+ technical indicators across multiple libraries:

### Moving Averages
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)

### Oscillators
- Relative Strength Index (RSI)
- MACD
- Stochastic Oscillator

### Volatility
- Bollinger Bands
- Average True Range (ATR)

### Volume
- On Balance Volume (OBV)
- Accumulation/Distribution

## Backtesting Features

- **Realistic Execution**: Includes commission, slippage, and market impact
- **Risk Management**: Position sizing, stop losses, take profits
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
- **Trade Analysis**: Detailed trade log and equity curve
- **Parameter Optimization**: Grid search and genetic algorithms

## AI Agent Features

The AgentStrategyManager provides a high-level interface for AI agents:

- **Strategy Registration**: Register custom strategy classes
- **Instance Management**: Create and manage strategy instances
- **Automated Backtesting**: Run backtests programmatically
- **Parameter Optimization**: Automated parameter tuning
- **Performance Analysis**: Comprehensive strategy reports
- **Real-time Monitoring**: Live strategy performance tracking

## Example Usage

See `examples/strategy_engine_demo.py` for a complete demonstration of all features.

```bash
python examples/strategy_engine_demo.py
```

## Testing

Run the test suite:

```bash
pytest tests/test_strategy_engine.py -v
```

## Integration with NyxTrade

The Strategy Engine integrates seamlessly with the NyxTrade ecosystem:

- **Secure Execution**: Uses NyxTrade's secure trading infrastructure
- **Multi-Agent Coordination**: Works with NyxTrade's agent system
- **Risk Management**: Integrates with portfolio and risk managers
- **Data Sources**: Uses NyxTrade's market data collectors

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure AI agent compatibility for new components

## License

Part of the NyxTrade project. See main project license for details.
