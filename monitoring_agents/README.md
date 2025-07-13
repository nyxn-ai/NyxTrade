# NyxTrade Monitoring Agents System

A comprehensive monitoring system with AI-powered analysis using Google Gemini AI for cryptocurrency market surveillance and automated decision support.

## 🎯 Overview

This system provides specialized monitoring agents for different aspects of cryptocurrency markets:

- **Market Regression Agent**: Monitors BTC/ETH mean reversion opportunities
- **Trend Tracking Agent**: Identifies and tracks market trends
- **Fund Flow Agent**: Monitors large capital movements and institutional activity
- **Indicator Collector Agent**: Aggregates technical and sentiment indicators
- **Hotspot Tracking Agent**: Tracks social media and news trends

Each agent integrates with Google Gemini AI for intelligent analysis and actionable insights.

## 🏗️ Architecture

```
monitoring_agents/
├── core/                           # Core infrastructure
│   ├── base_agent.py              # Base agent class
│   ├── gemini_client.py           # Gemini AI integration
│   ├── data_collector.py          # Data collection utilities
│   ├── alert_manager.py           # Alert management system
│   └── config_manager.py          # Configuration management
├── agents/                         # Monitoring agents
│   ├── market_regression/          # BTC/ETH mean reversion
│   ├── trend_tracking/             # Trend analysis
│   ├── fund_flow/                  # Capital flow monitoring
│   ├── indicator_collector/        # Technical indicators
│   └── hotspot_tracking/           # Social/news trends
├── config/                         # Configuration files
├── examples/                       # Demo scripts
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install google-generativeai aiohttp pandas numpy scipy pyyaml

# Set up environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_secret"
```

### 2. Configuration

Edit configuration files in `monitoring_agents/config/`:

```yaml
# gemini_config.yaml
api_key: "your_gemini_api_key"
model: "gemini-pro"
temperature: 0.7

# agents_config.yaml
market_regression:
  enabled: true
  update_interval: 300  # 5 minutes
  symbols: ["BTC", "ETH"]
  alert_thresholds:
    z_score: 2.0
    deviation_percent: 10.0
```

### 3. Run Demo

```bash
cd monitoring_agents
python examples/agent_demo.py
```

## 🤖 Agent Details

### Market Regression Agent

Monitors BTC and ETH for mean reversion opportunities:

- **Metrics**: Z-scores, Bollinger Bands, RSI, momentum
- **Signals**: Oversold/overbought conditions
- **AI Analysis**: Statistical significance, reversion probability
- **Alerts**: Extreme deviations, high-confidence opportunities

```python
from agents.market_regression import BTCETHRegressionAgent

agent = BTCETHRegressionAgent()
result = await agent.run_analysis_cycle()
```

### Trend Tracking Agent

Identifies and tracks market trends across multiple timeframes:

- **Metrics**: Trend strength, direction, momentum
- **Signals**: Trend reversals, breakouts
- **AI Analysis**: Trend sustainability, reversal probability
- **Alerts**: Strong trend changes, reversal signals

### Fund Flow Agent

Monitors large capital movements and institutional activity:

- **Data Sources**: On-chain data, exchange flows, whale movements
- **Metrics**: Net flows, large transactions, institutional indicators
- **AI Analysis**: Flow interpretation, market impact assessment
- **Alerts**: Unusual flow patterns, whale activity

### Indicator Collector Agent

Aggregates various market indicators:

- **Technical**: RSI, MACD, moving averages
- **Sentiment**: Fear & Greed Index, social sentiment
- **On-chain**: MVRV, NVT, active addresses
- **AI Analysis**: Indicator synthesis, market regime identification

### Hotspot Tracking Agent

Tracks social media and news trends:

- **Sources**: Twitter, Reddit, news feeds
- **Metrics**: Mention volume, sentiment, viral content
- **AI Analysis**: Trend impact assessment, narrative analysis
- **Alerts**: Viral content, sentiment shifts

## 🔧 Gemini AI Integration

### Setup

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable: `export GEMINI_API_KEY="your_key"`
3. Configure in `config/gemini_config.yaml`

### Features

- **Intelligent Analysis**: Context-aware market interpretation
- **Structured Output**: JSON-formatted insights and recommendations
- **Caching**: Response caching for efficiency
- **Error Handling**: Graceful fallbacks and retry logic

### Example Prompt

```python
prompt = """
Analyze BTC mean reversion data:
- Current price: $45,000
- 20-day MA: $47,000
- Z-score: -1.8
- RSI: 35

Provide analysis on reversion probability and timing.
"""

insights = await gemini_client.analyze(prompt)
```

## 📊 Alert System

### Alert Types

- **Threshold Exceeded**: Metric crosses predefined threshold
- **Anomaly Detected**: Unusual pattern identified
- **Trend Change**: Significant trend shift
- **High Confidence Signal**: Strong trading opportunity

### Alert Handlers

```python
from core.alert_manager import AlertManager, console_alert_handler

alert_manager = AlertManager()
alert_manager.add_alert_handler(console_alert_handler)

# Custom handler
async def custom_handler(alert):
    print(f"Custom alert: {alert.message}")

alert_manager.add_alert_handler(custom_handler)
```

## 📈 Data Sources

### Supported APIs

- **Binance**: Price data, futures data
- **CoinGecko**: Market overview, historical data
- **Alternative.me**: Fear & Greed Index
- **Glassnode**: On-chain metrics (with API key)
- **Social APIs**: Twitter, Reddit (with credentials)

### Mock Data

System includes mock data generators for testing without API keys.

## 🔍 Monitoring & Debugging

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Agent logs
logger = logging.getLogger("monitoring_agent.btc_eth_regression")
```

### Status Monitoring

```python
# Get agent status
status = agent.get_status()
print(f"Agent: {status['name']}, Running: {status['is_running']}")

# Get results history
history = agent.get_results_history(hours_back=24)
print(f"Results in last 24h: {len(history)}")

# Get alert statistics
stats = alert_manager.get_alert_stats()
print(f"Active alerts: {stats['active_alerts_count']}")
```

## 🚀 Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY monitoring_agents/ ./monitoring_agents/
CMD ["python", "monitoring_agents/examples/agent_demo.py"]
```

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_key

# Optional (for real data)
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
COINGECKO_API_KEY=your_coingecko_key
TWITTER_BEARER_TOKEN=your_twitter_token
```

### Scaling

- Run agents in separate processes/containers
- Use Redis for inter-agent communication
- Implement load balancing for API calls
- Set up monitoring dashboards

## 🔧 Customization

### Creating Custom Agents

```python
from core.base_agent import BaseMonitoringAgent, AgentConfig

class CustomAgent(BaseMonitoringAgent):
    async def collect_data(self):
        # Implement data collection
        return {"custom_data": "value"}
    
    async def analyze_data(self, data):
        # Implement analysis logic
        return {"analysis": "result"}
    
    def get_gemini_prompt(self, data, analysis):
        # Create AI prompt
        return "Analyze this custom data..."
```

### Custom Alert Handlers

```python
async def slack_alert_handler(alert):
    # Send alert to Slack
    await send_slack_message(alert.message)

alert_manager.add_alert_handler(slack_alert_handler)
```

## 📚 API Reference

### BaseMonitoringAgent

- `run_analysis_cycle()`: Execute complete analysis
- `start_monitoring()`: Begin continuous monitoring
- `stop_monitoring()`: Stop monitoring
- `get_status()`: Get agent status
- `get_results_history()`: Get historical results

### GeminiClient

- `analyze(prompt, context)`: Get AI analysis
- `get_usage_stats()`: Get usage statistics

### AlertManager

- `send_alerts(alerts)`: Process alerts
- `get_active_alerts()`: Get active alerts
- `get_alert_stats()`: Get alert statistics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📄 License

Part of the NyxTrade project. See main project license.
