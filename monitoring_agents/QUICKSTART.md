# NyxTrade Monitoring Agents - Quick Start Guide

## 🚀 Quick Start

### 1. Simple Demo (No Configuration Required)

```bash
cd monitoring_agents
python simple_demo.py
```

This will run a simplified demonstration showing the basic functionality of monitoring agents using simulated data, no API keys required.

### 2. Complete Environment Setup

#### Step 1: Setup Virtual Environment

```bash
cd monitoring_agents
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Create configuration file templates
- Setup Docker support

#### Step 2: Configure API Keys

```bash
cp .env.template .env
# Edit .env file and add your API keys
```

Required API keys:
- `GEMINI_API_KEY` - Google Gemini AI API key
- `BINANCE_API_KEY` - Binance API key (optional, for real-time data)

#### Step 3: Activate Environment

```bash
./activate.sh
```

#### Step 4: Run Complete Demo

```bash
python examples/agent_demo.py
```

### 3. Production Environment Deployment

#### Local Execution

```bash
./activate.sh
python run_agents.py
```

#### Docker Deployment

```bash
# Ensure .env file is configured
docker-compose up -d
```

#### Health Check

```bash
python scripts/health_check.py
```

## 📊 Monitoring Agent Features

### 1. BTC ETH Mean Reversion Monitoring
- Monitor BTC and ETH price deviation from historical mean
- Calculate Z-score and statistical significance
- Identify overbought/oversold opportunities
- Generate mean reversion trading signals

### 2. Trend Tracking Monitoring
- Identify market trend direction and strength
- Monitor trend reversal points
- Multi-timeframe analysis

### 3. Fund Flow Monitoring
- Monitor large capital inflows and outflows
- On-chain data analysis
- Institutional fund movements

### 4. Indicator Collection Monitoring
- Technical indicator aggregation
- Market sentiment indicators
- Macroeconomic data

### 5. Hotspot Tracking Monitoring
- Social media hotspots
- News event tracking
- Trending topic identification

## 🤖 Gemini AI Integration

### Get API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to `.env` file

### AI Analysis Features
- Intelligent market analysis
- Trend prediction
- Risk assessment
- Trading recommendation generation

## 🔧 Configuration Options

### Agent Configuration (`config/agents_config.yaml`)

```yaml
market_regression:
  enabled: true
  update_interval: 300  # 5 minutes
  symbols: ["BTC", "ETH"]
  alert_thresholds:
    z_score: 2.0
    deviation_percent: 10.0
  gemini_enabled: true
```

### Gemini Configuration (`config/gemini_config.yaml`)

```yaml
api_key: "your_api_key_here"
model: "gemini-pro"
temperature: 0.7
max_tokens: 2048
```

## 📈 Usage Examples

### Create Custom Agent

```python
from core.base_agent import BaseMonitoringAgent, AgentConfig

class CustomAgent(BaseMonitoringAgent):
    async def collect_data(self):
        # Implement data collection logic
        return {"custom_data": "value"}

    async def analyze_data(self, data):
        # Implement analysis logic
        return {"analysis": "result"}

    def get_gemini_prompt(self, data, analysis):
        return "Analyze this custom data..."

# Use agent
config = AgentConfig(name="custom_agent")
agent = CustomAgent(config)
result = await agent.run_analysis_cycle()
```

### Setup Alert Handling

```python
from core.alert_manager import AlertManager

async def custom_alert_handler(alert):
    print(f"Custom alert: {alert.message}")

alert_manager = AlertManager()
alert_manager.add_alert_handler(custom_alert_handler)
```

## 🧪 Testing

### Run Tests

```bash
./activate.sh
python -m pytest tests/ -v
```

### Basic Functionality Test

```bash
python -c "
from core.config_manager import ConfigManager
config = ConfigManager()
print('✅ Configuration manager working properly')
"
```

## 📁 Project Structure

```
monitoring_agents/
├── core/                    # Core infrastructure
├── agents/                  # Monitoring agent implementations
├── config/                  # Configuration files
├── examples/               # Demo scripts
├── tests/                  # Test files
├── logs/                   # Log files
├── scripts/                # Utility scripts
├── simple_demo.py          # Simple demo
├── setup.sh               # Environment setup script
├── activate.sh            # Environment activation script
├── run_agents.py          # Production run script
└── requirements.txt       # Python dependencies
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   ./activate.sh
   ```

2. **API Key Errors**
   ```bash
   # Check .env file configuration
   cat .env | grep GEMINI_API_KEY
   ```

3. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

### Log Viewing

```bash
# View real-time logs
tail -f logs/monitoring_agents.log

# View error logs
grep ERROR logs/monitoring_agents.log
```

## 📞 Support

If you encounter issues:
1. Check log files
2. Run health check script
3. Verify configuration files
4. Confirm API keys are valid

## 🔄 Updates

```bash
# Pull latest code
git pull

# Update dependencies
./activate.sh
pip install -r requirements.txt --upgrade
```
