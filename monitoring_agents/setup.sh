#!/bin/bash

# NyxTrade Monitoring Agents Setup Script
# Creates virtual environment and installs dependencies

set -e

echo "🚀 Setting up NyxTrade Monitoring Agents System"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "📁 Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "📦 Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install \
    aiohttp>=3.8.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    PyYAML>=6.0 \
    python-dateutil>=2.8.0

# Install AI dependencies
echo "🤖 Installing AI dependencies..."
pip install google-generativeai>=0.3.0

# Install optional dependencies for enhanced functionality
echo "📊 Installing optional dependencies..."
pip install \
    pandas-ta>=0.3.14b \
    finta>=1.3 \
    requests>=2.28.0 \
    websockets>=11.0 \
    redis>=4.5.0 \
    python-dotenv>=1.0.0

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0

# Create requirements.txt
echo "📝 Generating requirements.txt..."
pip freeze > requirements.txt

# Create .env template
echo "⚙️  Creating environment template..."
cat > .env.template << 'EOF'
# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret

# Data Provider API Keys
COINGECKO_API_KEY=your_coingecko_api_key
GLASSNODE_API_KEY=your_glassnode_api_key
NEWSAPI_KEY=your_newsapi_key

# Social Media API Keys
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Database Configuration (Optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Monitoring Configuration
LOG_LEVEL=INFO
ALERT_WEBHOOK_URL=your_webhook_url
EOF

# Create activation script
echo "🔧 Creating activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
# Activation script for NyxTrade Monitoring Agents

echo "🔌 Activating NyxTrade Monitoring Agents environment..."

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    echo "📁 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Copy .env.template to .env and configure your API keys."
fi

echo "✅ Environment activated!"
echo "💡 Run 'python examples/agent_demo.py' to test the system"
EOF

chmod +x activate.sh

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {logs,data,tests,scripts}

# Create basic test file
echo "🧪 Creating basic test..."
cat > tests/test_basic.py << 'EOF'
"""
Basic tests for monitoring agents system
"""

import pytest
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import ConfigManager
from core.alert_manager import AlertManager
from core.data_collector import DataCollector


def test_config_manager():
    """Test configuration manager"""
    config_manager = ConfigManager()
    
    # Test loading configurations
    agents_config = config_manager.get_agents_config()
    assert isinstance(agents_config, dict)
    assert "market_regression" in agents_config
    
    gemini_config = config_manager.get_gemini_config()
    assert isinstance(gemini_config, dict)
    assert "model" in gemini_config


def test_alert_manager():
    """Test alert manager"""
    alert_manager = AlertManager()
    
    # Test basic functionality
    assert len(alert_manager.active_alerts) == 0
    assert len(alert_manager.alert_history) == 0
    
    stats = alert_manager.get_alert_stats()
    assert stats["active_alerts_count"] == 0


@pytest.mark.asyncio
async def test_data_collector():
    """Test data collector"""
    async with DataCollector() as collector:
        # Test mock data generation
        price = await collector.get_current_price("BTCUSDT")
        assert isinstance(price, float)
        assert price > 0
        
        # Test historical data
        historical = await collector.get_historical_prices("BTCUSDT", limit=10)
        assert len(historical) == 10
        assert "close" in historical.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create run script
echo "🚀 Creating run script..."
cat > run_agents.py << 'EOF'
#!/usr/bin/env python3
"""
Production runner for NyxTrade Monitoring Agents
"""

import asyncio
import logging
import signal
import sys
import os
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.base_agent import BaseMonitoringAgent, AgentConfig
from core.alert_manager import AlertManager, console_alert_handler, log_alert_handler
from agents.market_regression.btc_eth_regression_agent import BTCETHRegressionAgent


class AgentRunner:
    """Manages multiple monitoring agents"""
    
    def __init__(self):
        self.agents: List[BaseMonitoringAgent] = []
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/monitoring_agents.log')
            ]
        )
        
        self.logger = logging.getLogger("agent_runner")
    
    def add_agent(self, agent: BaseMonitoringAgent):
        """Add agent to runner"""
        self.agents.append(agent)
        self.logger.info(f"Added agent: {agent.name}")
    
    async def start_all_agents(self):
        """Start all agents"""
        self.running = True
        self.logger.info(f"Starting {len(self.agents)} agents...")
        
        # Start all agents concurrently
        tasks = []
        for agent in self.agents:
            task = asyncio.create_task(agent.start_monitoring())
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            await self.stop_all_agents()
    
    async def stop_all_agents(self):
        """Stop all agents"""
        self.running = False
        self.logger.info("Stopping all agents...")
        
        for agent in self.agents:
            agent.stop_monitoring()
        
        self.logger.info("All agents stopped")


async def main():
    """Main function"""
    print("🚀 Starting NyxTrade Monitoring Agents")
    
    # Create runner
    runner = AgentRunner()
    
    # Create and add agents
    btc_eth_agent = BTCETHRegressionAgent()
    runner.add_agent(btc_eth_agent)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal")
        runner.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start agents
    try:
        await runner.start_all_agents()
    except Exception as e:
        print(f"❌ Error running agents: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
EOF

chmod +x run_agents.py

# Create Docker support
echo "🐳 Creating Docker support..."
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Expose port for health checks
EXPOSE 8080

# Run the application
CMD ["python", "run_agents.py"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  monitoring-agents:
    build: .
    container_name: nyxtrade-monitoring-agents
    restart: unless-stopped
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8080:8080"
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    container_name: nyxtrade-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
EOF

# Create health check script
cat > scripts/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Health check script for monitoring agents
"""

import asyncio
import aiohttp
import sys
import os

async def check_health():
    """Check if monitoring agents are healthy"""
    try:
        # Add your health check logic here
        # For example, check if agents are responding
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(check_health())
    sys.exit(0 if result else 1)
EOF

chmod +x scripts/health_check.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and configure your API keys"
echo "2. Run './activate.sh' to activate the environment"
echo "3. Run 'python examples/agent_demo.py' to test the system"
echo "4. Run 'python run_agents.py' to start production monitoring"
echo ""
echo "🐳 Docker deployment:"
echo "1. Configure .env file with your API keys"
echo "2. Run 'docker-compose up -d' to start with Docker"
echo ""
echo "🧪 Testing:"
echo "1. Run 'python -m pytest tests/ -v' to run tests"
echo ""
echo "📁 Important files created:"
echo "   • activate.sh - Environment activation script"
echo "   • .env.template - Environment variables template"
echo "   • run_agents.py - Production runner"
echo "   • Dockerfile & docker-compose.yml - Docker deployment"
echo "   • requirements.txt - Python dependencies"
echo ""
