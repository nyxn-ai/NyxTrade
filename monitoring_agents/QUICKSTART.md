# NyxTrade Monitoring Agents - Quick Start Guide

## 🚀 快速开始

### 1. 简单演示（无需配置）

```bash
cd monitoring_agents
python simple_demo.py
```

这将运行一个简化的演示，展示监控agent的基本功能，使用模拟数据，无需API密钥。

### 2. 完整环境设置

#### 步骤 1: 设置虚拟环境

```bash
cd monitoring_agents
chmod +x setup.sh
./setup.sh
```

这将：
- 创建Python虚拟环境
- 安装所有依赖
- 创建配置文件模板
- 设置Docker支持

#### 步骤 2: 配置API密钥

```bash
cp .env.template .env
# 编辑 .env 文件，添加您的API密钥
```

必需的API密钥：
- `GEMINI_API_KEY` - Google Gemini AI API密钥
- `BINANCE_API_KEY` - Binance API密钥（可选，用于实时数据）

#### 步骤 3: 激活环境

```bash
./activate.sh
```

#### 步骤 4: 运行完整演示

```bash
python examples/agent_demo.py
```

### 3. 生产环境部署

#### 本地运行

```bash
./activate.sh
python run_agents.py
```

#### Docker部署

```bash
# 确保 .env 文件已配置
docker-compose up -d
```

#### 健康检查

```bash
python scripts/health_check.py
```

## 📊 监控Agent功能

### 1. BTC ETH均价回归监控
- 监控BTC和ETH价格偏离历史均值
- 计算Z-score和统计显著性
- 识别超买超卖机会
- 生成回归交易信号

### 2. 趋势追踪监控
- 识别市场趋势方向和强度
- 监控趋势转换点
- 多时间框架分析

### 3. 资金动向监控
- 监控大额资金流入流出
- 链上数据分析
- 机构资金动向

### 4. 指标收集监控
- 技术指标聚合
- 市场情绪指标
- 宏观经济数据

### 5. 热点追踪监控
- 社交媒体热点
- 新闻事件追踪
- 趋势话题识别

## 🤖 Gemini AI集成

### 获取API密钥
1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的API密钥
3. 将密钥添加到 `.env` 文件

### AI分析功能
- 智能市场分析
- 趋势预测
- 风险评估
- 交易建议生成

## 🔧 配置选项

### Agent配置 (`config/agents_config.yaml`)

```yaml
market_regression:
  enabled: true
  update_interval: 300  # 5分钟
  symbols: ["BTC", "ETH"]
  alert_thresholds:
    z_score: 2.0
    deviation_percent: 10.0
  gemini_enabled: true
```

### Gemini配置 (`config/gemini_config.yaml`)

```yaml
api_key: "your_api_key_here"
model: "gemini-pro"
temperature: 0.7
max_tokens: 2048
```

## 📈 使用示例

### 创建自定义Agent

```python
from core.base_agent import BaseMonitoringAgent, AgentConfig

class CustomAgent(BaseMonitoringAgent):
    async def collect_data(self):
        # 实现数据收集逻辑
        return {"custom_data": "value"}
    
    async def analyze_data(self, data):
        # 实现分析逻辑
        return {"analysis": "result"}
    
    def get_gemini_prompt(self, data, analysis):
        return "分析这些自定义数据..."

# 使用agent
config = AgentConfig(name="custom_agent")
agent = CustomAgent(config)
result = await agent.run_analysis_cycle()
```

### 设置警报处理

```python
from core.alert_manager import AlertManager

async def custom_alert_handler(alert):
    print(f"自定义警报: {alert.message}")

alert_manager = AlertManager()
alert_manager.add_alert_handler(custom_alert_handler)
```

## 🧪 测试

### 运行测试

```bash
./activate.sh
python -m pytest tests/ -v
```

### 基本功能测试

```bash
python -c "
from core.config_manager import ConfigManager
config = ConfigManager()
print('✅ 配置管理器工作正常')
"
```

## 📁 项目结构

```
monitoring_agents/
├── core/                    # 核心基础设施
├── agents/                  # 监控agent实现
├── config/                  # 配置文件
├── examples/               # 演示脚本
├── tests/                  # 测试文件
├── logs/                   # 日志文件
├── scripts/                # 工具脚本
├── simple_demo.py          # 简单演示
├── setup.sh               # 环境设置脚本
├── activate.sh            # 环境激活脚本
├── run_agents.py          # 生产运行脚本
└── requirements.txt       # Python依赖
```

## 🔍 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保激活了虚拟环境
   ./activate.sh
   ```

2. **API密钥错误**
   ```bash
   # 检查 .env 文件配置
   cat .env | grep GEMINI_API_KEY
   ```

3. **依赖问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt
   ```

### 日志查看

```bash
# 查看实时日志
tail -f logs/monitoring_agents.log

# 查看错误日志
grep ERROR logs/monitoring_agents.log
```

## 📞 支持

如果遇到问题：
1. 查看日志文件
2. 运行健康检查脚本
3. 检查配置文件
4. 确认API密钥有效

## 🔄 更新

```bash
# 拉取最新代码
git pull

# 更新依赖
./activate.sh
pip install -r requirements.txt --upgrade
```
