"""
Monitoring agents for various market analysis tasks
"""

from .market_regression.btc_eth_regression_agent import BTCETHRegressionAgent
from .trend_tracking.trend_agent import TrendTrackingAgent
from .fund_flow.fund_flow_agent import FundFlowAgent
from .indicator_collector.indicator_agent import IndicatorCollectorAgent
from .hotspot_tracking.hotspot_agent import HotspotTrackingAgent

__all__ = [
    "BTCETHRegressionAgent",
    "TrendTrackingAgent", 
    "FundFlowAgent",
    "IndicatorCollectorAgent",
    "HotspotTrackingAgent"
]
