"""
Backtesting Engine for NyxTrade Strategy Engine
Supports comprehensive strategy backtesting with performance analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .strategy_base import StrategyBase, TradingSignal, MarketData, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_positions: int = 1
    risk_per_trade: float = 0.02    # 2% risk per trade


@dataclass
class BacktestResult:
    """Backtesting results"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    profit_factor: float
    equity_curve: pd.Series
    trade_log: List[Dict]
    performance_metrics: Dict[str, float]


class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    Designed for AI agent optimization and analysis
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.daily_returns = []
        
        logger.info(f"Initialized BacktestEngine with capital: ${config.initial_capital:,.2f}")
    
    def run_backtest(self, 
                    strategy: StrategyBase,
                    market_data: pd.DataFrame) -> BacktestResult:
        """
        Run comprehensive backtest for a strategy
        
        Args:
            strategy: Strategy instance to test
            market_data: Historical OHLCV data
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Reset strategy and engine state
        strategy.reset_strategy()
        self._reset_engine_state()
        
        # Filter data by date range
        filtered_data = self._filter_data_by_date(market_data)
        
        if filtered_data.empty:
            raise ValueError("No data available for specified date range")
        
        # Run simulation
        for i in range(len(filtered_data)):
            current_row = filtered_data.iloc[i]
            historical_data = filtered_data.iloc[:i+1]
            
            # Create market data object
            market_data_obj = MarketData(
                symbol=strategy.symbol,
                timestamp=current_row.name,
                open=current_row['open'],
                high=current_row['high'],
                low=current_row['low'],
                close=current_row['close'],
                volume=current_row['volume'],
                timeframe=strategy.timeframe
            )
            
            # Generate signal
            try:
                signal = strategy.generate_signal(market_data_obj, historical_data)
                
                if signal and strategy.validate_signal(signal):
                    self._process_signal(strategy, signal, current_row)
                    
            except Exception as e:
                logger.error(f"Error processing signal at {current_row.name}: {e}")
                continue
            
            # Update equity curve
            self._update_equity_curve(current_row.name, current_row['close'])
        
        # Calculate final results
        return self._calculate_results(strategy)
    
    def _reset_engine_state(self):
        """Reset engine state for new backtest"""
        self.current_capital = self.config.initial_capital
        self.positions.clear()
        self.trade_log.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by backtest date range"""
        return data[
            (data.index >= self.config.start_date) & 
            (data.index <= self.config.end_date)
        ].copy()
    
    def _process_signal(self, strategy: StrategyBase, signal: TradingSignal, 
                       current_data: pd.Series):
        """Process trading signal and execute trades"""
        symbol = signal.symbol
        current_price = current_data['close']
        
        # Apply slippage
        if signal.signal_type in [SignalType.BUY]:
            execution_price = current_price * (1 + self.config.slippage_rate)
        else:
            execution_price = current_price * (1 - self.config.slippage_rate)
        
        if signal.signal_type == SignalType.BUY and symbol not in self.positions:
            self._execute_buy(strategy, signal, execution_price, current_data.name)
            
        elif signal.signal_type == SignalType.SELL and symbol in self.positions:
            self._execute_sell(strategy, signal, execution_price, current_data.name)
            
        elif signal.signal_type == SignalType.CLOSE_LONG and symbol in self.positions:
            self._execute_sell(strategy, signal, execution_price, current_data.name)
    
    def _execute_buy(self, strategy: StrategyBase, signal: TradingSignal,
                    price: float, timestamp: datetime):
        """Execute buy order"""
        # Calculate position size
        position_size = strategy.calculate_position_size(signal, self.current_capital)
        
        if position_size <= 0:
            return
        
        # Calculate costs
        trade_value = position_size * price
        commission = trade_value * self.config.commission_rate
        total_cost = trade_value + commission
        
        if total_cost > self.current_capital:
            logger.warning(f"Insufficient capital for trade: ${total_cost:,.2f} > ${self.current_capital:,.2f}")
            return
        
        # Execute trade
        self.positions[signal.symbol] = {
            'quantity': position_size,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_signal': signal
        }
        
        self.current_capital -= total_cost
        
        # Log trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': signal.symbol,
            'action': 'BUY',
            'quantity': position_size,
            'price': price,
            'commission': commission,
            'capital_after': self.current_capital
        }
        
        self.trade_log.append(trade_record)
        logger.debug(f"Executed BUY: {trade_record}")
    
    def _execute_sell(self, strategy: StrategyBase, signal: TradingSignal,
                     price: float, timestamp: datetime):
        """Execute sell order"""
        if signal.symbol not in self.positions:
            return
        
        position = self.positions[signal.symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # Calculate trade results
        trade_value = quantity * price
        commission = trade_value * self.config.commission_rate
        net_proceeds = trade_value - commission
        
        # Calculate P&L
        entry_cost = quantity * entry_price * (1 + self.config.commission_rate)
        pnl = net_proceeds - entry_cost
        pnl_percentage = (pnl / entry_cost) * 100
        
        # Update capital
        self.current_capital += net_proceeds
        
        # Calculate trade duration
        trade_duration = (timestamp - position['entry_time']).total_seconds() / 3600  # hours
        
        # Record trade in strategy
        strategy.record_trade(signal, price, quantity, pnl)
        
        # Log trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': signal.symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'commission': commission,
            'duration_hours': trade_duration,
            'capital_after': self.current_capital
        }
        
        self.trade_log.append(trade_record)
        
        # Remove position
        del self.positions[signal.symbol]
        
        logger.debug(f"Executed SELL: {trade_record}")
    
    def _update_equity_curve(self, timestamp: datetime, current_price: float):
        """Update equity curve with current portfolio value"""
        portfolio_value = self.current_capital
        
        # Add unrealized P&L from open positions
        for symbol, position in self.positions.items():
            unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            portfolio_value += unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value
        })
    
    def _calculate_results(self, strategy: StrategyBase) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not self.equity_curve:
            raise ValueError("No equity curve data available")
        
        # Convert equity curve to pandas Series
        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = equity_df.set_index('timestamp')['portfolio_value']
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] / self.config.initial_capital - 1) * 100
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_series)
        
        # Trade metrics
        winning_trades = [t for t in self.trade_log if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_log if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(self.trade_log) * 100 if self.trade_log else 0
        
        avg_trade_duration = np.mean([t.get('duration_hours', 0) for t in self.trade_log]) if self.trade_log else 0
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Performance metrics
        performance_metrics = {
            'total_trades': len(self.trade_log),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0,
        }
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trade_log),
            avg_trade_duration=avg_trade_duration,
            profit_factor=profit_factor,
            equity_curve=equity_series,
            trade_log=self.trade_log,
            performance_metrics=performance_metrics
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100
