"""
Unified Technical Indicator Service
Integrates TA-Lib, pandas-ta, FinTA, and custom indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Optional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None

try:
    from finta import TA as finta
    FINTA_AVAILABLE = True
except ImportError:
    FINTA_AVAILABLE = False
    finta = None

from utils.logger import get_logger
from .simple_indicators import (
    simple_moving_average, exponential_moving_average, relative_strength_index,
    bollinger_bands, macd, average_true_range, stochastic_oscillator, on_balance_volume
)

logger = get_logger(__name__)


class IndicatorLibrary(Enum):
    """Available indicator libraries"""
    TALIB = "talib"
    PANDAS_TA = "pandas_ta"
    FINTA = "finta"
    SIMPLE = "simple"
    CUSTOM = "custom"


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculation"""
    name: str
    library: IndicatorLibrary
    parameters: Dict[str, Any]
    period: Optional[int] = None


class IndicatorService:
    """
    Unified service for calculating technical indicators
    Supports multiple libraries and provides consistent interface
    """
    
    def __init__(self):
        self.supported_indicators = self._initialize_supported_indicators()
        logger.info("Initialized IndicatorService with multiple libraries")
    
    def _initialize_supported_indicators(self) -> Dict[str, Dict]:
        """Initialize mapping of supported indicators across libraries"""
        indicators = {}

        # Moving Averages
        sma_config = {"description": "Simple Moving Average", "simple": simple_moving_average}
        if TALIB_AVAILABLE:
            sma_config["talib"] = talib.SMA
        if PANDAS_TA_AVAILABLE:
            sma_config["pandas_ta"] = "sma"
        if FINTA_AVAILABLE:
            sma_config["finta"] = finta.SMA
        indicators["sma"] = sma_config

        ema_config = {"description": "Exponential Moving Average", "simple": exponential_moving_average}
        if TALIB_AVAILABLE:
            ema_config["talib"] = talib.EMA
        if PANDAS_TA_AVAILABLE:
            ema_config["pandas_ta"] = "ema"
        if FINTA_AVAILABLE:
            ema_config["finta"] = finta.EMA
        indicators["ema"] = ema_config

        wma_config = {"description": "Weighted Moving Average"}
        if TALIB_AVAILABLE:
            wma_config["talib"] = talib.WMA
        if PANDAS_TA_AVAILABLE:
            wma_config["pandas_ta"] = "wma"
        if FINTA_AVAILABLE:
            wma_config["finta"] = finta.WMA
        indicators["wma"] = wma_config

        # Oscillators
        rsi_config = {"description": "Relative Strength Index", "simple": relative_strength_index}
        if TALIB_AVAILABLE:
            rsi_config["talib"] = talib.RSI
        if PANDAS_TA_AVAILABLE:
            rsi_config["pandas_ta"] = "rsi"
        if FINTA_AVAILABLE:
            rsi_config["finta"] = finta.RSI
        indicators["rsi"] = rsi_config

        macd_config = {"description": "MACD"}
        if TALIB_AVAILABLE:
            macd_config["talib"] = talib.MACD
        if PANDAS_TA_AVAILABLE:
            macd_config["pandas_ta"] = "macd"
        if FINTA_AVAILABLE:
            macd_config["finta"] = finta.MACD
        indicators["macd"] = macd_config

        # Add more indicators as available libraries permit
        return indicators
    
    def calculate_indicator(self, 
                          data: pd.DataFrame,
                          indicator_name: str,
                          library: IndicatorLibrary = IndicatorLibrary.TALIB,
                          **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate technical indicator using specified library
        
        Args:
            data: OHLCV DataFrame
            indicator_name: Name of indicator to calculate
            library: Which library to use
            **kwargs: Indicator-specific parameters
        
        Returns:
            Calculated indicator values
        """
        try:
            if indicator_name not in self.supported_indicators:
                raise ValueError(f"Unsupported indicator: {indicator_name}")

            indicator_info = self.supported_indicators[indicator_name]

            if library == IndicatorLibrary.TALIB:
                if not TALIB_AVAILABLE:
                    raise ValueError("TA-Lib is not available. Please install it or use another library.")
                return self._calculate_talib_indicator(data, indicator_name, indicator_info, **kwargs)
            elif library == IndicatorLibrary.PANDAS_TA:
                if not PANDAS_TA_AVAILABLE:
                    raise ValueError("pandas-ta is not available. Please install it or use another library.")
                return self._calculate_pandas_ta_indicator(data, indicator_name, indicator_info, **kwargs)
            elif library == IndicatorLibrary.FINTA:
                if not FINTA_AVAILABLE:
                    raise ValueError("FinTA is not available. Please install it or use another library.")
                return self._calculate_finta_indicator(data, indicator_name, indicator_info, **kwargs)
            elif library == IndicatorLibrary.SIMPLE:
                return self._calculate_simple_indicator(data, indicator_name, indicator_info, **kwargs)
            else:
                raise ValueError(f"Unsupported library: {library}")

        except Exception as e:
            logger.error(f"Error calculating {indicator_name} with {library.value}: {e}")
            raise
    
    def _calculate_talib_indicator(self, data: pd.DataFrame, name: str, 
                                 info: Dict, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using TA-Lib"""
        func = info["talib"]
        
        if name in ["sma", "ema", "wma", "rsi"]:
            return func(data['close'], **kwargs)
        elif name == "macd":
            macd, signal, hist = func(data['close'], **kwargs)
            return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': hist})
        elif name == "bbands":
            upper, middle, lower = func(data['close'], **kwargs)
            return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})
        elif name == "stoch":
            slowk, slowd = func(data['high'], data['low'], data['close'], **kwargs)
            return pd.DataFrame({'slowk': slowk, 'slowd': slowd})
        elif name == "atr":
            return func(data['high'], data['low'], data['close'], **kwargs)
        elif name in ["obv", "ad"]:
            if name == "obv":
                return func(data['close'], data['volume'])
            else:
                return func(data['high'], data['low'], data['close'], data['volume'])
    
    def _calculate_pandas_ta_indicator(self, data: pd.DataFrame, name: str,
                                     info: Dict, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using pandas-ta"""
        indicator_func = getattr(ta, info["pandas_ta"])
        
        if name in ["sma", "ema", "wma", "rsi", "atr"]:
            return indicator_func(data['close'], **kwargs)
        elif name == "macd":
            return indicator_func(data['close'], **kwargs)
        elif name == "bbands":
            return indicator_func(data['close'], **kwargs)
        elif name == "stoch":
            return indicator_func(data['high'], data['low'], data['close'], **kwargs)
        elif name in ["obv", "ad"]:
            return indicator_func(data['close'], data['volume'], **kwargs)
    
    def _calculate_finta_indicator(self, data: pd.DataFrame, name: str,
                                 info: Dict, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using FinTA"""
        func = info["finta"]
        return func(data, **kwargs)

    def _calculate_simple_indicator(self, data: pd.DataFrame, name: str,
                                  info: Dict, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using simple implementations"""
        func = info["simple"]

        if name in ["sma", "ema", "rsi"]:
            return func(data['close'], **kwargs)
        elif name == "macd":
            return func(data['close'], **kwargs)
        elif name == "bbands":
            return bollinger_bands(data['close'], **kwargs)
        elif name == "atr":
            return func(data['high'], data['low'], data['close'], **kwargs)
        elif name == "stoch":
            return func(data['high'], data['low'], data['close'], **kwargs)
        elif name == "obv":
            return func(data['close'], data['volume'], **kwargs)
        else:
            return func(data['close'], **kwargs)
    
    def calculate_multiple_indicators(self, 
                                    data: pd.DataFrame,
                                    indicators: List[IndicatorConfig]) -> Dict[str, Any]:
        """
        Calculate multiple indicators at once
        Useful for strategy development and AI agent analysis
        """
        results = {}
        
        for config in indicators:
            try:
                result = self.calculate_indicator(
                    data, 
                    config.name, 
                    config.library,
                    **config.parameters
                )
                results[config.name] = result
                
            except Exception as e:
                logger.error(f"Failed to calculate {config.name}: {e}")
                results[config.name] = None
        
        return results
    
    def get_indicator_info(self, indicator_name: str) -> Optional[Dict]:
        """Get information about a specific indicator"""
        return self.supported_indicators.get(indicator_name)
    
    def list_supported_indicators(self) -> List[str]:
        """Get list of all supported indicators"""
        return list(self.supported_indicators.keys())
    
    def create_indicator_config(self, name: str, library: str, 
                              **parameters) -> IndicatorConfig:
        """Helper method to create indicator configuration"""
        return IndicatorConfig(
            name=name,
            library=IndicatorLibrary(library),
            parameters=parameters
        )
