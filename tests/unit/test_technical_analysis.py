"""Unit tests for enhanced backtesting technical analysis components."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import sys
sys.path.append('.')

from enhanced_backtest import (
    TechnicalAnalyzer, StrategySelector, TechnicalIndicators,
    StrategyType, MarketSignal, StrategySelection
)
from tests.conftest import TestDataGenerator, assert_valid_technical_indicators


class TestTechnicalAnalyzer:
    """Test technical analysis calculations."""
    
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()
        self.sample_prices = pd.Series([
            4500, 4510, 4505, 4515, 4520, 4525, 4530, 4535, 4540, 4545,
            4550, 4555, 4560, 4565, 4570, 4575, 4580, 4585, 4590, 4595,
            4600, 4605, 4610, 4615, 4620, 4625, 4630, 4635, 4640, 4645,
            4650, 4655, 4660, 4665, 4670, 4675, 4680, 4685, 4690, 4695,
            4700, 4705, 4710, 4715, 4720, 4725, 4730, 4735, 4740, 4745
        ])
    
    def test_calculate_rsi_valid_range(self):
        """Test RSI calculation returns value in valid range."""
        rsi = self.analyzer.calculate_rsi(self.sample_prices)
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)
    
    def test_calculate_rsi_trending_up(self):
        """Test RSI for uptrending prices."""
        uptrend_prices = pd.Series(range(4500, 4600, 2))  # Consistent uptrend
        rsi = self.analyzer.calculate_rsi(uptrend_prices)
        assert rsi > 50  # Should be above 50 for uptrend
    
    def test_calculate_rsi_trending_down(self):
        """Test RSI for downtrending prices."""
        downtrend_prices = pd.Series(range(4600, 4500, -2))  # Consistent downtrend
        rsi = self.analyzer.calculate_rsi(downtrend_prices)
        assert rsi < 50  # Should be below 50 for downtrend
    
    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        short_series = pd.Series([4500, 4510, 4505])
        rsi = self.analyzer.calculate_rsi(short_series, period=14)
        assert rsi == 50.0  # Should return default value
    
    def test_calculate_macd_returns_tuple(self):
        """Test MACD calculation returns proper tuple."""
        macd_line, macd_signal, macd_histogram = self.analyzer.calculate_macd(self.sample_prices)
        
        assert isinstance(macd_line, float)
        assert isinstance(macd_signal, float)
        assert isinstance(macd_histogram, float)
        assert macd_histogram == macd_line - macd_signal
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_upper, bb_middle, bb_lower, bb_position = self.analyzer.calculate_bollinger_bands(
            self.sample_prices
        )
        
        # Validate band ordering
        assert bb_upper > bb_middle > bb_lower
        assert 0 <= bb_position <= 1
        
        # Current price should be close to upper band for uptrend
        current_price = self.sample_prices.iloc[-1]
        assert abs(current_price - bb_middle) / (bb_upper - bb_middle) <= 2
    
    def test_analyze_market_conditions_complete(self):
        """Test complete market analysis."""
        indicators = self.analyzer.analyze_market_conditions(self.sample_prices)
        
        assert_valid_technical_indicators(indicators)
        assert isinstance(indicators, TechnicalIndicators)
    
    def test_analyze_market_conditions_insufficient_data(self):
        """Test market analysis with insufficient data."""
        short_prices = pd.Series([4500, 4510])
        indicators = self.analyzer.analyze_market_conditions(short_prices)
        
        # Should return default values
        assert indicators.rsi == 50.0
        assert indicators.bb_position == 0.5
        assert_valid_technical_indicators(indicators)


class TestStrategySelector:
    """Test strategy selection logic."""
    
    def setup_method(self):
        self.selector = StrategySelector()
    
    def test_select_strategy_neutral_market(self):
        """Test strategy selection for neutral market conditions."""
        # Create neutral indicators
        indicators = TechnicalIndicators(
            rsi=50.0,  # Neutral RSI
            macd_line=0.1,
            macd_signal=0.1,  # Neutral MACD
            macd_histogram=0.0,
            bb_upper=4600,
            bb_middle=4550,
            bb_lower=4500,
            bb_position=0.5  # Middle of bands
        )
        
        selection = self.selector.select_strategy(indicators)
        
        assert isinstance(selection, StrategySelection)
        assert selection.strategy_type == StrategyType.IRON_CONDOR
        assert selection.market_signal == MarketSignal.NEUTRAL
        assert 0 <= selection.confidence <= 1
        assert "neutral" in selection.reason.lower()
    
    def test_select_strategy_bullish_market(self):
        """Test strategy selection for bullish market conditions."""
        indicators = TechnicalIndicators(
            rsi=25.0,  # Oversold = bullish signal
            macd_line=0.5,
            macd_signal=0.0,  # Positive MACD = bullish
            macd_histogram=0.5,
            bb_upper=4600,
            bb_middle=4550,
            bb_lower=4500,
            bb_position=0.1  # Near lower band = bullish
        )
        
        selection = self.selector.select_strategy(indicators)
        
        assert selection.strategy_type == StrategyType.CALL_SPREAD
        assert selection.market_signal == MarketSignal.BULLISH
        assert selection.confidence > 0
        assert "bullish" in selection.reason.lower()
    
    def test_select_strategy_bearish_market(self):
        """Test strategy selection for bearish market conditions."""
        indicators = TechnicalIndicators(
            rsi=75.0,  # Overbought = bearish signal
            macd_line=-0.5,
            macd_signal=0.0,  # Negative MACD = bearish
            macd_histogram=-0.5,
            bb_upper=4600,
            bb_middle=4550,
            bb_lower=4500,
            bb_position=0.9  # Near upper band = bearish
        )
        
        selection = self.selector.select_strategy(indicators)
        
        assert selection.strategy_type == StrategyType.PUT_SPREAD
        assert selection.market_signal == MarketSignal.BEARISH
        assert selection.confidence > 0
        assert "bearish" in selection.reason.lower()
    
    def test_select_strategy_mixed_signals(self):
        """Test strategy selection with mixed signals."""
        indicators = TechnicalIndicators(
            rsi=65.0,  # Slightly overbought
            macd_line=0.2,
            macd_signal=0.1,  # Slightly bullish MACD
            macd_histogram=0.1,
            bb_upper=4600,
            bb_middle=4550,
            bb_lower=4500,
            bb_position=0.6  # Slightly above middle
        )
        
        selection = self.selector.select_strategy(indicators)
        
        # Should handle mixed signals gracefully
        assert selection.strategy_type in [StrategyType.IRON_CONDOR, StrategyType.PUT_SPREAD, StrategyType.CALL_SPREAD]
        assert isinstance(selection.confidence, float)
        assert 0 <= selection.confidence <= 1


class TestTechnicalIndicatorEdgeCases:
    """Test edge cases and error handling in technical analysis."""
    
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()
    
    def test_empty_price_series(self):
        """Test handling of empty price series."""
        empty_series = pd.Series([], dtype=float)
        
        # Should not crash and return sensible defaults
        with pytest.raises((IndexError, ValueError)):
            self.analyzer.calculate_rsi(empty_series)
    
    def test_constant_prices(self):
        """Test handling of constant prices."""
        constant_prices = pd.Series([4500] * 50)
        
        rsi = self.analyzer.calculate_rsi(constant_prices)
        # RSI should be 50 for no price movement
        assert rsi == 50.0
        
        macd_line, macd_signal, macd_histogram = self.analyzer.calculate_macd(constant_prices)
        # MACD should be near 0 for no trend
        assert abs(macd_line) < 0.01
        assert abs(macd_signal) < 0.01
        assert abs(macd_histogram) < 0.01
    
    def test_extreme_volatility(self):
        """Test handling of extreme price volatility."""
        volatile_prices = pd.Series([4500, 5000, 4000, 5500, 3500] * 10)
        
        indicators = self.analyzer.analyze_market_conditions(volatile_prices)
        assert_valid_technical_indicators(indicators)
        
        # Should still produce valid values despite volatility
        assert 0 <= indicators.rsi <= 100
        assert 0 <= indicators.bb_position <= 1
    
    def test_nan_in_price_series(self):
        """Test handling of NaN values in price series."""
        prices_with_nan = pd.Series([4500, 4510, np.nan, 4520, 4530] * 10)
        
        # Should handle NaN gracefully
        indicators = self.analyzer.analyze_market_conditions(prices_with_nan)
        
        # All indicator values should be valid (not NaN)
        assert not pd.isna(indicators.rsi)
        assert not pd.isna(indicators.bb_position)
        assert not pd.isna(indicators.macd_line)