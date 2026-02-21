"""Performance and benchmark tests for the SPX AI trading system."""

import pytest
import time
import psutil
import os
from datetime import datetime
import pandas as pd
import numpy as np

import sys
sys.path.append('.')

from enhanced_backtest import TechnicalAnalyzer, StrategySelector
from tests.conftest import TestDataGenerator


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for core system components."""
    
    def setup_method(self):
        """Set up performance testing environment."""
        self.analyzer = TechnicalAnalyzer()
        self.selector = StrategySelector()
        
        # Create performance test data
        self.small_dataset = pd.Series(np.random.normal(4550, 20, 100))
        self.medium_dataset = pd.Series(np.random.normal(4550, 20, 1000))
        self.large_dataset = pd.Series(np.random.normal(4550, 20, 10000))
    
    @pytest.mark.fast
    def test_rsi_calculation_performance(self):
        """Test RSI calculation performance across dataset sizes."""
        
        # Small dataset benchmark
        start_time = time.time()
        rsi_small = self.analyzer.calculate_rsi(self.small_dataset)
        small_time = time.time() - start_time
        
        # Medium dataset benchmark
        start_time = time.time()
        rsi_medium = self.analyzer.calculate_rsi(self.medium_dataset)
        medium_time = time.time() - start_time
        
        # Large dataset benchmark
        start_time = time.time()
        rsi_large = self.analyzer.calculate_rsi(self.large_dataset)
        large_time = time.time() - start_time
        
        # Performance assertions
        assert small_time < 0.01  # Should be very fast for small data
        assert medium_time < 0.05  # Should be fast for medium data
        assert large_time < 0.2   # Should be reasonable for large data
        
        # Results should be valid regardless of dataset size
        assert 0 <= rsi_small <= 100
        assert 0 <= rsi_medium <= 100
        assert 0 <= rsi_large <= 100
    
    @pytest.mark.fast
    def test_macd_calculation_performance(self):
        """Test MACD calculation performance."""
        
        start_time = time.time()
        for _ in range(100):  # Run 100 times to get meaningful measurement
            macd_line, macd_signal, macd_histogram = self.analyzer.calculate_macd(
                self.medium_dataset
            )
        execution_time = time.time() - start_time
        
        # Should complete 100 calculations in reasonable time
        assert execution_time < 1.0
        
        # Results should be valid
        assert isinstance(macd_line, float)
        assert isinstance(macd_signal, float) 
        assert isinstance(macd_histogram, float)
    
    @pytest.mark.fast
    def test_bollinger_bands_performance(self):
        """Test Bollinger Bands calculation performance."""
        
        start_time = time.time()
        bb_upper, bb_middle, bb_lower, bb_position = self.analyzer.calculate_bollinger_bands(
            self.large_dataset
        )
        execution_time = time.time() - start_time
        
        # Should complete quickly even for large dataset
        assert execution_time < 0.1
        
        # Results should be valid
        assert bb_upper > bb_middle > bb_lower
        assert 0 <= bb_position <= 1
    
    @pytest.mark.fast
    def test_complete_technical_analysis_performance(self):
        """Test complete technical analysis performance."""
        
        start_time = time.time()
        indicators = self.analyzer.analyze_market_conditions(self.medium_dataset)
        execution_time = time.time() - start_time
        
        # Complete analysis should be fast
        assert execution_time < 0.1
        
        # Results should be comprehensive and valid
        assert 0 <= indicators.rsi <= 100
        assert indicators.bb_upper > indicators.bb_lower
        assert 0 <= indicators.bb_position <= 1
    
    @pytest.mark.fast
    def test_strategy_selection_performance(self):
        """Test strategy selection performance."""
        
        # Create various indicator scenarios
        scenarios = []
        for _ in range(1000):
            from enhanced_backtest import TechnicalIndicators
            indicators = TechnicalIndicators(
                rsi=np.random.uniform(0, 100),
                macd_line=np.random.normal(0, 1),
                macd_signal=np.random.normal(0, 1),
                macd_histogram=np.random.normal(0, 0.5),
                bb_upper=4600,
                bb_middle=4550,
                bb_lower=4500,
                bb_position=np.random.uniform(0, 1)
            )
            scenarios.append(indicators)
        
        # Time strategy selection for all scenarios
        start_time = time.time()
        selections = []
        for indicators in scenarios:
            selection = self.selector.select_strategy(indicators)
            selections.append(selection)
        execution_time = time.time() - start_time
        
        # Should process 1000 selections quickly
        assert execution_time < 1.0
        assert len(selections) == 1000
        
        # All selections should be valid
        for selection in selections:
            assert selection.strategy_type is not None
            assert 0 <= selection.confidence <= 1


@pytest.mark.performance
@pytest.mark.slow
class TestMemoryUsageBenchmarks:
    """Memory usage benchmarks for system components."""
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_technical_analysis_memory_usage(self):
        """Test memory usage of technical analysis components."""
        
        initial_memory = self.get_memory_usage()
        
        # Create large datasets and analyze them
        analyzer = TechnicalAnalyzer()
        large_datasets = []
        
        for i in range(10):
            # Create 10 large datasets (10k points each)
            dataset = pd.Series(np.random.normal(4550, 20, 10000))
            large_datasets.append(dataset)
            
            # Analyze each dataset
            indicators = analyzer.analyze_market_conditions(dataset)
            assert indicators is not None
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100
        
        # Clean up
        del large_datasets
        
        # Memory should be manageable after cleanup
        final_memory = self.get_memory_usage()
        assert final_memory - initial_memory < 50
    
    def test_options_data_memory_usage(self):
        """Test memory usage when processing large options datasets."""
        
        initial_memory = self.get_memory_usage()
        
        # Create large options dataset
        large_options_datasets = []
        for i in range(5):
            options_data = TestDataGenerator.create_sample_options_data()
            # Expand dataset (simulate full options chain)
            expanded_data = pd.concat([options_data] * 50, ignore_index=True)
            large_options_datasets.append(expanded_data)
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Should handle large options datasets efficiently
        assert memory_increase < 200  # Less than 200MB increase
        
        # Clean up
        del large_options_datasets
        
        final_memory = self.get_memory_usage()
        assert final_memory - initial_memory < 50


@pytest.mark.edge_cases
class TestEdgeCaseHandling:
    """Test system behavior under edge cases and extreme conditions."""
    
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()
        self.selector = StrategySelector()
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        
        empty_series = pd.Series([], dtype=float)
        
        # Should handle empty data gracefully (may raise exceptions, which is acceptable)
        try:
            rsi = self.analyzer.calculate_rsi(empty_series)
        except (IndexError, ValueError):
            pass  # Acceptable to raise exception for empty data
    
    def test_single_data_point_handling(self):
        """Test handling of single data point."""
        
        single_point = pd.Series([4550.0])
        
        # Should handle single point gracefully
        try:
            indicators = self.analyzer.analyze_market_conditions(single_point)
        except (IndexError, ValueError):
            pass  # May not be able to calculate with single point
    
    def test_constant_price_handling(self):
        """Test handling of constant prices (no volatility)."""
        
        constant_prices = pd.Series([4550.0] * 100)
        
        # Should handle constant prices without error
        indicators = self.analyzer.analyze_market_conditions(constant_prices)
        
        # RSI should be neutral for no movement
        assert indicators.rsi == 50.0
        
        # MACD should be near zero
        assert abs(indicators.macd_line) < 0.01
        assert abs(indicators.macd_signal) < 0.01
        assert abs(indicators.macd_histogram) < 0.01
        
        # Bollinger bands should be tight
        band_width = indicators.bb_upper - indicators.bb_lower
        assert band_width < 1.0  # Very tight bands for no volatility
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements."""
        
        # Create extreme volatility scenario
        extreme_prices = pd.Series([
            4500, 5000, 4000, 5500, 3500, 6000, 3000, 6500, 2500, 7000
        ] * 10)  # Repeat pattern
        
        # Should handle extreme movements without crashing
        indicators = self.analyzer.analyze_market_conditions(extreme_prices)
        selection = self.selector.select_strategy(indicators)
        
        # Results should still be valid despite extreme conditions
        assert 0 <= indicators.rsi <= 100
        assert 0 <= indicators.bb_position <= 1
        assert selection.strategy_type is not None
    
    def test_nan_and_infinite_value_handling(self):
        """Test handling of NaN and infinite values."""
        
        # Create dataset with problematic values
        problematic_data = pd.Series([
            4500, 4510, np.nan, 4520, np.inf, 4530, -np.inf, 4540
        ] * 10)
        
        # Should handle NaN and infinite values gracefully
        try:
            indicators = self.analyzer.analyze_market_conditions(problematic_data)
            
            # Results should not contain NaN values
            assert not pd.isna(indicators.rsi)
            assert not pd.isna(indicators.bb_position)
            assert not np.isinf(indicators.rsi)
            assert not np.isinf(indicators.bb_position)
            
        except (ValueError, TypeError):
            # Acceptable to raise exception for invalid data
            pass
    
    def test_very_large_numbers(self):
        """Test handling of very large price values."""
        
        # Create dataset with very large numbers
        large_numbers = pd.Series([1e6, 1e6 + 100, 1e6 - 50] * 50)
        
        # Should handle large numbers without overflow
        try:
            indicators = self.analyzer.analyze_market_conditions(large_numbers)
            assert 0 <= indicators.rsi <= 100
            assert 0 <= indicators.bb_position <= 1
        except (OverflowError, ValueError):
            # May not handle extremely large numbers
            pass
    
    def test_very_small_numbers(self):
        """Test handling of very small price values."""
        
        # Create dataset with very small numbers
        small_numbers = pd.Series([0.001, 0.0011, 0.0009] * 50)
        
        # Should handle small numbers without underflow
        try:
            indicators = self.analyzer.analyze_market_conditions(small_numbers)
            assert 0 <= indicators.rsi <= 100
            assert 0 <= indicators.bb_position <= 1
        except (ValueError, ZeroDivisionError):
            # May have issues with very small numbers
            pass


@pytest.mark.performance
class TestConcurrencyAndStress:
    """Test system behavior under concurrent load and stress conditions."""
    
    def test_multiple_concurrent_analyses(self):
        """Simulate multiple concurrent technical analyses."""
        
        import threading
        import concurrent.futures
        
        analyzer = TechnicalAnalyzer()
        
        def run_analysis(dataset_id):
            """Run technical analysis on a dataset."""
            np.random.seed(dataset_id)  # Deterministic per thread
            prices = pd.Series(np.random.normal(4550, 20, 1000))
            return analyzer.analyze_market_conditions(prices)
        
        # Run multiple analyses concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_analysis, i) for i in range(20)]
            results = [future.result() for future in futures]
        
        # All results should be valid
        assert len(results) == 20
        for indicators in results:
            assert 0 <= indicators.rsi <= 100
            assert 0 <= indicators.bb_position <= 1
    
    def test_rapid_strategy_selections(self):
        """Test rapid strategy selection calls."""
        
        selector = StrategySelector()
        
        # Create many different indicator scenarios
        scenarios = []
        for i in range(1000):
            from enhanced_backtest import TechnicalIndicators
            indicators = TechnicalIndicators(
                rsi=np.random.uniform(20, 80),
                macd_line=np.random.normal(0, 1),
                macd_signal=np.random.normal(0, 1),
                macd_histogram=np.random.normal(0, 0.5),
                bb_upper=4600,
                bb_middle=4550,
                bb_lower=4500,
                bb_position=np.random.uniform(0.2, 0.8)
            )
            scenarios.append(indicators)
        
        # Process all scenarios rapidly
        start_time = time.time()
        selections = []
        for indicators in scenarios:
            selection = selector.select_strategy(indicators)
            selections.append(selection)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 2.0  # Less than 2 seconds for 1000 selections
        
        # All selections should be valid
        assert len(selections) == 1000
        for selection in selections:
            assert selection.strategy_type is not None
            assert 0 <= selection.confidence <= 1


if __name__ == "__main__":
    # Run performance tests standalone
    pytest.main([__file__, "-v", "--tb=short"])