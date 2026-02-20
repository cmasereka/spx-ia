#!/usr/bin/env python3
"""
Comprehensive example demonstrating the integrated backtesting system.
Shows how to use parquet data, caching, and optimized Iron Condor strategies.
"""
import sys
sys.path.append('.')

import pandas as pd
from datetime import datetime
from loguru import logger

from src.backtesting.backtest_engine import OptimizedBacktestEngine, BacktestConfig, quick_iron_condor_backtest
from src.backtesting.strategy_adapter import EnhancedStrategyBuilder, create_strategy_builder_for_backtest
from src.backtesting.iron_condor_loader import IronCondorDataLoader, create_iron_condor_loader, find_best_iron_condor_entry
from src.backtesting.caching import BacktestingCacheManager


def example_1_basic_iron_condor_backtest():
    """Example 1: Basic Iron Condor backtest using the quick function."""
    print("=" * 60)
    print("Example 1: Quick Iron Condor Backtest")
    print("=" * 60)
    
    # Run a quick backtest with default parameters
    results = quick_iron_condor_backtest(
        start_date="2026-02-09",
        end_date="2026-02-13",
        data_path="data/processed/parquet_1m"
    )
    
    # Display results
    print(f"Backtest Results:")
    print(f"  Total P&L: ${results.total_pnl:,.2f}")
    print(f"  Return: {results.total_return_pct:+.2f}%")
    print(f"  Win Rate: {results.win_rate:.1f}%")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Profit Factor: {results.profit_factor:.2f}")
    print(f"  Max Drawdown: ${results.max_drawdown:,.2f}")
    print(f"  Execution Time: {results.execution_time_seconds:.2f}s")
    print()
    
    # Show individual trades
    if results.trades:
        print("Sample Trades:")
        for trade in results.trades[:3]:
            print(f"  {trade.entry_date.date()} -> {trade.exit_date.date()}: "
                  f"${trade.pnl:+,.2f} ({trade.exit_reason})")
    print()


def example_2_advanced_configuration():
    """Example 2: Advanced backtesting with custom configuration."""
    print("=" * 60)
    print("Example 2: Advanced Configuration Backtest")
    print("=" * 60)
    
    # Create advanced configuration
    config = BacktestConfig(
        start_date="2026-02-09",
        end_date="2026-02-13",
        initial_capital=50000.0,
        max_positions=5,
        
        # Strategy parameters
        strategy_type="iron_condor",
        entry_times=["09:45:00", "10:15:00"],
        exit_times=["15:30:00"],
        
        # Iron Condor settings
        put_distances=[40, 60, 80],
        call_distances=[40, 60, 80], 
        spread_widths=[25, 50],
        min_credit=1.00,
        
        # Risk management
        max_loss_pct=0.40,  # Close at 40% of max loss
        profit_target_pct=0.30,  # Close at 30% of max profit
        
        # Data quality
        use_liquid_options=True,
        min_bid=0.15,
        max_spread_pct=15.0,
        
        # Performance
        preload_data=True,
        cache_enabled=True
    )
    
    # Run backtest
    engine = OptimizedBacktestEngine("data/processed/parquet_1m")
    results = engine.run_backtest(config)
    
    # Display detailed results
    print(f"Advanced Backtest Results:")
    print(f"  Capital: ${config.initial_capital:,.0f}")
    print(f"  Final Value: ${config.initial_capital + results.total_pnl:,.2f}")
    print(f"  Total Return: {results.total_return_pct:+.2f}%")
    print(f"  Annualized Return: {results.total_return_pct * (252/len(results.trades)) if results.trades else 0:+.2f}%")
    print(f"  Win Rate: {results.win_rate:.1f}% ({results.winning_trades}/{results.total_trades})")
    print(f"  Avg Win: ${results.avg_win:.2f}")
    print(f"  Avg Loss: ${results.avg_loss:.2f}")
    print(f"  Profit Factor: {results.profit_factor:.2f}")
    print(f"  Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown/config.initial_capital*100:.1f}%)")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Cache Hit Rate: {results.cache_hit_rate:.1f}%")
    print()


def example_3_iron_condor_optimization():
    """Example 3: Finding optimal Iron Condor setups."""
    print("=" * 60) 
    print("Example 3: Iron Condor Setup Optimization")
    print("=" * 60)
    
    # Create Iron Condor loader
    ic_loader = create_iron_condor_loader("data/processed/parquet_1m")
    
    # Find best setups for a specific day
    test_date = "2026-02-09"
    test_time = "10:00:00"
    
    print(f"Finding Iron Condor opportunities for {test_date} at {test_time}")
    
    # Get all viable setups
    setups = ic_loader.get_viable_iron_condor_setups(
        date=test_date,
        timestamp=test_time,
        put_distances=[25, 50, 75, 100],
        call_distances=[25, 50, 75, 100],
        spread_widths=[25, 50],
        min_credit=0.50
    )
    
    print(f"Found {len(setups)} viable setups")
    
    if setups:
        print("\nTop 5 setups by credit:")
        for i, setup in enumerate(setups[:5]):
            print(f"  {i+1}. Put: {setup.put_long_strike}/{setup.put_short_strike}, "
                  f"Call: {setup.call_short_strike}/{setup.call_long_strike}")
            print(f"     Credit: ${setup.net_credit:.2f}, "
                  f"Max Loss: ${setup.max_loss:.2f}, "
                  f"Liquidity: {setup.liquidity_score:.2f}")
        
        # Get best setup by different criteria
        best_credit = ic_loader.get_best_iron_condor_setup(
            test_date, test_time, optimize_for='credit'
        )
        
        best_liquidity = ic_loader.get_best_iron_condor_setup(
            test_date, test_time, optimize_for='liquidity'
        )
        
        print(f"\nBest by credit: ${best_credit.net_credit:.2f}")
        print(f"Best by liquidity: Score {best_liquidity.liquidity_score:.2f}")
    
    print()


def example_4_strategy_comparison():
    """Example 4: Compare different strategy parameters."""
    print("=" * 60)
    print("Example 4: Strategy Parameter Comparison")
    print("=" * 60)
    
    engine = OptimizedBacktestEngine("data/processed/parquet_1m")
    
    # Base configuration
    base_config = BacktestConfig(
        start_date="2026-02-09",
        end_date="2026-02-13",
        initial_capital=25000.0,
        strategy_type="iron_condor",
        entry_times=["10:00:00"]
    )
    
    # Test different parameter combinations
    test_configs = [
        {"put_distances": [50], "call_distances": [50], "spread_widths": [25], "min_credit": 0.50},
        {"put_distances": [75], "call_distances": [75], "spread_widths": [25], "min_credit": 0.75},
        {"put_distances": [50], "call_distances": [50], "spread_widths": [50], "min_credit": 1.00},
        {"put_distances": [100], "call_distances": [100], "spread_widths": [25], "min_credit": 0.50}
    ]
    
    print("Comparing Iron Condor configurations:")
    print(f"{'Config':<8} {'Return%':<8} {'Win%':<6} {'Trades':<7} {'PF':<6} {'MaxDD':<8}")
    print("-" * 50)
    
    for i, params in enumerate(test_configs, 1):
        # Create config with new parameters
        config_dict = base_config.__dict__.copy()
        config_dict.update(params)
        test_config = BacktestConfig(**config_dict)
        
        # Run backtest
        results = engine.run_backtest(test_config)
        
        print(f"{i:<8} {results.total_return_pct:+6.1f}%  "
              f"{results.win_rate:5.1f}% {results.total_trades:<7} "
              f"{results.profit_factor:5.2f} ${results.max_drawdown:>6.0f}")
    
    print()


def example_5_data_exploration():
    """Example 5: Explore the data access capabilities."""
    print("=" * 60)
    print("Example 5: Data Access and Exploration")
    print("=" * 60)
    
    # Create strategy builder for data access
    builder = create_strategy_builder_for_backtest("data/processed/parquet_1m")
    
    test_date = "2026-02-09"
    test_time = "14:00:00"
    
    # Get SPX price
    spx_price = builder.query_engine.get_fastest_spx_price(test_date, test_time)
    print(f"SPX Price at {test_date} {test_time}: ${spx_price:,.2f}")
    
    # Get liquid options
    liquid_options = builder.data_adapter.get_liquid_options_for_strategy(
        test_date, test_time, min_bid=0.10, max_spread_pct=25.0
    )
    
    print(f"Found {len(liquid_options)} liquid options")
    
    # Show some example liquid options
    if liquid_options:
        print("\nSample liquid options:")
        sample_keys = list(liquid_options.keys())[:5]
        
        for key in sample_keys:
            option = liquid_options[key]
            print(f"  {key}: Bid ${option['bid']:.2f}, Ask ${option['ask']:.2f}, "
                  f"Spread {option['spread_pct']:.1f}%")
    
    # Test Iron Condor creation
    print(f"\nTesting Iron Condor creation...")
    ic = builder.build_iron_condor_optimized(
        date=test_date,
        timestamp=test_time,
        put_distance=50,
        call_distance=50,
        spread_width=25,
        use_liquid_options=True
    )
    
    if ic:
        print(f"Successfully created Iron Condor:")
        print(f"  Entry Credit: ${ic.entry_credit:.2f}")
        print(f"  Max Profit: ${ic.max_profit:.2f}")
        print(f"  Max Loss: ${ic.max_loss:.2f}")
        print(f"  Number of legs: {len(ic.legs)}")
        print(f"  Breakeven points: {[f'${bp:.2f}' for bp in ic.breakeven_points]}")
    else:
        print("Could not create Iron Condor with current parameters")
    
    print()


def example_6_performance_monitoring():
    """Example 6: Performance monitoring and caching demonstration."""
    print("=" * 60)
    print("Example 6: Performance and Caching")
    print("=" * 60)
    
    engine = OptimizedBacktestEngine("data/processed/parquet_1m")
    
    # Show initial cache stats
    initial_stats = engine.cache_manager.get_cache_stats()
    print("Initial cache stats:")
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")
    
    # Run a small backtest to populate cache
    print("\nRunning backtest to populate cache...")
    
    config = BacktestConfig(
        start_date="2026-02-09",
        end_date="2026-02-11",  # Just 3 days
        preload_data=True,
        cache_enabled=True
    )
    
    results = engine.run_backtest(config)
    
    # Show final cache stats
    final_stats = engine.cache_manager.get_cache_stats()
    print("\nFinal cache stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Show performance improvement on second run
    print("\nRunning same backtest again (should be faster with cache)...")
    
    start_time = pd.Timestamp.now()
    results2 = engine.run_backtest(config)
    end_time = pd.Timestamp.now()
    
    cached_time = (end_time - start_time).total_seconds()
    
    print(f"First run: {results.execution_time_seconds:.2f}s")
    print(f"Cached run: {cached_time:.2f}s")
    print(f"Speed improvement: {results.execution_time_seconds / cached_time:.1f}x faster")
    print()


def run_all_examples():
    """Run all backtesting integration examples."""
    try:
        example_1_basic_iron_condor_backtest()
        example_2_advanced_configuration()
        example_3_iron_condor_optimization()
        example_4_strategy_comparison()
        example_5_data_exploration()
        example_6_performance_monitoring()
        
        print("=" * 60)
        print("All Examples Completed Successfully!")
        print("=" * 60)
        print()
        print("Your integrated backtesting system is ready for:")
        print("✓ High-performance Iron Condor backtesting")
        print("✓ Advanced parameter optimization")
        print("✓ Intelligent caching for speed")
        print("✓ Liquid options filtering")
        print("✓ Comprehensive performance analytics")
        print("✓ Multi-threading support")
        print()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error running examples: {e}")
        print("Make sure your parquet data is available in data/processed/parquet_1m/")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    print("SPX 0DTE Backtesting Integration Examples")
    print("Using optimized parquet data with intelligent caching")
    print("=" * 60)
    print()
    
    run_all_examples()