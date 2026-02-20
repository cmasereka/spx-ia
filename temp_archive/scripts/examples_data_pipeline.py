#!/usr/bin/env python3
"""
Example usage of the Data Access Pipeline for SPX 0DTE backtesting.
Demonstrates key functionality for strategy development.
"""
import sys
sys.path.append('.')

import pandas as pd
from datetime import datetime, time
from loguru import logger

from src.data.parquet_loader import ParquetDataLoader, load_data_for_backtest
from src.data.query_engine import BacktestQueryEngine, create_fast_query_engine


def example_basic_data_access():
    """Basic data loading and access examples"""
    print("=== Basic Data Access Examples ===\n")
    
    # Initialize data loader
    loader = ParquetDataLoader("data/processed/parquet_1m")
    
    # Get data summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # Load SPX data for a specific date
    test_date = "2026-02-09"
    print(f"Loading SPX data for {test_date}:")
    spx_data = loader.load_spx_data(test_date)
    print(f"  Loaded {len(spx_data)} SPX price points")
    print(f"  Time range: {spx_data.index[0]} to {spx_data.index[-1]}")
    print(f"  Price range: ${spx_data['price'].min():.2f} - ${spx_data['price'].max():.2f}")
    print()
    
    # Load options data for the same date
    print(f"Loading options data for {test_date}:")
    options_data = loader.load_options_data(test_date)
    print(f"  Loaded {len(options_data)} option quotes")
    print(f"  Strike range: {options_data.index.get_level_values('strike').min()} - {options_data.index.get_level_values('strike').max()}")
    print(f"  Option types: {options_data.index.get_level_values('right').unique().tolist()}")
    print()


def example_fast_queries():
    """Fast query examples using the query engine"""
    print("=== Fast Query Engine Examples ===\n")
    
    # Create query engine with pre-built indexes
    engine = create_fast_query_engine("data/processed/parquet_1m")
    
    test_date = "2026-02-09"
    test_time = "10:00:00"
    
    # Get SPX price at specific time (ultra-fast lookup)
    print(f"SPX price at {test_date} {test_time}:")
    spx_price = engine.get_fastest_spx_price(test_date, test_time)
    print(f"  SPX Price: ${spx_price:.2f}")
    print()
    
    # Get ATM options chain (pre-indexed for speed)
    print(f"ATM options at {test_date} {test_time}:")
    atm_options = engine.get_atm_options_fast(test_date, test_time, strike_count=3)
    if not atm_options.empty:
        print(f"  Found {len(atm_options)} ATM options")
        print("  Sample ATM options:")
        cols = ['strike', 'right', 'bid', 'ask', 'mid']
        available_cols = [col for col in cols if col in atm_options.columns]
        print(atm_options[available_cols].head())
    print()
    
    # Find liquid options
    print(f"Liquid options at {test_date} {test_time}:")
    liquid_options = engine.find_liquid_options(test_date, test_time, min_bid=0.10, max_spread_pct=20)
    if not liquid_options.empty:
        print(f"  Found {len(liquid_options)} liquid options")
        print("  Most liquid options (lowest spread %):")
        cols = ['strike', 'right', 'bid', 'ask', 'spread_pct']
        available_cols = [col for col in cols if col in liquid_options.columns]
        print(liquid_options[available_cols].head())
    print()


def example_option_selection():
    """Examples of finding specific options for trading strategies"""
    print("=== Option Selection Examples ===\n")
    
    engine = create_fast_query_engine("data/processed/parquet_1m")
    
    test_date = "2026-02-09"
    test_time = "14:00:00"  # Afternoon trading
    
    # Get current SPX price for context
    spx_price = engine.get_fastest_spx_price(test_date, test_time)
    print(f"SPX Price at {test_date} {test_time}: ${spx_price:.2f}")
    print()
    
    # Find options by moneyness (useful for systematic strategies)
    print("Finding options by moneyness:")
    
    # 1% OTM call
    otm_call = engine.get_option_by_moneyness(test_date, test_time, 1.01, 'call')
    if otm_call is not None:
        print(f"  1% OTM Call: Strike ${otm_call['strike']:.0f}, Bid: ${otm_call['bid']:.2f}, Ask: ${otm_call['ask']:.2f}")
    
    # 1% OTM put  
    otm_put = engine.get_option_by_moneyness(test_date, test_time, 0.99, 'put')
    if otm_put is not None:
        print(f"  1% OTM Put:  Strike ${otm_put['strike']:.0f}, Bid: ${otm_put['bid']:.2f}, Ask: ${otm_put['ask']:.2f}")
    
    # ATM straddle components
    atm_call = engine.get_option_by_moneyness(test_date, test_time, 1.0, 'call')
    atm_put = engine.get_option_by_moneyness(test_date, test_time, 1.0, 'put')
    if atm_call is not None and atm_put is not None:
        straddle_cost = atm_call['ask'] + atm_put['ask']  # Buying both
        print(f"  ATM Straddle: Call ${atm_call['strike']:.0f} + Put ${atm_put['strike']:.0f}, Total Cost: ${straddle_cost:.2f}")
    print()


def example_session_analysis():
    """Example of analyzing a full trading session"""
    print("=== Trading Session Analysis ===\n")
    
    engine = create_fast_query_engine("data/processed/parquet_1m")
    
    test_date = "2026-02-09"
    
    # Get trading session data
    session_data = engine.get_trading_session_data(test_date, "09:30:00", "16:00:00")
    
    if 'spx' in session_data and not session_data['spx'].empty:
        spx_session = session_data['spx']
        
        print(f"SPX Session Analysis for {test_date}:")
        print(f"  Session Open:  ${spx_session['price'].iloc[0]:.2f}")
        print(f"  Session High:  ${spx_session['price'].max():.2f}")
        print(f"  Session Low:   ${spx_session['price'].min():.2f}")
        print(f"  Session Close: ${spx_session['price'].iloc[-1]:.2f}")
        
        # Calculate session statistics
        session_change = spx_session['price'].iloc[-1] - spx_session['price'].iloc[0]
        session_change_pct = (session_change / spx_session['price'].iloc[0]) * 100
        
        print(f"  Session Change: ${session_change:+.2f} ({session_change_pct:+.2f}%)")
        
        # Volatility measure (std of 1-minute returns)
        returns = spx_session['price'].pct_change().dropna()
        realized_vol = returns.std() * (252 * 390) ** 0.5 * 100  # Annualized
        print(f"  Realized Volatility: {realized_vol:.1f}%")
        print()
    
    # Options activity summary
    if 'options' in session_data and not session_data['options'].empty:
        options_session = session_data['options']
        
        print(f"Options Activity Summary:")
        print(f"  Total Quotes: {len(options_session):,}")
        
        # Analyze by option type
        calls = options_session[options_session.index.get_level_values('right') == 'C']
        puts = options_session[options_session.index.get_level_values('right') == 'P']
        
        print(f"  Call Quotes: {len(calls):,}")
        print(f"  Put Quotes:  {len(puts):,}")
        
        if 'bid' in options_session.columns and 'ask' in options_session.columns:
            # Average bid-ask spreads
            options_session_reset = options_session.reset_index()
            spreads = options_session_reset['ask'] - options_session_reset['bid']
            avg_spread = spreads.mean()
            print(f"  Avg Spread: ${avg_spread:.3f}")
        print()


def example_backtest_data_prep():
    """Example of preparing data for a backtest"""
    print("=== Backtest Data Preparation ===\n")
    
    # Load data for multiple days
    start_date = "2026-02-09"
    end_date = "2026-02-13"
    
    loader = load_data_for_backtest(start_date, end_date, "data/processed/parquet_1m")
    
    # Load multi-day data
    multi_day_data = loader.load_date_range(start_date, end_date, 'both')
    
    print(f"Multi-day backtest data ({start_date} to {end_date}):")
    
    if 'spx' in multi_day_data:
        spx_multi = multi_day_data['spx']
        print(f"  SPX Data: {len(spx_multi)} data points")
        print(f"  Date range: {spx_multi.index[0].date()} to {spx_multi.index[-1].date()}")
        
    if 'options' in multi_day_data:
        options_multi = multi_day_data['options']
        print(f"  Options Data: {len(options_multi)} quotes")
        
        # Unique trading days
        unique_days = options_multi.index.get_level_values('timestamp').date
        trading_days = len(pd.Series(unique_days).unique())
        print(f"  Trading Days: {trading_days}")
    print()
    
    # Performance tip
    print("Performance Tips for Backtesting:")
    print("1. Use get_fastest_spx_price() for SPX price lookups")
    print("2. Use get_atm_options_fast() for quick ATM option chains")  
    print("3. Use find_liquid_options() to filter for tradeable options")
    print("4. Cache frequently accessed data using the built-in LRU cache")
    print("5. Clear caches periodically with loader.clear_cache() to manage memory")
    print()


def run_all_examples():
    """Run all example functions"""
    try:
        example_basic_data_access()
        example_fast_queries() 
        example_option_selection()
        example_session_analysis()
        example_backtest_data_prep()
        
        print("=== Examples completed successfully! ===")
        print("\nYour data access pipeline is ready for 0DTE SPX backtesting!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error running examples: {e}")
        print("Make sure your parquet data is available in data/processed/parquet_1m/")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    print("SPX 0DTE Data Access Pipeline Examples")
    print("=" * 50)
    print()
    
    run_all_examples()