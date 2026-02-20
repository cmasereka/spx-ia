#!/usr/bin/env python3
"""
Example usage of the SPX Trading Bot

This script demonstrates how to:
1. Download SPX options data
2. Run a backtest
3. Analyze results
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingBot

def example_download_and_backtest():
    """Example: Download data and run a backtest"""
    bot = TradingBot()
    
    print("="*50)
    print("SPX Trading Bot - Example Usage")
    print("="*50)
    
    # 1. Check current data status
    print("\n1. Checking data status...")
    status = bot.get_data_status()
    print(f"Current data status: {status}")
    
    # 2. Download recent data (if needed)
    print("\n2. Downloading recent data...")
    try:
        download_summary = bot.download_data(days_back=30, update_only=True)
        print(f"Download summary: {download_summary}")
    except Exception as e:
        print(f"Download failed (this is normal if you don't have ThetaData credentials): {e}")
    
    # 3. Run a sample backtest with existing data
    print("\n3. Running Iron Condor backtest...")
    
    # Use a recent date range (adjust based on your data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        analysis = bot.run_iron_condor_backtest(start_date, end_date)
        bot.print_analysis(analysis)
    except Exception as e:
        print(f"Backtest failed: {e}")
        print("This might be because you don't have options data yet.")
        print("Run the download command first with valid ThetaData credentials.")

def example_custom_strategy():
    """Example: Create a custom strategy configuration"""
    from config.settings import IRON_CONDOR_PARAMS
    
    print("\n" + "="*50)
    print("CUSTOM STRATEGY EXAMPLE")
    print("="*50)
    
    # Custom Iron Condor parameters
    custom_params = {
        'put_strike_distance': 75,    # Wider strikes
        'call_strike_distance': 75,
        'profit_target': 0.25,        # Take profit at 25%
        'stop_loss': 3.0              # Stop loss at 3x credit
    }
    
    print(f"Default Iron Condor params: {IRON_CONDOR_PARAMS}")
    print(f"Custom Iron Condor params: {custom_params}")
    
    # You would pass custom_params to run_iron_condor_backtest()
    print("\nTo use custom parameters, modify the strategy_params argument:")
    print("bot.run_iron_condor_backtest(start_date, end_date, custom_params)")

def example_data_analysis():
    """Example: Analyze stored data"""
    from src.data.storage import DataStorage
    from src.indicators.technical_indicators import TechnicalIndicators
    
    print("\n" + "="*50)
    print("DATA ANALYSIS EXAMPLE")
    print("="*50)
    
    storage = DataStorage()
    indicators = TechnicalIndicators()
    
    # Get underlying data
    print("\n1. Getting underlying SPX data...")
    underlying_df = storage.get_underlying_data()
    
    if not underlying_df.empty:
        print(f"Found {len(underlying_df)} days of underlying data")
        print(f"Date range: {underlying_df['date'].min()} to {underlying_df['date'].max()}")
        
        # Calculate and display technical indicators
        print("\n2. Calculating technical indicators...")
        indicators_df = indicators.calculate_all_indicators(underlying_df)
        
        print(f"Latest SPX close: ${underlying_df['close'].iloc[-1]:.2f}")
        print(f"Latest RSI: {indicators_df['rsi'].iloc[-1]:.1f}")
        print(f"Latest MACD: {indicators_df['macd'].iloc[-1]:.2f}")
        
        # Get indicator summary
        summary = indicators.get_indicator_summary(indicators_df)
        print(f"\nTechnical indicators summary:")
        for key, value in summary.items():
            if isinstance(value, dict) and 'current' in value:
                print(f"  {key}: {value['current']:.2f}")
    else:
        print("No underlying data found. Run download first.")

if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Download data and run backtest")
    print("2. Custom strategy parameters")
    print("3. Data analysis")
    print("4. Run all examples")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        example_download_and_backtest()
    elif choice == '2':
        example_custom_strategy()
    elif choice == '3':
        example_data_analysis()
    elif choice == '4':
        example_download_and_backtest()
        example_custom_strategy()
        example_data_analysis()
    else:
        print("Invalid choice")
    
    print("\nExample complete!")