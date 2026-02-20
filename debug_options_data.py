#!/usr/bin/env python3
"""
Debug script to test the enhanced get_options_data_for_strategy method.
Use this to troubleshoot data availability issues during monitoring periods.
"""
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from backtesting.strategy_adapter import create_strategy_builder_for_backtest

def test_options_data_retrieval(date: str, time: str, data_path: str = "data/processed/parquet_1m"):
    """
    Test the get_options_data_for_strategy method with enhanced debugging.
    
    Args:
        date: Date string (YYYY-MM-DD)
        time: Time string (HH:MM:SS)
        data_path: Path to parquet data
    """
    logger.info(f"Testing options data retrieval for {date} {time}")
    
    try:
        # Create strategy builder
        builder = create_strategy_builder_for_backtest(data_path)
        
        # Test data adapter diagnosis
        logger.info("Running data availability diagnosis...")
        diagnosis = builder.data_adapter.diagnose_data_availability(date, time)
        
        logger.info("=== DATA DIAGNOSIS ===")
        logger.info(f"Date: {diagnosis['date']}")
        logger.info(f"Timestamp: {diagnosis['timestamp']}")
        logger.info(f"SPX Available: {diagnosis['spx_available']}")
        logger.info(f"SPX Price: {diagnosis['spx_price']}")
        logger.info(f"Options Available: {diagnosis['options_available']}")
        logger.info(f"Options Count: {diagnosis['options_count']}")
        
        if diagnosis['time_range']:
            logger.info(f"Time Range: {diagnosis['time_range'][0]} to {diagnosis['time_range'][1]}")
        
        if diagnosis['available_strikes']:
            logger.info(f"Strike Range: {min(diagnosis['available_strikes'])} to {max(diagnosis['available_strikes'])}")
            logger.info(f"Total Strikes: {len(diagnosis['available_strikes'])}")
        
        if 'timestamps_within_5min' in diagnosis:
            logger.info(f"Timestamps within 5min: {diagnosis['timestamps_within_5min']}")
        
        if diagnosis['errors']:
            logger.error(f"Errors: {diagnosis['errors']}")
        
        # Test actual options data retrieval
        logger.info("\n=== TESTING get_options_data_for_strategy ===")
        
        # Set logger to DEBUG level to see all debug messages
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", 
                  format="{time} | {level} | {message}")
        
        options_data = builder.data_adapter.get_options_data_for_strategy(
            date=date,
            timestamp=time,
            center_strike=None,  # Let it use SPX price
            strike_range=200
        )
        
        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Retrieved {len(options_data)} options")
        
        if options_data:
            # Analyze the retrieved data
            strikes = [float(key.split('_')[0]) for key in options_data.keys()]
            unique_strikes = sorted(set(strikes))
            
            logger.info(f"Strikes: {unique_strikes}")
            logger.info(f"Strike range: {min(unique_strikes)} to {max(unique_strikes)}")
            
            # Show sample options
            sample_keys = list(options_data.keys())[:10]
            logger.info(f"Sample options: {sample_keys}")
            
            # Show sample data structure
            if sample_keys:
                sample_option = options_data[sample_keys[0]]
                logger.info(f"Sample option data keys: {list(sample_option.keys())}")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Example usage - modify these for your testing
    test_date = "2024-01-03"  # Modify to a date you have data for
    test_time = "10:30:00"   # Modify to a time during market hours
    
    # You can override these via command line
    if len(sys.argv) >= 3:
        test_date = sys.argv[1]
        test_time = sys.argv[2]
    
    logger.info(f"Starting debug test with date={test_date}, time={test_time}")
    logger.info("To use different date/time: python debug_options_data.py YYYY-MM-DD HH:MM:SS")
    
    test_options_data_retrieval(test_date, test_time)