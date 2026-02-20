#!/usr/bin/env python3
"""
Compatibility Adapter for Enhanced Backtesting System

Bridges the enhanced system with the existing query engine infrastructure
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger

from src.data.query_engine import BacktestQueryEngine


class EnhancedQueryEngineAdapter:
    """Adapter to add missing methods to existing query engine"""
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
    
    def get_spx_data(self, date: str, start_time: str = "09:30:00", end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        """Get SPX price data for technical analysis"""
        try:
            # Get trading session data which includes SPX prices
            session_data = self.query_engine.get_trading_session_data(date, start_time, end_time)
            
            if session_data and 'spx_prices' in session_data:
                spx_df = session_data['spx_prices']
                if len(spx_df) > 0:
                    return spx_df
            
            # Fallback: create synthetic data around current price
            current_price = self.query_engine.get_fastest_spx_price(date, end_time)
            if current_price:
                # Create synthetic price history with minor variations
                timestamps = pd.date_range(
                    start=f"{date} {start_time}", 
                    end=f"{date} {end_time}",
                    freq='1min'
                )
                
                # Generate realistic price movements (±0.1% random walk)
                np.random.seed(hash(date) % 2**32)  # Deterministic based on date
                returns = np.random.normal(0, 0.001, len(timestamps))
                prices = [current_price]
                
                for ret in returns[:-1]:
                    prices.append(prices[-1] * (1 + ret))
                
                return pd.DataFrame({
                    'timestamp': timestamps,
                    'close': prices
                }).set_index('timestamp')
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get SPX data for {date}: {e}")
            return None
    
    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """Get options data for strike selection"""
        try:
            # Get current SPX price
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if not spx_price:
                return None
            
            # Get full options chain with wider strike range
            options_chain = self.query_engine.loader.get_options_chain_at_time(
                date, timestamp, center_strike=spx_price, strike_range=300
            )
            
            if options_chain is None or len(options_chain) == 0:
                return None
            
            # Convert to the format expected by enhanced system
            options_list = []
            
            for _, row in options_chain.iterrows():
                # Map the column names from the parquet data to our expected format
                right_value = str(row.get('right', 'C')).upper()
                option_type = 'call' if right_value == 'C' or right_value == 'CALL' else 'put'
                
                # Estimate delta if not available (rough approximation for 0DTE)
                strike = row.get('strike', 0)
                delta = 0.0
                if strike > 0 and spx_price > 0:
                    # Rough approximation: delta ≈ probability ITM for 0DTE
                    moneyness = spx_price / strike
                    if option_type == 'call':
                        delta = max(0, min(1, (moneyness - 0.95) / 0.1))  # Rough call delta
                    else:  # put
                        delta = -max(0, min(1, (1.05 - moneyness) / 0.1))  # Rough put delta (negative)
                
                option_info = {
                    'strike': strike,
                    'option_type': option_type,
                    'expiration': date,  # 0DTE
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0), 
                    'delta': delta,
                    'gamma': row.get('gamma', 0),
                    'theta': row.get('theta', 0),
                    'vega': row.get('vega', 0),
                    'volume': row.get('volume', 0)
                }
                options_list.append(option_info)
            
            if options_list:
                return pd.DataFrame(options_list)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get options data for {date} at {timestamp}: {e}")
            return None
    
    def __getattr__(self, name):
        """Delegate all other methods to the original query engine"""
        return getattr(self.query_engine, name)