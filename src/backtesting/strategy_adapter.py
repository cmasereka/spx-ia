#!/usr/bin/env python3
"""
Enhanced Strategy Adapter for integrating parquet data with existing strategy framework.
Converts DataFrame-based data to the dictionary format expected by existing strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time
from functools import lru_cache
from loguru import logger

from ..data.parquet_loader import ParquetDataLoader
from ..data.query_engine import BacktestQueryEngine
from ..strategies.options_strategies import OptionsStrategy, IronCondor, VerticalSpread, StrategyBuilder


class ParquetDataAdapter:
    """
    Adapter that converts parquet DataFrame data to dictionary format 
    expected by existing strategy classes.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
        self._option_cache = {}
        self._spx_cache = {}
    
    def convert_options_dataframe_to_dict(self, options_df: pd.DataFrame, 
                                        spx_price: Optional[float] = None) -> Dict[str, Dict]:
        """
        Convert options DataFrame to strategy-expected dictionary format.
        
        Args:
            options_df: Options DataFrame from parquet loader
            spx_price: Current SPX price for calculations
            
        Returns:
            Dictionary in format: {"strike_optiontype": {"mid_price": ..., "delta": ..., ...}}
        """
        if options_df.empty:
            return {}
        
        options_dict = {}
        
        for _, row in options_df.iterrows():
            # Normalize option type
            option_type = str(row['right']).lower()
            if option_type in ['c', 'call']:
                option_type = 'call'
            elif option_type in ['p', 'put']:
                option_type = 'put'
            else:
                continue
            
            # Create key
            strike = float(row['strike'])
            key = f"{strike}_{option_type}"
            
            # Calculate mid price
            bid = float(row.get('bid', 0))
            ask = float(row.get('ask', 0))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            
            # Extract Greeks (may not be available in all data)
            delta = float(row.get('delta', 0))
            gamma = float(row.get('gamma', 0)) 
            theta = float(row.get('theta', 0))
            vega = float(row.get('vega', 0))
            iv = float(row.get('iv', 0))
            
            # Add calculated metrics
            bid_ask_spread = ask - bid if ask > bid else 0.0
            spread_pct = (bid_ask_spread / mid_price * 100) if mid_price > 0 else 0.0
            
            # Estimate moneyness if SPX price available
            moneyness = strike / spx_price if spx_price and spx_price > 0 else 1.0
            
            options_dict[key] = {
                'mid_price': mid_price,
                'bid': bid,
                'ask': ask,
                'strike': strike,
                'option_type': option_type,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'iv': iv,
                'bid_ask_spread': bid_ask_spread,
                'spread_pct': spread_pct,
                'moneyness': moneyness,
                'volume': int(row.get('volume', 0)),
                'open_interest': int(row.get('open_interest', 0))
            }
        
        return options_dict
    
    def get_options_data_for_strategy(self, date: Union[str, datetime], 
                                    timestamp: Union[str, datetime, time],
                                    center_strike: Optional[float] = None,
                                    strike_range: float = 200) -> Dict[str, Dict]:
        """
        Get options data formatted for strategy creation.
        
        Args:
            date: Trading date
            timestamp: Time for data lookup
            center_strike: Center strike (uses SPX price if None)
            strike_range: Strike range around center
            
        Returns:
            Dictionary formatted for strategy classes
        """
        # Get SPX price
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            logger.warning(f"No SPX price found for {date} {timestamp}")
            return {}
        
        # Use SPX price as center if not provided
        if center_strike is None:
            center_strike = spx_price
        
        # Get options chain
        options_df = self.query_engine.loader.get_options_chain_at_time(
            date, timestamp, center_strike, strike_range
        )
        
        if options_df.empty:
            logger.warning(f"No options data found for {date} {timestamp}")
            return {}
        
        # Convert to dictionary format
        return self.convert_options_dataframe_to_dict(options_df, spx_price)
    
    def get_liquid_options_for_strategy(self, date: Union[str, datetime],
                                      timestamp: Union[str, datetime, time],
                                      min_bid: float = 0.05,
                                      max_spread_pct: float = 30.0,
                                      center_strike: Optional[float] = None,
                                      strike_range: float = 150) -> Dict[str, Dict]:
        """
        Get liquid options data formatted for strategy creation.
        
        Args:
            date: Trading date
            timestamp: Time for data lookup
            min_bid: Minimum bid price for liquidity
            max_spread_pct: Maximum bid/ask spread percentage
            center_strike: Center strike for filtering
            strike_range: Strike range around center
            
        Returns:
            Dictionary with liquid options only
        """
        # Get liquid options using query engine
        liquid_df = self.query_engine.find_liquid_options(
            date, timestamp, min_bid, max_spread_pct
        )
        
        if liquid_df.empty:
            logger.warning(f"No liquid options found for {date} {timestamp}")
            return {}
        
        # Apply strike filtering if requested
        if center_strike is not None:
            strike_mask = (
                (liquid_df['strike'] >= center_strike - strike_range) &
                (liquid_df['strike'] <= center_strike + strike_range)
            )
            liquid_df = liquid_df[strike_mask]
        
        # Get SPX price for calculations
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        
        return self.convert_options_dataframe_to_dict(liquid_df, spx_price)


class EnhancedStrategyBuilder:
    """
    Enhanced strategy builder that uses parquet data for efficient strategy creation.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
        self.data_adapter = ParquetDataAdapter(query_engine)
        self._strategy_cache = {}
    
    def build_iron_condor_optimized(self, 
                                  date: Union[str, datetime],
                                  timestamp: Union[str, datetime, time],
                                  put_distance: int = 50,
                                  call_distance: int = 50,
                                  spread_width: int = 25,
                                  quantity: int = 1,
                                  use_liquid_options: bool = True,
                                  min_bid: float = 0.10,
                                  max_spread_pct: float = 20.0) -> Optional[IronCondor]:
        """
        Build Iron Condor using optimized parquet data access.
        
        Args:
            date: Trading date
            timestamp: Entry timestamp
            put_distance: Distance for put short strike from underlying
            call_distance: Distance for call short strike from underlying  
            spread_width: Width of each spread
            quantity: Number of contracts
            use_liquid_options: Only use liquid options
            min_bid: Minimum bid for liquidity filter
            max_spread_pct: Maximum spread % for liquidity filter
            
        Returns:
            IronCondor strategy or None if insufficient data
        """
        # Get SPX price
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            logger.error(f"No SPX price found for {date} {timestamp}")
            return None
        
        # Get options data
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, min_bid, max_spread_pct, spx_price, 200
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 200
            )
        
        if not options_data:
            logger.warning(f"No suitable options data found for Iron Condor at {date} {timestamp}")
            return None
        
        # Build strategy using existing builder
        try:
            iron_condor = StrategyBuilder.build_iron_condor(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                put_distance=put_distance,
                call_distance=call_distance,
                spread_width=spread_width,
                quantity=quantity,
                expiration=pd.to_datetime(date)  # 0DTE
            )
            
            # Add metadata
            iron_condor.entry_spx_price = spx_price
            iron_condor.entry_timestamp = timestamp
            iron_condor.liquidity_filtered = use_liquid_options
            
            return iron_condor
            
        except Exception as e:
            logger.error(f"Failed to build Iron Condor: {e}")
            return None
    
    def build_call_spread_optimized(self,
                                  date: Union[str, datetime],
                                  timestamp: Union[str, datetime, time],
                                  strike_distance: int = 25,
                                  spread_width: int = 25,
                                  is_debit: bool = True,
                                  quantity: int = 1,
                                  use_liquid_options: bool = True) -> Optional[VerticalSpread]:
        """Build Call Spread using optimized data access."""
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            return None
        
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, 0.05, 30.0, spx_price, 150
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 150
            )
        
        if not options_data:
            return None
        
        try:
            return StrategyBuilder.build_call_spread(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                strike_distance=strike_distance,
                spread_width=spread_width,
                is_debit=is_debit,
                quantity=quantity,
                expiration=pd.to_datetime(date)
            )
        except Exception as e:
            logger.error(f"Failed to build Call Spread: {e}")
            return None
    
    def build_put_spread_optimized(self,
                                 date: Union[str, datetime],
                                 timestamp: Union[str, datetime, time],
                                 strike_distance: int = 25,
                                 spread_width: int = 25,
                                 is_debit: bool = True,
                                 quantity: int = 1,
                                 use_liquid_options: bool = True) -> Optional[VerticalSpread]:
        """Build Put Spread using optimized data access."""
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            return None
        
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, 0.05, 30.0, spx_price, 150
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 150
            )
        
        if not options_data:
            return None
        
        try:
            return StrategyBuilder.build_put_spread(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                strike_distance=strike_distance,
                spread_width=spread_width,
                is_debit=is_debit,
                quantity=quantity,
                expiration=pd.to_datetime(date)
            )
        except Exception as e:
            logger.error(f"Failed to build Put Spread: {e}")
            return None
    
    def update_strategy_prices_optimized(self, 
                                       strategy: OptionsStrategy,
                                       date: Union[str, datetime],
                                       timestamp: Union[str, datetime, time]) -> bool:
        """
        Update strategy prices using optimized data access.
        
        Args:
            strategy: Strategy to update
            date: Trading date
            timestamp: Current timestamp
            
        Returns:
            Success boolean
        """
        try:
            # Get current options data for all legs
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if spx_price is None:
                return False
            
            # Get strikes from strategy legs
            strikes = [leg.strike for leg in strategy.legs]
            if not strikes:
                return False
            
            center_strike = spx_price
            strike_range = max(abs(max(strikes) - center_strike), 
                             abs(min(strikes) - center_strike)) + 50
            
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, center_strike, strike_range
            )
            
            if not options_data:
                return False
            
            # Update strategy with new prices
            strategy.update_prices(options_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy prices: {e}")
            return False
    
    def clear_cache(self):
        """Clear strategy cache to free memory."""
        self._strategy_cache.clear()
        self.data_adapter._option_cache.clear()
        self.data_adapter._spx_cache.clear()
        logger.info("Cleared strategy adapter cache")


# Convenience functions for backtesting
def create_strategy_builder_for_backtest(data_path: str = "data/processed/parquet_1m") -> EnhancedStrategyBuilder:
    """
    Create strategy builder configured for backtesting.
    
    Args:
        data_path: Path to parquet data
        
    Returns:
        Configured EnhancedStrategyBuilder
    """
    from ..data.query_engine import create_fast_query_engine
    
    query_engine = create_fast_query_engine(data_path)
    return EnhancedStrategyBuilder(query_engine)


def quick_iron_condor_test(date: str, time_str: str, 
                          data_path: str = "data/processed/parquet_1m") -> Optional[IronCondor]:
    """
    Quick test function to create an Iron Condor.
    
    Args:
        date: Date string (YYYY-MM-DD)
        time_str: Time string (HH:MM:SS)
        data_path: Path to data
        
    Returns:
        IronCondor strategy or None
    """
    builder = create_strategy_builder_for_backtest(data_path)
    
    return builder.build_iron_condor_optimized(
        date=date,
        timestamp=time_str,
        put_distance=50,
        call_distance=50,
        spread_width=25,
        quantity=1,
        use_liquid_options=True
    )