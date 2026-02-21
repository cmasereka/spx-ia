#!/usr/bin/env python3
"""
High-performance data access pipeline for 1-minute SPX options backtesting.
Optimized for fast loading, querying, and filtering of Parquet data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
from loguru import logger


class ParquetDataLoader:
    """
    High-performance data loader for 1-minute SPX and options data.
    Designed for fast backtesting with chunked loading and indexed lookups.
    """
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        self.data_path = Path(data_path)
        self._cache = {}
        self._index_cache = {}
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
        logger.info(f"Initialized ParquetDataLoader with path: {data_path}")
        self._scan_available_data()
    
    def _scan_available_data(self):
        """Scan available data files and build metadata"""
        self.spx_files = list(self.data_path.glob("SPX_index_price_1m_*.parquet"))
        self.options_files = list(self.data_path.glob("SPXW_option_quotes_1m_*.parquet"))
        
        # Extract dates from filenames
        self.available_dates = set()
        for file in self.spx_files + self.options_files:
            date_str = file.stem.split('_')[-1]  # Get YYYYMMDD from filename
            if len(date_str) == 8 and date_str.isdigit():
                self.available_dates.add(pd.to_datetime(date_str))
        
        self.available_dates = sorted(list(self.available_dates))
        
        logger.info(f"Found data for {len(self.available_dates)} dates")
        if self.available_dates:
            logger.info(f"Date range: {self.available_dates[0].strftime('%Y-%m-%d')} to "
                       f"{self.available_dates[-1].strftime('%Y-%m-%d')}")
        else:
            logger.warning("No valid date files found in data directory")
        logger.info(f"SPX files: {len(self.spx_files)}, Options files: {len(self.options_files)}")
    
    @lru_cache(maxsize=32)
    def load_spx_data(self, date: Union[str, datetime]) -> pd.DataFrame:
        """
        Load SPX index data for a specific date with caching.
        
        Args:
            date: Date as string (YYYY-MM-DD) or datetime
            
        Returns:
            DataFrame with columns: timestamp, price
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        date_str = date.strftime('%Y%m%d')
        file_path = self.data_path / f"SPX_index_price_1m_{date_str}.parquet"
        
        if not file_path.exists():
            logger.warning(f"SPX data not found for {date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            logger.debug(f"Loaded SPX data for {date.strftime('%Y-%m-%d')}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading SPX data for {date}: {e}")
            return pd.DataFrame()
    
    @lru_cache(maxsize=32)
    def load_options_data(self, date: Union[str, datetime]) -> pd.DataFrame:
        """
        Load options data for a specific date with caching and indexing.
        
        Args:
            date: Date as string (YYYY-MM-DD) or datetime
            
        Returns:
            DataFrame with options data, indexed by timestamp
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        date_str = date.strftime('%Y%m%d')
        file_path = self.data_path / f"SPXW_option_quotes_1m_{date_str}_exp{date_str}_sr200.parquet"
        
        if not file_path.exists():
            logger.warning(f"Options data not found for {date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create multi-index for fast filtering
            df = df.set_index(['timestamp', 'strike', 'right']).sort_index()
            
            # Calculate mid prices if not present
            if 'mid' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
            
            logger.debug(f"Loaded options data for {date.strftime('%Y-%m-%d')}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading options data for {date}: {e}")
            return pd.DataFrame()
    
    def get_spx_price_at_time(self, date: Union[str, datetime], 
                             time: Union[str, datetime]) -> Optional[float]:
        """
        Get SPX price at a specific date and time.
        
        Args:
            date: Date as string (YYYY-MM-DD) or datetime
            time: Time as string (HH:MM:SS) or datetime
            
        Returns:
            SPX price or None if not found
        """
        spx_data = self.load_spx_data(date)
        if spx_data.empty:
            return None
        
        # Convert time to timestamp for that date
        if isinstance(time, str):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            timestamp = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {time}")
        else:
            timestamp = time
        
        # Find nearest timestamp (forward fill for missing minutes)
        try:
            if timestamp in spx_data.index:
                return float(spx_data.loc[timestamp, 'price'])
            else:
                # Forward fill to nearest available time
                nearest_data = spx_data[spx_data.index <= timestamp]
                if not nearest_data.empty:
                    return float(nearest_data.iloc[-1]['price'])
                return None
        except Exception as e:
            logger.error(f"Error getting SPX price at {timestamp}: {e}")
            return None
    
    def filter_options_by_strikes(self, df: pd.DataFrame, 
                                 center_strike: float, 
                                 strike_range: float = 100) -> pd.DataFrame:
        """
        Filter options data by strike range around a center strike.
        
        Args:
            df: Options DataFrame (must be indexed by timestamp, strike, right)
            center_strike: Center strike price
            strike_range: Range around center (±strike_range)
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        min_strike = center_strike - strike_range
        max_strike = center_strike + strike_range
        
        try:
            # Use index slicing for fast filtering
            strikes = df.index.get_level_values('strike')
            mask = (strikes >= min_strike) & (strikes <= max_strike)
            return df.loc[mask]
        except Exception as e:
            logger.error(f"Error filtering options by strikes: {e}")
            return df
    
    def filter_options_by_time_range(self, df: pd.DataFrame,
                                   start_time: str = "09:30:00",
                                   end_time: str = "16:00:00") -> pd.DataFrame:
        """
        Filter options data by time range within the trading day.
        
        Args:
            df: Options DataFrame (must be indexed by timestamp)
            start_time: Start time (HH:MM:SS)
            end_time: End time (HH:MM:SS)
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        try:
            timestamps = df.index.get_level_values('timestamp')
            
            # Filter by time of day
            start_filter = timestamps.time >= pd.to_datetime(start_time).time()
            end_filter = timestamps.time <= pd.to_datetime(end_time).time()
            
            mask = start_filter & end_filter
            return df.loc[mask]
        except Exception as e:
            logger.error(f"Error filtering options by time range: {e}")
            return df
    
    def get_options_chain_at_time(self, date: Union[str, datetime],
                                 time: Union[str, datetime],
                                 center_strike: Optional[float] = None,
                                 strike_range: float = 100) -> pd.DataFrame:
        """
        Get complete options chain at a specific time, optionally filtered by strikes.
        
        Args:
            date: Date as string or datetime
            time: Time as string or datetime  
            center_strike: Center strike for filtering (uses SPX price if None)
            strike_range: Range around center strike
            
        Returns:
            DataFrame with options chain
        """
        # Enhanced debug logging for troubleshooting
        logger.debug(f"get_options_chain_at_time called: date={date}, time={time}, "
                    f"center_strike={center_strike}, strike_range={strike_range}")
        
        # Load options data for the date
        options_df = self.load_options_data(date)
        if options_df.empty:
            logger.debug(f"No options data loaded for {date}")
            return pd.DataFrame()
        
        logger.debug(f"Loaded options data: {len(options_df)} total rows")
        
        # Convert time to timestamp
        if isinstance(time, str):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            timestamp = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {time}")
        else:
            timestamp = time
        
        logger.debug(f"Target timestamp: {timestamp}")
        
        # Get options at specific timestamp
        try:
            available_timestamps = options_df.index.get_level_values('timestamp').unique()
            logger.debug(f"Available timestamps: {len(available_timestamps)} from "
                        f"{available_timestamps.min()} to {available_timestamps.max()}")
            
            if timestamp in available_timestamps:
                logger.debug(f"Exact timestamp match found for {timestamp}")
                chain = options_df.loc[timestamp]
            else:
                # Find nearest timestamp
                nearest_ts = available_timestamps[available_timestamps <= timestamp]
                if len(nearest_ts) == 0:
                    logger.debug(f"No timestamps <= {timestamp} found")
                    return pd.DataFrame()
                nearest_ts = nearest_ts.max()
                logger.debug(f"Using nearest timestamp: {nearest_ts} (diff: {timestamp - nearest_ts})")
                chain = options_df.loc[nearest_ts]
            
            logger.debug(f"Chain before filtering: {len(chain)} rows")
            
            # Filter by strikes if requested
            if center_strike is None:
                # Use current SPX price as center
                center_strike = self.get_spx_price_at_time(date, time)
                if center_strike is None:
                    logger.debug("No SPX price found, returning unfiltered chain")
                    filtered_chain = chain.reset_index()
                    logger.debug(f"Returning unfiltered chain with {len(filtered_chain)} rows")
                    return filtered_chain
                logger.debug(f"Using SPX price as center_strike: {center_strike}")
            
            # Apply strike filtering
            strikes = chain.index.get_level_values('strike')
            min_strike = center_strike - strike_range
            max_strike = center_strike + strike_range
            
            logger.debug(f"Strike filtering: {min_strike} <= strike <= {max_strike}")
            logger.debug(f"Available strikes range: {strikes.min()} to {strikes.max()}")
            
            mask = (strikes >= min_strike) & (strikes <= max_strike)
            filtered_chain = chain.loc[mask]
            
            logger.debug(f"Chain after strike filtering: {len(filtered_chain)} rows")
            
            # Reset index to make strikes and option type regular columns
            filtered_chain = filtered_chain.reset_index()
            
            logger.debug(f"Final result: {len(filtered_chain)} options with strikes: "
                        f"{filtered_chain['strike'].unique()}")
            
            return filtered_chain
            
        except Exception as e:
            logger.error(f"Error getting options chain at {timestamp}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def load_date_range(self, start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       data_type: str = 'both') -> Dict[str, pd.DataFrame]:
        """
        Load data for a date range - useful for multi-day backtests.
        
        Args:
            start_date: Start date
            end_date: End date  
            data_type: 'spx', 'options', or 'both'
            
        Returns:
            Dictionary with 'spx' and/or 'options' DataFrames
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Filter available dates to range
        date_range = [d for d in self.available_dates 
                     if start_date <= d <= end_date]
        
        result = {}
        
        if data_type in ['spx', 'both']:
            spx_dfs = []
            for date in date_range:
                df = self.load_spx_data(date)
                if not df.empty:
                    spx_dfs.append(df)
            
            if spx_dfs:
                result['spx'] = pd.concat(spx_dfs).sort_index()
                logger.info(f"Loaded SPX data: {len(result['spx'])} rows across {len(spx_dfs)} days")
        
        if data_type in ['options', 'both']:
            options_dfs = []
            for date in date_range:
                df = self.load_options_data(date)
                if not df.empty:
                    options_dfs.append(df)
            
            if options_dfs:
                result['options'] = pd.concat(options_dfs).sort_index()
                logger.info(f"Loaded options data: {len(result['options'])} rows across {len(options_dfs)} days")
        
        return result
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        return {
            'available_dates': [d.strftime('%Y-%m-%d') for d in self.available_dates],
            'date_range': {
                'start': self.available_dates[0].strftime('%Y-%m-%d') if self.available_dates else None,
                'end': self.available_dates[-1].strftime('%Y-%m-%d') if self.available_dates else None,
                'count': len(self.available_dates)
            },
            'files': {
                'spx': len(self.spx_files),
                'options': len(self.options_files)
            }
        }
    
    def clear_cache(self):
        """Clear all cached data to free memory"""
        self._cache.clear()
        self._index_cache.clear()
        self.load_spx_data.cache_clear()
        self.load_options_data.cache_clear()
        logger.info("Cleared all data caches")


# Convenience functions for quick data access
def load_data_for_backtest(start_date: str, end_date: str, 
                          data_path: str = "data/processed/parquet_1m") -> ParquetDataLoader:
    """
    Convenience function to set up data loader for backtesting.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)  
        data_path: Path to parquet data
        
    Returns:
        Configured ParquetDataLoader
    """
    loader = ParquetDataLoader(data_path)
    
    # Validate date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    available_in_range = [d for d in loader.available_dates 
                         if start_dt <= d <= end_dt]
    
    if not available_in_range:
        logger.warning(f"No data available in range {start_date} to {end_date}")
    else:
        logger.info(f"Data available for {len(available_in_range)} days in backtest range")
    
    return loader