import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import os
from loguru import logger
from config.settings import DATABASE_PATH, PROCESSED_DATA_DIR

class DataStorage:
    """Local data storage system for SPX options and underlying data"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._ensure_directory_exists()
        self._initialize_database()
    
    def _ensure_directory_exists(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _initialize_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # SPX underlying data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spx_underlying (
                    date TEXT PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # SPX options data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spx_options (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    strike REAL NOT NULL,
                    option_type TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    bid REAL,
                    ask REAL,
                    mid_price REAL,
                    bid_ask_spread REAL,
                    spread_pct REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    iv REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, expiration, strike, option_type)
                )
            """)
            
            # Technical indicators table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    date TEXT PRIMARY KEY,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    rsi REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    bb_width REAL,
                    bb_position REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Backtesting results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    max_loss REAL,
                    max_profit REAL,
                    dte INTEGER,
                    strategy_params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_options_date ON spx_options(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_options_expiration ON spx_options(expiration)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_options_strike ON spx_options(strike)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_options_type ON spx_options(option_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_entry_date ON backtest_results(entry_date)")
            
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_underlying_data(self, df: pd.DataFrame) -> bool:
        """
        Save SPX underlying data to database
        
        Args:
            df: DataFrame with columns: date, open, high, low, close, volume
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_clean = df.copy()
                df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')
                
                df_clean.to_sql('spx_underlying', conn, if_exists='replace', index=False,
                              method='multi', chunksize=1000)
                
                logger.info(f"Saved {len(df)} underlying data records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving underlying data: {e}")
            return False
    
    def save_options_data(self, df: pd.DataFrame) -> bool:
        """
        Save SPX options data to database
        
        Args:
            df: DataFrame with options data
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_clean = df.copy()
                df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')
                
                # Handle potential expiration column
                if 'expiration' in df_clean.columns:
                    df_clean['expiration'] = pd.to_datetime(df_clean['expiration']).dt.strftime('%Y-%m-%d')
                
                df_clean.to_sql('spx_options', conn, if_exists='append', index=False,
                              method='multi', chunksize=1000)
                
                logger.info(f"Saved {len(df)} options data records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving options data: {e}")
            return False
    
    def get_underlying_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve SPX underlying data from database
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with underlying data
        """
        query = "SELECT * FROM spx_underlying"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            conditions = []
            
            if start_date:
                conditions.append(" date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append(" date <= ?")
                params.append(end_date)
            
            query += " AND".join(conditions)
        
        query += " ORDER BY date"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            logger.error(f"Error retrieving underlying data: {e}")
            return pd.DataFrame()
    
    def get_options_data(self, date: str = None, expiration: str = None, 
                        option_type: str = None, min_strike: float = None, 
                        max_strike: float = None) -> pd.DataFrame:
        """
        Retrieve SPX options data from database with filters
        
        Args:
            date: Trading date (YYYY-MM-DD)
            expiration: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            DataFrame with options data
        """
        query = "SELECT * FROM spx_options WHERE 1=1"
        params = []
        
        if date:
            query += " AND date = ?"
            params.append(date)
        
        if expiration:
            query += " AND expiration = ?"
            params.append(expiration)
        
        if option_type:
            query += " AND option_type = ?"
            params.append(option_type)
        
        if min_strike is not None:
            query += " AND strike >= ?"
            params.append(min_strike)
        
        if max_strike is not None:
            query += " AND strike <= ?"
            params.append(max_strike)
        
        query += " ORDER BY date, strike"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df['expiration'] = pd.to_datetime(df['expiration'])
                return df
        except Exception as e:
            logger.error(f"Error retrieving options data: {e}")
            return pd.DataFrame()
    
    def save_technical_indicators(self, df: pd.DataFrame) -> bool:
        """
        Save technical indicators to database
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_clean = df.copy()
                df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')
                
                df_clean.to_sql('technical_indicators', conn, if_exists='replace', 
                              index=False, method='multi', chunksize=1000)
                
                logger.info(f"Saved {len(df)} technical indicator records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving technical indicators: {e}")
            return False
    
    def get_technical_indicators(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve technical indicators from database
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with technical indicators
        """
        query = "SELECT * FROM technical_indicators"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            conditions = []
            
            if start_date:
                conditions.append(" date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append(" date <= ?")
                params.append(end_date)
            
            query += " AND".join(conditions)
        
        query += " ORDER BY date"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            logger.error(f"Error retrieving technical indicators: {e}")
            return pd.DataFrame()
    
    def save_backtest_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Save backtesting results to database
        
        Args:
            results: List of backtest result dictionaries
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.DataFrame(results)
                df.to_sql('backtest_results', conn, if_exists='append', 
                         index=False, method='multi', chunksize=1000)
                
                logger.info(f"Saved {len(results)} backtest results")
                return True
                
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of stored data
        
        Returns:
            Dictionary with data summary
        """
        summary = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Underlying data summary
                underlying_query = """
                    SELECT COUNT(*) as count, MIN(date) as start_date, 
                           MAX(date) as end_date FROM spx_underlying
                """
                underlying_result = conn.execute(underlying_query).fetchone()
                summary['underlying'] = {
                    'count': underlying_result[0],
                    'start_date': underlying_result[1],
                    'end_date': underlying_result[2]
                }
                
                # Options data summary
                options_query = """
                    SELECT COUNT(*) as count, COUNT(DISTINCT date) as trading_days,
                           MIN(date) as start_date, MAX(date) as end_date,
                           COUNT(DISTINCT expiration) as expirations
                    FROM spx_options
                """
                options_result = conn.execute(options_query).fetchone()
                summary['options'] = {
                    'count': options_result[0],
                    'trading_days': options_result[1],
                    'start_date': options_result[2],
                    'end_date': options_result[3],
                    'expirations': options_result[4]
                }
                
                # Technical indicators summary
                indicators_query = "SELECT COUNT(*) FROM technical_indicators"
                indicators_count = conn.execute(indicators_query).fetchone()[0]
                summary['technical_indicators'] = {'count': indicators_count}
                
                # Backtest results summary
                backtest_query = """
                    SELECT COUNT(*) as count, COUNT(DISTINCT strategy_name) as strategies
                    FROM backtest_results
                """
                backtest_result = conn.execute(backtest_query).fetchone()
                summary['backtest_results'] = {
                    'count': backtest_result[0],
                    'strategies': backtest_result[1]
                }
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
        
        return summary