import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import os
from loguru import logger
from src.data.storage import DataStorage

class CSVDataImporter:
    """Import SPX options data from CSV files"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.storage = DataStorage()
        os.makedirs(data_dir, exist_ok=True)
    
    def import_spx_underlying(self, csv_path: str) -> bool:
        """
        Import SPX underlying data from CSV
        
        Expected CSV format:
        date,open,high,low,close,volume
        2024-01-01,4500.00,4520.00,4495.00,4515.00,1000000
        """
        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns: {required_cols}")
                return False
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            success = self.storage.save_underlying_data(df)
            if success:
                logger.info(f"Imported {len(df)} underlying data records from {csv_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error importing underlying data: {e}")
            return False
    
    def import_spx_options(self, csv_path: str) -> bool:
        """
        Import SPX options data from CSV
        
        Expected CSV format:
        date,expiration,strike,option_type,open,high,low,close,volume,bid,ask,delta,gamma,theta,vega,iv
        """
        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df['expiration'] = pd.to_datetime(df['expiration'])
            
            # Validate required columns
            required_cols = ['date', 'expiration', 'strike', 'option_type', 'bid', 'ask']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns: {required_cols}")
                return False
            
            # Add calculated fields
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df['spread_pct'] = (df['bid_ask_spread'] / df['mid_price']) * 100
            
            # Fill missing OHLC with mid price
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df['mid_price']
            
            # Fill missing Greeks with 0
            for col in ['delta', 'gamma', 'theta', 'vega', 'iv']:
                if col not in df.columns:
                    df[col] = 0.0
            
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            success = self.storage.save_options_data(df)
            if success:
                logger.info(f"Imported {len(df)} options records from {csv_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error importing options data: {e}")
            return False
    
    def generate_sample_data(self, days_back: int = 30) -> bool:
        """
        Generate sample SPX data for testing (DO NOT USE FOR REAL TRADING)
        """
        logger.warning("Generating SAMPLE DATA - DO NOT USE FOR REAL TRADING")
        
        # Generate sample underlying data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # Only weekdays
        
        # Sample SPX data around 4500
        np.random.seed(42)
        base_price = 4500
        
        underlying_data = []
        current_price = base_price
        
        for date in dates:
            daily_return = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            high = current_price * np.random.uniform(1.001, 1.02)
            low = current_price * np.random.uniform(0.98, 0.999)
            open_price = np.random.uniform(low, high)
            close_price = current_price
            volume = np.random.randint(1000000, 5000000)
            
            underlying_data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        underlying_df = pd.DataFrame(underlying_data)
        
        # Generate sample options data (0DTE)
        options_data = []
        
        for _, row in underlying_df.iterrows():
            underlying_price = row['close']
            date = row['date']
            
            # Generate strikes around current price
            strikes = range(int(underlying_price - 200), int(underlying_price + 200), 25)
            
            for strike in strikes:
                for option_type in ['call', 'put']:
                    # Simple Black-Scholes approximation for demo
                    if option_type == 'call':
                        intrinsic = max(0, underlying_price - strike)
                    else:
                        intrinsic = max(0, strike - underlying_price)
                    
                    time_value = np.random.uniform(0, 10) if intrinsic == 0 else np.random.uniform(0, 5)
                    mid_price = intrinsic + time_value
                    
                    bid = mid_price * 0.95
                    ask = mid_price * 1.05
                    
                    options_data.append({
                        'date': date,
                        'expiration': date,  # 0DTE
                        'strike': strike,
                        'option_type': option_type,
                        'open': mid_price,
                        'high': mid_price * 1.1,
                        'low': mid_price * 0.9,
                        'close': mid_price,
                        'volume': np.random.randint(0, 1000),
                        'bid': bid,
                        'ask': ask,
                        'delta': np.random.uniform(-1, 1),
                        'gamma': np.random.uniform(0, 0.1),
                        'theta': np.random.uniform(-5, 0),
                        'vega': np.random.uniform(0, 10),
                        'iv': np.random.uniform(0.1, 0.5)
                    })
        
        options_df = pd.DataFrame(options_data)
        
        # Save to database
        underlying_success = self.storage.save_underlying_data(underlying_df)
        options_success = self.storage.save_options_data(options_df)
        
        if underlying_success and options_success:
            logger.info(f"Generated sample data: {len(underlying_df)} underlying, {len(options_df)} options records")
            return True
        
        return False