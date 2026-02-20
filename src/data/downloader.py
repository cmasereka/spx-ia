import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import time
from loguru import logger
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.theta_client import ThetaDataClient
from src.data.storage import DataStorage
from config.settings import THETA_USERNAME, THETA_PASSWORD, THETA_TERMINAL_PORT, SPX_SYMBOL

class DataDownloader:
    """Orchestrates downloading and storing SPX options and underlying data"""
    
    def __init__(self, theta_username: str = None, theta_password: str = None):
        self.theta_username = theta_username or THETA_USERNAME
        self.theta_password = theta_password or THETA_PASSWORD
        
        if not self.theta_username or not self.theta_password:
            raise ValueError("ThetaData credentials must be provided")
        
        self.theta_client = None
        self.storage = DataStorage()
        
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def connect(self):
        """Connect to ThetaData API"""
        self.theta_client = ThetaDataClient(
            self.theta_username, 
            self.theta_password, 
            THETA_TERMINAL_PORT,
            SPX_SYMBOL
        )
    
    def disconnect(self):
        """Disconnect from ThetaData API"""
        if self.theta_client:
            self.theta_client.disconnect()
    
    def download_underlying_data(self, start_date: datetime, end_date: datetime, 
                               save_to_db: bool = True) -> pd.DataFrame:
        """
        Download SPX underlying data for date range
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            save_to_db: Whether to save to database
            
        Returns:
            DataFrame with underlying data
        """
        logger.info(f"Downloading SPX underlying data from {start_date} to {end_date}")
        
        try:
            df = self.theta_client.get_spx_underlying_data(start_date, end_date)
            
            if df.empty:
                logger.warning("No underlying data retrieved")
                return df
            
            if save_to_db:
                success = self.storage.save_underlying_data(df)
                if success:
                    logger.info(f"Successfully saved {len(df)} underlying data records")
                else:
                    logger.error("Failed to save underlying data")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading underlying data: {e}")
            return pd.DataFrame()
    
    def download_options_data(self, start_date: datetime, end_date: datetime, 
                            dte: int = 0, save_to_db: bool = True) -> List[pd.DataFrame]:
        """
        Download SPX options data for date range
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            dte: Days to expiration (0 for 0DTE)
            save_to_db: Whether to save to database
            
        Returns:
            List of DataFrames with options data
        """
        logger.info(f"Downloading SPX {dte}DTE options data from {start_date} to {end_date}")
        
        try:
            options_data_list = self.theta_client.get_options_data_range(
                start_date, end_date, dte
            )
            
            if not options_data_list:
                logger.warning("No options data retrieved")
                return []
            
            if save_to_db:
                total_saved = 0
                for df in options_data_list:
                    if not df.empty:
                        # Add expiration date (same as date for 0DTE)
                        if dte == 0:
                            df['expiration'] = df['date']
                        else:
                            df['expiration'] = df['date'] + timedelta(days=dte)
                        
                        success = self.storage.save_options_data(df)
                        if success:
                            total_saved += len(df)
                
                logger.info(f"Successfully saved {total_saved} options records")
            
            return options_data_list
            
        except Exception as e:
            logger.error(f"Error downloading options data: {e}")
            return []
    
    def download_complete_dataset(self, start_date: datetime, end_date: datetime,
                                dte_list: List[int] = None) -> Dict[str, Any]:
        """
        Download complete dataset including underlying and options for multiple DTEs
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            dte_list: List of DTEs to download (defaults to [0])
            
        Returns:
            Summary dictionary of downloaded data
        """
        if dte_list is None:
            dte_list = [0]  # Default to 0DTE
        
        summary = {
            'start_date': start_date,
            'end_date': end_date,
            'underlying_records': 0,
            'options_records': 0,
            'errors': []
        }
        
        logger.info(f"Starting complete dataset download from {start_date} to {end_date}")
        
        try:
            # Download underlying data
            underlying_df = self.download_underlying_data(start_date, end_date)
            summary['underlying_records'] = len(underlying_df)
            
            # Download options data for each DTE
            total_options_records = 0
            for dte in dte_list:
                logger.info(f"Downloading {dte}DTE options data...")
                options_data_list = self.download_options_data(start_date, end_date, dte)
                
                dte_records = sum(len(df) for df in options_data_list if not df.empty)
                total_options_records += dte_records
                
                logger.info(f"Downloaded {dte_records} records for {dte}DTE")
                
                # Add delay between DTE downloads to avoid rate limits
                if len(dte_list) > 1:
                    time.sleep(1)
            
            summary['options_records'] = total_options_records
            
        except Exception as e:
            error_msg = f"Error in complete dataset download: {e}"
            logger.error(error_msg)
            summary['errors'].append(error_msg)
        
        return summary
    
    def download_recent_data(self, days_back: int = 30, dte_list: List[int] = None) -> Dict[str, Any]:
        """
        Download recent data (last N trading days)
        
        Args:
            days_back: Number of days to look back
            dte_list: List of DTEs to download
            
        Returns:
            Summary dictionary
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        return self.download_complete_dataset(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time()),
            dte_list
        )
    
    def update_data(self) -> Dict[str, Any]:
        """
        Update data by downloading any missing recent data
        
        Returns:
            Summary dictionary
        """
        try:
            # Check what data we already have
            summary = self.storage.get_data_summary()
            
            if not summary['underlying']['end_date']:
                # No existing data, download last 30 days
                return self.download_recent_data(30)
            
            # Get the last date we have data for
            last_date_str = summary['underlying']['end_date']
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            
            # Download from last date to today
            start_date = last_date + timedelta(days=1)
            end_date = datetime.now().date()
            
            if start_date >= end_date:
                logger.info("Data is up to date")
                return {'message': 'Data is up to date', 'underlying_records': 0, 'options_records': 0}
            
            return self.download_complete_dataset(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time())
            )
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return {'errors': [str(e)]}
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get current data status and summary
        
        Returns:
            Data status dictionary
        """
        return self.storage.get_data_summary()

def main():
    """Command line interface for data downloader"""
    parser = argparse.ArgumentParser(description='Download SPX options data from ThetaData')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=30, help='Days back from today')
    parser.add_argument('--dte', type=str, default='0', help='Comma-separated list of DTEs')
    parser.add_argument('--update', action='store_true', help='Update existing data')
    parser.add_argument('--status', action='store_true', help='Show data status')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/data_download_{time}.log", rotation="1 day", retention="30 days")
    
    try:
        with DataDownloader() as downloader:
            if args.status:
                status = downloader.get_data_status()
                print("Data Status:")
                for table, info in status.items():
                    print(f"  {table}: {info}")
                return
            
            if args.update:
                summary = downloader.update_data()
            elif args.start_date and args.end_date:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
                dte_list = [int(x.strip()) for x in args.dte.split(',')]
                summary = downloader.download_complete_dataset(start_date, end_date, dte_list)
            else:
                dte_list = [int(x.strip()) for x in args.dte.split(',')]
                summary = downloader.download_recent_data(args.days_back, dte_list)
            
            print("Download Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()