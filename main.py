#!/usr/bin/env python3
"""
SPX Options Trading Bot - Main Entry Point

This script provides a command-line interface for:
1. Downloading SPX options data from ThetaData
2. Running backtests on 0DTE options strategies
3. Analyzing results and generating reports

Usage:
    python main.py download --days-back 30
    python main.py backtest --strategy iron_condor --start-date 2024-01-01 --end-date 2024-12-31
    python main.py status
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.downloader import DataDownloader
from src.data.storage import DataStorage
from src.data.csv_importer import CSVDataImporter
from src.indicators.technical_indicators import TechnicalIndicators
from src.strategies.options_strategies import StrategyBuilder, IronCondor, VerticalSpread
from config.settings import *

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.storage = DataStorage()
        self.indicators = TechnicalIndicators()
        self.csv_importer = CSVDataImporter()
        self.downloader = None
        
        # Setup logging
        logger.add("logs/bot_{time}.log", rotation="1 day", retention="30 days")
        logger.info("SPX Trading Bot initialized")
    
    def setup_downloader(self):
        """Initialize data downloader with credentials"""
        try:
            self.downloader = DataDownloader()
            self.downloader.connect()
            logger.info("Data downloader connected successfully")
        except Exception as e:
            logger.error(f"Failed to setup downloader: {e}")
            raise
    
    def download_data(self, days_back: int = 30, update_only: bool = False):
        """Download SPX data"""
        if not self.downloader:
            self.setup_downloader()
        
        try:
            if update_only:
                summary = self.downloader.update_data()
            else:
                summary = self.downloader.download_recent_data(days_back)
            
            logger.info(f"Download completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {"error": str(e)}
        finally:
            if self.downloader:
                self.downloader.disconnect()
    
    def get_data_status(self):
        """Get current data status"""
        return self.storage.get_data_summary()
    
    def generate_sample_data(self, days_back: int = 30):
        """Generate sample data for testing"""
        return self.csv_importer.generate_sample_data(days_back)
    
    def prepare_data_for_backtest(self, start_date: str, end_date: str):
        """Prepare data with technical indicators for backtesting"""
        logger.info(f"Preparing data from {start_date} to {end_date}")
        
        # Get underlying data
        underlying_df = self.storage.get_underlying_data(start_date, end_date)
        
        if underlying_df.empty:
            logger.error("No underlying data found for the specified period")
            return None, None
        
        # Calculate technical indicators
        indicators_df = self.indicators.calculate_all_indicators(underlying_df)
        signals_df = self.indicators.get_trading_signals(indicators_df)
        
        # Save indicators to database
        self.storage.save_technical_indicators(signals_df)
        
        return signals_df, underlying_df
    
    def run_iron_condor_backtest(self, start_date: str, end_date: str, 
                                strategy_params: dict = None):
        """Run Iron Condor strategy backtest"""
        logger.info("Running Iron Condor backtest")
        
        params = strategy_params or IRON_CONDOR_PARAMS
        
        # Prepare data
        signals_df, underlying_df = self.prepare_data_for_backtest(start_date, end_date)
        
        if signals_df is None:
            return {"error": "Failed to prepare data"}
        
        results = []
        
        for idx, row in signals_df.iterrows():
            if pd.isna(row['neutral_signal']) or not row['neutral_signal']:
                continue
            
            trade_date = row['date'].strftime('%Y-%m-%d')
            underlying_price = row['close']
            
            # Get options data for this date
            options_df = self.storage.get_options_data(
                date=trade_date,
                expiration=trade_date  # 0DTE
            )
            
            if options_df.empty:
                continue
            
            # Convert options data to dictionary format
            options_data = {}
            for _, opt_row in options_df.iterrows():
                key = f"{opt_row['strike']}_{opt_row['option_type']}"
                options_data[key] = opt_row.to_dict()
            
            try:
                # Create Iron Condor strategy
                ic = StrategyBuilder.build_iron_condor(
                    entry_date=row['date'],
                    underlying_price=underlying_price,
                    options_data=options_data,
                    put_distance=params['put_strike_distance'],
                    call_distance=params['call_strike_distance']
                )
                
                # Calculate P&L at expiration (0DTE)
                final_pnl = ic.get_profit_at_expiration(underlying_price)
                
                # Store result
                result = {
                    'strategy_name': 'Iron Condor',
                    'entry_date': trade_date,
                    'exit_date': trade_date,  # 0DTE
                    'entry_price': ic.entry_credit,
                    'exit_price': 0.0,  # Expired
                    'pnl': final_pnl,
                    'pnl_pct': (final_pnl / ic.entry_credit * 100) if ic.entry_credit > 0 else 0,
                    'max_loss': ic.max_loss,
                    'max_profit': ic.max_profit,
                    'dte': 0,
                    'strategy_params': str(params)
                }
                
                results.append(result)
                logger.debug(f"IC trade on {trade_date}: P&L=${final_pnl:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to create IC strategy for {trade_date}: {e}")
                continue
        
        # Save results to database
        if results:
            self.storage.save_backtest_results(results)
            logger.info(f"Completed Iron Condor backtest: {len(results)} trades")
        
        return self.analyze_backtest_results(results)
    
    def analyze_backtest_results(self, results: list):
        """Analyze backtest results and generate summary"""
        if not results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame(results)
        
        analysis = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': len(df[df['pnl'] > 0]) / len(df) * 100,
            'total_pnl': df['pnl'].sum(),
            'avg_pnl_per_trade': df['pnl'].mean(),
            'max_win': df['pnl'].max(),
            'max_loss': df['pnl'].min(),
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(df[df['pnl'] < 0]) > 0 else float('inf'),
            'sharpe_ratio': df['pnl'].mean() / df['pnl'].std() if df['pnl'].std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df['pnl'].cumsum()),
            'start_date': df['entry_date'].min(),
            'end_date': df['entry_date'].max()
        }
        
        return analysis
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() * 100
    
    def print_analysis(self, analysis: dict):
        """Print formatted analysis results"""
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        print(f"Period: {analysis['start_date']} to {analysis['end_date']}")
        print(f"Total Trades: {analysis['total_trades']}")
        print(f"Win Rate: {analysis['win_rate']:.1f}%")
        print(f"Total P&L: ${analysis['total_pnl']:.2f}")
        print(f"Avg P&L per Trade: ${analysis['avg_pnl_per_trade']:.2f}")
        print(f"Max Win: ${analysis['max_win']:.2f}")
        print(f"Max Loss: ${analysis['max_loss']:.2f}")
        print(f"Profit Factor: {analysis['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {analysis['max_drawdown']:.1f}%")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='SPX Options Trading Bot')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download SPX data')
    download_parser.add_argument('--days-back', type=int, default=30, help='Days back from today')
    download_parser.add_argument('--update', action='store_true', help='Update existing data only')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtest')
    backtest_parser.add_argument('--strategy', choices=['iron_condor', 'call_spread', 'put_spread'], 
                                default='iron_condor', help='Strategy to backtest')
    backtest_parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show data status')
    
    # Sample data command
    sample_parser = subparsers.add_parser('sample', help='Generate sample data for testing')
    sample_parser.add_argument('--days-back', type=int, default=30, help='Days of sample data to generate')
    
    # Test ThetaData connection
    test_parser = subparsers.add_parser('test-theta', help='Test ThetaData Terminal connection')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    bot = TradingBot()
    
    try:
        if args.command == 'download':
            print("Downloading SPX data...")
            summary = bot.download_data(args.days_back, args.update)
            print(f"Download completed: {summary}")
            
        elif args.command == 'backtest':
            print(f"Running {args.strategy} backtest...")
            
            if args.strategy == 'iron_condor':
                analysis = bot.run_iron_condor_backtest(args.start_date, args.end_date)
            else:
                print(f"Strategy {args.strategy} not implemented yet")
                return
            
            bot.print_analysis(analysis)
            
        elif args.command == 'status':
            status = bot.get_data_status()
            print("\nData Status:")
            for table, info in status.items():
                print(f"  {table}: {info}")
        
        elif args.command == 'sample':
            print("Generating sample data for testing...")
            success = bot.generate_sample_data(args.days_back)
            if success:
                print(f"Sample data generated successfully ({args.days_back} days)")
            else:
                print("Failed to generate sample data")
        
        elif args.command == 'test-theta':
            print("Testing ThetaData Terminal connection...")
            try:
                from src.data.theta_client import ThetaDataClient
                from config.settings import THETA_TERMINAL_PORT, SPX_SYMBOL
                client = ThetaDataClient(
                    username=os.getenv('THETA_USERNAME', 'test'),
                    password=os.getenv('THETA_PASSWORD', 'test'),
                    port=THETA_TERMINAL_PORT,
                    symbol=SPX_SYMBOL
                )
                
                connected = client.connect()
                if connected:
                    print("✅ ThetaData Terminal connection successful!")
                    print("You can now download real market data.")
                else:
                    print("❌ ThetaData Terminal connection failed.")
                    print("\nTo fix this:")
                    print("1. Download ThetaData Terminal from https://thetadata.com")
                    print("2. Launch the Terminal and login")
                    print("3. Ensure it's running on localhost:25503")
                    print("4. Update your .env file with correct credentials")
                
                client.disconnect()
            except Exception as e:
                print(f"Error testing connection: {e}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()