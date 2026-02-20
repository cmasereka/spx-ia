#!/usr/bin/env python3
"""
Quick analysis script for tick data structure and compression potential
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_spx_data():
    """Analyze SPX index data structure"""
    spx_file = "/Users/cmasereka/Personal/Trading/spx-ai/data/raw/tick/theta_tick_week_2026-02-09/SPX_index_price_tick_20260209.csv"
    
    print("=== SPX INDEX DATA ANALYSIS ===")
    df = pd.read_csv(spx_file)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"File size: {os.path.getsize(spx_file) / 1024**2:.2f} MB")
    print("\nFirst few rows:")
    print(df.head(3))
    print(f"\nPrice range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
def sample_options_data():
    """Sample and analyze options data structure"""
    options_file = "~/temp_tick_data/SPXW_option_quotes_tick_20260209_exp20260209_sr200.csv"
    options_file = os.path.expanduser(options_file)
    
    print("\n=== OPTIONS DATA ANALYSIS (Sample) ===")
    
    # Read just first 100k rows to avoid memory issues
    df = pd.read_csv(options_file, nrows=100000)
    print(f"Sample rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Get full file size
    file_size = os.path.getsize(options_file) / 1024**3  # GB
    print(f"Full file size: {file_size:.2f} GB")
    
    # Estimate total rows based on sample
    estimated_rows = len(df) * (file_size * 1024) / (df.memory_usage(deep=True).sum() / 1024**2)
    print(f"Estimated total rows: ~{estimated_rows:,.0f}")
    
    print("\nData structure:")
    print(df.dtypes)
    
    print("\nUnique values:")
    print(f"  Strikes: {df['strike'].nunique()} unique ({df['strike'].min():.0f} - {df['strike'].max():.0f})")
    print(f"  Rights: {df['right'].unique()}")
    print(f"  Bid exchanges: {df['bid_exchange'].nunique()}")
    print(f"  Ask exchanges: {df['ask_exchange'].nunique()}")
    
    print("\nSample data:")
    print(df[['strike', 'right', 'timestamp', 'bid', 'ask']].head(10))
    
def estimate_compression_potential():
    """Estimate compression ratios for different approaches"""
    print("\n=== COMPRESSION ESTIMATES ===")
    
    # Test with sample data
    options_file = os.path.expanduser("~/temp_tick_data/SPXW_option_quotes_tick_20260209_exp20260209_sr200.csv")
    
    # Read small sample for testing
    df_sample = pd.read_csv(options_file, nrows=50000)
    
    # Convert to different formats and measure
    print("Testing compression on 50k row sample:")
    
    # Original CSV size
    csv_size = df_sample.memory_usage(deep=True).sum() / 1024**2
    print(f"  In-memory CSV: {csv_size:.2f} MB")
    
    # Optimize data types
    df_optimized = df_sample.copy()
    df_optimized['strike'] = df_optimized['strike'].astype('float32')
    df_optimized['bid'] = df_optimized['bid'].astype('float32') 
    df_optimized['ask'] = df_optimized['ask'].astype('float32')
    df_optimized['bid_size'] = df_optimized['bid_size'].astype('uint16')
    df_optimized['ask_size'] = df_optimized['ask_size'].astype('uint16')
    df_optimized['right'] = df_optimized['right'].astype('category')
    df_optimized['bid_exchange'] = df_optimized['bid_exchange'].astype('category')
    df_optimized['ask_exchange'] = df_optimized['ask_exchange'].astype('category')
    
    optimized_size = df_optimized.memory_usage(deep=True).sum() / 1024**2
    print(f"  Optimized types: {optimized_size:.2f} MB ({csv_size/optimized_size:.1f}x smaller)")
    
    # Test parquet
    try:
        parquet_file = "/tmp/test_options.parquet"
        df_optimized.to_parquet(parquet_file, compression='snappy')
        parquet_size = os.path.getsize(parquet_file) / 1024**2
        print(f"  Parquet (snappy): {parquet_size:.2f} MB ({csv_size/parquet_size:.1f}x smaller)")
        os.remove(parquet_file)
    except Exception as e:
        print(f"  Parquet test failed: {e}")

if __name__ == "__main__":
    analyze_spx_data()
    sample_options_data() 
    estimate_compression_potential()