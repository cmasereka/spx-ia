#!/usr/bin/env python3
"""
Memory-efficient CSV to Parquet converter for large financial data files.
Processes files in chunks to avoid memory issues.
"""
import pandas as pd
import os
from pathlib import Path
import argparse
from typing import Dict, Any


def get_optimized_dtypes() -> Dict[str, Any]:
    """Return optimized data types for common financial data columns."""
    return {
        'strike': 'float32',
        'bid': 'float32', 
        'ask': 'float32',
        'price': 'float32',
        'bid_size': 'uint16',
        'ask_size': 'uint16',
        'size': 'uint16',
        'right': 'category',
        'bid_exchange': 'category',
        'ask_exchange': 'category',
        'exchange': 'category'
    }


def convert_csv_to_parquet_chunked(csv_path: str, parquet_path: str, 
                                 chunk_size: int = 50000, compression: str = 'snappy'):
    """
    Convert large CSV to Parquet using chunked processing.
    
    Args:
        csv_path: Path to input CSV file
        parquet_path: Path for output Parquet file  
        chunk_size: Number of rows to process at once
        compression: Compression algorithm ('snappy', 'gzip', 'brotli')
    """
    print(f"Converting {csv_path} to {parquet_path}")
    print(f"Chunk size: {chunk_size:,} rows, Compression: {compression}")
    
    # Get file size
    file_size_mb = os.path.getsize(csv_path) / 1024**2
    print(f"Input file size: {file_size_mb:.1f} MB")
    
    # Read first chunk to determine schema and optimize dtypes
    first_chunk = pd.read_csv(csv_path, nrows=1000)
    
    # Apply optimized dtypes where possible
    dtype_mapping = get_optimized_dtypes()
    dtypes_to_use = {}
    
    for col in first_chunk.columns:
        if col in dtype_mapping:
            try:
                # Test if conversion is possible
                first_chunk[col].astype(dtype_mapping[col])
                dtypes_to_use[col] = dtype_mapping[col]
                print(f"  Optimizing {col} -> {dtype_mapping[col]}")
            except (ValueError, TypeError):
                print(f"  Keeping {col} as default type")
    
    # Process file in chunks
    chunk_files = []
    chunk_num = 0
    
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtypes_to_use):
            chunk_num += 1
            chunk_file = f"{parquet_path}.chunk_{chunk_num}.parquet"
            
            print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...")
            
            # Convert remaining columns to categories if they have low cardinality
            for col in chunk.select_dtypes(include=['object']).columns:
                if col not in dtypes_to_use:
                    unique_ratio = chunk[col].nunique() / len(chunk)
                    if unique_ratio < 0.1:  # Less than 10% unique values
                        chunk[col] = chunk[col].astype('category')
                        print(f"    Converting {col} to category (cardinality: {unique_ratio:.1%})")
            
            # Save chunk
            chunk.to_parquet(chunk_file, compression=compression, index=False)
            chunk_files.append(chunk_file)
            
        # Combine all chunks into final file
        print(f"Combining {len(chunk_files)} chunks...")
        combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        combined_df.to_parquet(parquet_path, compression=compression, index=False)
        
        # Clean up chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
            
        # Report results
        output_size_mb = os.path.getsize(parquet_path) / 1024**2
        compression_ratio = file_size_mb / output_size_mb
        
        print(f"✅ Conversion complete!")
        print(f"  Output size: {output_size_mb:.1f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Total rows: {len(combined_df):,}")
        
    except Exception as e:
        # Clean up on error
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        raise e


def batch_convert_directory(input_dir: str, output_dir: str, pattern: str = "*.csv",
                          chunk_size: int = 50000, compression: str = 'snappy'):
    """Convert all CSV files in a directory to Parquet."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob(pattern))
    print(f"Found {len(csv_files)} CSV files to convert")
    
    for csv_file in csv_files:
        parquet_file = output_path / f"{csv_file.stem}.parquet"
        convert_csv_to_parquet_chunked(str(csv_file), str(parquet_file), 
                                     chunk_size, compression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet efficiently")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("output", help="Output Parquet file or directory")
    parser.add_argument("--chunk-size", type=int, default=50000, 
                       help="Rows per chunk (default: 50,000)")
    parser.add_argument("--compression", choices=['snappy', 'gzip', 'brotli'], 
                       default='snappy', help="Compression algorithm")
    parser.add_argument("--pattern", default="*.csv", 
                       help="File pattern for directory mode")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        convert_csv_to_parquet_chunked(args.input, args.output, 
                                     args.chunk_size, args.compression)
    elif os.path.isdir(args.input):
        batch_convert_directory(args.input, args.output, args.pattern,
                              args.chunk_size, args.compression)
    else:
        print(f"Error: {args.input} is not a valid file or directory")