#!/usr/bin/env python3
"""
Advanced caching layer for backtesting with intelligent cache management.
Optimizes frequently accessed data patterns for ultra-fast strategy backtesting.
"""
import pandas as pd
import numpy as np
import pickle
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, time, timedelta
from functools import lru_cache, wraps
from dataclasses import dataclass, asdict
import sqlite3
from threading import Lock
from loguru import logger

from ..data.query_engine import BacktestQueryEngine
from .iron_condor_loader import IronCondorSetup


@dataclass 
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SmartCache:
    """
    Intelligent cache that learns access patterns and pre-loads frequently needed data.
    """
    
    def __init__(self, max_memory_mb: int = 500, cache_dir: str = "cache/backtesting"):
        self.max_memory_mb = max_memory_mb
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches with different strategies
        self._memory_cache = {}  # LRU for hot data
        self._pattern_cache = {}  # Pattern-based prediction cache
        self._preload_cache = {}  # Pre-loaded data cache
        
        # Access pattern tracking
        self._access_patterns = {}
        self._access_frequencies = {}
        self._access_times = {}
        
        # Cache stats
        self.stats = CacheStats()
        self._lock = Lock()
        
        # Persistent cache database
        self._init_persistent_cache()
        
        logger.info(f"Initialized SmartCache with {max_memory_mb}MB limit")
    
    def _init_persistent_cache(self):
        """Initialize persistent SQLite cache for cross-session data."""
        self.db_path = self.cache_dir / "cache.db"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    access_count INTEGER DEFAULT 0,
                    last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    size_bytes INTEGER,
                    data_type TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_patterns (
                    pattern_key TEXT PRIMARY KEY,
                    pattern_data TEXT,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with pattern learning."""
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                self.stats.hits += 1
                self._record_access(key)
                return self._memory_cache[key]
            
            # Check persistent cache
            cached_data = self._get_from_persistent(key)
            if cached_data is not None:
                self.stats.hits += 1
                self._memory_cache[key] = cached_data  # Promote to memory
                self._record_access(key)
                return cached_data
            
            self.stats.misses += 1
            return default
    
    def put(self, key: str, value: Any, data_type: str = "general", ttl_hours: int = 24):
        """Put item in cache with intelligent placement."""
        with self._lock:
            # Serialize data
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Memory cache (for hot data)
            self._memory_cache[key] = value
            
            # Persistent cache
            self._put_to_persistent(key, serialized, size_bytes, data_type)
            
            # Update patterns
            self._record_access(key)
            
            # Manage memory usage
            self._manage_memory_usage()
    
    def _get_from_persistent(self, key: str) -> Any:
        """Get data from persistent SQLite cache."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT data FROM cache_entries WHERE key = ?", (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update access statistics
                    conn.execute("""
                        UPDATE cache_entries 
                        SET access_count = access_count + 1, 
                            last_access = CURRENT_TIMESTAMP 
                        WHERE key = ?
                    """, (key,))
                    
                    return pickle.loads(result[0])
                
                return None
                
        except Exception as e:
            logger.error(f"Error reading from persistent cache: {e}")
            return None
    
    def _put_to_persistent(self, key: str, data: bytes, size_bytes: int, data_type: str):
        """Put data to persistent SQLite cache."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, size_bytes, data_type) 
                    VALUES (?, ?, ?, ?)
                """, (key, data, size_bytes, data_type))
                
        except Exception as e:
            logger.error(f"Error writing to persistent cache: {e}")
    
    def _record_access(self, key: str):
        """Record access pattern for predictive caching."""
        now = datetime.now()
        
        # Update frequency tracking
        self._access_frequencies[key] = self._access_frequencies.get(key, 0) + 1
        self._access_times[key] = now
        
        # Pattern detection (e.g., sequential date access)
        self._detect_patterns(key, now)
    
    def _detect_patterns(self, key: str, access_time: datetime):
        """Detect access patterns for predictive pre-loading."""
        # Extract date/time patterns from keys
        if "_" in key:
            parts = key.split("_")
            for part in parts:
                if len(part) == 8 and part.isdigit():  # Date pattern YYYYMMDD
                    try:
                        date = datetime.strptime(part, "%Y%m%d")
                        pattern_key = key.replace(part, "DATE_PATTERN")
                        
                        if pattern_key not in self._pattern_cache:
                            self._pattern_cache[pattern_key] = []
                        
                        self._pattern_cache[pattern_key].append({
                            'date': date,
                            'access_time': access_time,
                            'original_key': key
                        })
                        
                    except ValueError:
                        pass
    
    def preload_predicted_data(self, query_engine: BacktestQueryEngine, 
                             current_date: Union[str, datetime]) -> int:
        """
        Pre-load data based on detected access patterns.
        
        Args:
            query_engine: Query engine for data access
            current_date: Current date being processed
            
        Returns:
            Number of items pre-loaded
        """
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, "%Y-%m-%d")
        
        preloaded_count = 0
        
        # Predict next dates based on patterns
        for pattern_key, accesses in self._pattern_cache.items():
            if len(accesses) < 2:
                continue
            
            # Sort by date
            accesses.sort(key=lambda x: x['date'])
            
            # Detect sequential access pattern
            if len(accesses) >= 3:
                dates = [a['date'] for a in accesses[-3:]]
                if self._is_sequential_pattern(dates):
                    # Pre-load next few dates
                    next_dates = self._predict_next_dates(current_date, 3)
                    
                    for next_date in next_dates:
                        predicted_keys = self._generate_predicted_keys(pattern_key, next_date)
                        
                        for pred_key in predicted_keys:
                            if pred_key not in self._memory_cache:
                                data = self._load_predicted_data(query_engine, pred_key, next_date)
                                if data is not None:
                                    self._preload_cache[pred_key] = data
                                    preloaded_count += 1
        
        logger.info(f"Pre-loaded {preloaded_count} data items based on patterns")
        return preloaded_count
    
    def _is_sequential_pattern(self, dates: List[datetime]) -> bool:
        """Check if dates follow a sequential pattern."""
        if len(dates) < 2:
            return False
        
        # Check for daily sequence
        diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        return all(d == 1 for d in diffs) or all(d in [1, 3] for d in diffs)  # Daily or skip weekends
    
    def _predict_next_dates(self, current_date: datetime, count: int) -> List[datetime]:
        """Predict next trading dates."""
        next_dates = []
        date = current_date
        
        for _ in range(count):
            date += timedelta(days=1)
            # Skip weekends (basic business day logic)
            while date.weekday() >= 5:
                date += timedelta(days=1)
            next_dates.append(date)
        
        return next_dates
    
    def _generate_predicted_keys(self, pattern_key: str, date: datetime) -> List[str]:
        """Generate predicted cache keys for a date."""
        date_str = date.strftime("%Y%m%d")
        
        # Common key patterns for backtesting
        patterns = [
            pattern_key.replace("DATE_PATTERN", date_str),
            f"spx_price_{date_str}_10:00:00",
            f"spx_price_{date_str}_14:00:00",
            f"options_chain_{date_str}_10:00:00",
            f"liquid_options_{date_str}_10:00:00",
            f"iron_condor_setups_{date_str}_10:00:00"
        ]
        
        return patterns
    
    def _load_predicted_data(self, query_engine: BacktestQueryEngine, 
                           key: str, date: datetime) -> Any:
        """Load predicted data using query engine."""
        try:
            if "spx_price" in key:
                # Extract time from key
                parts = key.split("_")
                if len(parts) >= 4:
                    time_str = parts[-1].replace(":", ":")
                    return query_engine.get_fastest_spx_price(date, time_str)
            
            elif "options_chain" in key:
                parts = key.split("_")
                if len(parts) >= 4:
                    time_str = parts[-1].replace(":", ":")
                    return query_engine.loader.get_options_chain_at_time(date, time_str)
            
            elif "liquid_options" in key:
                parts = key.split("_")
                if len(parts) >= 4:
                    time_str = parts[-1].replace(":", ":")
                    return query_engine.find_liquid_options(date, time_str)
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not pre-load data for {key}: {e}")
            return None
    
    def _manage_memory_usage(self):
        """Manage memory usage by evicting least used items."""
        # Estimate current memory usage
        current_usage = sum(
            len(pickle.dumps(value)) for value in self._memory_cache.values()
        ) / (1024 * 1024)  # Convert to MB
        
        if current_usage > self.max_memory_mb:
            # Evict least frequently accessed items
            items_to_evict = []
            
            for key in self._memory_cache:
                frequency = self._access_frequencies.get(key, 0)
                last_access = self._access_times.get(key, datetime.now())
                
                # Score based on frequency and recency
                age_hours = (datetime.now() - last_access).total_seconds() / 3600
                score = frequency / (1 + age_hours)
                
                items_to_evict.append((score, key))
            
            # Sort by score and evict lowest
            items_to_evict.sort()
            evict_count = len(items_to_evict) // 4  # Evict 25%
            
            for _, key in items_to_evict[:evict_count]:
                del self._memory_cache[key]
                self.stats.evictions += 1
            
            logger.info(f"Evicted {evict_count} items from memory cache")
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        # Update memory usage
        memory_usage = sum(
            len(pickle.dumps(value)) for value in self._memory_cache.values()
        ) / (1024 * 1024)
        
        self.stats.memory_usage_mb = memory_usage
        return self.stats
    
    def clear(self, clear_persistent: bool = False):
        """Clear cache."""
        with self._lock:
            self._memory_cache.clear()
            self._pattern_cache.clear()
            self._preload_cache.clear()
            self._access_patterns.clear()
            self._access_frequencies.clear()
            self._access_times.clear()
            
            if clear_persistent:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_entries")
                    conn.execute("DELETE FROM access_patterns")
            
            self.stats = CacheStats()
            logger.info("Cache cleared")


class BacktestingCacheManager:
    """
    High-level cache manager specifically designed for backtesting workflows.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine, 
                 max_memory_mb: int = 1000, cache_dir: str = "cache/backtesting"):
        self.query_engine = query_engine
        self.cache = SmartCache(max_memory_mb, cache_dir)
        
        # Specialized caches for different data types
        self._spx_price_cache = {}
        self._options_chain_cache = {} 
        self._iron_condor_cache = {}
        self._liquid_options_cache = {}
        
    def get_cached_spx_price(self, date: Union[str, datetime], 
                           timestamp: Union[str, datetime, time]) -> Optional[float]:
        """Get SPX price with intelligent caching."""
        cache_key = f"spx_price_{date}_{timestamp}"
        
        # Check cache first
        cached_price = self.cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        # Load from query engine
        price = self.query_engine.get_fastest_spx_price(date, timestamp)
        
        if price is not None:
            self.cache.put(cache_key, price, "spx_price")
        
        return price
    
    def get_cached_options_chain(self, date: Union[str, datetime],
                               timestamp: Union[str, datetime, time],
                               center_strike: Optional[float] = None,
                               strike_range: float = 200) -> pd.DataFrame:
        """Get options chain with caching."""
        cache_key = f"options_chain_{date}_{timestamp}_{center_strike}_{strike_range}"
        
        cached_chain = self.cache.get(cache_key)
        if cached_chain is not None:
            return cached_chain
        
        # Load from query engine
        chain = self.query_engine.loader.get_options_chain_at_time(
            date, timestamp, center_strike, strike_range
        )
        
        if not chain.empty:
            self.cache.put(cache_key, chain, "options_chain")
        
        return chain
    
    def get_cached_liquid_options(self, date: Union[str, datetime],
                                timestamp: Union[str, datetime, time],
                                min_bid: float = 0.05,
                                max_spread_pct: float = 30.0) -> pd.DataFrame:
        """Get liquid options with caching."""
        cache_key = f"liquid_options_{date}_{timestamp}_{min_bid}_{max_spread_pct}"
        
        cached_options = self.cache.get(cache_key)
        if cached_options is not None:
            return cached_options
        
        # Load from query engine
        liquid_options = self.query_engine.find_liquid_options(
            date, timestamp, min_bid, max_spread_pct
        )
        
        if not liquid_options.empty:
            self.cache.put(cache_key, liquid_options, "liquid_options")
        
        return liquid_options
    
    def get_cached_iron_condor_setups(self, date: Union[str, datetime],
                                    timestamp: Union[str, datetime, time],
                                    put_distances: List[int] = None,
                                    call_distances: List[int] = None) -> List[IronCondorSetup]:
        """Get Iron Condor setups with caching."""
        # Create cache key from parameters
        put_dist_str = "_".join(map(str, put_distances or [25, 50, 75]))
        call_dist_str = "_".join(map(str, call_distances or [25, 50, 75]))
        cache_key = f"ic_setups_{date}_{timestamp}_{put_dist_str}_{call_dist_str}"
        
        cached_setups = self.cache.get(cache_key)
        if cached_setups is not None:
            return cached_setups
        
        # This would require the iron_condor_loader to be integrated
        # For now, return empty list
        setups = []
        
        if setups:
            self.cache.put(cache_key, setups, "iron_condor_setups")
        
        return setups
    
    def preload_for_date_range(self, start_date: Union[str, datetime],
                             end_date: Union[str, datetime],
                             times: List[str] = None) -> int:
        """
        Pre-load commonly accessed data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            times: List of times to pre-load (defaults to key trading times)
            
        Returns:
            Number of items pre-loaded
        """
        times = times or ["09:30:00", "10:00:00", "11:00:00", "12:00:00", 
                         "13:00:00", "14:00:00", "15:00:00", "15:30:00"]
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        preload_count = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                for time_str in times:
                    # Pre-load SPX prices
                    price = self.get_cached_spx_price(current_date, time_str)
                    if price is not None:
                        preload_count += 1
                    
                    # Pre-load liquid options for key times
                    if time_str in ["10:00:00", "14:00:00"]:
                        liquid_opts = self.get_cached_liquid_options(current_date, time_str)
                        if not liquid_opts.empty:
                            preload_count += 1
            
            current_date += timedelta(days=1)
        
        logger.info(f"Pre-loaded {preload_count} items for date range")
        return preload_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.cache.get_stats()
        
        return {
            'hit_rate': f"{stats.hit_rate:.2%}",
            'total_requests': stats.hits + stats.misses,
            'memory_usage_mb': f"{stats.memory_usage_mb:.1f} MB",
            'evictions': stats.evictions,
            'cache_items': len(self.cache._memory_cache)
        }
    
    def optimize_for_strategy(self, strategy_type: str):
        """Optimize cache for specific strategy type."""
        if strategy_type == "iron_condor":
            # Pre-load common Iron Condor data patterns
            logger.info("Optimizing cache for Iron Condor strategies")
            # Implementation would depend on known patterns
        
        elif strategy_type == "0dte":
            # Optimize for 0DTE patterns
            logger.info("Optimizing cache for 0DTE strategies")
            # Focus on end-of-day data patterns
        
        elif strategy_type == "scalping":
            # Optimize for high-frequency patterns
            logger.info("Optimizing cache for scalping strategies") 
            # Focus on minute-by-minute data
    
    def clear_cache(self, clear_persistent: bool = False):
        """Clear all caches."""
        self.cache.clear(clear_persistent)
        self._spx_price_cache.clear()
        self._options_chain_cache.clear()
        self._iron_condor_cache.clear()
        self._liquid_options_cache.clear()


# Decorator for automatic caching
def cached_backtest_data(cache_manager: BacktestingCacheManager, 
                        cache_type: str = "general", ttl_hours: int = 24):
    """Decorator to automatically cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache_manager.cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.cache.put(cache_key, result, cache_type, ttl_hours)
            
            return result
        
        return wrapper
    return decorator