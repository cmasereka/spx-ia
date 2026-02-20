#!/usr/bin/env python3
"""
Iron Condor optimized data loader for ultra-fast backtesting.
Pre-calculates and caches Iron Condor specific option combinations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from datetime import datetime, time
from functools import lru_cache
from dataclasses import dataclass
from loguru import logger

from ..data.query_engine import BacktestQueryEngine
from .strategy_adapter import ParquetDataAdapter


@dataclass
class IronCondorSetup:
    """Iron Condor setup parameters and metadata"""
    put_short_strike: float
    put_long_strike: float
    call_short_strike: float
    call_long_strike: float
    spx_price: float
    timestamp: datetime
    put_spread_width: float
    call_spread_width: float
    max_profit: float
    max_loss: float
    put_short_bid: float = 0.0
    put_short_ask: float = 0.0
    put_long_bid: float = 0.0
    put_long_ask: float = 0.0
    call_short_bid: float = 0.0
    call_short_ask: float = 0.0
    call_long_bid: float = 0.0
    call_long_ask: float = 0.0
    net_credit: float = 0.0
    is_valid: bool = False
    liquidity_score: float = 0.0


class IronCondorDataLoader:
    """
    Specialized data loader optimized for Iron Condor backtesting.
    Pre-calculates viable Iron Condor setups for rapid strategy evaluation.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
        self.data_adapter = ParquetDataAdapter(query_engine)
        
        # Caching for Iron Condor setups
        self._ic_setup_cache = {}
        self._strike_combinations_cache = {}
        
        # Default parameters
        self.default_put_distances = [25, 50, 75, 100]
        self.default_call_distances = [25, 50, 75, 100]
        self.default_spread_widths = [25, 50]
        
    def get_viable_iron_condor_setups(self, 
                                    date: Union[str, datetime],
                                    timestamp: Union[str, datetime, time],
                                    put_distances: Optional[List[int]] = None,
                                    call_distances: Optional[List[int]] = None,
                                    spread_widths: Optional[List[int]] = None,
                                    min_credit: float = 0.50,
                                    min_bid: float = 0.05,
                                    max_spread_pct: float = 25.0) -> List[IronCondorSetup]:
        """
        Get all viable Iron Condor setups for given parameters.
        
        Args:
            date: Trading date
            timestamp: Entry timestamp
            put_distances: List of put short distances from SPX
            call_distances: List of call short distances from SPX
            spread_widths: List of spread widths to test
            min_credit: Minimum net credit required
            min_bid: Minimum bid for liquidity
            max_spread_pct: Maximum bid/ask spread percentage
            
        Returns:
            List of viable IronCondorSetup objects
        """
        # Use defaults if not provided
        put_distances = put_distances or self.default_put_distances
        call_distances = call_distances or self.default_call_distances
        spread_widths = spread_widths or self.default_spread_widths
        
        # Get SPX price
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            logger.warning(f"No SPX price for {date} {timestamp}")
            return []
        
        # Get liquid options data
        liquid_options = self.data_adapter.get_liquid_options_for_strategy(
            date, timestamp, min_bid, max_spread_pct, spx_price, 300
        )
        
        if not liquid_options:
            logger.warning(f"No liquid options data for {date} {timestamp}")
            return []
        
        viable_setups = []
        
        # Test all combinations
        for put_distance in put_distances:
            for call_distance in call_distances:
                for spread_width in spread_widths:
                    
                    setup = self._evaluate_iron_condor_setup(
                        liquid_options, spx_price, timestamp,
                        put_distance, call_distance, spread_width,
                        min_credit
                    )
                    
                    if setup and setup.is_valid:
                        viable_setups.append(setup)
        
        # Sort by net credit (highest first)
        viable_setups.sort(key=lambda x: x.net_credit, reverse=True)
        
        logger.info(f"Found {len(viable_setups)} viable Iron Condor setups")
        return viable_setups
    
    def _evaluate_iron_condor_setup(self, 
                                   options_data: Dict[str, Dict],
                                   spx_price: float,
                                   timestamp: Union[datetime, str, time],
                                   put_distance: int,
                                   call_distance: int,
                                   spread_width: int,
                                   min_credit: float) -> Optional[IronCondorSetup]:
        """
        Evaluate a specific Iron Condor setup configuration.
        """
        # Calculate target strikes
        put_short_target = spx_price - put_distance
        put_long_target = put_short_target - spread_width
        call_short_target = spx_price + call_distance
        call_long_target = call_short_target + spread_width
        
        # Find closest available strikes
        put_short_strike = self._find_closest_strike(options_data, put_short_target, 'put')
        put_long_strike = self._find_closest_strike(options_data, put_long_target, 'put')
        call_short_strike = self._find_closest_strike(options_data, call_short_target, 'call')
        call_long_strike = self._find_closest_strike(options_data, call_long_target, 'call')
        
        # Check if all strikes are available
        required_keys = [
            f"{put_short_strike}_put",
            f"{put_long_strike}_put", 
            f"{call_short_strike}_call",
            f"{call_long_strike}_call"
        ]
        
        if not all(key in options_data for key in required_keys):
            return None
        
        # Get option prices
        put_short_data = options_data[f"{put_short_strike}_put"]
        put_long_data = options_data[f"{put_long_strike}_put"]
        call_short_data = options_data[f"{call_short_strike}_call"]
        call_long_data = options_data[f"{call_long_strike}_call"]
        
        # Calculate net credit (sell short strikes, buy long strikes)
        net_credit = (
            put_short_data['bid'] - put_long_data['ask'] +
            call_short_data['bid'] - call_long_data['ask']
        )
        
        # Check minimum credit requirement
        if net_credit < min_credit:
            return None
        
        # Calculate spread widths
        actual_put_width = put_short_strike - put_long_strike
        actual_call_width = call_long_strike - call_short_strike
        
        # Calculate max profit/loss
        max_profit = net_credit * 100  # Options are x100
        max_loss = max(actual_put_width, actual_call_width) * 100 - max_profit
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score([
            put_short_data, put_long_data, call_short_data, call_long_data
        ])
        
        # Convert timestamp to datetime if needed
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, time):
            timestamp = pd.to_datetime(f"{pd.to_datetime(date).date()} {timestamp}")
        
        return IronCondorSetup(
            put_short_strike=put_short_strike,
            put_long_strike=put_long_strike,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            spx_price=spx_price,
            timestamp=timestamp,
            put_spread_width=actual_put_width,
            call_spread_width=actual_call_width,
            max_profit=max_profit,
            max_loss=max_loss,
            put_short_bid=put_short_data['bid'],
            put_short_ask=put_short_data['ask'],
            put_long_bid=put_long_data['bid'],
            put_long_ask=put_long_data['ask'],
            call_short_bid=call_short_data['bid'],
            call_short_ask=call_short_data['ask'],
            call_long_bid=call_long_data['bid'],
            call_long_ask=call_long_data['ask'],
            net_credit=net_credit,
            is_valid=True,
            liquidity_score=liquidity_score
        )
    
    def _find_closest_strike(self, options_data: Dict[str, Dict], 
                           target_strike: float, option_type: str) -> Optional[float]:
        """Find the closest available strike to target."""
        available_strikes = []
        
        for key in options_data.keys():
            if key.endswith(f'_{option_type}'):
                strike = float(key.split('_')[0])
                available_strikes.append(strike)
        
        if not available_strikes:
            return None
        
        return min(available_strikes, key=lambda x: abs(x - target_strike))
    
    def _calculate_liquidity_score(self, option_data_list: List[Dict]) -> float:
        """
        Calculate liquidity score based on bid/ask spreads and bid sizes.
        Higher score = more liquid.
        """
        total_score = 0.0
        
        for data in option_data_list:
            bid = data['bid']
            ask = data['ask']
            
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            
            # Spread penalty (lower is better)
            spread_pct = (ask - bid) / bid * 100
            spread_score = max(0, 50 - spread_pct) / 50
            
            # Bid size bonus (if available)
            bid_size = data.get('bid_size', 1)
            size_score = min(bid_size / 10, 1.0)  # Normalize to max 1.0
            
            # Price level bonus (higher prices generally more liquid)
            price_score = min(bid / 10, 1.0)
            
            leg_score = (spread_score * 0.6 + size_score * 0.3 + price_score * 0.1)
            total_score += leg_score
        
        return total_score / len(option_data_list) if option_data_list else 0.0
    
    def get_best_iron_condor_setup(self,
                                 date: Union[str, datetime],
                                 timestamp: Union[str, datetime, time],
                                 put_distances: Optional[List[int]] = None,
                                 call_distances: Optional[List[int]] = None,
                                 spread_widths: Optional[List[int]] = None,
                                 min_credit: float = 0.50,
                                 optimize_for: str = 'credit') -> Optional[IronCondorSetup]:
        """
        Get the best Iron Condor setup based on optimization criteria.
        
        Args:
            date: Trading date
            timestamp: Entry timestamp
            put_distances: Put short distances to test
            call_distances: Call short distances to test  
            spread_widths: Spread widths to test
            min_credit: Minimum credit required
            optimize_for: 'credit', 'liquidity', or 'risk_reward'
            
        Returns:
            Best IronCondorSetup or None
        """
        setups = self.get_viable_iron_condor_setups(
            date, timestamp, put_distances, call_distances, 
            spread_widths, min_credit
        )
        
        if not setups:
            return None
        
        if optimize_for == 'credit':
            return max(setups, key=lambda x: x.net_credit)
        elif optimize_for == 'liquidity':
            return max(setups, key=lambda x: x.liquidity_score)
        elif optimize_for == 'risk_reward':
            # Optimize for profit/loss ratio while considering liquidity
            return max(setups, key=lambda x: (x.max_profit / x.max_loss) * x.liquidity_score)
        else:
            return setups[0]  # Default to highest credit
    
    def get_iron_condor_for_target_credit(self,
                                        date: Union[str, datetime],
                                        timestamp: Union[str, datetime, time],
                                        target_credit: float,
                                        tolerance: float = 0.10) -> Optional[IronCondorSetup]:
        """
        Find Iron Condor setup closest to target credit amount.
        
        Args:
            date: Trading date
            timestamp: Entry timestamp
            target_credit: Target net credit
            tolerance: Acceptable deviation from target
            
        Returns:
            Closest IronCondorSetup or None
        """
        setups = self.get_viable_iron_condor_setups(
            date, timestamp, min_credit=target_credit * (1 - tolerance)
        )
        
        if not setups:
            return None
        
        # Find closest to target
        closest = min(setups, key=lambda x: abs(x.net_credit - target_credit))
        
        # Check if within tolerance
        if abs(closest.net_credit - target_credit) <= target_credit * tolerance:
            return closest
        
        return None
    
    def batch_load_iron_condor_opportunities(self,
                                           date: Union[str, datetime],
                                           start_time: str = "09:30:00",
                                           end_time: str = "15:30:00",
                                           interval_minutes: int = 30) -> Dict[str, List[IronCondorSetup]]:
        """
        Batch load Iron Condor opportunities throughout a trading day.
        
        Args:
            date: Trading date
            start_time: Start time for scanning
            end_time: End time for scanning
            interval_minutes: Minutes between scans
            
        Returns:
            Dictionary mapping timestamps to Iron Condor setups
        """
        opportunities = {}
        
        # Generate time range
        start_dt = pd.to_datetime(f"{date} {start_time}")
        end_dt = pd.to_datetime(f"{date} {end_time}")
        
        current_time = start_dt
        while current_time <= end_dt:
            time_str = current_time.strftime("%H:%M:%S")
            
            setups = self.get_viable_iron_condor_setups(
                date, time_str, min_credit=0.50
            )
            
            if setups:
                opportunities[time_str] = setups[:5]  # Keep top 5
                logger.info(f"Found {len(setups)} opportunities at {time_str}")
            
            current_time += pd.Timedelta(minutes=interval_minutes)
        
        return opportunities
    
    def clear_cache(self):
        """Clear all cached data."""
        self._ic_setup_cache.clear()
        self._strike_combinations_cache.clear()
        self.data_adapter.clear_cache()
        logger.info("Cleared Iron Condor data loader cache")


# Convenience functions
def create_iron_condor_loader(data_path: str = "data/processed/parquet_1m") -> IronCondorDataLoader:
    """Create Iron Condor loader with query engine."""
    from ..data.query_engine import create_fast_query_engine
    
    query_engine = create_fast_query_engine(data_path)
    return IronCondorDataLoader(query_engine)


def find_best_iron_condor_entry(date: str, 
                               time_range: Tuple[str, str] = ("10:00:00", "14:00:00"),
                               data_path: str = "data/processed/parquet_1m") -> Optional[IronCondorSetup]:
    """
    Find the best Iron Condor entry opportunity in a time range.
    
    Args:
        date: Trading date (YYYY-MM-DD)
        time_range: (start_time, end_time) tuple
        data_path: Path to parquet data
        
    Returns:
        Best IronCondorSetup or None
    """
    loader = create_iron_condor_loader(data_path)
    
    start_time, end_time = time_range
    opportunities = loader.batch_load_iron_condor_opportunities(
        date, start_time, end_time, interval_minutes=15
    )
    
    # Find overall best across all times
    all_setups = []
    for time_str, setups in opportunities.items():
        all_setups.extend(setups)
    
    if not all_setups:
        return None
    
    # Return highest credit setup
    return max(all_setups, key=lambda x: x.net_credit)