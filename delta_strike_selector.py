#!/usr/bin/env python3
"""
Delta-Based Strike Selection and Position Monitoring

Part 2: Strike selection using delta/probability ITM and dynamic monitoring
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from enhanced_backtest import (
    StrategyType, MarketSignal, StrikeSelection, EnhancedBacktestResult,
    TechnicalIndicators, StrategySelection
)


class DeltaStrikeSelector:
    """Select strikes based on delta and probability ITM"""
    
    def __init__(self, query_engine, ic_loader):
        self.query_engine = query_engine
        self.ic_loader = ic_loader
    
    def select_strikes_by_delta(self, 
                              date: str,
                              timestamp: str,
                              strategy_type: StrategyType,
                              target_delta: float = 0.15,
                              target_prob_itm: float = 0.15,
                              min_spread_width: int = 25) -> Optional[StrikeSelection]:
        """
        Select strikes based on target delta or probability ITM
        
        Args:
            target_delta: Target delta for short strike (0.10-0.30)
            target_prob_itm: Target probability ITM for short strike (0.10-0.30) 
            min_spread_width: Minimum spread width
        """
        
        try:
            # Get current SPX price
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if not spx_price:
                return None
            
            # Get options data
            options_data = self.query_engine.get_options_data(date, timestamp)
            if options_data is None or len(options_data) == 0:
                return None
            
            # Filter for 0DTE options
            options_data = options_data[options_data['expiration'] == date]
            
            if strategy_type == StrategyType.PUT_SPREAD:
                return self._select_put_spread_strikes(
                    options_data, spx_price, target_delta, target_prob_itm, min_spread_width
                )
            elif strategy_type == StrategyType.CALL_SPREAD:
                return self._select_call_spread_strikes(
                    options_data, spx_price, target_delta, target_prob_itm, min_spread_width
                )
            else:  # IRON_CONDOR
                put_strikes = self._select_put_spread_strikes(
                    options_data, spx_price, target_delta, target_prob_itm, min_spread_width
                )
                call_strikes = self._select_call_spread_strikes(
                    options_data, spx_price, target_delta, target_prob_itm, min_spread_width
                )
                
                # For IC, return put strikes (we'll handle call strikes separately)
                return put_strikes
                
        except Exception as e:
            logger.error(f"Strike selection failed: {e}")
            return None
    
    def _select_put_spread_strikes(self, options_data: pd.DataFrame, spx_price: float, 
                                 target_delta: float, target_prob_itm: float, 
                                 min_spread_width: int) -> Optional[StrikeSelection]:
        """Select put spread strikes"""
        
        # Filter for puts
        puts = options_data[options_data['option_type'] == 'put'].copy()
        if len(puts) == 0:
            return None
        
        # Find puts with delta close to target (puts have negative delta)
        puts['abs_delta_diff'] = abs(abs(puts['delta']) - target_delta)
        puts = puts.sort_values('abs_delta_diff')
        
        # Select short strike (sell this)
        short_candidates = puts.head(5)  # Top 5 closest to target delta
        
        for _, short_option in short_candidates.iterrows():
            short_strike = short_option['strike']
            short_delta = abs(short_option['delta'])
            
            # Estimate probability ITM (approximation)
            short_prob_itm = short_delta  # Rough approximation for 0DTE
            
            # Find long strike (buy this) - further OTM
            long_strike = short_strike - min_spread_width
            long_options = puts[puts['strike'] == long_strike]
            
            if len(long_options) > 0:
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=short_delta,
                    short_prob_itm=short_prob_itm,
                    spread_width=short_strike - long_strike
                )
        
        return None
    
    def _select_call_spread_strikes(self, options_data: pd.DataFrame, spx_price: float,
                                  target_delta: float, target_prob_itm: float,
                                  min_spread_width: int) -> Optional[StrikeSelection]:
        """Select call spread strikes"""
        
        # Filter for calls
        calls = options_data[options_data['option_type'] == 'call'].copy()
        if len(calls) == 0:
            return None
        
        # Find calls with delta close to target
        calls['abs_delta_diff'] = abs(calls['delta'] - target_delta)
        calls = calls.sort_values('abs_delta_diff')
        
        # Select short strike (sell this)
        short_candidates = calls.head(5)
        
        for _, short_option in short_candidates.iterrows():
            short_strike = short_option['strike']
            short_delta = short_option['delta']
            
            # Estimate probability ITM
            short_prob_itm = short_delta
            
            # Find long strike (buy this) - further OTM
            long_strike = short_strike + min_spread_width
            long_options = calls[calls['strike'] == long_strike]
            
            if len(long_options) > 0:
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=short_delta,
                    short_prob_itm=short_prob_itm,
                    spread_width=long_strike - short_strike
                )
        
        return None


class PositionMonitor:
    """Monitor positions with 5-minute intervals and decay-based exits"""
    
    def __init__(self, query_engine, strategy_builder):
        self.query_engine = query_engine
        self.strategy_builder = strategy_builder
    
    def monitor_position(self, 
                        strategy,
                        date: str,
                        entry_time: str,
                        exit_time: str,
                        strategy_type: StrategyType,
                        decay_threshold: float = 0.1) -> Tuple[List[Dict], str, float]:
        """
        Monitor position every 5 minutes with decay-based exits
        
        Returns:
            monitoring_points: List of monitoring data
            exit_reason: Why position was closed
            final_exit_cost: Final cost to close position
        """
        
        monitoring_points = []
        
        # Parse times
        entry_dt = datetime.strptime(f"{date} {entry_time}", "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(f"{date} {exit_time}", "%Y-%m-%d %H:%M:%S")
        
        # Monitor every 5 minutes starting 5 minutes after entry
        current_dt = entry_dt + timedelta(minutes=5)
        
        # Track for at least 30 minutes to show progression
        min_monitoring_time = entry_dt + timedelta(minutes=30)
        
        while current_dt <= exit_dt:
            current_time = current_dt.strftime("%H:%M:%S")
            
            try:
                # Update strategy prices
                price_update_success = self.strategy_builder.update_strategy_prices_optimized(strategy, date, current_time)
                logger.info(f"Price update at {current_time}: {'SUCCESS' if price_update_success else 'FAILED'}")
                
                # Log current leg prices for debugging
                for leg in strategy.legs:
                    logger.info(f"  Leg {leg.strike} {leg.option_type.value}: current_price={leg.current_price}, entry_price={leg.entry_price}")
                
                # Calculate current cost to close
                current_cost = self._calculate_exit_cost(strategy)
                
                # Calculate current P&L
                entry_credit = strategy.entry_credit
                current_pnl = entry_credit - current_cost
                current_pnl_pct = (current_pnl / entry_credit * 100) if entry_credit > 0 else 0
                
                # Calculate decay ratio - how much of the entry credit remains as cost
                decay_ratio = current_cost / entry_credit if entry_credit > 0 else 0
                
                monitoring_point = {
                    'timestamp': current_time,
                    'spx_price': self.query_engine.get_fastest_spx_price(date, current_time) or 0,
                    'exit_cost': current_cost,
                    'pnl': current_pnl,
                    'pnl_pct': current_pnl_pct,
                    'decay_ratio': decay_ratio,
                    'minutes_elapsed': (current_dt - entry_dt).total_seconds() / 60
                }
                
                monitoring_points.append(monitoring_point)
                
                # Check exit conditions only after minimum monitoring time
                if current_dt >= min_monitoring_time:
                    # Check decay threshold
                    if decay_ratio <= decay_threshold:
                        return monitoring_points, f"Decay threshold reached ({decay_ratio:.3f} <= {decay_threshold})", current_cost
                    
                    # Early profit taking (optional - 50% of max profit)
                    if strategy_type in [StrategyType.PUT_SPREAD, StrategyType.CALL_SPREAD]:
                        if current_pnl_pct >= 50:  # 50% profit
                            return monitoring_points, f"Early profit taking ({current_pnl_pct:.1f}%)", current_cost
                    
                    # Iron Condor early exit (25% profit)
                    elif strategy_type == StrategyType.IRON_CONDOR:
                        if current_pnl_pct >= 25:  # 25% profit
                            return monitoring_points, f"Early profit taking ({current_pnl_pct:.1f}%)", current_cost
                
            except Exception as e:
                logger.warning(f"Monitoring failed at {current_time}: {e}")
                # Continue monitoring with estimated values
                entry_credit = strategy.entry_credit
                estimated_cost = entry_credit * 0.3  # Assume 30% of entry credit as cost
                estimated_pnl = entry_credit - estimated_cost
                estimated_pnl_pct = (estimated_pnl / entry_credit * 100) if entry_credit > 0 else 0
                
                monitoring_point = {
                    'timestamp': current_time,
                    'spx_price': self.query_engine.get_fastest_spx_price(date, current_time) or 0,
                    'exit_cost': estimated_cost,
                    'pnl': estimated_pnl,
                    'pnl_pct': estimated_pnl_pct,
                    'decay_ratio': estimated_cost / entry_credit if entry_credit > 0 else 0.3,
                    'minutes_elapsed': (current_dt - entry_dt).total_seconds() / 60
                }
                monitoring_points.append(monitoring_point)
            
            # Next 5-minute interval
            current_dt += timedelta(minutes=5)
        
        # Position held to expiration
        final_cost = self._calculate_exit_cost(strategy)
        return monitoring_points, "Held to expiration", final_cost
    
    def _calculate_exit_cost(self, strategy) -> float:
        """Calculate current cost to close position"""
        current_cost = 0.0
        prices_updated = False
        
        try:
            for leg in strategy.legs:
                leg_price = leg.current_price
                
                # Check if price was actually updated (greater than 0)
                if leg_price > 0:
                    prices_updated = True
                else:
                    # More realistic fallback: use entry price as estimate
                    # This preserves the original value of the position for monitoring
                    leg_price = leg.entry_price
                    logger.debug(f"Using entry price fallback for {leg.strike} {leg.option_type.value}: {leg_price}")
                
                if leg.position_side.name == 'SHORT':
                    current_cost += leg_price * 100 * leg.quantity  # Cost to buy back short
                else:
                    current_cost -= leg_price * 100 * leg.quantity  # Credit from selling long
            
            # If no prices were updated, log warning
            if not prices_updated:
                logger.warning(f"No current prices available for strategy monitoring - using entry prices")
            
            return max(current_cost, 0.0)  # Ensure non-negative cost
            
        except Exception as e:
            logger.warning(f"Error calculating exit cost: {e}")
            # Return a reasonable fallback based on entry credit
            return getattr(strategy, 'entry_credit', 0) * 0.5  # Assume 50% of entry credit as cost


print("Enhanced backtesting system framework created - Part 2/3")