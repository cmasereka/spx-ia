#!/usr/bin/env python3
"""
Enhanced Multi-Strategy Backtesting Engine

Part 3: Main backtesting engine with all enhancements integrated
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import argparse

from enhanced_backtest import (
    StrategyType, MarketSignal, TechnicalIndicators, StrategySelection,
    EnhancedBacktestResult, TechnicalAnalyzer, StrategySelector,
    EnhancedMultiStrategyBacktester
)
from delta_strike_selector import DeltaStrikeSelector, PositionMonitor, StrikeSelection
from query_engine_adapter import EnhancedQueryEngineAdapter


class EnhancedBacktestingEngine(EnhancedMultiStrategyBacktester):
    """Complete enhanced backtesting engine"""
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        super().__init__(data_path)
        # Wrap query engine with enhanced adapter
        self.enhanced_query_engine = EnhancedQueryEngineAdapter(self.query_engine)
        self.delta_selector = DeltaStrikeSelector(self.enhanced_query_engine, self.ic_loader)
        self.position_monitor = PositionMonitor(self.enhanced_query_engine, self.strategy_builder)
    
    def enhanced_backtest_single_day(self,
                                   date: str,
                                   entry_time: str = "10:00:00",
                                   exit_time: str = "15:45:00",
                                   target_delta: float = 0.15,
                                   target_prob_itm: float = 0.15,
                                   min_spread_width: int = 25,
                                   decay_threshold: float = 0.1,
                                   quantity: int = 1) -> EnhancedBacktestResult:
        """
        Run enhanced backtest with technical analysis and dynamic monitoring
        """
        
        logger.info(f"Enhanced backtesting {date} - Target Delta: {target_delta}, Decay: {decay_threshold}")
        
        # Validate date
        if date not in self.available_dates:
            return self._create_failed_result(date, entry_time, exit_time, "No data available")
        
        try:
            # Step 1: Get SPX price and history
            entry_spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, entry_time)
            if not entry_spx_price:
                return self._create_failed_result(date, entry_time, exit_time, "No SPX price at entry")
            
            spx_history = self.get_spx_price_history(date, entry_time, lookback_minutes=60)
            
            # Step 2: Technical analysis
            technical_indicators = self.technical_analyzer.analyze_market_conditions(spx_history)
            strategy_selection = self.strategy_selector.select_strategy(technical_indicators)
            
            logger.info(f"Strategy selected: {strategy_selection.strategy_type.value} - {strategy_selection.reason}")
            
            # Step 3: Delta-based strike selection
            strike_selection = self.delta_selector.select_strikes_by_delta(
                date=date,
                timestamp=entry_time,
                strategy_type=strategy_selection.strategy_type,
                target_delta=target_delta,
                target_prob_itm=target_prob_itm,
                min_spread_width=min_spread_width
            )
            
            if not strike_selection:
                return self._create_failed_result(date, entry_time, exit_time, 
                                               f"No viable {strategy_selection.strategy_type.value} strikes found")
            
            # Step 4: Build strategy
            if strategy_selection.strategy_type == StrategyType.IRON_CONDOR:
                strategy = self._build_iron_condor_strategy(date, entry_time, strike_selection, quantity)
            elif strategy_selection.strategy_type == StrategyType.PUT_SPREAD:
                strategy = self._build_put_spread_strategy(date, entry_time, strike_selection, quantity)
            else:  # CALL_SPREAD
                strategy = self._build_call_spread_strategy(date, entry_time, strike_selection, quantity)
            
            if not strategy or not strategy.legs:
                return self._create_failed_result(date, entry_time, exit_time, 
                                               f"Could not build {strategy_selection.strategy_type.value} strategy")
            
            # Step 5: Position monitoring with 5-minute intervals
            monitoring_points, exit_reason, final_exit_cost = self.position_monitor.monitor_position(
                strategy=strategy,
                date=date,
                entry_time=entry_time,
                exit_time=exit_time,
                strategy_type=strategy_selection.strategy_type,
                decay_threshold=decay_threshold
            )
            
            # Step 6: Final P&L calculation
            entry_credit = strategy.entry_credit
            pnl = entry_credit - final_exit_cost
            pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
            
            # Get final SPX price
            exit_spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, exit_time) or entry_spx_price
            
            # Extract final exit time from monitoring
            actual_exit_time = exit_time
            if monitoring_points and exit_reason != "Held to expiration":
                actual_exit_time = monitoring_points[-1]['timestamp']
            
            return EnhancedBacktestResult(
                date=date,
                strategy_type=strategy_selection.strategy_type,
                market_signal=strategy_selection.market_signal,
                entry_time=entry_time,
                exit_time=actual_exit_time,
                exit_reason=exit_reason,
                entry_spx_price=entry_spx_price,
                exit_spx_price=exit_spx_price,
                technical_indicators=technical_indicators,
                strike_selection=strike_selection,
                entry_credit=entry_credit,
                exit_cost=final_exit_cost,
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_profit=strategy.max_profit if hasattr(strategy, 'max_profit') else entry_credit,
                max_loss=strategy.max_loss if hasattr(strategy, 'max_loss') else -final_exit_cost,
                monitoring_points=monitoring_points,
                success=True,
                confidence=strategy_selection.confidence,
                notes=f"{strategy_selection.reason} | Delta: {strike_selection.short_delta:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Enhanced backtest failed for {date}: {e}")
            return self._create_failed_result(date, entry_time, exit_time, f"Error: {str(e)}")
    
    def _build_iron_condor_strategy(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int):
        """Build Iron Condor using existing infrastructure"""
        # Use existing Iron Condor builder with put strikes from delta selection
        put_distance = abs(strike_selection.short_strike - self.enhanced_query_engine.get_fastest_spx_price(date, timestamp))
        call_distance = put_distance  # Symmetric
        spread_width = int(strike_selection.spread_width)
        
        return self.strategy_builder.build_iron_condor_optimized(
            date=date,
            timestamp=timestamp,
            put_distance=put_distance,
            call_distance=call_distance,
            spread_width=spread_width,
            quantity=quantity,
            use_liquid_options=True
        )
    
    def _build_put_spread_strategy(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int):
        """Build Put Spread strategy"""
        # Build put spread using existing infrastructure
        # This is a simplified implementation - you may want to create a dedicated put spread builder
        return self._build_single_spread(date, timestamp, strike_selection, quantity, 'put')
    
    def _build_call_spread_strategy(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int):
        """Build Call Spread strategy"""
        # Build call spread using existing infrastructure  
        return self._build_single_spread(date, timestamp, strike_selection, quantity, 'call')
    
    def _build_single_spread(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int, option_type: str):
        """Build single spread (put or call)"""
        try:
            # Get options data
            options_data = self.enhanced_query_engine.get_options_data(date, timestamp)
            if options_data is None:
                return None
            
            # Filter for the specific option type and strikes
            options = options_data[
                (options_data['option_type'] == option_type) &
                (options_data['expiration'] == date) &
                (options_data['strike'].isin([strike_selection.short_strike, strike_selection.long_strike]))
            ]
            
            if len(options) < 2:
                return None
            
            # Create a simple strategy object (you may need to adapt this based on your strategy infrastructure)
            from src.strategies.options_strategies import OptionsStrategy, OptionsLeg, PositionSide
            
            short_option = options[options['strike'] == strike_selection.short_strike].iloc[0]
            long_option = options[options['strike'] == strike_selection.long_strike].iloc[0]
            
            # Create legs
            short_leg = OptionsLeg(
                option_type=option_type,
                strike=strike_selection.short_strike,
                expiration_date=date,
                position_side=PositionSide.SHORT,
                quantity=quantity,
                entry_price=(short_option['bid'] + short_option['ask']) / 2,
                current_price=(short_option['bid'] + short_option['ask']) / 2
            )
            
            long_leg = OptionsLeg(
                option_type=option_type,
                strike=strike_selection.long_strike,
                expiration_date=date,
                position_side=PositionSide.LONG,
                quantity=quantity,
                entry_price=(long_option['bid'] + long_option['ask']) / 2,
                current_price=(long_option['bid'] + long_option['ask']) / 2
            )
            
            # Create strategy
            strategy = OptionsStrategy(
                strategy_type=f"{option_type.upper()}_SPREAD",
                legs=[short_leg, long_leg],
                entry_date=date,
                entry_time=timestamp
            )
            
            # Calculate entry credit
            entry_credit = (short_leg.entry_price - long_leg.entry_price) * quantity * 100
            strategy.entry_credit = max(0, entry_credit)  # Ensure positive credit
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to build {option_type} spread: {e}")
            return None
    
    def _create_failed_result(self, date: str, entry_time: str, exit_time: str, reason: str) -> EnhancedBacktestResult:
        """Create a failed result object"""
        return EnhancedBacktestResult(
            date=date,
            strategy_type=StrategyType.IRON_CONDOR,
            market_signal=MarketSignal.NEUTRAL,
            entry_time=entry_time,
            exit_time=exit_time,
            exit_reason=reason,
            entry_spx_price=0,
            exit_spx_price=0,
            technical_indicators=TechnicalIndicators(0, 0, 0, 0, 0, 0, 0, 0.5),
            strike_selection=StrikeSelection(0, 0, 0, 0, 0),
            entry_credit=0,
            exit_cost=0,
            pnl=0,
            pnl_pct=0,
            max_profit=0,
            max_loss=0,
            monitoring_points=[],
            success=False,
            confidence=0,
            notes=reason
        )
    
    def print_enhanced_results(self, results: List[EnhancedBacktestResult], show_monitoring: bool = False):
        """Print enhanced results with technical analysis details"""
        
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*140}")
        print(f"ENHANCED MULTI-STRATEGY BACKTEST RESULTS - {len(results)} Days")
        print(f"{'='*140}")
        
        # Summary table
        print(f"{'Date':<12} {'Strategy':<12} {'Signal':<8} {'Delta':<6} {'RSI':<5} {'P&L':<10} {'%':<7} {'Exit':<15} {'Status'}")
        print(f"{'-'*140}")
        
        for result in results:
            if result.success:
                status = "✓ WIN" if result.pnl > 0 else "✗ LOSS"
                delta_str = f"{result.strike_selection.short_delta:.3f}" if result.success else "N/A"
                rsi_str = f"{result.technical_indicators.rsi:.0f}" if result.success else "N/A"
                strategy_short = result.strategy_type.value.replace(" ", "")[:10]
                signal_short = result.market_signal.value[:6]
                
                print(f"{result.date:<12} {strategy_short:<12} {signal_short:<8} {delta_str:<6} {rsi_str:<5} "
                      f"${result.pnl:<9.2f} {result.pnl_pct:<6.1f}% {result.exit_reason[:13]:<15} {status}")
            else:
                print(f"{result.date:<12} {'SKIP':<12} {'N/A':<8} {'N/A':<6} {'N/A':<5} "
                      f"{'$0.00':<10} {'0.0%':<7} {result.exit_reason[:13]:<15} {'SKIP'}")
        
        # Enhanced statistics
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            print(f"\n{'-'*140}")
            print(f"ENHANCED STATISTICS")
            print(f"{'-'*140}")
            
            # Strategy breakdown
            strategy_stats = {}
            for result in successful_results:
                strategy = result.strategy_type.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                strategy_stats[strategy]['count'] += 1
                if result.pnl > 0:
                    strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['total_pnl'] += result.pnl
            
            print(f"Strategy Performance:")
            for strategy, stats in strategy_stats.items():
                win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {strategy}: {stats['count']} trades, {win_rate:.1f}% win rate, ${avg_pnl:.2f} avg P&L")
            
            # Technical indicator summary
            avg_rsi = sum(r.technical_indicators.rsi for r in successful_results) / len(successful_results)
            avg_delta = sum(r.strike_selection.short_delta for r in successful_results) / len(successful_results)
            
            print(f"\\nTechnical Summary:")
            print(f"  Average RSI: {avg_rsi:.1f}")
            print(f"  Average Short Delta: {avg_delta:.3f}")
            print(f"  Setup Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
            
            # Show monitoring details if requested
            if show_monitoring:
                print(f"\\n{'-'*140}")
                print(f"DETAILED POSITION MONITORING")
                print(f"{'-'*140}")
                
                for result in successful_results:  # Show all results, not just first 3
                    print(f"\\n📊 {result.date} - {result.strategy_type.value} Strategy:")
                    print(f"   Entry SPX: ${result.entry_spx_price:.2f} → Exit SPX: ${result.exit_spx_price:.2f}")
                    print(f"   Strike Selection: Short {result.strike_selection.short_strike:.0f} | Long {result.strike_selection.long_strike:.0f} | Delta {result.strike_selection.short_delta:.3f}")
                    print(f"   Entry Credit: ${result.entry_credit:.2f} → Exit Cost: ${result.exit_cost:.2f} → P&L: ${result.pnl:.2f}")
                    print(f"   Exit Reason: {result.exit_reason}")
                    print(f"   Monitoring Points: {len(result.monitoring_points)}")
                    
                    if result.monitoring_points:
                        print(f"   ⏱️  5-Minute Checkpoint Details:")
                        print(f"      {'Time':<8} {'SPX':<8} {'ExitCost':<10} {'P&L':<8} {'P&L%':<7} {'Decay':<6} {'ΔP&L':<8}")
                        print(f"      {'-'*58}")
                        
                        prev_pnl = result.entry_credit  # Starting P&L
                        for i, point in enumerate(result.monitoring_points):
                            pnl_change = point['pnl'] - prev_pnl if i > 0 else 0
                            status_icon = "🔴" if point['decay_ratio'] <= 0.1 else "🟡" if point['decay_ratio'] <= 0.3 else "🟢"
                            print(f"      {point['timestamp']:<8} ${point['spx_price']:<7.0f} ${point['exit_cost']:<9.2f} ${point['pnl']:<7.2f} {point['pnl_pct']:<6.1f}% {point['decay_ratio']:<5.3f} ${pnl_change:>+6.2f} {status_icon}")
                            prev_pnl = point['pnl']
                        
                        last_point = result.monitoring_points[-1]
                        print(f"      💡 Final Status: {result.exit_reason} (Decay: {last_point.get('decay_ratio', 0):.3f})")
                    
                    print()  # Empty line between days
        
        print(f"{'='*140}\\n")


def run_enhanced_backtest():
    """Run enhanced backtesting with command line interface"""
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-Strategy SPX 0DTE Backtester")
    parser.add_argument("--date", "-d", help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--target-delta", type=float, default=0.15, help="Target delta for short strikes")
    parser.add_argument("--target-prob-itm", type=float, default=0.15, help="Target probability ITM")
    parser.add_argument("--decay-threshold", type=float, default=0.1, help="Decay threshold for exits")
    parser.add_argument("--min-spread-width", type=int, default=25, help="Minimum spread width")
    parser.add_argument("--show-monitoring", action="store_true", help="Show detailed monitoring")
    
    args = parser.parse_args()
    
    # Initialize enhanced engine
    engine = EnhancedBacktestingEngine()
    
    if args.date:
        # Single day backtest
        result = engine.enhanced_backtest_single_day(
            date=args.date,
            target_delta=args.target_delta,
            target_prob_itm=args.target_prob_itm,
            decay_threshold=args.decay_threshold,
            min_spread_width=args.min_spread_width
        )
        engine.print_enhanced_results([result], show_monitoring=args.show_monitoring)
        
    elif args.start_date and args.end_date:
        # Date range backtest
        results = []
        
        # Generate date range
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        # Filter to available dates
        test_dates = []
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in engine.available_dates:
                test_dates.append(date_str)
        
        logger.info(f"Testing {len(test_dates)} available days in enhanced mode")
        
        # Run enhanced backtests
        for i, date in enumerate(test_dates, 1):
            logger.info(f"Enhanced testing {i}/{len(test_dates)}: {date}")
            result = engine.enhanced_backtest_single_day(
                date=date,
                target_delta=args.target_delta,
                target_prob_itm=args.target_prob_itm,
                decay_threshold=args.decay_threshold,
                min_spread_width=args.min_spread_width
            )
            results.append(result)
        
        engine.print_enhanced_results(results, show_monitoring=args.show_monitoring)
    
    else:
        print("Enhanced SPX 0DTE Multi-Strategy Backtester")
        print("Available commands:")
        print("  --date YYYY-MM-DD                           # Single day enhanced backtest")
        print("  --start-date YYYY-MM-DD --end-date YYYY-MM-DD  # Enhanced date range")
        print("  --target-delta 0.15                        # Target delta for strikes")
        print("  --decay-threshold 0.1                      # Exit when position decays to 10%")
        print("  --show-monitoring                           # Show 5-min monitoring details")
        print("\\nExample:")
        print("  python enhanced_multi_strategy.py --date 2026-02-09 --target-delta 0.20")
        print("  python enhanced_multi_strategy.py --start-date 2026-02-09 --end-date 2026-02-13 --show-monitoring")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    run_enhanced_backtest()


print("Enhanced Multi-Strategy Backtesting Engine Complete - Part 3/3")
print("\\n✅ All enhancements implemented:")
print("1. ✅ Multi-strategy selection (IC, Put Spreads, Call Spreads)")
print("2. ✅ Technical indicators (RSI, MACD, Bollinger Bands)")
print("3. ✅ Delta/Probability ITM based strike selection") 
print("4. ✅ Dynamic position monitoring (5-min intervals)")
print("5. ✅ Decay-based exits (0.1 threshold)")