#!/usr/bin/env python3
"""
Interactive SPX 0DTE Backtesting Pipeline

Enhanced version with parameter testing, optimization, and interactive features.
"""
import sys
sys.path.append('.')

import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import argparse
from loguru import logger

from simple_backtest import SimpleBacktester, SimpleBacktestResult


class InteractiveBacktester(SimpleBacktester):
    """
    Enhanced backtester with interactive features and parameter optimization.
    """
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        super().__init__(data_path)
    
    def test_parameters(self, 
                       date: str,
                       put_distances: List[int] = [25, 50, 75],
                       call_distances: List[int] = [25, 50, 75],
                       spread_widths: List[int] = [25, 50],
                       entry_times: List[str] = ["10:00:00"],
                       min_credit: float = 0.50) -> List[SimpleBacktestResult]:
        """
        Test multiple parameter combinations for a single day.
        
        Args:
            date: Date to test
            put_distances: List of put distances to test
            call_distances: List of call distances to test
            spread_widths: List of spread widths to test
            entry_times: List of entry times to test
            min_credit: Minimum credit threshold
            
        Returns:
            List of results for all parameter combinations
        """
        logger.info(f"Testing parameter combinations for {date}")
        
        results = []
        total_combinations = len(put_distances) * len(call_distances) * len(spread_widths) * len(entry_times)
        
        count = 0
        for entry_time in entry_times:
            for put_dist in put_distances:
                for call_dist in call_distances:
                    for width in spread_widths:
                        count += 1
                        logger.info(f"Testing {count}/{total_combinations}: P{put_dist}/C{call_dist}/W{width} @ {entry_time}")
                        
                        result = self.backtest_single_day(
                            date=date,
                            entry_time=entry_time,
                            put_distance=put_dist,
                            call_distance=call_dist,
                            spread_width=width,
                            min_credit=min_credit
                        )
                        
                        # Add parameter info to notes
                        result.notes += f" | P{put_dist}/C{call_dist}/W{width}"
                        results.append(result)
        
        return results
    
    def find_best_parameters(self, 
                           date: str,
                           optimization_target: str = "pnl") -> Tuple[SimpleBacktestResult, Dict]:
        """
        Find the best parameters for a given day.
        
        Args:
            date: Date to optimize for
            optimization_target: 'pnl', 'pnl_pct', or 'credit'
            
        Returns:
            Tuple of (best_result, parameter_summary)
        """
        logger.info(f"Optimizing parameters for {date} (target: {optimization_target})")
        
        # Test comprehensive parameter range
        results = self.test_parameters(
            date=date,
            put_distances=[25, 50, 75, 100],
            call_distances=[25, 50, 75, 100],
            spread_widths=[25, 50],
            entry_times=["09:45:00", "10:00:00", "10:15:00"],
            min_credit=0.25  # Lower threshold for optimization
        )
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            logger.warning(f"No successful parameter combinations found for {date}")
            return None, {"error": "No successful combinations"}
        
        # Find best result based on target
        if optimization_target == "pnl":
            best_result = max(successful_results, key=lambda r: r.pnl)
        elif optimization_target == "pnl_pct":
            best_result = max(successful_results, key=lambda r: r.pnl_pct)
        elif optimization_target == "credit":
            best_result = max(successful_results, key=lambda r: r.entry_credit)
        else:
            best_result = max(successful_results, key=lambda r: r.pnl)
        
        # Create parameter summary
        summary = {
            "date": date,
            "total_tested": len(results),
            "successful_setups": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "best_result": best_result,
            "optimization_target": optimization_target,
            "parameter_range": {
                "put_distances": [25, 50, 75, 100],
                "call_distances": [25, 50, 75, 100], 
                "spread_widths": [25, 50],
                "entry_times": ["09:45:00", "10:00:00", "10:15:00"]
            }
        }
        
        return best_result, summary
    
    def parameter_sensitivity_analysis(self, date: str) -> Dict[str, Any]:
        """
        Analyze how sensitive results are to parameter changes.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary with sensitivity analysis
        """
        logger.info(f"Running parameter sensitivity analysis for {date}")
        
        # Get comprehensive results
        results = self.test_parameters(
            date=date,
            put_distances=[25, 50, 75, 100],
            call_distances=[25, 50, 75, 100],
            spread_widths=[25, 50],
            entry_times=["10:00:00"]
        )
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful results for sensitivity analysis"}
        
        # Analyze by parameter
        analysis = {
            "put_distance": {},
            "call_distance": {},
            "spread_width": {}
        }
        
        # Group by put distance
        for put_dist in [25, 50, 75, 100]:
            put_results = [r for r in successful_results if f"P{put_dist}/" in r.notes]
            if put_results:
                analysis["put_distance"][put_dist] = {
                    "count": len(put_results),
                    "avg_pnl": sum(r.pnl for r in put_results) / len(put_results),
                    "avg_credit": sum(r.entry_credit for r in put_results) / len(put_results),
                    "best_pnl": max(r.pnl for r in put_results)
                }
        
        # Group by call distance
        for call_dist in [25, 50, 75, 100]:
            call_results = [r for r in successful_results if f"C{call_dist}/" in r.notes]
            if call_results:
                analysis["call_distance"][call_dist] = {
                    "count": len(call_results),
                    "avg_pnl": sum(r.pnl for r in call_results) / len(call_results),
                    "avg_credit": sum(r.entry_credit for r in call_results) / len(call_results),
                    "best_pnl": max(r.pnl for r in call_results)
                }
        
        # Group by spread width
        for width in [25, 50]:
            width_results = [r for r in successful_results if f"W{width}" in r.notes]
            if width_results:
                analysis["spread_width"][width] = {
                    "count": len(width_results),
                    "avg_pnl": sum(r.pnl for r in width_results) / len(width_results),
                    "avg_credit": sum(r.entry_credit for r in width_results) / len(width_results),
                    "best_pnl": max(r.pnl for r in width_results)
                }
        
        return analysis
    
    def print_optimization_results(self, best_result: SimpleBacktestResult, summary: Dict):
        """Print formatted optimization results."""
        print(f"\n{'='*80}")
        print(f"PARAMETER OPTIMIZATION RESULTS - {summary['date']}")
        print(f"{'='*80}")
        
        print(f"Tested Combinations: {summary['total_tested']}")
        print(f"Successful Setups:   {summary['successful_setups']} ({summary['success_rate']:.1f}%)")
        print(f"Optimization Target: {summary['optimization_target']}")
        
        if best_result:
            print(f"\nBEST RESULT:")
            print(f"Entry Time:    {best_result.entry_time}")
            print(f"Parameters:    {best_result.notes.split('|')[-1].strip()}")
            print(f"Entry Credit:  ${best_result.entry_credit:.2f}")
            print(f"P&L:           ${best_result.pnl:.2f} ({best_result.pnl_pct:.1f}%)")
            print(f"SPX Movement:  {best_result.entry_spx_price:.0f} → {best_result.exit_spx_price:.0f}")
        
        print(f"{'='*80}")
    
    def print_sensitivity_analysis(self, analysis: Dict[str, Any]):
        """Print formatted sensitivity analysis."""
        print(f"\n{'='*80}")
        print(f"PARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        for param_name, param_data in analysis.items():
            if param_name == "error":
                print(f"Error: {param_data}")
                continue
                
            print(f"\n{param_name.upper().replace('_', ' ')}:")
            print(f"{'Value':<8} {'Count':<6} {'Avg P&L':<10} {'Avg Credit':<12} {'Best P&L':<10}")
            print(f"{'-'*50}")
            
            for value, stats in param_data.items():
                print(f"{value:<8} {stats['count']:<6} ${stats['avg_pnl']:<9.2f} "
                      f"${stats['avg_credit']:<11.2f} ${stats['best_pnl']:<9.2f}")
        
        print(f"{'='*80}")


def run_interactive_mode():
    """Run the backtester in interactive mode."""
    parser = argparse.ArgumentParser(description="Interactive SPX 0DTE Backtester")
    parser.add_argument("--date", "-d", help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--optimize", "-o", action="store_true", help="Run parameter optimization")
    parser.add_argument("--sensitivity", "-s", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--detailed", action="store_true", help="Show detailed trade information")
    parser.add_argument("--put-distance", type=int, default=50, help="Put distance from SPX")
    parser.add_argument("--call-distance", type=int, default=50, help="Call distance from SPX")
    parser.add_argument("--spread-width", type=int, default=25, help="Spread width")
    parser.add_argument("--entry-time", default="10:00:00", help="Entry time (HH:MM:SS)")
    parser.add_argument("--min-credit", type=float, default=0.50, help="Minimum credit required")
    
    args = parser.parse_args()
    
    backtester = InteractiveBacktester()
    
    # Single date with optimization
    if args.date and args.optimize:
        best_result, summary = backtester.find_best_parameters(args.date)
        backtester.print_optimization_results(best_result, summary)
        
    # Single date with sensitivity analysis
    elif args.date and args.sensitivity:
        analysis = backtester.parameter_sensitivity_analysis(args.date)
        backtester.print_sensitivity_analysis(analysis)
        
    # Single date backtest
    elif args.date:
        result = backtester.backtest_single_day(
            date=args.date,
            entry_time=args.entry_time,
            put_distance=args.put_distance,
            call_distance=args.call_distance,
            spread_width=args.spread_width,
            min_credit=args.min_credit
        )
        backtester.print_results([result], detailed_trades=args.detailed)
        
    # Date range backtest
    elif args.start_date and args.end_date:
        results = backtester.backtest_date_range(
            start_date=args.start_date,
            end_date=args.end_date,
            entry_time=args.entry_time,
            put_distance=args.put_distance,
            call_distance=args.call_distance,
            spread_width=args.spread_width,
            min_credit=args.min_credit
        )
        backtester.print_results(results, detailed_trades=args.detailed)
        
    else:
        print("SPX 0DTE Interactive Backtester")
        print("Available commands:")
        print("  --date YYYY-MM-DD                    # Single day backtest")
        print("  --date YYYY-MM-DD --detailed         # Single day with detailed trades")
        print("  --date YYYY-MM-DD --optimize         # Find best parameters")
        print("  --date YYYY-MM-DD --sensitivity      # Parameter sensitivity")
        print("  --start-date YYYY-MM-DD --end-date YYYY-MM-DD  # Date range")
        print("  --start-date YYYY-MM-DD --end-date YYYY-MM-DD --detailed  # Detailed range")
        print("\nExample:")
        print("  python interactive_backtest.py --date 2026-02-09 --detailed")
        print("  python interactive_backtest.py --date 2026-02-09 --optimize")
        print("  python interactive_backtest.py --start-date 2026-02-09 --end-date 2026-02-13 --detailed")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    run_interactive_mode()