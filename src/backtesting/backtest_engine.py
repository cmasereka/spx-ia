#!/usr/bin/env python3
"""
Optimized backtesting engine for SPX 0DTE options strategies.
Integrates parquet data, caching, and specialized loaders for maximum performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..data.query_engine import BacktestQueryEngine, create_fast_query_engine
from ..strategies.options_strategies import OptionsStrategy, IronCondor, VerticalSpread
from .strategy_adapter import EnhancedStrategyBuilder, ParquetDataAdapter
from .iron_condor_loader import IronCondorDataLoader, IronCondorSetup
from .caching import BacktestingCacheManager, cached_backtest_data


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    max_positions: int = 10
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.01
    
    # Strategy parameters
    strategy_type: str = "iron_condor"
    entry_times: List[str] = None
    exit_times: List[str] = None
    
    # Iron Condor specific
    put_distances: List[int] = None
    call_distances: List[int] = None
    spread_widths: List[int] = None
    min_credit: float = 0.50
    max_loss_pct: float = 0.50  # Close at 50% of max loss
    profit_target_pct: float = 0.25  # Close at 25% of max profit
    
    # Data filtering
    use_liquid_options: bool = True
    min_bid: float = 0.10
    max_spread_pct: float = 20.0
    
    # Performance
    use_multiprocessing: bool = False
    max_workers: int = None
    preload_data: bool = True
    cache_enabled: bool = True
    
    def __post_init__(self):
        if self.entry_times is None:
            self.entry_times = ["10:00:00"]
        if self.exit_times is None:
            self.exit_times = ["15:45:00"]
        if self.put_distances is None:
            self.put_distances = [50, 75]
        if self.call_distances is None:
            self.call_distances = [50, 75]
        if self.spread_widths is None:
            self.spread_widths = [25]


@dataclass
class TradeResult:
    """Single trade result"""
    strategy_name: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    max_loss: float
    max_profit: float
    days_held: int
    exit_reason: str
    underlying_entry: float
    underlying_exit: float
    
    # Strategy specific details
    strategy_details: Dict[str, Any] = None


@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    config: BacktestConfig
    trades: List[TradeResult]
    
    # Performance metrics
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Execution stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    execution_time_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    
    def calculate_metrics(self):
        """Calculate performance metrics from trades."""
        if not self.trades:
            return
        
        # Basic metrics
        self.total_trades = len(self.trades)
        pnls = [trade.pnl for trade in self.trades]
        self.total_pnl = sum(pnls)
        self.total_return_pct = (self.total_pnl / self.config.initial_capital) * 100
        
        # Win/loss analysis
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        self.avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        self.avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        cumulative_pnl = np.cumsum([0] + pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        self.max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Sharpe ratio (simplified - assumes daily returns)
        if len(pnls) > 1:
            returns = np.array(pnls)
            self.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0


class OptimizedBacktestEngine:
    """
    High-performance backtesting engine optimized for SPX 0DTE strategies.
    """
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        self.data_path = data_path
        
        # Initialize components
        logger.info("Initializing backtesting engine...")
        self.query_engine = create_fast_query_engine(data_path)
        self.strategy_builder = EnhancedStrategyBuilder(self.query_engine)
        self.iron_condor_loader = IronCondorDataLoader(self.query_engine)
        self.cache_manager = BacktestingCacheManager(self.query_engine)
        
        # State
        self.active_positions: Dict[str, OptionsStrategy] = {}
        self.closed_trades: List[TradeResult] = []
        self.current_capital = 0.0
        
        logger.info("Backtesting engine initialized")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        """
        Run a complete backtest with the given configuration.
        
        Args:
            config: Backtesting configuration
            
        Returns:
            Comprehensive backtest results
        """
        start_time = datetime.now()
        
        logger.info(f"Starting backtest: {config.start_date} to {config.end_date}")
        logger.info(f"Strategy: {config.strategy_type}, Capital: ${config.initial_capital:,.2f}")
        
        # Initialize
        self.current_capital = config.initial_capital
        self.active_positions.clear()
        self.closed_trades.clear()
        
        # Pre-load data if enabled
        if config.preload_data:
            logger.info("Pre-loading data for backtest period...")
            self.cache_manager.preload_for_date_range(config.start_date, config.end_date)
        
        # Generate trading dates
        trading_dates = self._generate_trading_dates(config.start_date, config.end_date)
        logger.info(f"Processing {len(trading_dates)} trading days")
        
        # Run backtest by date
        if config.use_multiprocessing:
            self._run_parallel_backtest(config, trading_dates)
        else:
            self._run_sequential_backtest(config, trading_dates)
        
        # Calculate results
        execution_time = (datetime.now() - start_time).total_seconds()
        
        results = BacktestResults(
            config=config,
            trades=self.closed_trades.copy(),
            execution_time_seconds=execution_time
        )
        
        # Add cache stats
        cache_stats = self.cache_manager.get_cache_stats()
        results.cache_hit_rate = float(cache_stats.get('hit_rate', '0%').rstrip('%'))
        
        # Calculate performance metrics
        results.calculate_metrics()
        
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        logger.info(f"Total P&L: ${results.total_pnl:,.2f} ({results.total_return_pct:+.2f}%)")
        logger.info(f"Win Rate: {results.win_rate:.1f}% ({results.winning_trades}/{results.total_trades})")
        
        return results
    
    def _run_sequential_backtest(self, config: BacktestConfig, trading_dates: List[datetime]):
        """Run backtest sequentially (single-threaded)."""
        for i, date in enumerate(trading_dates):
            if i % 10 == 0:
                logger.info(f"Processing day {i+1}/{len(trading_dates)}: {date.strftime('%Y-%m-%d')}")
            
            self._process_trading_day(config, date)
    
    def _run_parallel_backtest(self, config: BacktestConfig, trading_dates: List[datetime]):
        """Run backtest in parallel (multi-threaded)."""
        max_workers = config.max_workers or min(mp.cpu_count(), 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_trading_day, config, date): date 
                for date in trading_dates
            }
            
            for future in as_completed(futures):
                date = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {date}: {e}")
    
    def _process_trading_day(self, config: BacktestConfig, date: datetime):
        """Process a single trading day."""
        date_str = date.strftime('%Y-%m-%d')
        
        # Check for new entries
        for entry_time in config.entry_times:
            self._check_entry_signals(config, date_str, entry_time)
        
        # Update existing positions throughout the day
        self._update_active_positions(config, date_str)
        
        # Check for exits
        for exit_time in config.exit_times:
            self._check_exit_signals(config, date_str, exit_time)
        
        # Force close at end of day for 0DTE strategies
        if config.strategy_type == "iron_condor":
            self._force_close_0dte_positions(config, date_str)
    
    def _check_entry_signals(self, config: BacktestConfig, date: str, entry_time: str):
        """Check for new position entry signals."""
        if len(self.active_positions) >= config.max_positions:
            return
        
        if config.strategy_type == "iron_condor":
            self._enter_iron_condor_position(config, date, entry_time)
    
    def _enter_iron_condor_position(self, config: BacktestConfig, date: str, entry_time: str):
        """Enter a new Iron Condor position."""
        try:
            # Find best Iron Condor setup
            best_setup = self.iron_condor_loader.get_best_iron_condor_setup(
                date, entry_time,
                put_distances=config.put_distances,
                call_distances=config.call_distances,
                spread_widths=config.spread_widths,
                min_credit=config.min_credit,
                optimize_for='credit'
            )
            
            if not best_setup or not best_setup.is_valid:
                logger.debug(f"No viable Iron Condor setup found at {date} {entry_time}")
                return
            
            # Create strategy using enhanced builder
            iron_condor = self.strategy_builder.build_iron_condor_optimized(
                date, entry_time,
                put_distance=int(best_setup.spx_price - best_setup.put_short_strike),
                call_distance=int(best_setup.call_short_strike - best_setup.spx_price),
                spread_width=int(min(best_setup.put_spread_width, best_setup.call_spread_width)),
                quantity=1,
                use_liquid_options=config.use_liquid_options
            )
            
            if iron_condor and iron_condor.legs:
                position_id = f"IC_{date}_{entry_time}_{len(self.active_positions)}"
                self.active_positions[position_id] = iron_condor
                
                logger.debug(f"Entered Iron Condor: {position_id}, Credit: ${iron_condor.entry_credit:.2f}")
                
        except Exception as e:
            logger.error(f"Error entering Iron Condor position at {date} {entry_time}: {e}")
    
    def _update_active_positions(self, config: BacktestConfig, date: str):
        """Update prices for all active positions."""
        if not self.active_positions:
            return
        
        # Update at key times during the day
        update_times = ["10:30:00", "12:00:00", "13:30:00", "15:00:00"]
        
        for update_time in update_times:
            for position_id, strategy in list(self.active_positions.items()):
                success = self.strategy_builder.update_strategy_prices_optimized(
                    strategy, date, update_time
                )
                
                if success:
                    # Check profit/loss targets
                    self._check_profit_loss_targets(config, position_id, strategy, date, update_time)
    
    def _check_profit_loss_targets(self, config: BacktestConfig, position_id: str, 
                                 strategy: OptionsStrategy, date: str, time_str: str):
        """Check if position should be closed due to profit/loss targets."""
        if strategy.current_pnl == 0:
            return
        
        close_reason = None
        
        # Profit target (e.g., 25% of max profit)
        if strategy.max_profit > 0:
            profit_target = strategy.max_profit * config.profit_target_pct
            if strategy.current_pnl >= profit_target:
                close_reason = f"Profit target ({config.profit_target_pct:.0%})"
        
        # Loss limit (e.g., 50% of max loss)
        if strategy.max_loss > 0:
            loss_limit = -strategy.max_loss * config.max_loss_pct
            if strategy.current_pnl <= loss_limit:
                close_reason = f"Loss limit ({config.max_loss_pct:.0%})"
        
        if close_reason:
            self._close_position(position_id, strategy, date, time_str, close_reason)
    
    def _check_exit_signals(self, config: BacktestConfig, date: str, exit_time: str):
        """Check for position exit signals."""
        # For 0DTE strategies, this is mainly end-of-day exits
        for position_id, strategy in list(self.active_positions.items()):
            # Update final prices
            self.strategy_builder.update_strategy_prices_optimized(strategy, date, exit_time)
            
            # Close position
            self._close_position(position_id, strategy, date, exit_time, "End of day")
    
    def _force_close_0dte_positions(self, config: BacktestConfig, date: str):
        """Force close all 0DTE positions at expiration."""
        if not self.active_positions:
            return
        
        expiration_time = "16:00:00"
        
        for position_id, strategy in list(self.active_positions.items()):
            # Calculate expiration P&L
            spx_price = self.cache_manager.get_cached_spx_price(date, expiration_time)
            
            if spx_price and hasattr(strategy, 'get_profit_at_expiration'):
                expiration_pnl = strategy.get_profit_at_expiration(spx_price)
                strategy.current_pnl = expiration_pnl
            
            self._close_position(position_id, strategy, date, expiration_time, "Expiration")
    
    def _close_position(self, position_id: str, strategy: OptionsStrategy, 
                       date: str, time_str: str, reason: str):
        """Close a position and record the trade."""
        if position_id not in self.active_positions:
            return
        
        # Mark strategy as closed
        exit_datetime = pd.to_datetime(f"{date} {time_str}")
        strategy.close_position(exit_datetime, reason)
        
        # Get SPX prices for context
        entry_spx = getattr(strategy, 'entry_spx_price', 0.0)
        exit_spx = self.cache_manager.get_cached_spx_price(date, time_str) or 0.0
        
        # Calculate days held
        days_held = (exit_datetime - strategy.entry_date).days
        
        # Create trade result
        trade_result = TradeResult(
            strategy_name=strategy.strategy_name,
            entry_date=strategy.entry_date,
            exit_date=exit_datetime,
            entry_price=strategy.entry_credit - strategy.entry_debit,
            exit_price=strategy.current_pnl + strategy.entry_credit - strategy.entry_debit,
            pnl=strategy.current_pnl,
            pnl_pct=(strategy.current_pnl / abs(strategy.entry_credit - strategy.entry_debit)) * 100 if (strategy.entry_credit - strategy.entry_debit) != 0 else 0,
            max_loss=strategy.max_loss,
            max_profit=strategy.max_profit,
            days_held=days_held,
            exit_reason=reason,
            underlying_entry=entry_spx,
            underlying_exit=exit_spx,
            strategy_details={
                'legs': len(strategy.legs),
                'entry_credit': strategy.entry_credit,
                'entry_debit': strategy.entry_debit
            }
        )
        
        # Record trade and remove from active positions
        self.closed_trades.append(trade_result)
        del self.active_positions[position_id]
        
        logger.debug(f"Closed {position_id}: P&L ${strategy.current_pnl:.2f} ({reason})")
    
    def _generate_trading_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """Generate list of trading dates (excluding weekends)."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate business days
        business_days = pd.bdate_range(start_dt, end_dt)
        
        # Filter to only dates where we have data
        available_dates = self.query_engine.loader.available_dates
        trading_dates = [date for date in business_days if date in available_dates]
        
        return trading_dates
    
    def optimize_strategy_parameters(self, base_config: BacktestConfig,
                                   parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            base_config: Base configuration
            parameter_ranges: Dict of parameter names to lists of values to test
            
        Returns:
            Best parameters and their results
        """
        logger.info("Starting parameter optimization...")
        
        best_result = None
        best_params = None
        all_results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"Testing combination {i+1}/{len(param_combinations)}")
            
            # Create config with new parameters
            test_config = self._create_config_with_params(base_config, params)
            
            # Run backtest
            result = self.run_backtest(test_config)
            
            # Store result
            result_summary = {
                'parameters': params,
                'total_return_pct': result.total_return_pct,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio
            }
            all_results.append(result_summary)
            
            # Check if this is the best result
            if best_result is None or result.total_return_pct > best_result.total_return_pct:
                best_result = result
                best_params = params
        
        logger.info(f"Optimization complete. Best return: {best_result.total_return_pct:.2f}%")
        
        return {
            'best_parameters': best_params,
            'best_result': best_result,
            'all_results': all_results
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _create_config_with_params(self, base_config: BacktestConfig, params: Dict) -> BacktestConfig:
        """Create new config with updated parameters."""
        config_dict = asdict(base_config)
        config_dict.update(params)
        return BacktestConfig(**config_dict)
    
    def export_results(self, results: BacktestResults, output_path: str):
        """Export backtest results to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary = {
            'config': asdict(results.config),
            'performance': {
                'total_pnl': results.total_pnl,
                'total_return_pct': results.total_return_pct,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'max_drawdown': results.max_drawdown,
                'sharpe_ratio': results.sharpe_ratio,
                'total_trades': results.total_trades
            }
        }
        
        with open(output_dir / "backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export trades
        trades_df = pd.DataFrame([asdict(trade) for trade in results.trades])
        trades_df.to_csv(output_dir / "trades.csv", index=False)
        
        logger.info(f"Results exported to {output_path}")
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache_manager.clear_cache()
        self.query_engine.clear_indexes()


# Convenience functions for quick backtesting
def quick_iron_condor_backtest(start_date: str, end_date: str, 
                              data_path: str = "data/processed/parquet_1m") -> BacktestResults:
    """
    Quick Iron Condor backtest with default parameters.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_path: Path to parquet data
        
    Returns:
        Backtest results
    """
    engine = OptimizedBacktestEngine(data_path)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy_type="iron_condor",
        entry_times=["10:00:00"],
        put_distances=[50],
        call_distances=[50],
        spread_widths=[25],
        min_credit=0.75
    )
    
    return engine.run_backtest(config)