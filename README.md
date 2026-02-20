# SPX 0DTE Options Backtesting System

A high-performance Python backtesting system for SPX (S&P 500 Index) 0DTE (0 Days to Expiration) options strategies, featuring Iron Condors with comprehensive parameter optimization and parquet-based data pipeline.

## 🚀 Current Features

- **Production-Ready Backtesting**: Single day and date range backtesting
- **Parameter Optimization**: Systematic testing of Iron Condor configurations
- **High-Performance Data Pipeline**: Parquet-based storage with optimized query engine
- **Comprehensive Analytics**: P&L tracking, win rates, sensitivity analysis
- **Interactive CLI**: Command-line interface for all backtesting operations

## 📁 Project Structure

```
spx-ai/
├── simple_backtest.py      # Main backtesting pipeline
├── interactive_backtest.py # Enhanced backtesting with optimization
├── config/
│   └── settings.py         # Configuration parameters
├── src/
│   ├── data/
│   │   ├── parquet_loader.py    # Parquet data loading
│   │   └── query_engine.py      # Optimized query engine
│   ├── backtesting/
│   │   ├── iron_condor_loader.py    # Iron Condor data loader
│   │   └── strategy_adapter.py      # Strategy building
│   └── utils/
├── data/
│   └── processed/
│       └── parquet_1m/     # Parquet data files (1-minute resolution)
└── logs/                   # Application logs
```

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Parquet data files in `data/processed/parquet_1m/`
- Dependencies: pandas, pyarrow, loguru

### Installation
```bash
# Clone repository
git clone <repo-url>
cd spx-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies  
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Backtesting

**Single Day Backtest:**
```bash
python simple_backtest.py
```

**Custom Parameters:**
```bash
python interactive_backtest.py --date 2026-02-09 --put-distance 50 --call-distance 50
```

**Date Range Backtest:**
```bash
python interactive_backtest.py --start-date 2026-02-09 --end-date 2026-02-13
```

### Parameter Optimization

**Find Best Parameters:**
```bash
python interactive_backtest.py --date 2026-02-09 --optimize
```

**Sensitivity Analysis:**
```bash
python interactive_backtest.py --date 2026-02-09 --sensitivity
```

## 📊 Backtesting Results

### Current Performance (2026-02-09 to 2026-02-13)

```
================================================================================
BACKTEST RESULTS - 5 Days
================================================================================
Date         Entry    SPX      Credit   P&L        %       Status    
--------------------------------------------------------------------------------
2026-02-09   10:00:00 6925     $269.90  $207.40    76.8  % ✓ WIN     
2026-02-10   10:00:00 6978     $129.90  $117.40    90.4  % ✓ WIN     
2026-02-11   10:00:00 6961     $219.90  $219.90    100.0 % ✓ WIN     
2026-02-12   10:00:00 6949     $189.90  $-2295.10  -1208.6% ✗ LOSS    
2026-02-13   10:00:00 6818     $972.40  $959.90    98.7  % ✓ WIN     

Setup Success Rate:  80.0% (4/5)
Trading Win Rate:    100.0% (4/4) 
Total P&L:           $-790.50
Avg P&L per Day:     $-158.10
```

### Parameter Optimization Results

**Best Configuration Found:** P25/C50/W50 (Put 25pt, Call 50pt, Width 50pt)
- **Entry Credit:** $722.40
- **P&L:** $714.90 (99.0% return)
- **Success Rate:** 66.7% of 96 combinations tested

## 🔧 Advanced Configuration

### Strategy Parameters

Edit parameters in backtesting functions:

```python
# Iron Condor Configuration
put_distance = 50        # Put short strike distance from SPX
call_distance = 50       # Call short strike distance from SPX  
spread_width = 25        # Spread width for both sides
min_credit = 0.50        # Minimum credit required
entry_time = "10:00:00"  # Entry time
exit_time = "15:45:00"   # Exit time
```

### Optimization Ranges

```python
# Parameter ranges for optimization
put_distances = [25, 50, 75, 100]
call_distances = [25, 50, 75, 100]
spread_widths = [25, 50]
entry_times = ["09:45:00", "10:00:00", "10:15:00"]
```

## 🏗️ Architecture

### Core Components

1. **SimpleBacktester** (`simple_backtest.py:42`): Main backtesting engine
2. **InteractiveBacktester** (`interactive_backtest.py:19`): Enhanced optimization features
3. **FastQueryEngine** (`src/data/query_engine.py`): Optimized data queries
4. **IronCondorDataLoader** (`src/backtesting/iron_condor_loader.py`): Strategy-specific data loading
5. **EnhancedStrategyBuilder** (`src/backtesting/strategy_adapter.py`): Strategy construction

### Data Pipeline

- **Input**: Parquet files with 1-minute SPX and options data
- **Processing**: Optimized indexing and caching for fast backtesting
- **Output**: Detailed P&L analysis and performance metrics

## 📈 Strategy Logic

### Iron Condor Implementation

**Entry Process:**
1. Get SPX price at entry time
2. Find viable Iron Condor setups based on parameters
3. Build strategy with liquid options
4. Calculate entry credit

**Exit Process:**
1. Get SPX price at exit time
2. Update option prices
3. Calculate exit cost
4. Compute P&L = Entry Credit - Exit Cost

**Success Criteria:**
- Setup found with minimum credit requirements
- Positive P&L at exit

## 🔍 Performance Metrics

### Key Analytics
- **Setup Success Rate**: Percentage of days with viable strategies
- **Trading Win Rate**: Percentage of profitable trades (successful setups only)
- **Total P&L**: Cumulative profit/loss
- **Average P&L per Day**: Mean daily performance
- **Credit Collection**: Total premiums collected

### Optimization Targets
- **P&L**: Maximize absolute profit
- **P&L %**: Maximize percentage returns
- **Credit**: Maximize premium collection

## 📊 Data Requirements

### Parquet File Structure
```
data/processed/parquet_1m/
├── spx_2026-02-09.parquet     # SPX underlying data
├── options_2026-02-09.parquet # Options chain data
├── spx_2026-02-10.parquet
├── options_2026-02-10.parquet
└── ...
```

### Required Columns
**SPX Data**: timestamp, close
**Options Data**: timestamp, strike, option_type, bid, ask, expiration, delta, gamma, theta, vega

## ⚠️ Important Notes

1. **Paper Trading Only**: This system is for backtesting and analysis only
2. **Historical Data**: Results based on historical market conditions
3. **Risk Management**: No built-in position sizing or risk controls
4. **Market Hours**: Designed for regular trading hours (9:30 AM - 4:00 PM ET)
5. **0DTE Focus**: Optimized for same-day expiration strategies

## 🚀 Next Steps

### Potential Enhancements
- Multi-strategy backtesting (Put spreads, Call spreads)
- Dynamic position sizing based on volatility
- Risk management rules (stop losses, profit targets)
- Portfolio-level analysis across multiple positions
- Export functionality for results (CSV, JSON)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

MIT License - See LICENSE file for details

---

**Ready for Production Backtesting! 🚀📊**

*Comprehensive SPX 0DTE Iron Condor backtesting with parameter optimization and high-performance data pipeline.*