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

**Detailed Trade Information:**
```bash
python interactive_backtest.py --date 2026-02-09 --detailed
```

**Date Range Backtest:**
```bash
python interactive_backtest.py --start-date 2026-02-09 --end-date 2026-02-13
```

**Detailed Multi-Day Analysis:**
```bash
python interactive_backtest.py --start-date 2026-02-09 --end-date 2026-02-11 --detailed
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

**All Available Options:**
```bash
# Show help with all available flags
python interactive_backtest.py --help

# Common usage patterns
python interactive_backtest.py --date 2026-02-09 --detailed --put-distance 75
python interactive_backtest.py --start-date 2026-02-09 --end-date 2026-02-13 --detailed
```

## 📊 Backtesting Results

### Detailed Trade Analysis (--detailed flag)

```
🟢 TRADE 1: 2026-02-09 - ✓ WIN
------------------------------------------------------------
Entry Time:     10:00:00
SPX Price:      $6925 → $6968 (Δ+43)

IRON CONDOR STRIKES:
Put Spread:     6850.0P / 6875.0P  (Buy 6850.0 / Sell 6875.0)
Call Spread:    6975.0C / 7000.0C  (Sell 6975.0 / Buy 7000.0)

TRADE FINANCIALS:
Entry Credit:   $  269.90  (Premium received)
Exit Cost:      $   62.50  (Cost to close)
Net P&L:        $  207.40  (+76.8%)
Max Profit:     $  269.90
Max Loss:       $ 2230.10
```

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

## 🔍 Detailed Trade Information

The `--detailed` flag provides comprehensive trade analysis:

### Trade Display Format

**Strike Prices:**
- **Put Spread**: Long Strike / Short Strike (e.g., 6850P/6875P)
- **Call Spread**: Short Strike / Long Strike (e.g., 6975C/7000C)

**Financial Breakdown:**
- **Entry Credit**: Premium received when opening the Iron Condor
- **Exit Cost**: Cost to close/buy back the position at exit time
- **Net P&L**: Entry Credit - Exit Cost
- **SPX Movement**: Underlying price change during trade

**Risk Metrics:**
- **Max Profit**: Maximum possible profit (usually entry credit)
- **Max Loss**: Maximum possible loss (spread width - entry credit)

### Example Trade Analysis

```
🟢 TRADE 1: 2026-02-09 - ✓ WIN
SPX Price: $6925 → $6968 (Δ+43)

IRON CONDOR STRIKES:
Put Spread:  6850P/6875P  (Bought 6850P, Sold 6875P)
Call Spread: 6975C/7000C  (Sold 6975C, Bought 7000C)

TRADE FINANCIALS:
Entry Credit: $269.90  (Collected premium)
Exit Cost:    $ 62.50  (Cost to close)
Net P&L:      $207.40  (+76.8% return)
```

This shows SPX moved +43 points but stayed within the profitable range (6875-6975), allowing most of the credit to be kept as profit.

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

### Recently Added
- ✓ **Detailed Trade Analysis**: Complete strike prices, entry/exit costs, P&L breakdown
- ✓ **Enhanced CLI**: `--detailed` flag for comprehensive trade information  
- ✓ **Multi-format Display**: Summary tables for ranges, detailed view for individual trades

### Potential Enhancements
- **Multi-Strategy Support**: Put spreads, Call spreads, Strangles
- **Dynamic Position Sizing**: Based on volatility and risk tolerance
- **Risk Management Rules**: Stop losses, profit targets, delta hedging
- **Portfolio-Level Analysis**: Multiple positions, correlation analysis
- **Export Functionality**: CSV, JSON, Excel formats for results
- **Real-time Integration**: Live trading capabilities with broker APIs
- **Advanced Greeks Analysis**: Delta, gamma, theta decay tracking
- **Market Regime Detection**: Trending vs. ranging market identification

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