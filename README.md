# SPX 0DTE Options Trading Bot

A comprehensive Python trading bot for SPX (S&P 500 Index) 0DTE (0 Days to Expiration) options strategies, featuring Iron Condors, Call/Put Spreads, and technical analysis.

## 🚀 Features

- **Multiple Strategies**: Iron Condor, Call Spreads, Put Spreads
- **Technical Analysis**: MACD, RSI, Bollinger Bands for entry/exit signals
- **Local Data Storage**: SQLite database for fast backtesting
- **Risk Management**: Position sizing, portfolio limits, performance tracking
- **Data Sources**: ThetaData Terminal REST API + CSV import capability
- **Comprehensive Backtesting**: P&L analysis, win rates, Sharpe ratios

## 📁 Project Structure

```
spx-ai/
├── main.py                 # Main CLI interface
├── setup.sh               # Setup script
├── requirements.txt       # Python dependencies
├── examples.py           # Usage examples
├── config/
│   ├── settings.py       # Configuration parameters
│   └── __init__.py
├── src/
│   ├── data/
│   │   ├── theta_client.py    # ThetaData API client
│   │   ├── storage.py         # SQLite data storage
│   │   ├── downloader.py      # Data download orchestrator
│   │   └── csv_importer.py    # CSV data import
│   ├── indicators/
│   │   └── technical_indicators.py  # MACD, RSI, Bollinger Bands
│   ├── strategies/
│   │   └── options_strategies.py    # Options strategy classes
│   ├── utils/
│   │   └── risk_management.py       # Risk management & performance
│   └── backtesting/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # SQLite database
└── logs/                 # Application logs
```

## ⚙️ Setup & Installation

### 1. Quick Setup
```bash
# Clone and setup
git clone <repo-url>
cd spx-ai
chmod +x setup.sh
./setup.sh
```

### 2. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env
```

### 3. Configure Credentials
Edit `.env` file with your ThetaData credentials:
```bash
THETA_USERNAME=your_username
THETA_PASSWORD=your_password
```

## 🎯 Usage

### Basic Commands

```bash
# Generate sample data for testing
python main.py sample --days-back 30

# Check data status
python main.py status

# Download real data (requires ThetaData credentials)
python main.py download --days-back 30

# Run Iron Condor backtest
python main.py backtest --strategy iron_condor --start-date 2024-01-01 --end-date 2024-12-31

# Update existing data
python main.py download --update
```

### Strategy Parameters

Edit `config/settings.py` to customize strategies:

```python
IRON_CONDOR_PARAMS = {
    "put_strike_distance": 50,    # Distance from current price
    "call_strike_distance": 50,   # Distance from current price
    "profit_target": 0.5,         # Take profit at 50% of credit
    "stop_loss": 2.0             # Stop loss at 200% of credit
}

RSI_PARAMS = {
    "period": 14,
    "overbought": 70,
    "oversold": 30
}
```

## 📊 Backtesting Results

The bot provides comprehensive analysis:

```
==================================================
BACKTEST RESULTS SUMMARY
==================================================
Period: 2024-01-01 to 2024-12-31
Total Trades: 157
Win Rate: 68.2%
Total P&L: $12,450.00
Avg P&L per Trade: $79.30
Max Win: $850.00
Max Loss: $-2,100.00
Profit Factor: 1.85
Sharpe Ratio: 1.42
Max Drawdown: -8.3%
==================================================
```

## 🔧 Advanced Usage

### Custom Strategy Development

```python
from src.strategies.options_strategies import StrategyBuilder

# Build custom Iron Condor
ic = StrategyBuilder.build_iron_condor(
    entry_date=datetime(2024, 1, 15),
    underlying_price=4500,
    options_data=options_data,
    put_distance=75,      # Custom parameters
    call_distance=75,
    spread_width=25
)

# Calculate P&L at different prices
pnl_4400 = ic.get_profit_at_expiration(4400)
pnl_4500 = ic.get_profit_at_expiration(4500)
pnl_4600 = ic.get_profit_at_expiration(4600)
```

### CSV Data Import

```python
from src.data.csv_importer import CSVDataImporter

importer = CSVDataImporter()

# Import underlying SPX data
importer.import_spx_underlying('data/spx_daily.csv')

# Import options chain data
importer.import_spx_options('data/spx_options.csv')
```

### Technical Analysis

```python
from src.indicators.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators()

# Calculate all indicators
df_with_indicators = indicators.calculate_all_indicators(price_df)

# Generate trading signals
signals_df = indicators.get_trading_signals(df_with_indicators)

# Check for Iron Condor entry signals
neutral_days = signals_df[signals_df['neutral_signal'] == True]
```

## 📈 Strategy Logic

### Iron Condor Entry Criteria
- RSI between 40-60 (neutral market)
- Price within middle 60% of Bollinger Bands
- Low implied volatility environment
- No major economic events

### Entry/Exit Rules
1. **Entry**: When technical indicators show neutral/range-bound market
2. **Profit Target**: Close at 25-50% of maximum profit
3. **Stop Loss**: Close if loss exceeds 200% of credit received
4. **Time Decay**: Automatic closure at expiration (0DTE)

### Risk Management
- Maximum 2% risk per trade
- Maximum 20% portfolio risk
- Position sizing based on confidence score
- Correlation limits across strategies

## 🛠️ Data Sources

### ThetaData Terminal (Recommended)
- **Setup**: Download and run ThetaData Terminal locally
- **Connection**: REST API via http://127.0.0.1:25503
- **No Python Library**: Uses direct HTTP requests (no installation issues)
- **Real-time Data**: Live and historical options chains with OHLC
- **Full Options Greeks**: Delta, gamma, theta, vega, IV (when available)
- **Subscription Required**: Need ThetaData account

**Quick Setup:**
1. Download ThetaData Terminal from [thetadata.com](https://thetadata.com)
2. Login and keep terminal running
3. Update `.env` with your credentials
4. See `THETADATA_SETUP.md` for detailed instructions

### CSV Import (Alternative)
Support for standard CSV formats:

**Underlying Data Format:**
```csv
date,open,high,low,close,volume
2024-01-01,4500.00,4520.00,4495.00,4515.00,1000000
```

**Options Data Format:**
```csv
date,expiration,strike,option_type,bid,ask,delta,gamma,theta,vega,iv
2024-01-01,2024-01-01,4500,call,2.50,2.70,0.45,0.02,-0.8,12.5,0.25
```

## 🔍 Performance Monitoring

### Key Metrics Tracked
- Win Rate & Profit Factor
- Average P&L per Trade
- Maximum Drawdown
- Sharpe Ratio
- Risk-Adjusted Returns
- Strategy-Specific Analytics

### Logging & Alerts
- Comprehensive logging to `logs/` directory
- Trade execution logs
- Error handling and debugging
- Performance monitoring

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This bot is for educational and research purposes
2. **Paper Trading**: Test thoroughly with paper trading before live use
3. **Risk Warning**: Options trading involves substantial risk
4. **No Financial Advice**: This is not investment advice
5. **Sample Data**: Generated sample data is NOT for real trading decisions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-strategy`)
3. Commit changes (`git commit -am 'Add new strategy'`)
4. Push to branch (`git push origin feature/new-strategy`)
5. Create Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Resources

- [ThetaData API Documentation](https://http-docs.thetadata.us/)
- [SPX Options Basics](https://www.cboe.com/tradable_products/sp_500/)
- [0DTE Trading Strategies](https://www.tastytrade.com/concepts-strategies/0dte)
- [Options Greeks Explained](https://www.investopedia.com/trading/using-the-greeks-to-understand-options/)

## 🆘 Support

- 📧 Issues: Create GitHub issue
- 📚 Documentation: See `examples.py`
- 💬 Discussions: GitHub discussions tab

---

**Happy Trading! 🚀📈**

*Remember: Past performance does not guarantee future results. Trade responsibly.*