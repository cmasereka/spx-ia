# ThetaData Terminal Setup Guide

## Overview

The trading bot now uses the ThetaData Terminal REST API instead of the Python library, which eliminates installation issues and provides more reliable data access.

## Setup Steps

### 1. Download ThetaData Terminal
- Visit [ThetaData.com](https://thetadata.com)
- Download the ThetaData Terminal application
- Install on your local machine

### 2. Configure Terminal
1. Launch ThetaData Terminal
2. Login with your ThetaData credentials
3. The terminal runs a local REST API on port 25503
4. Ensure the terminal stays running while using the bot

### 3. Update Bot Configuration
Edit your `.env` file:
```bash
THETA_USERNAME=your_username_here
THETA_PASSWORD=your_password_here

# Optional: Change ThetaData Terminal port (default is 25503)
# THETA_TERMINAL_PORT=25503
```

### 4. Test Connection
```bash
# Test if ThetaData Terminal is running
curl http://127.0.0.1:25503/v2/list/expirations?root=SPX

# Or test with the bot
python main.py status
```

## API Endpoints Used

The bot uses these ThetaData Terminal API endpoints:

- **Underlying Data**: `GET /v2/hist/stock/ohlc`
- **Options Chains**: `GET /v2/hist/option/ohlc`
- **Expirations**: `GET /v2/list/expirations`

## Troubleshooting

### Terminal Not Running
```
Error: ThetaData Terminal not running. Please start ThetaData Terminal first.
```
**Solution**: Launch ThetaData Terminal application

### Connection Failed
```
Error: Failed to connect: HTTP 401
```
**Solution**: Check credentials in `.env` file

### No Data Returned
```
Warning: No underlying data found for SPX
```
**Solution**: 
- Ensure you have the correct data subscription
- Check date ranges (weekends/holidays)
- Verify SPX symbol access

## Data Format

### Underlying Data Response
```json
{
  "response": [
    [1640995200000, 4766000, 4818000, 4750000, 4793000, 2883000000]
  ]
}
```
Format: `[timestamp_ms, open, high, low, close, volume]`

### Options Data Response
```json
{
  "response": [
    {
      "contract": ".SPX240119C04700000",
      "data": [[1640995200000, 250, 300, 200, 275, 1500]]
    }
  ]
}
```

## Advanced Configuration

### Custom Terminal Port
If running ThetaData Terminal on a different port, set in your `.env` file:

```bash
THETA_TERMINAL_PORT=YOUR_PORT
```

The bot will automatically use this port for all API requests.

### Authentication Headers
The bot automatically handles Basic Auth:
```python
Authorization: Basic base64(username:password)
```

## Performance Tips

1. **Keep Terminal Running**: Don't close ThetaData Terminal during bot operation
2. **Rate Limiting**: Built-in 1-second delays between requests
3. **Data Caching**: Consider caching frequently accessed data
4. **Error Handling**: Bot automatically retries failed connections

## Alternative: CSV Import

If you don't have ThetaData access, use the CSV import feature:

```bash
# Generate sample data for testing
python main.py sample --days-back 30

# Import real CSV data
python -c "
from src.data.csv_importer import CSVDataImporter
importer = CSVDataImporter()
importer.import_spx_underlying('your_data.csv')
"
```