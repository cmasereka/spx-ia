# SPX AI Trading Platform API

## 🚀 FastAPI Web Service

This is the web service version of the SPX AI Trading Platform, providing REST API endpoints and real-time WebSocket communication for backtesting SPX options strategies.

## ✨ Features

- **RESTful API** for managing backtests
- **Real-time WebSocket** updates for live monitoring
- **Async backtesting** with progress tracking
- **Interactive API documentation** with Swagger UI
- **Multiple strategy types** (Iron Condor, Put Spreads, Call Spreads)
- **Configurable parameters** for entry, risk management, and monitoring

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
# Option 1: Use the startup script
./start_api.sh

# Option 2: Start manually
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### System
- `GET /` - Health check
- `GET /api/v1/status` - System status and data availability

### Backtesting
- `POST /api/v1/backtest/start` - Start a new backtest
- `GET /api/v1/backtest/{id}/status` - Get backtest status
- `GET /api/v1/backtest/{id}/results` - Get backtest results
- `DELETE /api/v1/backtest/{id}` - Cancel a backtest
- `GET /api/v1/backtest` - List all backtests

### WebSocket
- `WS /ws/{client_id}` - Real-time updates

## 🔧 Configuration

### Backtest Request Example
```json
{
  "mode": "date_range",
  "start_date": "2026-02-09",
  "end_date": "2026-02-13",
  "strategy_type": "iron_condor",
  "target_delta": 0.15,
  "put_distance": 50,
  "call_distance": 50,
  "spread_width": 25,
  "decay_threshold": 0.1,
  "entry_time": "10:00:00"
}
```

## 🌐 WebSocket Usage

Connect to `ws://localhost:8000/ws/{your_client_id}` to receive real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client-123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
    
    // Handle different message types
    switch(data.type) {
        case 'progress':
            // Update progress bar
            break;
        case 'trade_result':
            // Display new trade result
            break;
        case 'backtest_completed':
            // Show completion notification
            break;
    }
};
```

## 🎯 Next Steps

This FastAPI service provides the foundation for building a React frontend. Key integration points:

1. **React App** can consume the REST API for CRUD operations
2. **WebSocket connection** enables real-time trade monitoring
3. **Progress updates** allow for live backtest tracking
4. **Result streaming** provides immediate feedback

### Recommended Frontend Architecture:
- **React** with TypeScript for the UI
- **Material-UI** or **Chakra UI** for components  
- **React Query** for API state management
- **WebSocket hook** for real-time updates
- **Charts.js** or **Recharts** for performance visualization

## 📊 Example Usage

1. Start a backtest via POST to `/api/v1/backtest/start`
2. Monitor progress via WebSocket connection
3. Retrieve results via GET to `/api/v1/backtest/{id}/results`
4. Visualize performance in your frontend

The API is designed to be frontend-agnostic and can power web, mobile, or desktop applications.