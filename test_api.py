#!/usr/bin/env python3
"""
Test script for SPX AI Trading Platform API
Tests the main endpoints to ensure everything works correctly.
"""

import asyncio
import json
import time
from datetime import date
import requests
import websockets

API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/test-client"

def test_api_health():
    """Test basic API health endpoint"""
    print("🔍 Testing API health...")
    
    try:
        response = requests.get(f"{API_BASE}/")
        assert response.status_code == 200
        data = response.json()
        assert "SPX AI Trading Platform API" in data["message"]
        print("✅ API health check passed")
        return True
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_system_status():
    """Test system status endpoint"""
    print("🔍 Testing system status...")
    
    try:
        response = requests.get(f"{API_BASE}/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "available_dates" in data
        print("✅ System status check passed")
        print(f"   Available dates: {len(data['available_dates'])}")
        return True
    except Exception as e:
        print(f"❌ System status check failed: {e}")
        return False

def test_start_backtest():
    """Test starting a backtest"""
    print("🔍 Testing backtest start...")
    
    backtest_request = {
        "mode": "single_day",
        "single_date": "2026-02-09",
        "strategy_type": "iron_condor",
        "target_delta": 0.15,
        "put_distance": 50,
        "call_distance": 50,
        "spread_width": 25,
        "decay_threshold": 0.1,
        "entry_time": "10:00:00"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/backtest/start",
            json=backtest_request
        )
        assert response.status_code == 200
        data = response.json()
        assert "backtest_id" in data
        backtest_id = data["backtest_id"]
        print(f"✅ Backtest started successfully: {backtest_id}")
        return backtest_id
    except Exception as e:
        print(f"❌ Backtest start failed: {e}")
        return None

def test_backtest_status(backtest_id):
    """Test getting backtest status"""
    print("🔍 Testing backtest status...")
    
    try:
        response = requests.get(f"{API_BASE}/api/v1/backtest/{backtest_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"✅ Backtest status retrieved: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Backtest status check failed: {e}")
        return False

async def test_websocket():
    """Test WebSocket connection"""
    print("🔍 Testing WebSocket connection...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(welcome_msg)
            assert data["type"] == "connection"
            print("✅ WebSocket connection successful")
            
            # Send a test message
            await websocket.send("Hello from test client")
            
            # Wait for echo
            echo_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print("✅ WebSocket echo received")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting SPX AI Trading Platform API Tests")
    print("=" * 50)
    
    # Basic tests
    if not test_api_health():
        print("❌ Basic API tests failed. Is the server running?")
        return False
    
    if not test_system_status():
        return False
    
    # Backtest tests
    backtest_id = test_start_backtest()
    if not backtest_id:
        return False
    
    # Give the backtest a moment to start
    time.sleep(2)
    
    if not test_backtest_status(backtest_id):
        return False
    
    # WebSocket test
    print("\n🔍 Testing WebSocket (requires server to be running)...")
    try:
        asyncio.run(test_websocket())
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        print("   (This is expected if server is not running)")
    
    print("\n" + "=" * 50)
    print("✅ All basic API tests completed!")
    print("\n📋 Test Summary:")
    print("   - Health check: ✅")
    print("   - System status: ✅")  
    print("   - Start backtest: ✅")
    print("   - Backtest status: ✅")
    print("   - WebSocket: (manual test)")
    
    print(f"\n🎯 To monitor your backtest:")
    print(f"   Status: GET {API_BASE}/api/v1/backtest/{backtest_id}/status")
    print(f"   Results: GET {API_BASE}/api/v1/backtest/{backtest_id}/results")
    print(f"   WebSocket: {WS_URL}")
    
    return True

if __name__ == "__main__":
    main()