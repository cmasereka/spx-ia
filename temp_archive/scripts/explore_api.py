#!/usr/bin/env python3
"""
ThetaData Terminal API Explorer
Helps discover working endpoints in your ThetaData Terminal
"""

import requests
import json
from datetime import datetime

def test_endpoint(base_url, endpoint, params=None, auth=None):
    """Test an API endpoint and return results"""
    try:
        session = requests.Session()
        if auth:
            session.headers.update({'Authorization': f'Basic {auth}'})
        
        full_url = f"{base_url}/{endpoint.lstrip('/')}"
        response = session.get(full_url, params=params or {}, timeout=5)
        
        return {
            'url': full_url,
            'params': params,
            'status': response.status_code,
            'headers': dict(response.headers),
            'content': response.text[:500] + '...' if len(response.text) > 500 else response.text
        }
    except Exception as e:
        return {
            'url': full_url if 'full_url' in locals() else endpoint,
            'error': str(e)
        }

def explore_thetadata_api():
    """Explore ThetaData API endpoints"""
    base_url = "http://127.0.0.1:25503"
    
    # Test common endpoint patterns
    endpoints_to_test = [
        # Root endpoints
        "/",
        "/v2/",
        "/v3/",
        "/api/",
        
        # List endpoints
        "/v2/list/expirations",
        "/v3/list/expirations", 
        "/list/expirations",
        
        # Historical data endpoints
        "/v2/hist/stock/ohlc",
        "/v3/hist/stock/ohlc",
        "/hist/stock/ohlc",
        
        # Snapshot endpoints
        "/v2/snapshot/stock",
        "/v3/snapshot/stock",
        "/snapshot/stock",
        
        # Option endpoints
        "/v2/hist/option/ohlc",
        "/v3/hist/option/ohlc",
        "/hist/option/ohlc",
    ]
    
    symbols_to_test = ['SPX', 'SPXW']
    
    print("🔍 ThetaData Terminal API Explorer")
    print("=" * 50)
    print(f"Testing connection to: {base_url}")
    print()
    
    results = []
    
    for endpoint in endpoints_to_test:
        print(f"Testing: {endpoint}")
        
        # Test without parameters
        result = test_endpoint(base_url, endpoint)
        results.append(result)
        
        if result.get('status') == 200:
            print(f"  ✅ SUCCESS: {result['status']}")
        elif result.get('status') == 404:
            print(f"  ❌ Not Found: {result['status']}")
        elif result.get('status') == 410:
            print(f"  ⚠️  Gone (deprecated): {result['status']}")
        elif 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ❓ Status: {result.get('status', 'unknown')}")
        
        # Test with common parameters for data endpoints
        if 'hist' in endpoint or 'list' in endpoint or 'snapshot' in endpoint:
            for symbol in symbols_to_test:
                params = {}
                if 'list' in endpoint:
                    params = {'symbol': symbol}
                elif 'hist' in endpoint:
                    params = {
                        'symbol': symbol,
                        'start_date': '20240101',
                        'end_date': '20240102'
                    }
                elif 'snapshot' in endpoint:
                    params = {'symbol': symbol}
                
                if params:
                    param_result = test_endpoint(base_url, endpoint, params)
                    if param_result.get('status') == 200:
                        print(f"    ✅ {symbol}: SUCCESS")
                        # Show sample response for working endpoints
                        if len(param_result.get('content', '')) > 0:
                            try:
                                json_data = json.loads(param_result['content'])
                                print(f"      📄 Sample response: {str(json_data)[:200]}...")
                            except:
                                print(f"      📄 Response: {param_result['content'][:100]}...")
                    elif param_result.get('status') not in [404, 410]:
                        print(f"    ❓ {symbol}: Status {param_result.get('status')}")
        
        print()
    
    # Summary of working endpoints
    working_endpoints = [r for r in results if r.get('status') == 200]
    
    print("\n📋 SUMMARY")
    print("=" * 50)
    if working_endpoints:
        print("✅ Working endpoints found:")
        for endpoint in working_endpoints:
            print(f"  - {endpoint['url']}")
    else:
        print("❌ No working endpoints found")
        print("\n💡 Troubleshooting suggestions:")
        print("1. Verify ThetaData Terminal is fully loaded")
        print("2. Check Terminal is logged in and connected")
        print("3. Try restarting ThetaData Terminal")
        print("4. Verify port 25503 is correct in Terminal settings")
        print("5. Check if Terminal requires different authentication")

if __name__ == "__main__":
    explore_thetadata_api()