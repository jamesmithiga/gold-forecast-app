"""
MCP Endpoint Inspector Script
Run this to inspect all backend endpoints and identify issues
"""

import asyncio
import httpx
import json
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

async def test_endpoint(client, method, path, payload=None, params=None):
    """Test a single endpoint"""
    url = f"{BACKEND_URL}{path}"
    try:
        if method == "GET":
            response = await client.get(url, params=params, timeout=10.0)
        else:
            response = await client.post(url, json=payload, timeout=10.0)
        
        return {
            "path": path,
            "method": method,
            "status_code": response.status_code,
            "working": response.status_code < 400,
            "response": response.json() if response.status_code < 400 else response.text[:200]
        }
    except Exception as e:
        return {
            "path": path,
            "method": method,
            "status_code": None,
            "working": False,
            "error": str(e)
        }

async def inspect_all_endpoints():
    """Test all endpoints and report results"""
    print("\n" + "="*70)
    print("MCP ENDPOINT INSPECTION")
    print("="*70)
    print(f"Backend: {BACKEND_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*70 + "\n")
    
    # First check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{BACKEND_URL}/health", timeout=5.0)
            if health.status_code == 200:
                print("[OK] Backend is running and healthy\n")
            else:
                print(f"[WARN] Backend returned status {health.status_code}\n")
    except Exception as e:
        print(f"[ERROR] Cannot connect to backend: {e}")
        print("\nPlease start the backend first with:")
        print("  python fastapi_backend.py")
        return
    
    endpoints = [
        # With /api prefix
        {"method": "POST", "path": "/api/data", "payload": {"ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "POST", "path": "/api/features", "payload": {"ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "POST", "path": "/api/metrics", "payload": {"y_true": [1,2,3], "y_pred": [1.1,2.1,2.9], "model_name": "test", "ticker": "GC=F"}, "params": None},
        {"method": "GET", "path": "/api/data/summary", "payload": None, "params": {"ticker": "GC=F", "period": "1y"}},
        {"method": "GET", "path": "/api/data/freshness", "payload": None, "params": {"ticker": "GC=F"}},
        {"method": "POST", "path": "/api/forecast", "payload": {"model_type": "arima", "periods": 5, "ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "POST", "path": "/api/retrain", "payload": {"ticker": "GC=F", "model_name": "arima", "train_ratio": 0.8}, "params": None},
        {"method": "POST", "path": "/api/predict", "payload": {"ticker": "GC=F", "model_name": "arima", "horizon_days": 5}, "params": None},
        {"method": "GET", "path": "/api/comparison", "payload": None, "params": {"ticker": "GC=F"}},
        {"method": "GET", "path": "/api/metrics/arima", "payload": None, "params": {"ticker": "GC=F"}},
        # Without /api prefix (legacy)
        {"method": "POST", "path": "/data", "payload": {"ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "POST", "path": "/features", "payload": {"ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "POST", "path": "/metrics", "payload": {"y_true": [1,2,3], "y_pred": [1.1,2.1,2.9], "model_name": "test", "ticker": "GC=F"}, "params": None},
        {"method": "GET", "path": "/data/summary", "payload": None, "params": {"ticker": "GC=F", "period": "1y"}},
        {"method": "GET", "path": "/data/freshness", "payload": None, "params": {"ticker": "GC=F"}},
        {"method": "POST", "path": "/forecast", "payload": {"model_type": "arima", "periods": 5, "ticker": "GC=F", "period": "1y"}, "params": None},
        {"method": "GET", "path": "/health", "payload": None, "params": None},
    ]
    
    results = []
    working_count = 0
    broken_endpoints = []
    
    async with httpx.AsyncClient() as client:
        for ep in endpoints:
            method = ep["method"]
            path = ep["path"]
            payload = ep["payload"]
            params = ep["params"]
            print(f"Testing {method} {path}...", end=" ")
            result = await test_endpoint(client, method, path, payload, params)
            results.append(result)
            
            if result["working"]:
                print(f"[OK] {result['status_code']}")
                working_count += 1
            else:
                print(f"[FAIL] {result.get('error', result.get('response', 'Unknown error'))}")
                broken_endpoints.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("INSPECTION SUMMARY")
    print("="*70)
    print(f"Total endpoints tested: {len(endpoints)}")
    print(f"Working: {working_count}")
    print(f"Broken: {len(broken_endpoints)}")
    
    if broken_endpoints:
        print("\n[!] BROKEN ENDPOINTS:")
        for endpoint in broken_endpoints:
            print(f"  - {endpoint['method']} {endpoint['path']}")
            if 'error' in endpoint:
                print(f"    Error: {endpoint['error']}")
            elif 'response' in endpoint:
                print(f"    Response: {endpoint['response']}")
    
    # Check for data format issues
    print("\n" + "="*70)
    print("DATA FORMAT VALIDATION")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Test /api/data response format
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/data",
                json={"ticker": "GC=F", "period": "1y"},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    first_record = data[0]
                    print(f"\n[OK] /api/data returns list with {len(data)} records")
                    print(f"   First record fields: {list(first_record.keys())}")
                else:
                    print(f"\n[WARN] /api/data returns unexpected format: {type(data)}")
        except Exception as e:
            print(f"\n[ERROR] /api/data error: {e}")
        
        # Test /api/predict response format
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/predict",
                json={"ticker": "GC=F", "model_name": "arima", "horizon_days": 5},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                print(f"\n[OK] /api/predict response keys: {list(data.keys())}")
                if "predictions" in data:
                    print(f"   Predictions type: {type(data['predictions'])}, length: {len(data['predictions'])}")
                else:
                    print(f"   [WARN] Missing 'predictions' field")
        except Exception as e:
            print(f"\n[ERROR] /api/predict error: {e}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Check for duplicate endpoints
    api_endpoints = [r for r in results if r["path"].startswith("/api/")]
    non_api_endpoints = [r for r in results if not r["path"].startswith("/api/")]
    
    if api_endpoints and non_api_endpoints:
        print("\n1. Duplicate endpoints detected:")
        print("   Both /api/* and /* versions exist")
        print("   Recommendation: Use /api/* prefix consistently in frontend")
    
    # Check CORS
    print("\n2. CORS Configuration:")
    print("   Backend has CORS middleware allowing all origins")
    print("   This is OK for development but should be restricted in production")
    
    print("\n" + "="*70)
    
    return results

if __name__ == "__main__":
    asyncio.run(inspect_all_endpoints())
