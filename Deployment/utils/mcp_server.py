"""
MCP (Model Context Protocol) Server for HTTP Traffic Monitoring
Intercepts and analyzes all HTTP traffic between Streamlit frontend and FastAPI backend

This MCP server provides:
- HTTP proxy functionality to intercept traffic
- Request/Response logging and inspection
- Endpoint health checking
- Error detection and diagnostics
- Data format validation
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import uvicorn
import httpx
import json
import logging
import time
from datetime import datetime
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCP-Server")

# Configuration
BACKEND_URL = "http://localhost:8000"
MCP_PORT = 8001

# In-memory storage for traffic logs
class TrafficLogger:
    def __init__(self, max_logs=1000):
        self.lock = threading.Lock()
        self.max_logs = max_logs
        self.request_logs = []
        self.endpoint_stats = defaultdict(lambda: {"requests": 0, "errors": 0, "avg_response_time": 0})
    
    def log_request(self, method: str, path: str, request_body: Any, response_status: int, 
                   response_body: Any, response_time: float, error: str = None):
        with self.lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "path": path,
                "request_body": request_body,
                "response_status": response_status,
                "response_body": response_body,
                "response_time_ms": round(response_time * 1000, 2),
                "error": error
            }
            
            # Add to logs
            self.request_logs.append(log_entry)
            
            # Trim if exceeding max
            if len(self.request_logs) > self.max_logs:
                self.request_logs = self.request_logs[-self.max_logs:]
            
            # Update stats
            endpoint = f"{method} {path}"
            self.endpoint_stats[endpoint]["requests"] += 1
            if response_status >= 400:
                self.endpoint_stats[endpoint]["errors"] += 1
            
            # Update average response time
            stats = self.endpoint_stats[endpoint]
            n = stats["requests"]
            stats["avg_response_time"] = (stats["avg_response_time"] * (n - 1) + response_time * 1000) / n
    
    def get_logs(self, limit: int = 100):
        with self.lock:
            return self.request_logs[-limit:]
    
    def get_stats(self):
        with self.lock:
            return dict(self.endpoint_stats)
    
    def clear_logs(self):
        with self.lock:
            self.request_logs.clear()
            self.endpoint_stats.clear()

# Initialize traffic logger
traffic_logger = TrafficLogger()

# Create FastAPI app for MCP Server
app = FastAPI(
    title="MCP Traffic Monitor Server",
    description="HTTP Traffic Monitoring and Inspection Server for Streamlit-FastAPI communication",
    version="1.0.0"
)

# Health check endpoint
@app.get("/mcp/health")
async def mcp_health():
    """MCP server health check"""
    return {
        "status": "ok",
        "service": "MCP Traffic Monitor",
        "timestamp": datetime.now().isoformat()
    }

# Backend health check
@app.get("/mcp/backend/health")
async def backend_health():
    """Check if backend is reachable"""
    try:
        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.get(f"{BACKEND_URL}/health", timeout=5.0)
            response_time = time.time() - start
            return {
                "status": "ok" if response.status_code == 200 else "error",
                "backend_url": BACKEND_URL,
                "response_time_ms": round(response_time * 1000, 2),
                "backend_response": response.json() if response.status_code == 200 else None
            }
    except Exception as e:
        return {
            "status": "error",
            "backend_url": BACKEND_URL,
            "error": str(e)
        }

# Endpoint inspection
@app.get("/mcp/inspect/endpoints")
async def inspect_endpoints():
    """Inspect all available endpoints on the backend"""
    endpoints = [
        # With /api prefix
        {"path": "/api/data", "method": "POST", "description": "Get historical data"},
        {"path": "/api/features", "method": "POST", "description": "Get engineered features"},
        {"path": "/api/metrics", "method": "POST", "description": "Calculate metrics"},
        {"path": "/api/data/summary", "method": "GET", "description": "Get data summary"},
        {"path": "/api/data/freshness", "method": "GET", "description": "Check data freshness"},
        {"path": "/api/forecast", "method": "POST", "description": "Generate forecast"},
        {"path": "/api/retrain", "method": "POST", "description": "Retrain model"},
        {"path": "/api/predict", "method": "POST", "description": "Make predictions"},
        {"path": "/api/comparison", "method": "GET", "description": "Model comparison"},
        {"path": "/api/metrics/{model_name}", "method": "GET", "description": "Get model metrics"},
        # Without /api prefix (legacy)
        {"path": "/data", "method": "POST", "description": "Get historical data (legacy)"},
        {"path": "/features", "method": "POST", "description": "Get engineered features (legacy)"},
        {"path": "/metrics", "method": "POST", "description": "Calculate metrics (legacy)"},
        {"path": "/data/summary", "method": "GET", "description": "Get data summary (legacy)"},
        {"path": "/data/freshness", "method": "GET", "description": "Check data freshness (legacy)"},
        {"path": "/forecast", "method": "POST", "description": "Generate forecast (legacy)"},
        {"path": "/health", "method": "GET", "description": "Backend health check"},
    ]
    
    results = []
    async with httpx.AsyncClient() as client:
        for endpoint in endpoints:
            try:
                if endpoint["method"] == "GET":
                    response = await client.get(
                        f"{BACKEND_URL}{endpoint['path']}",
                        params={"ticker": "GC=F"},
                        timeout=10.0
                    )
                else:
                    # For POST, send a minimal valid payload
                    if "data" in endpoint["path"]:
                        payload = {"ticker": "GC=F", "period": "1y"}
                    elif "forecast" in endpoint["path"]:
                        payload = {"model_type": "arima", "periods": 5, "ticker": "GC=F"}
                    elif "retrain" in endpoint["path"]:
                        payload = {"ticker": "GC=F", "model_name": "arima", "train_ratio": 0.8}
                    elif "predict" in endpoint["path"]:
                        payload = {"ticker": "GC=F", "model_name": "arima", "horizon_days": 5}
                    elif "metrics" in endpoint["path"] and "model_name" not in endpoint["path"]:
                        payload = {"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9], "model_name": "test", "ticker": "GC=F"}
                    else:
                        payload = {}
                    
                    response = await client.post(
                        f"{BACKEND_URL}{endpoint['path']}",
                        json=payload,
                        timeout=10.0
                    )
                
                results.append({
                    "path": endpoint["path"],
                    "method": endpoint["method"],
                    "description": endpoint["description"],
                    "status_code": response.status_code,
                    "working": response.status_code < 400,
                    "response_preview": str(response.text)[:200] if response.status_code < 400 else response.text
                })
            except Exception as e:
                results.append({
                    "path": endpoint["path"],
                    "method": endpoint["method"],
                    "description": endpoint["description"],
                    "status_code": None,
                    "working": False,
                    "error": str(e)
                })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "backend_url": BACKEND_URL,
        "total_endpoints": len(results),
        "working_endpoints": sum(1 for r in results if r["working"]),
        "broken_endpoints": [r for r in results if not r["working"]],
        "results": results
    }

# Traffic logs endpoint
@app.get("/mcp/logs")
async def get_logs(limit: int = 100):
    """Get recent traffic logs"""
    return {
        "logs": traffic_logger.get_logs(limit),
        "count": len(traffic_logger.get_logs(limit))
    }

# Traffic statistics endpoint
@app.get("/mcp/stats")
async def get_stats():
    """Get traffic statistics"""
    return {
        "stats": traffic_logger.get_stats(),
        "total_requests": sum(s["requests"] for s in traffic_logger.get_stats().values()),
        "total_errors": sum(s["errors"] for s in traffic_logger.get_stats().values())
    }

# Clear logs endpoint
@app.post("/mcp/logs/clear")
async def clear_logs():
    """Clear traffic logs"""
    traffic_logger.clear_logs()
    return {"status": "ok", "message": "Logs cleared"}

# Proxy endpoints that intercept and log traffic

@app.api_route("/mcp/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(path: str, request: Request):
    """Proxy endpoint that forwards requests to backend and logs them"""
    # Build the full URL
    full_path = f"/{path}"
    url = f"{BACKEND_URL}{full_path}"
    
    # Get request details
    method = request.method
    query_params = dict(request.query_params)
    
    # Read body for POST/PUT/PATCH
    body = None
    if method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except:
            body = None
    
    # Forward request to backend
    start_time = time.time()
    error = None
    response_status = 500
    response_body = None
    
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, params=query_params, timeout=30.0)
            elif method == "POST":
                response = await client.post(url, json=body, timeout=30.0)
            elif method == "PUT":
                response = await client.put(url, json=body, timeout=30.0)
            elif method == "DELETE":
                response = await client.delete(url, timeout=30.0)
            elif method == "PATCH":
                response = await client.patch(url, json=body, timeout=30.0)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
            
            response_status = response.status_code
            
            # Try to parse response
            try:
                response_body = response.json()
            except:
                response_body = response.text
        
        response_time = time.time() - start_time
        
    except Exception as e:
        error = str(e)
        response_time = time.time() - start_time
    
    # Log the request
    traffic_logger.log_request(
        method=method,
        path=full_path,
        request_body=body,
        response_status=response_status,
        response_body=response_body,
        response_time=response_time,
        error=error
    )
    
    # Return response
    if error:
        raise HTTPException(status_code=response_status, detail=error)
    
    return JSONResponse(content=response_body, status_code=response_status)

# Diagnostic endpoints

@app.get("/mcp/diagnose/format")
async def diagnose_format():
    """Diagnose data format issues between frontend and backend"""
    issues = []
    
    # Test data endpoint format
    try:
        async with httpx.AsyncClient() as client:
            # Test POST /api/data
            response = await client.post(
                f"{BACKEND_URL}/api/data",
                json={"ticker": "GC=F", "period": "1y"},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    # Check if it's a list of dicts with expected fields
                    first_record = data[0]
                    expected_fields = ["Date", "Open", "High", "Low", "Close", "Volume"]
                    missing_fields = [f for f in expected_fields if f not in first_record]
                    if missing_fields:
                        issues.append({
                            "endpoint": "/api/data",
                            "issue": f"Missing fields in response: {missing_fields}",
                            "severity": "warning"
                        })
                else:
                    issues.append({
                        "endpoint": "/api/data",
                        "issue": "Response is not a list of records",
                        "severity": "error"
                    })
            else:
                issues.append({
                    "endpoint": "/api/data",
                    "issue": f"HTTP {response.status_code}: {response.text}",
                    "severity": "error"
                })
    except Exception as e:
        issues.append({
            "endpoint": "/api/data",
            "issue": str(e),
            "severity": "error"
        })
    
    # Test predict endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/predict",
                json={"ticker": "GC=F", "model_name": "arima", "horizon_days": 5},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if "predictions" not in data:
                    issues.append({
                        "endpoint": "/api/predict",
                        "issue": "Response missing 'predictions' field",
                        "severity": "error"
                    })
            else:
                issues.append({
                    "endpoint": "/api/predict",
                    "issue": f"HTTP {response.status_code}: {response.text}",
                    "severity": "error"
                })
    except Exception as e:
        issues.append({
            "endpoint": "/api/predict",
            "issue": str(e),
            "severity": "error"
        })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "issues_found": len(issues),
        "issues": issues,
        "status": "healthy" if len(issues) == 0 else "issues_detected"
    }

@app.get("/mcp/diagnose/communication")
async def diagnose_communication():
    """Diagnose communication issues between frontend and backend"""
    issues = []
    
    # Check CORS configuration
    try:
        async with httpx.AsyncClient() as client:
            # Check if backend accepts requests from different origins
            response = await client.get(
                f"{BACKEND_URL}/health",
                headers={"Origin": "http://localhost:8501"},
                timeout=5.0
            )
            
            if "access-control-allow-origin" not in [h.lower() for h in response.headers]:
                # This is expected for simple requests, check if CORS is configured
                pass
    except Exception as e:
        issues.append({
            "type": "connection",
            "issue": f"Cannot connect to backend: {str(e)}",
            "severity": "critical"
        })
    
    # Check if API key is required but not validated
    try:
        async with httpx.AsyncClient() as client:
            # Try without API key
            response_no_key = await client.get(
                f"{BACKEND_URL}/api/comparison",
                params={"ticker": "GC=F"},
                timeout=5.0
            )
            
            # Try with API key
            response_with_key = await client.get(
                f"{BACKEND_URL}/api/comparison",
                params={"ticker": "GC=F"},
                headers={"X-API-Key": "xgold-forecast-key-2026-3b7f8a9c2d1e5f6g"},
                timeout=5.0
            )
            
            if response_no_key.status_code == 200 and response_with_key.status_code == 200:
                if response_no_key.json() != response_with_key.json():
                    issues.append({
                        "type": "authentication",
                        "issue": "API key validation causes different responses",
                        "severity": "warning"
                    })
    except Exception as e:
        pass  # This is optional check
    
    return {
        "timestamp": datetime.now().isoformat(),
        "issues_found": len(issues),
        "issues": issues,
        "status": "healthy" if len(issues) == 0 else "issues_detected"
    }

if __name__ == "__main__":
    print("=" * 70)
    print("MCP Traffic Monitor Server")
    print("=" * 70)
    print(f"Starting MCP Server on http://localhost:{MCP_PORT}")
    print(f"Proxying traffic to: {BACKEND_URL}")
    print("")
    print("Available endpoints:")
    print("  - GET  /mcp/health              - MCP server health check")
    print("  - GET  /mcp/backend/health     - Backend health check")
    print("  - GET  /mcp/inspect/endpoints  - Inspect all backend endpoints")
    print("  - GET  /mcp/logs               - Get traffic logs")
    print("  - GET  /mcp/stats              - Get traffic statistics")
    print("  - POST /mcp/logs/clear         - Clear traffic logs")
    print("  - GET  /mcp/diagnose/format   - Diagnose data format issues")
    print("  - GET  /mcp/diagnose/communication - Diagnose communication")
    print("  - ANY /mcp/proxy/*             - Proxy and log traffic")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)
