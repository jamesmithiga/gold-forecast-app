#!/bin/bash

# Gold Price Forecasting - Deployment Startup Script
# Author: James Mithiga
# Date: March 2026
# Usage: bash run_deployment.sh

set -e

echo ""
echo "============================================================"
echo "  Gold Price Forecasting Dashboard - Deployment Startup"
echo "  Student: James Mithiga (Admission No: 58200)"
echo "  Version: 1.0.0"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ using: brew install python3"
    exit 1
fi

echo "[OK] Python found"
python3 --version

# Check if virtual environment exists
if [ ! -d venv ]; then
    echo ""
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[INFO] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"

# Install dependencies
echo ""
echo "[INFO] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "[OK] Dependencies installed"

# Check models directory
echo ""
echo "[INFO] Checking models directory..."
if [ ! -d models ]; then
    echo "[WARNING] Models directory not found"
    echo "Please ensure trained models are exported to models/ directory"
    echo ""
    echo "To export models from your notebook, run:"
    echo "  import joblib"
    echo "  joblib.dump(model, 'models/GC=F_modelname.pkl')"
    echo ""
fi

# Display API Key
echo ""
echo "============================================================"
echo "  API Configuration"
echo "============================================================"
echo "API Key: xgold-forecast-key-2026-3b7f8a9c2d1e5f6g"
echo ""
echo "Add this header to API requests:"
echo "  X-API-Key: xgold-forecast-key-2026-3b7f8a9c2d1e5f6g"
echo "============================================================"
echo ""

# Start FastAPI in background
echo "[INFO] Starting FastAPI backend..."
python fastapi_backend.py &
FASTAPI_PID=$!

# Wait for FastAPI to start
echo "[INFO] Waiting for FastAPI to initialize (3 seconds)..."
sleep 3

# Test API connectivity
echo "[INFO] Testing API connectivity..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "[OK] FastAPI is running and healthy"
else
    echo "[WARNING] FastAPI may not have started successfully"
fi

# Start Streamlit
echo ""
echo "[INFO] Starting Streamlit dashboard..."
echo ""
echo "============================================================"
echo "  Dashboard URLs"
echo "============================================================"
echo "Streamlit Dashboard: http://localhost:8501"
echo "FastAPI Docs:        http://localhost:8000/docs"
echo "API Base URL:        http://localhost:8000"
echo "============================================================"
echo ""

streamlit run streamlit_dashboard.py

# Cleanup on exit
trap "kill $FASTAPI_PID" EXIT
