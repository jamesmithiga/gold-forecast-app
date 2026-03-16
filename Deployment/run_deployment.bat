@echo off
REM Gold Price Forecasting - Deployment Startup Script
REM Author: James Mithiga
REM Date: March 2026

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  Gold Price Forecasting Dashboard - Deployment Startup
echo  Student: James Mithiga (Admission No: 58200)
echo  Version: 1.0.0
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo [OK] Python found
python --version


REM Ensure we are in the Deployment directory
cd /d %~dp0

REM Check if virtual environment exists in Deployment
if not exist .venv (
    echo [INFO] Creating virtual environment in Deployment/.venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Install dependencies
echo [INFO] Installing dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Check models directory
echo.
echo [INFO] Checking models directory...
if not exist models (
    echo [WARNING] Models directory not found
    echo Please ensure trained models are exported to models/ directory
    echo.
    echo To export models from your notebook, run:
    echo   import joblib
    echo   joblib.dump(model, 'models/GC=F_modelname.pkl')
    echo.
)

REM Display API Key
echo.
echo ============================================================
echo  API Configuration
echo ============================================================
echo API Key: xgold-forecast-key-2026-3b7f8a9c2d1e5f6g
echo.
echo Add this header to API requests:
echo   X-API-Key: xgold-forecast-key-2026-3b7f8a9c2d1e5f6g
echo ============================================================
echo.

REM Start FastAPI in background
echo [INFO] Starting FastAPI backend...
timeout /t 2 /nobreak
start "FastAPI Backend - Gold Forecast" cmd /k ".venv\Scripts\activate.bat && python utils\fastapi_backend.py"

REM Wait for FastAPI to start
echo [INFO] Waiting for FastAPI to initialize (5 seconds)...
timeout /t 5 /nobreak

REM Test API connectivity
echo [INFO] Testing API connectivity...
curl -s http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo [WARNING] FastAPI may not have started successfully
    echo Please check the FastAPI window for errors
) else (
    echo [OK] FastAPI is running and healthy
)

REM Start Streamlit
echo.
echo [INFO] Starting Streamlit dashboard...
echo.
echo ============================================================
echo  Dashboard URLs
echo ============================================================
echo Streamlit Dashboard: http://localhost:8501
echo FastAPI Docs:        http://localhost:8000/docs
echo API Base URL:        http://localhost:8000
echo ============================================================
echo.


REM Start Streamlit dashboard
echo [INFO] Starting Streamlit dashboard...
call .venv\Scripts\activate.bat && streamlit run app.py

pause
