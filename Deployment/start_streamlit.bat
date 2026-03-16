@echo off
REM ============================================================
REM Local Streamlit Deployment - Setup & Run
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  Local Streamlit Deployment Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)


REM Ensure we are in the Deployment directory
cd /d %~dp0

REM Create virtual environment if needed
if not exist .venv (
    echo [INFO] Creating virtual environment in Deployment/.venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate venv
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo [INFO] Installing/updating dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies ready

REM Check models directory
if not exist models (
    echo.
    echo [WARNING] Models directory not found!
    echo Please export models from your notebook first:
    echo.
    echo   1. In Jupyter: Run the last cell (model export cell)
    echo   2. This will create models/ directory with all trained models
    echo.
)

REM Display API Key
echo.
echo ============================================================
echo Configuration
echo ============================================================
echo API Key: xgold-forecast-key-2026-3b7f8a9c2d1e5f6g
echo.
echo Two terminals will open:
echo   Terminal 1: FastAPI server (http://localhost:8000)
echo   Terminal 2: Streamlit dashboard (http://localhost:8501)
echo ============================================================
echo.

REM Start FastAPI
echo [INFO] Starting FastAPI backend...
start "FastAPI - Gold Forecast" cmd /k ".venv\Scripts\activate.bat && python utils\fastapi_backend.py"
timeout /t 3 /nobreak

REM Start Streamlit
echo [INFO] Starting Streamlit dashboard...
echo.
echo ============================================================
echo Dashboard should open automatically at:
echo http://localhost:8501
echo ============================================================
echo.

call .venv\Scripts\activate.bat && streamlit run app.py

pause
