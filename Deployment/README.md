# Deployment Guide: Gold Price Forecasting Dashboard

## Overview
This guide provides step-by-step instructions to deploy the Gold Price Forecasting Dashboard, including both the FastAPI backend and Streamlit frontend. The deployment scripts are designed for local Windows environments.

---

## Prerequisites
- **Python 3.8+** installed and added to your PATH
- **Git** (optional, for version control)
- **Internet connection** (for installing dependencies)

---

## Directory Structure
```
Deployment/
    components/
    config/
    models/
    pages/
    pydantic_models/
    services/
    tests/
    utils/
    metrics.json
    requirements.txt
    run_deployment.bat
    start_streamlit.bat
    streamlit_dashboard.py
    ...
```

---

## Quick Start

### 1. Using `run_deployment.bat`
This script sets up the environment, installs dependencies, starts the FastAPI backend, and launches the Streamlit dashboard.

**Steps:**
1. Open a Command Prompt in the `Deployment` directory.
2. Run:
   ```
   run_deployment.bat
   ```
3. Follow on-screen instructions. Two terminals will open:
   - FastAPI backend (http://localhost:8000)
   - Streamlit dashboard (http://localhost:8501)

### 2. Using `start_streamlit.bat`
This script assumes dependencies are already installed and the environment is set up.

**Steps:**
1. Open a Command Prompt in the `Deployment` directory.
2. Run:
   ```
   start_streamlit.bat
   ```

---

## Notes
- Ensure the `models/` directory contains all required trained model files. Export models from your notebook if missing.
- API Key for requests: `xgold-forecast-key-2026-3b7f8a9c2d1e5f6g`
- For troubleshooting, check the output in the FastAPI and Streamlit terminals.

---

## Cleaning Up
- Remove any `__pycache__/` folders before packaging for production.
- Ensure no sensitive data is present in config files or notebooks.

---

## Contact
For issues, contact: James Mithiga (Admission No: 58200)
