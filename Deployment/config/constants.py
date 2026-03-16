"""
Application Constants for Gold Price Forecasting
"""

from typing import Dict, List, Tuple, Final, Any
from pathlib import Path


# Commodity Configuration
COMMODITIES: Final[Dict[str, Dict[str, str]]] = {
    "Gold": {"symbol": "GC=F", "driver": "Safe-Haven & Inflation"},
}


# Available Models
AVAILABLE_MODELS: Final[Dict[str, str]] = {
    "LSTM": "lstm",
    "Prophet": "prophet",
    "ARIMA": "arima",
    "SARIMA": "sarima",
    "SARIMAX": "sarimax",
    "Linear Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}


# Model Information
MODEL_INFO: Final[Dict[str, Dict[str, str]]] = {
    "LSTM": {
        "type": "Deep Learning - Recurrent Neural Network",
        "pros": "Excellent for complex patterns, Captures long-term dependencies",
        "cons": "Requires more data, Computationally intensive",
        "best_for": "Volatile markets with complex patterns"
    },
    "Prophet": {
        "type": "Statistical - Additive Model",
        "pros": "Handles seasonal patterns, Works with limited data",
        "cons": "May miss sudden changes, Less flexible",
        "best_for": "Regular seasonal patterns"
    },
    "ARIMA": {
        "type": "Statistical - AutoRegressive Integrated Moving Average",
        "pros": "Theoretically sound, Good for stationary data",
        "cons": "Requires stationarity, Less flexible",
        "best_for": "Traditional time series"
    },
    "SARIMA": {
        "type": "Statistical - Seasonal ARIMA",
        "pros": "Handles seasonal patterns, Captures periodic trends",
        "cons": "More complex tuning, Computationally intensive",
        "best_for": "Time series with strong seasonal components"
    },
    "SARIMAX": {
        "type": "Statistical - Seasonal ARIMA with Exogenous variables",
        "pros": "Handles seasonality, Can include external factors",
        "cons": "Requires more data, Complex to configure",
        "best_for": "Seasonal data with external regressors"
    },
    "Linear Regression": {
        "type": "Statistical - Linear Regression",
        "pros": "Simple and interpretable, Fast to compute",
        "cons": "Assumes linear trends, Limited complexity",
        "best_for": "Simple trends and patterns"
    },
    "Random Forest": {
        "type": "Machine Learning - Ensemble",
        "pros": "Handles non-linearity, Robust to outliers",
        "cons": "Risk of overfitting, Less interpretable",
        "best_for": "Complex multivariate relationships"
    },
    "XGBoost": {
        "type": "Machine Learning - Gradient Boosting",
        "pros": "High performance, Handles missing data",
        "cons": "Computationally intensive, Complex tuning",
        "best_for": "High-dimensional data with complex relationships"
    },
}


# Metrics Information
METRICS_INFO: Final[Dict[str, Dict[str, str]]] = {
    "RMSE": {
        "description": "Root Mean Squared Error - lower is better",
        "range": "0 to ∞",
        "unit": "$"
    },
    "MAE": {
        "description": "Mean Absolute Error - lower is better",
        "range": "0 to ∞",
        "unit": "$"
    },
    "MAPE": {
        "description": "Mean Absolute Percentage Error - lower is better",
        "range": "0% to 100%",
        "unit": "%"
    },
    "R²": {
        "description": "Coefficient of Determination - higher is better",
        "range": "-∞ to 1",
        "unit": ""
    },
    "Directional Accuracy": {
        "description": "Percentage of correctly predicted price movements",
        "range": "0% to 100%",
        "unit": "%"
    },
}


# Chart Themes
CHART_THEMES: Final[List[str]] = [
    "plotly_white",
    "plotly_dark",
    "plotly",
    "seaborn",
    "ggplot2",
]


# Color Schemes
COLOR_SCHEMES: Final[Dict[str, List[str]]] = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "gold_trading": ["#FFD700", "#FFA500", "#FFC700", "#FFB300", "#FF9500", "#FF8C00", "#FF7F00", "#FF6B00", "#FF5500", "#FF4500"],
    "professional": ["#2E7D32", "#1976D2", "#D32F2F", "#7B1FA2", "#F57C00", "#689F38", "#039BE5", "#D81B60", "#FBC02D", "#616161"],
    "light": ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22", "#34495E", "#2C3E50", "#95A5A6"],
}


# File Extensions
VALID_FILE_EXTENSIONS: Final[List[str]] = [
    ".csv", ".xlsx", ".json", ".parquet", 
    ".pkl", ".pickle", ".txt", ".log"
]


# Data Intervals
VALID_INTERVALS: Final[List[str]] = [
    "1m", "5m", "15m", "30m", 
    "1h", "2h", "4h", "1d", 
    "1w", "1mo", "3mo"
]


# Time Periods
VALID_PERIODS: Final[List[str]] = [
    "1d", "5d", "1mo", "3mo", 
    "6mo", "1y", "2y", "5y", 
    "10y", "ytd", "max"
]


# Default Chart Settings
DEFAULT_CHART_SETTINGS: Final[Dict[str, Any]] = {
    "height": 600,
    "template": "plotly_white",
    "hovermode": "x unified",
    "legend": {"x": 0.01, "y": 0.99},
    "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
}


# Default Statistical Settings
DEFAULT_STATS_SETTINGS: Final[Dict[str, Any]] = {
    "window_size": 20,
    "confidence_level": 0.95,
    "volatility_adjustment": 1.0,
    "trend_strength": 0.7,
    "smoothing_factor": 0.3,
}


# Default Data Processing Settings
DEFAULT_PROCESSING_SETTINGS: Final[Dict[str, Any]] = {
    "missing_value_strategy": "ffill",
    "outlier_threshold": 3.0,
    "normalization": "z-score",
    "feature_engineering": True,
    "stationarity_test": True,
}


# Default Model Training Settings
DEFAULT_TRAINING_SETTINGS: Final[Dict[str, Any]] = {
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "random_state": 42,
    "early_stopping": True,
    "cross_validation": False,
}


# Default Forecast Settings
DEFAULT_FORECAST_SETTINGS: Final[Dict[str, Any]] = {
    "periods": 20,
    "confidence_level": 0.95,
    "include_historical": True,
    "include_confidence_bands": True,
    "show_trend_lines": True,
}


# Status Messages
STATUS_MESSAGES: Final[Dict[str, str]] = {
    "loading": "🔍 Loading data...",
    "processing": "⚠️ Processing data...",
    "training": "🚀 Training model...",
    "forecasting": "📊 Generating forecast...",
    "success": "✅ Success",
    "error": "❌ Error",
    "warning": "⚠️ Warning",
    "info": "ℹ️ Information",
}


# Error Messages
ERROR_MESSAGES: Final[Dict[str, str]] = {
    "data_fetch_failed": "Failed to fetch data from source",
    "model_training_failed": "Model training failed",
    "forecast_generation_failed": "Forecast generation failed",
    "api_connection_failed": "API connection failed",
    "invalid_input": "Invalid input provided",
    "missing_data": "Required data is missing",
    "permission_denied": "Permission denied",
    "timeout": "Request timed out",
}


# File Paths
BASE_DIR: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"
CONFIG_DIR: Final[Path] = BASE_DIR / "config"
PYDANTIC_MODELS_DIR: Final[Path] = BASE_DIR / "pydantic_models"
UTILS_DIR: Final[Path] = BASE_DIR / "utils"
SERVICES_DIR: Final[Path] = BASE_DIR / "services"
API_DIR: Final[Path] = BASE_DIR / "api"
STREAMLIT_DIR: Final[Path] = BASE_DIR / "streamlit"
TESTS_DIR: Final[Path] = BASE_DIR / "tests"


# Directory Structure
DIR_STRUCTURE: Final[Dict[str, Path]] = {
    "base": BASE_DIR,
    "data": DATA_DIR,
    "models": MODELS_DIR,
    "logs": LOGS_DIR,
    "config": CONFIG_DIR,
    "pydantic_models": PYDANTIC_MODELS_DIR,
    "utils": UTILS_DIR,
    "services": SERVICES_DIR,
    "api": API_DIR,
    "streamlit": STREAMLIT_DIR,
    "tests": TESTS_DIR,
}


# Default File Names
DEFAULT_FILES: Final[Dict[str, str]] = {
    "config": ".env",
    "log": "app.log",
    "data": "data.csv",
    "model": "model.pkl",
    "metrics": "metrics.json",
    "forecast": "forecast.csv",
}


# HTTP Status Codes
HTTP_STATUS_CODES: Final[Dict[str, int]] = {
    "OK": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "METHOD_NOT_ALLOWED": 405,
    "CONFLICT": 409,
    "UNPROCESSABLE_ENTITY": 422,
    "INTERNAL_SERVER_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503,
}


# MIME Types
MIME_TYPES: Final[Dict[str, str]] = {
    "json": "application/json",
    "csv": "text/csv",
    "excel": "application/vnd.ms-excel",
    "parquet": "application/octet-stream",
    "html": "text/html",
    "text": "text/plain",
    "javascript": "application/javascript",
    "css": "text/css",
}


# Cache Durations (in seconds)
CACHE_DURATIONS: Final[Dict[str, int]] = {
    "short": 60,           # 1 minute
    "medium": 300,         # 5 minutes
    "long": 3600,          # 1 hour
    "very_long": 86400,    # 1 day
    "forever": 31536000,   # 1 year
}


# Default Values
DEFAULT_VALUES: Final[Dict[str, Any]] = {
    "confidence_level": 0.95,
    "train_ratio": 0.8,
    "forecast_periods": 20,
    "volatility_adjustment": 1.0,
    "trend_strength": 0.7,
    "smoothing_factor": 0.3,
    "missing_value_strategy": "ffill",
    "outlier_threshold": 3.0,
    "normalization": "z-score",
    "feature_engineering": True,
    "stationarity_test": True,
    "early_stopping": True,
    "cross_validation": False,
}


# Gold Data Defaults (from yfinance GC=F 1-year data)
GOLD_DATA_DEFAULTS: Final[Dict[str, float]] = {
    "mean_price": 3858.46,
    "median_price": 3643.70,
    "std_dev": 653.11,
    "volatility": 1.66,  # in percentage
}