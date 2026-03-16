"""
Configuration Management for Gold Price Forecasting Application
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    API_BASE_URL: str = Field(default="http://localhost:8000", description="Base URL for API endpoints")
    API_KEY: str = Field(default="xgold-forecast-key-2026-3b7f8a9c2d1e5f6g", description="API authentication key")
    API_TIMEOUT: int = Field(default=30, description="API request timeout in seconds")
    
    # Data Configuration
    DEFAULT_TICKER: str = Field(default="GC=F", description="Default stock ticker symbol")
    DEFAULT_PERIOD: str = Field(default="5y", description="Default data period")
    DEFAULT_INTERVAL: str = Field(default="1d", description="Default data interval")
    MAX_DATA_AGE_DAYS: int = Field(default=7, description="Maximum data age before refresh")
    
    # Model Configuration
    DEFAULT_MODEL: str = Field(default="arima", description="Default forecasting model")
    DEFAULT_FORECAST_PERIODS: int = Field(default=20, description="Default forecast periods")
    MAX_FORECAST_PERIODS: int = Field(default=30, description="Maximum allowed forecast periods")
    DEFAULT_TRAIN_RATIO: float = Field(default=0.8, description="Default training data ratio")
    
    # FastAPI Configuration
    FASTAPI_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    FASTAPI_PORT: int = Field(default=8000, description="FastAPI port")
    ENABLE_CORS: bool = Field(default=True, description="Enable Cross-Origin Resource Sharing")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")
    
    # Streamlit Configuration
    STREAMLIT_PAGE_TITLE: str = Field(default="Gold Price Forecasting Dashboard", description="Streamlit page title")
    STREAMLIT_LAYOUT: str = Field(default="wide", description="Streamlit layout mode")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: Optional[str] = Field(None, description="Path to log file")
    
    # File Paths
    DATA_DIR: Path = Field(default=Path("data"), description="Directory for data files")
    MODELS_DIR: Path = Field(default=Path("models"), description="Directory for model files")
    LOGS_DIR: Path = Field(default=Path("logs"), description="Directory for log files")
    
    # External Services
    YAHOO_FINANCE_ENABLED: bool = Field(default=True, description="Enable Yahoo Finance data source")
    YAHOO_FINANCE_RETRIES: int = Field(default=3, description="Number of retries for Yahoo Finance")
    YAHOO_FINANCE_RETRY_DELAY: int = Field(default=1, description="Delay between retries in seconds")
    
    # Performance Configuration
    MAX_BATCH_SIZE: int = Field(default=1000, description="Maximum batch size for processing")
    WORKER_THREADS: int = Field(default=4, description="Number of worker threads")
    
    # Security Configuration
    ENABLE_API_AUTH: bool = Field(default=True, description="Enable API authentication")
    API_AUTH_HEADER: str = Field(default="X-API-Key", description="API authentication header")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        case_sensitive = True
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            """Customise settings sources"""
            return (
                init_settings,
                env_settings,
                file_secret_settings
            )


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def update_settings(**kwargs) -> None:
    """Update settings with new values"""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise AttributeError(f"Invalid setting: {key}")


def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary"""
    return settings.model_dump()