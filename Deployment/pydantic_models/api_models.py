"""
Pydantic API Models for Request/Response Schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .schemas import (
    PriceData, PriceSummary, Forecast, ForecastSummary, 
    ModelMetrics, ModelComparison, DataFreshness, DataSummary,
    RetrainResponse, PredictionResponse, FeatureData, FeatureSet
)


class DataRequest(BaseModel):
    """Request model for data endpoints"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    period: str = Field(default="5y", description="Data period (e.g., '5y', '1y', '6m')")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date in YYYY-MM-DD format")
    interval: Optional[str] = Field(None, description="Data interval (e.g., '1d', '1h', '1m')")


class FeatureRequest(BaseModel):
    """Request model for feature endpoints"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")
    feature_set: Optional[List[str]] = Field(None, description="Specific features to include")


class MetricsRequest(BaseModel):
    """Request model for metrics calculation"""
    y_true: List[float] = Field(..., description="Actual values")
    y_pred: List[float] = Field(..., description="Predicted values")
    model_name: str = Field(..., description="Name of the model")
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")


class ForecastRequest(BaseModel):
    """Request model for forecast endpoints"""
    model_type: str = Field(..., description="Type of model (e.g., 'arima', 'lstm', 'prophet')")
    periods: int = Field(default=20, description="Number of periods to forecast")
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    period: str = Field(default="5y", description="Data period for training")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")
    confidence_level: float = Field(default=0.95, description="Confidence level for prediction bands")


class RetrainRequest(BaseModel):
    """Request model for model retraining"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    model_name: str = Field(default="arima", description="Model to retrain")
    train_ratio: float = Field(default=0.8, description="Training data ratio")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    model_name: str = Field(default="arima", description="Model name")
    horizon_days: int = Field(default=20, description="Number of days to predict")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")


class ModelComparisonRequest(BaseModel):
    """Request model for model comparison"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    models: Optional[List[str]] = Field(None, description="List of models to compare")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")


class ModelMetricsRequest(BaseModel):
    """Request model for specific model metrics"""
    model_name: str = Field(..., description="Model name")
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")


class DataFreshnessRequest(BaseModel):
    """Request model for data freshness check"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    max_age_days: int = Field(default=7, description="Maximum age in days")


class DataSummaryRequest(BaseModel):
    """Request model for data summary"""
    ticker: str = Field(default="GC=F", description="Stock ticker symbol")
    period: str = Field(default="5y", description="Data period")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date")


class APIResponse(BaseModel):
    """Base response model for API endpoints"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")


class MultiModelComparisonResponse(BaseModel):
    """Response model for multi-model comparison"""
    success: bool = Field(True)
    data: List[ModelComparison] = Field(..., description="List of model comparisons")
    message: Optional[str] = Field(None)


class ForecastResponse(BaseModel):
    """Response model for forecast endpoints"""
    success: bool = Field(True)
    data: List[Forecast] = Field(..., description="List of forecast data points")
    summary: Optional[ForecastSummary] = Field(None, description="Forecast summary")
    confidence_level: float = Field(..., description="Confidence level used")
    message: Optional[str] = Field(None)


class RetrainResponse(BaseModel):
    """Response model for model retraining"""
    success: bool = Field(True)
    data: RetrainResponse = Field(..., description="Retraining results")
    message: Optional[str] = Field(None)


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints"""
    success: bool = Field(True)
    data: List[float] = Field(..., description="Predicted values")
    model: str = Field(..., description="Model name")
    ticker: str = Field(..., description="Stock ticker symbol")
    message: Optional[str] = Field(None)


class DataResponse(BaseModel):
    """Response model for data endpoints"""
    success: bool = Field(True)
    data: List[PriceData] = Field(..., description="List of price data points")
    summary: Optional[PriceSummary] = Field(None, description="Price summary")
    message: Optional[str] = Field(None)


class FeatureResponse(BaseModel):
    """Response model for feature endpoints"""
    success: bool = Field(True)
    data: FeatureSet = Field(..., description="Feature data set")
    message: Optional[str] = Field(None)


class MetricsResponse(BaseModel):
    """Response model for metrics endpoints"""
    success: bool = Field(True)
    data: ModelMetrics = Field(..., description="Model performance metrics")
    message: Optional[str] = Field(None)


class DataFreshnessResponse(BaseModel):
    """Response model for data freshness endpoints"""
    success: bool = Field(True)
    data: DataFreshness = Field(..., description="Data freshness status")
    message: Optional[str] = Field(None)


class DataSummaryResponse(BaseModel):
    """Response model for data summary endpoints"""
    success: bool = Field(True)
    data: DataSummary = Field(..., description="Data summary")
    message: Optional[str] = Field(None)


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    message: Optional[str] = Field(None)


class Config:
    """Pydantic configuration"""
    use_enum_values: bool = True
    validate_assignment: bool = True