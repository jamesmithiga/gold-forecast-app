
from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field

"""
Pydantic Schemas for Data Validation and API Models
"""

class ModelPrediction(BaseModel):
    """Schema for model prediction results"""
    model_name: str = Field(..., description="Name of the model")
    predictions: list = Field(..., description="List of predicted values")
    prediction_date: datetime = Field(..., description="Date of prediction")

class ModelEvaluation(BaseModel):
    """Schema for model evaluation results"""
    model_type: str = Field(..., description="Type of the model")
    model_name: Optional[str] = Field(default=None, description="Name of the model")
    target_column: str = Field(default="Close", description="Target column for evaluation")
    test_size: float = Field(default=0.2, description="Test size ratio")
    num_samples: int = Field(default=0, description="Number of samples")
    metrics: dict = Field(default_factory=dict, description="Metrics for the model")
    regression_metrics: dict = Field(default_factory=dict, description="Regression metrics for the model")
    evaluation_date: datetime = Field(..., description="Date of evaluation")

from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class PriceData(BaseModel):
    """Schema for price data points"""
    date: datetime = Field(..., description="Date of the price data")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price during the period")
    low: float = Field(..., description="Lowest price during the period")
    close: float = Field(..., description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")


class PriceSummary(BaseModel):
    """Schema for price summary statistics"""
    current_price: float = Field(..., description="Current price")
    min_price: float = Field(..., description="Minimum price in period")
    max_price: float = Field(..., description="Maximum price in period")
    avg_price: float = Field(..., description="Average price")
    std_dev: float = Field(..., description="Standard deviation of prices")
    price_change: float = Field(..., description="Price change from previous period")
    price_change_pct: float = Field(..., description="Price change percentage")


class Forecast(BaseModel):
    """Schema for forecast data"""
    date: datetime = Field(..., description="Forecast date")
    forecast: float = Field(..., description="Predicted price")
    lower_bound: Optional[float] = Field(None, description="Lower confidence bound")
    upper_bound: Optional[float] = Field(..., description="Upper confidence bound")


class ForecastSummary(BaseModel):
    """Schema for forecast summary"""
    current_price: float = Field(..., description="Current price")
    forecast_mean: float = Field(..., description="Average forecast value")
    forecast_high: float = Field(..., description="Highest forecast value")
    forecast_low: float = Field(..., description="Lowest forecast value")
    volatility: float = Field(..., description="Forecast volatility")


class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    model_name: str = Field(..., description="Name of the model")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R² Score")
    directional_accuracy: float = Field(..., description="Directional Accuracy Percentage")


class ModelComparison(BaseModel):
    """Schema for model comparison data"""
    model_name: str = Field(..., description="Name of the model")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R² Score")
    directional_accuracy: float = Field(..., description="Directional Accuracy Percentage")


class DataFreshness(BaseModel):
    """Schema for data freshness status"""
    ticker: str = Field(..., description="Stock ticker symbol")
    is_fresh: bool = Field(..., description="Whether data is fresh")
    last_update: Optional[datetime] = Field(None, description="Last update timestamp")
    days_since_update: Optional[int] = Field(None, description="Days since last update")
    current_price: Optional[float] = Field(None, description="Current price")
    message: str = Field(..., description="Status message")
    recommendation: str = Field(..., description="Action recommendation")


class DataSummary(BaseModel):
    """Schema for data summary"""
    ticker: str = Field(..., description="Stock ticker symbol")
    data_points: int = Field(..., description="Number of data points")
    start_date: datetime = Field(..., description="Start date of data")
    end_date: datetime = Field(..., description="End date of data")
    current_price: float = Field(..., description="Current price")
    min_price: float = Field(..., description="Minimum price")
    max_price: float = Field(..., description="Maximum price")
    avg_price: float = Field(..., description="Average price")
    std_dev: float = Field(..., description="Standard deviation")
    price_change: Optional[float] = Field(None, description="Price change")
    price_change_pct: Optional[float] = Field(None, description="Price change percentage")


class RetrainResponse(BaseModel):
    """Schema for model retraining response"""
    ticker: str = Field(..., description="Stock ticker symbol")
    model: str = Field(..., description="Model name")
    metrics: ModelMetrics = Field(..., description="Training metrics")
    train_samples: int = Field(..., description="Number of training samples")
    test_samples: int = Field(..., description="Number of test samples")


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    ticker: str = Field(..., description="Stock ticker symbol")
    model: str = Field(..., description="Model name")
    predictions: List[float] = Field(..., description="Predicted values")


class APIResponse(BaseModel):
    """Base schema for API responses"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Union[PriceData, Forecast, ModelMetrics, DataFreshness, DataSummary]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")


class MultiModelComparison(BaseModel):
    """Schema for multi-model comparison"""
    ticker: str = Field(..., description="Stock ticker symbol")
    models: List[ModelComparison] = Field(..., description="List of model comparisons")


class FeatureData(BaseModel):
    """Schema for feature data"""
    date: datetime = Field(..., description="Date of the feature data")
    features: dict = Field(..., description="Dictionary of feature values")
    target: Optional[float] = Field(None, description="Target value for supervised learning")


class FeatureSet(BaseModel):
    """Schema for a set of features"""
    features: List[FeatureData] = Field(..., description="List of feature data points")
    feature_names: List[str] = Field(..., description="List of feature names")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


class Config:
    """Pydantic configuration"""
    use_enum_values: bool = True
    validate_assignment: bool = True