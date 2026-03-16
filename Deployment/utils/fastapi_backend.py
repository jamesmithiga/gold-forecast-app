
"""
FastAPI Backend for Gold Price Forecasting Dashboard
Exposes endpoints for data, features, metrics, and forecasting
"""

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import sys
import os
import logging
import pandas as pd
import numpy as np

# Add parent directory to path for service imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_service import DataService
from services.model_service import ModelService
from services.forecast_service import ForecastService
from utils.feature_engineering import engineer_features
from utils.metrics import calculate_metrics
from utils.core_functions import get_data_summary

# Initialize services
from config.settings import Settings
from config.constants import COMMODITIES

logger = logging.getLogger(__name__)

config = Settings()
data_service = DataService(data_dir=config.DATA_DIR)
model_service = ModelService()
forecast_service = ForecastService(model_service, data_service)

app = FastAPI(title="Gold Price Forecasting API", version="1.0")

# Allow CORS for local frontend
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

class DataRequest(BaseModel):
	ticker: str = 'GC=F'
	period: str = '5y'
	cutoff_date: Optional[str] = None

class FeatureRequest(BaseModel):
	ticker: str = 'GC=F'
	period: str = '5y'
	cutoff_date: Optional[str] = None

class MetricsRequest(BaseModel):
	y_true: List[float]
	y_pred: List[float]
	model_name: str
	ticker: str = 'GC=F'

class ForecastRequest(BaseModel):
	model_type: str
	periods: int = 20
	ticker: str = 'GC=F'
	period: str = '5y'
	cutoff_date: Optional[str] = None


# --- API ROUTES ---
@app.get("/health")
def health():
	return {"status": "ok"}

# Move all endpoints under /api for consistency
@app.post("/api/data")
def get_data(req: DataRequest):
	try:
		data = data_service.load_data(ticker=req.ticker, period=req.period, cutoff_date=req.cutoff_date)
		return data.reset_index().to_dict(orient="records")
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/features")
def get_features(req: FeatureRequest):
	try:
		data = data_service.load_data(ticker=req.ticker, period=req.period, cutoff_date=req.cutoff_date)
		features = engineer_features(data)
		return features.reset_index().to_dict(orient="records")
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics")
def get_metrics(req: MetricsRequest):
	try:
		metrics = calculate_metrics(req.y_true, req.y_pred, req.model_name, req.ticker)
		return metrics
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/summary")
def data_summary(ticker: str = 'GC=F', period: str = '5y', cutoff_date: Optional[str] = None):
	try:
		data = data_service.load_data(ticker=ticker, period=period, cutoff_date=cutoff_date)
		summary = get_data_summary(data, ticker)
		return summary
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/freshness")
def data_freshness(ticker: str = 'GC=F', max_age_days: int = 7):
	try:
		freshness = data_service.check_data_freshness(ticker, max_age_days)
		return freshness
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast")
def forecast(req: ForecastRequest):
	try:
		forecast_result = forecast_service.generate_forecast(
			model_type=req.model_type, ticker=req.ticker, 
			periods=req.periods, confidence_level=0.95
		)
		return forecast_result
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# --- New endpoints using service layer ---
@app.post("/api/retrain")
def retrain_model(payload: dict = Body(...)):
	try:
		ticker = payload.get("ticker", "GC=F")
		model_name = payload.get("model_name", None)  # Get the specific model to retrain
		requested_train_ratio = payload.get("train_ratio", 0.2)
		force_retrain = payload.get("force_retrain", False)  # Option to force full retraining
		
		# Validate payload
		if not isinstance(payload, dict):
			raise ValueError(f"Expected dict payload but got {type(payload)}")
		
		# Load data
		data = data_service.load_data(ticker=ticker)
		
		# Validate data is a DataFrame
		if not isinstance(data, pd.DataFrame):
			raise ValueError(f"Expected DataFrame from data_service but got {type(data)}")
		
		# Determine which models to retrain
		if model_name:
			# Map frontend model names to internal names
			model_aliases = {
				"lstm": "LSTM",
				"prophet": "Prophet",
				"arima": "ARIMA",
				"sarima": "SARIMA",
				"lr": "LinearRegression",
				"linearregression": "LinearRegression",
				"rf": "RandomForest",
				"randomforest": "RandomForest"
			}
			model_type = model_aliases.get(str(model_name).lower(), model_name)
			
			# Check if saved model exists - use saved model for faster evaluation UNLESS force_retrain is True
			model_path = model_service._get_model_path(model_type, "Close", ticker)
			saved_model_exists = os.path.exists(model_path)
			
			if saved_model_exists and not force_retrain:
				# Use saved model - fast evaluation on fresh data
				logger.info(f"Using saved {model_type} model for fast evaluation (no full retraining)")
				try:
					result = model_service._evaluate_saved_model(
						model_type=model_type,
						data=data,
						target_column="Close",
						test_size=requested_train_ratio,
						ticker=ticker
					)
					
					# Get metrics from result
					raw_metrics = result.get('metrics', {})
					
					# Helper function to convert NaN to None for JSON serialization
					def _json_safe(val):
						if val is None or (isinstance(val, float) and np.isnan(val)):
							return None
						return float(val) if isinstance(val, (np.floating, np.integer)) else val
					
					# Convert metrics to format expected by frontend
					processed_metrics = {
						'rmse': _json_safe(raw_metrics.get('RMSE')),
						'mae': _json_safe(raw_metrics.get('MAE')),
						'mape': _json_safe(raw_metrics.get('MAPE')),
						'r2': _json_safe(raw_metrics.get('R2 Score')),
						'directional_accuracy': _json_safe(raw_metrics.get('Directional Accuracy (%)'))
					}
					
					# Return response with metrics at top level
					response = {
						'status': 'success',
						'ticker': ticker,
						'model_name': model_type,
						'metrics': processed_metrics,
						'metrics_by_model': {model_type: processed_metrics},
						'train_samples': result.get('X_train_shape', [0])[0] if result.get('X_train_shape') else 0,
						'test_samples': result.get('X_test_shape', [0])[0] if result.get('X_test_shape') else 0,
						'used_saved_model': True,
						'source': result.get('source', 'saved_model')
					}
					
					# Update metrics.json with new evaluation results
					try:
						model_service.export_metrics_to_json(
							[{'model_type': model_type, 'metrics': raw_metrics}],
							metrics_path="metrics.json",
							ticker=ticker
						)
						logger.info(f"Updated metrics.json with {model_type} evaluation results")
					except Exception as e:
						logger.warning(f"Could not update metrics.json: {str(e)}")
					
					return response
				except Exception as e:
					logger.warning(f"Error evaluating saved model {model_type}: {str(e)}, falling back to full retrain")
					# Fall through to full retrain if evaluation fails
			else:
				# No saved model or force_retrain=True - do full retraining
				logger.info(f"Full retraining {model_type} model (force_retrain={force_retrain}, saved_exists={saved_model_exists})")
			try:
				result = model_service.train_model(
					model_type=model_type,
					data=data,
					target_column="Close",
					test_size=requested_train_ratio,
					ticker=ticker
				)
				
				# Get metrics from result
				raw_metrics = result.get('metrics', {})
				
				# Helper function to convert NaN to None for JSON serialization
				def _json_safe(val):
					if val is None or (isinstance(val, float) and np.isnan(val)):
						return None
					return float(val) if isinstance(val, (np.floating, np.integer)) else val
				
				# Convert metrics to format expected by frontend (lowercase keys)
				processed_metrics = {
					'rmse': _json_safe(raw_metrics.get('RMSE')),
					'mae': _json_safe(raw_metrics.get('MAE')),
					'mape': _json_safe(raw_metrics.get('MAPE')),
					'r2': _json_safe(raw_metrics.get('R2 Score')),
					'directional_accuracy': _json_safe(raw_metrics.get('Directional Accuracy (%)'))
				}
				
				# Return response with metrics at top level for backwards compatibility
				# Also include model-keyed metrics for new frontend code
				response = {
					'status': 'success',
					'ticker': ticker,
					'model_name': model_type,
					'metrics': processed_metrics,  # Top level for backwards compatibility
					'metrics_by_model': {model_type: processed_metrics},  # For new frontend
					'train_samples': result.get('X_train_shape', [0])[0] if result.get('X_train_shape') else 0,
					'test_samples': result.get('X_test_shape', [0])[0] if result.get('X_test_shape') else 0,
					'used_saved_model': False
				}
				return response
			except Exception as e:
				logger.error(f"Error retraining model {model_type}: {str(e)}")
				raise HTTPException(status_code=500, detail=f"Error retraining {model_type}: {str(e)}")
		else:
			# No model specified - retrain all models (original behavior)
			logger.info(f"Retraining all models for {ticker}")
			result = model_service.retrain_all_models(data, target_column="Close", ticker=ticker)
			logger.info(f"Models retrained for {ticker}")
			
			# Extract metrics from results for API response
			if isinstance(result, dict) and 'results' in result:
				# Build response with metrics
				response = {
					'status': 'success',
					'ticker': ticker,
					'models_trained': result.get('models_trained', 0),
					'models_failed': result.get('models_failed', 0),
					'metrics': {}
				}
				
				# Extract metrics from each result
				for res in result.get('results', []):
					if 'error' not in res and 'metrics' in res:
						model_type = res.get('model_type', 'Unknown')
						response['metrics'][model_type] = res['metrics']
				
				return response
			
			return result

	except Exception as e:
		logger.error(f"Error retraining models: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict(payload: dict = Body(...)):
	try:
		ticker = payload.get("ticker", "GC=F")
		model_name = payload.get("model_name", "ARIMA")
		model_aliases = {
			"arima": "ARIMA",
			"sarima": "SARIMA",
			"prophet": "Prophet",
			"lr": "LinearRegression",
			"linear_regression": "LinearRegression",
			"rf": "RandomForest",
			"random_forest": "RandomForest",
			"lstm": "LSTM"
		}
		model_type = model_aliases.get(str(model_name).lower(), model_name)
		data = data_service.load_data(ticker=ticker)
		result = model_service.predict(
			model_type=model_type,
			data=data,
			target_column="Close",
			ticker=ticker
		)
		logger.info(f"Prediction generated for {model_type} on {ticker}")
		return result
	except Exception as e:
		logger.error(f"Error generating prediction: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/comparison")
def model_comparison(ticker: str = 'GC=F'):
	try:
		data = data_service.load_data(ticker=ticker)
		evaluations = model_service.compare_models(
			model_service.get_default_models(), data, target_column="Close"
		)
		summary = model_service.get_model_performance_summary(evaluations)
		logger.info(f"Model comparison completed for {ticker}: {summary}")
		return summary
	except Exception as e:
		logger.error(f"Error in model comparison: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{model_name}")
def get_model_metrics(model_name: str, ticker: str = 'GC=F'):
	try:
		model_aliases = {
			"arima": "ARIMA",
			"sarima": "SARIMA",
			"prophet": "Prophet",
			"lr": "LinearRegression",
			"linear_regression": "LinearRegression",
			"rf": "RandomForest",
			"random_forest": "RandomForest",
			"lstm": "LSTM"
		}
		model_type = model_aliases.get(str(model_name).lower(), model_name)
		data = data_service.load_data(ticker=ticker)
		evaluation = model_service.evaluate_model(model_type, data, target_column="Close", ticker=ticker)
		result = {
			"ticker": ticker,
			"model": model_type,
			"rmse": evaluation.regression_metrics.get("rmse", 0),
			"mae": evaluation.regression_metrics.get("mae", 0),
			"mape": evaluation.regression_metrics.get("mape", 0),
			"r2": evaluation.regression_metrics.get("r2", 0),
			"directional_accuracy": evaluation.regression_metrics.get("directional_accuracy", 0),
			"evaluation_date": evaluation.evaluation_date
		}
		logger.info(f"Metrics retrieved for {model_type} on {ticker}")
		return result
	except Exception as e:
		logger.error(f"Error retrieving metrics for {model_name}: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)
