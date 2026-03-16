"""
Feature Engineering Utilities for Machine Learning Models

Creates technical indicators and features for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import timedelta

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    ta = None
    PANDAS_TA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=length, min_periods=length).mean()
    avg_loss = losses.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "MACD_12_26_9": macd_line,
        "MACDs_12_26_9": signal_line,
        "MACDh_12_26_9": histogram,
    }


def _calculate_roc(series: pd.Series, length: int = 12) -> pd.Series:
    return series.pct_change(periods=length) * 100


def _append_manual_indicators(data_ml: pd.DataFrame) -> None:
    close = data_ml["Close"]
    data_ml["RSI_14"] = _calculate_rsi(close, length=14)
    macd_values = _calculate_macd(close, fast=12, slow=26, signal=9)
    for column_name, series in macd_values.items():
        data_ml[column_name] = series
    data_ml["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    data_ml["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    data_ml["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    data_ml["ROC_12"] = _calculate_roc(close, length=12)


def engineer_features(df: pd.DataFrame, feature_set: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create technical indicators and features for ML models.
    
    Args:
        df: DataFrame with OHLCV data
        feature_set: Specific features to include (optional)
    
    Returns:
        pd.DataFrame: Enhanced dataframe with engineered features
    """
    try:
        data_ml = df.copy()
        data_ml.columns = [c.capitalize() for c in data_ml.columns]
        
        # Price action and lags
        logger.info("Creating lag features...")
        for lag in [1, 2, 3, 5, 10, 21]:
            data_ml[f'Lag_{lag}'] = data_ml['Close'].shift(lag)
        
        # Simple Momentum
        data_ml['Manual_Momentum_5'] = data_ml['Close'] - data_ml['Close'].shift(5)
        
        if PANDAS_TA_AVAILABLE:
            logger.info("Creating technical indicators with pandas_ta...")
            data_ml.ta.rsi(length=14, append=True)
            data_ml.ta.macd(fast=12, slow=26, signal=9, append=True)
            data_ml.ta.sma(length=20, append=True)
            data_ml.ta.sma(length=50, append=True)
            data_ml.ta.ema(length=10, append=True)
            data_ml['ROC_12'] = ta.roc(data_ml['Close'], length=12)
        else:
            logger.warning("pandas_ta is not installed; using native pandas indicator calculations")
            _append_manual_indicators(data_ml)
        
        # Shift features by 1 to avoid look-ahead bias
        features = [c for c in data_ml.columns if c != 'Close']
        data_ml[features] = data_ml[features].shift(1)
        
        # Create target (next day up/down)
        data_ml['Target'] = (data_ml['Close'].shift(-1) > data_ml['Close']).astype(int)
        
        # Clean up NaNs from shifting
        data_ml = data_ml.dropna()
        
        # Filter to specific feature set if requested
        if feature_set:
            available_features = [c for c in data_ml.columns if c != 'Target']
            selected_features = [f for f in feature_set if f in available_features]
            if selected_features:
                data_ml = data_ml[['Target'] + selected_features]
        
        logger.info(f"Features engineered: {len(data_ml.columns)} total columns")
        logger.info(f"Valid data rows: {len(data_ml)}")
        
        return data_ml
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


def create_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Create chronological train-test split.
    
    Args:
        data: DataFrame with features
        train_ratio: Percentage of data for training (default: 0.8)
    
    Returns:
        dict: Train/test split data and metadata
    """
    try:
        train_size = int(len(data) * train_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        split_info = {
            'total_samples': len(data),
            'train': {
                'samples': len(train_data),
                'percentage': train_ratio * 100,
                'date_range': f"{train_data.index[0].date()} to {train_data.index[-1].date()}"
            },
            'test': {
                'samples': len(test_data),
                'percentage': (1 - train_ratio) * 100,
                'date_range': f"{test_data.index[0].date()} to {test_data.index[-1].date()}"
            }
        }
        
        logger.info(f"Train-Test Split created:")
        logger.info(f"  Training: {len(train_data)} days ({train_ratio*100:.0f}%)")
        logger.info(f"  Testing: {len(test_data)} days ({(1-train_ratio)*100:.0f}%)")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'split_info': split_info
        }
        
    except Exception as e:
        logger.error(f"Error in train-test split: {str(e)}")
        raise


def create_train_test_forecast_split(data: pd.DataFrame,
                                    target_column: str = 'Close',
                                    test_ratio: float = 0.2,
                                    train_ratio: float = 0.65,
                                    forecast_ratio: float = 0.15) -> Dict[str, Any]:
    """
    Create chronological 3-way split: Train | Test | Forecast.
    
    Args:
        data: DataFrame with features or raw prices
        train_ratio: Percentage for training (default: 0.65)
        test_ratio: Percentage for testing (default: 0.20)
        forecast_ratio: Percentage for future forecasting (default: 0.15)
    
    Returns:
        dict: Train/test/forecast split data and metadata
    """
    try:
        # Validate ratios sum to 1.0
        total_ratio = train_ratio + test_ratio + forecast_ratio
        if not np.isclose(total_ratio, 1.0):
            logger.warning(f"Ratios sum to {total_ratio:.2f}, normalizing...")
            train_ratio /= total_ratio
            test_ratio /= total_ratio
            forecast_ratio /= total_ratio
        
        total_len = len(data)
        train_size = int(total_len * train_ratio)
        test_size = int(total_len * test_ratio)
        forecast_size = total_len - train_size - test_size
        
        # Chronological split (preserves time order)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:train_size + test_size]
        forecast_data = data.iloc[train_size + test_size:]
        
        split_info = {
            'total_samples': total_len,
            'train': {
                'samples': len(train_data),
                'percentage': train_ratio * 100,
                'date_range': f"{train_data.index[0] if hasattr(train_data.index[0], 'date') else train_data.index[0]} to {train_data.index[-1] if hasattr(train_data.index[-1], 'date') else train_data.index[-1]}"
            },
            'test': {
                'samples': len(test_data),
                'percentage': test_ratio * 100,
                'date_range': f"{test_data.index[0] if hasattr(test_data.index[0], 'date') else test_data.index[0]} to {test_data.index[-1] if hasattr(test_data.index[-1], 'date') else test_data.index[-1]}"
            },
            'forecast': {
                'samples': len(forecast_data),
                'percentage': forecast_ratio * 100,
                'date_range': f"{forecast_data.index[0] if hasattr(forecast_data.index[0], 'date') else forecast_data.index[0]} to {forecast_data.index[-1] if hasattr(forecast_data.index[-1], 'date') else forecast_data.index[-1]}"
            }
        }
        
        logger.info(f"\n{'='*70}")
        logger.info("3-WAY TRAIN-TEST-FORECAST SPLIT (CORRECT PIPELINE)")
        logger.info(f"{'='*70}")
        logger.info(f"Training:   {len(train_data):5d} days ({train_ratio*100:5.1f}%) | {split_info['train']['date_range']}")
        logger.info(f"Testing:    {len(test_data):5d} days ({test_ratio*100:5.1f}%) | {split_info['test']['date_range']}")
        logger.info(f"Forecast:   {len(forecast_data):5d} days ({forecast_ratio*100:5.1f}%) | {split_info['forecast']['date_range']}")
        logger.info(f"{'='*70}\n")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'forecast_data': forecast_data,
            'split_info': split_info
        }
        
    except Exception as e:
        logger.error(f"Error in train-test-forecast split: {str(e)}")
        raise


def create_walk_forward_batches(data: pd.DataFrame, window_size: int = 252, step_size: int = 1):
    """
    Create walk-forward validation batches for time series evaluation.
    
    Walk-forward validation respects chronological order:
    - Never use future data to predict the past
    - Gradually expand training window
    
    Args:
        data: DataFrame with historical data
        window_size: Initial training window size (default: 252 trading days ~ 1 year)
        step_size: How many periods to step forward (default: 1)
    
    Yields:
        tuple: (train_batch, test_batch) for each iteration
    """
    try:
        total_len = len(data)
        
        for i in range(window_size, total_len, step_size):
            train_batch = data.iloc[:i]
            # Test batch is just the next step_size periods
            test_end = min(i + step_size, total_len)
            test_batch = data.iloc[i:test_end]
            
            if len(test_batch) > 0:
                yield train_batch, test_batch
        
        logger.info(f"Generated walk-forward batches: initial_window={window_size}, step={step_size}")
        
    except Exception as e:
        logger.error(f"Error creating walk-forward batches: {str(e)}")
        raise


def generate_forecast_future_periods(model, forecast_data: pd.DataFrame, 
                                    periods: int = 20, model_type: str = 'arima',
                                    max_periods: int = 30) -> pd.DataFrame:
    """
    Generate forecasts for FUTURE periods (periods NOT in training or test data).
    
    Args:
        model: Trained model object
        forecast_data: The forecast_data portion from 3-way split
        periods: Number of future periods to forecast (default: 20, max: 30)
        model_type: Type of model ('arima', 'sarima', 'prophet', 'lstm', 'rf', 'lr')
        max_periods: Maximum allowed forecast periods (default: 30 days)
    
    Returns:
        pd.DataFrame: Forecast with dates and predictions
    
    Raises:
        ValueError: If periods exceed max_periods limit
    """
    try:
        # Enforce maximum forecast horizon (30 days)
        if periods > max_periods:
            logger.warning(f"Requested {periods} periods exceeds maximum {max_periods} days")
            logger.warning(f"Capping forecast to {max_periods} days (confidence decreases with longer horizons)")
            periods = max_periods
        
        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING FORECASTS FOR FUTURE PERIODS ({periods} days ahead, max: {max_periods})")
        logger.info(f"Model Type: {model_type}")
        logger.info(f"{'='*70}")
        
        last_date = forecast_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        predictions = []
        
        if model_type.lower() in ['arima', 'sarima']:
            # For ARIMA/SARIMA: use get_forecast() method
            forecast_result = model.get_forecast(steps=periods)
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int(alpha=0.05)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': predictions,
                'Lower_CI': conf_int.iloc[:, 0].values,
                'Upper_CI': conf_int.iloc[:, 1].values
            })
            
        elif model_type.lower() == 'prophet':
            # For Prophet: use make_future_dataframe()
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            forecast = forecast.tail(periods)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast['yhat'].values,
                'Lower_CI': forecast['yhat_lower'].values,
                'Upper_CI': forecast['yhat_upper'].values
            })
            
        elif model_type.lower() in ['lstm', 'rf', 'lr']:
            # For ML models: generate simple forecasts based on last value trend
            last_value = forecast_data.iloc[-1, 0] if hasattr(forecast_data, 'iloc') else forecast_data[-1]
            
            # Calculate trend from last 10 periods
            if len(forecast_data) >= 10:
                recent_data = forecast_data.iloc[-10:, 0] if hasattr(forecast_data, 'iloc') else forecast_data[-10:]
                trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / 10 if hasattr(recent_data.iloc[-1], 'dtype') else (recent_data[-1] - recent_data[0]) / 10
            else:
                trend = 0
            
            # Generate predictions with trend
            predictions = np.array([last_value + trend * (i + 1) for i in range(periods)])
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': predictions,
                'Lower_CI': predictions * 0.95,
                'Upper_CI': predictions * 1.05
            })
        else:
            logger.warning(f"Unknown model type: {model_type}, generating simple forecasts")
            last_value = forecast_data.iloc[-1, 0]
            predictions = np.array([last_value] * periods)
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': predictions,
                'Lower_CI': predictions * 0.95,
                'Upper_CI': predictions * 1.05
            })
        
        logger.info(f"Generated {len(forecast_df)} future forecasts")
        logger.info(f"  Period: {forecast_df['Date'].iloc[0].date()} to {forecast_df['Date'].iloc[-1].date()}")
        logger.info(f"  Forecast Range: {forecast_df['Forecast'].min():.4f} to {forecast_df['Forecast'].max():.4f}")
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise


def evaluate_model_on_test_set(y_true, y_pred, model_name: str, ticker: str = 'GC=F') -> Dict[str, Any]:
    """
    CORRECT evaluation: evaluate model ONLY on the TEST set.
    Never evaluate on training data or future forecast data.
    
    Args:
        y_true: Actual test values
        y_pred: Model predictions on test set only
        model_name: Name of model
        ticker: Stock ticker
    
    Returns:
        dict: Test set performance metrics
    """
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATING {model_name.upper()} ON TEST SET ONLY")
        logger.info(f"{'='*70}")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE calculation
        nonzero_idx = y_true != 0
        mape = np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100 if np.any(nonzero_idx) else np.nan
        
        r2 = r2_score(y_true, y_pred)
        
        # Directional Accuracy
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100 if len(diff_true) > 0 else np.nan
        
        metrics = {
            'Ticker': ticker,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2,
            'Directional Accuracy (%)': da
        }
        
        logger.info(f"{model_name} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, DA: {da:.1f}%")
        logger.info(f"{'='*70}\n")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise