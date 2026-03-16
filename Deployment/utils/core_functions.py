# Global list to accumulate all model evaluation metrics
results_list = []
"""
Core Functions Module - Single Source of Truth
Extracted from: POA_Final_Project_James_Mithiga_58200 copy.ipynb
All functions use real data from yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns from yfinance data.
    Converts MultiIndex columns to simple single-level columns.
    
    Args:
        df: DataFrame potentially with MultiIndex columns
    
    Returns:
        DataFrame with flattened columns
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns by taking the first level if it represents ticker information
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        logger.info(f"Flattened MultiIndex columns: {list(df.columns)}")
    return df


def _extract_series(series_or_df, default_col: str = 'Close') -> pd.Series:
    """
    Extract a Series from potentially MultiIndex data.
    
    Args:
        series_or_df: Series or DataFrame
        default_col: Default column name for DataFrame
    
    Returns:
        A clean pandas Series
    """
    if isinstance(series_or_df, pd.DataFrame):
        if default_col in series_or_df.columns:
            col = series_or_df[default_col]
            # Handle case where column is still a DataFrame due to MultiIndex issue by taking the first column
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col
        else:
            # Take the first available column
            return series_or_df.iloc[:, 0]
    return series_or_df

# Section containing core data loading and preprocessing functions

def load_and_prepare_data(ticker: str = 'GC=F', cutoff_date: str = None, period: str = '5y') -> pd.DataFrame:
    """
    Load and prepare gold price data from Yahoo Finance
    Mirrors notebook cell 9 functionality
    
    Args:
        ticker: Stock ticker (default: GC=F for Gold Futures)
        cutoff_date: Date to cut off data (format: YYYY-MM-DD)
        period: Period to download (default: 5 years)
    
    Returns:
        pd.DataFrame: Clean dataframe with Close prices
    """
    try:
        logger.info(f"Loading {ticker} data from yfinance (period: {period})")
        
        # Download data
        raw = yf.download(ticker, period=period, progress=False)
        
        # Flatten MultiIndex columns if present
        raw = _flatten_columns(raw)
        
        # Extract Close Price
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw['Close'][[ticker]].copy()
            df.columns = ['Close']
        else:
            df = raw[['Close']].copy()
        
        # Apply cut-off date if provided
        if cutoff_date is None:
            cutoff_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        df = df.loc[df.index <= cutoff_date]
        
        # Handle missing values (forward fill then drop)
        missing_before = df.isnull().sum().sum()
        df = df.ffill().dropna()
        missing_after = df.isnull().sum().sum()
        
        logger.info(f"Data loaded: {len(df)} trading days")
        logger.info(f"Missing values - Before: {missing_before}, After: {missing_after}")
        logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Section containing stationarity testing functions

def check_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    Perform Augmented Dickey-Fuller test
    Mirrors notebook cell 15 functionality
    
    Args:
        series: Time series to test
        name: Name of series for logging
    
    Returns:
        dict: Test results including p-value and stationarity status
    """
    try:
        series_clean = series.dropna()
        result = adfuller(series_clean)
        
        p_value = result[1]
        is_stationary = p_value <= 0.05
        
        status = 'Stationary' if is_stationary else 'Non-Stationary'
        
        logger.info(f"{name} ADF Test - p-value: {p_value:.4f} | {status}")
        
        return {
            'name': name,
            'p_value': p_value,
            'is_stationary': is_stationary,
            'status': status,
            'test_statistic': result[0],
            'critical_values': result[4]
        }
    
    except Exception as e:
        logger.error(f"Error in stationarity test: {str(e)}")
        raise

# Section containing feature engineering functions

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators and features for ML models
    Mirrors notebook cell 22 functionality
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: Enhanced dataframe with 18+ engineered features
    """
    try:
        import pandas_ta as ta
        data_ml = df.copy()
        data_ml.columns = [c.capitalize() for c in data_ml.columns]
        # Price action and lags
        logger.info("Creating lag features...")
        for lag in [1, 2, 3, 5, 10, 21]:
            data_ml[f'Lag_{lag}'] = data_ml['Close'].shift(lag)
        # Simple Momentum
        data_ml['Manual_Momentum_5'] = data_ml['Close'] - data_ml['Close'].shift(5)
        # Technical indicators using pandas_ta only
        logger.info("Creating technical indicators with pandas_ta...")
        data_ml.ta.rsi(length=14, append=True)
        data_ml.ta.macd(fast=12, slow=26, signal=9, append=True)
        data_ml.ta.sma(length=20, append=True)
        data_ml.ta.sma(length=50, append=True)
        data_ml.ta.ema(length=10, append=True)
        # pandas-ta equivalents for ROC and trendline
        data_ml['ROC_12'] = ta.roc(data_ml['Close'], length=12)
        # pandas-ta does not have direct Hilbert Transform, so omit HT_TRENDLINE and HT_DCPERIOD
        # Shift features by 1 to avoid look-ahead bias
        features = [c for c in data_ml.columns if c != 'Close']
        data_ml[features] = data_ml[features].shift(1)
        # Create target (next day up/down)
        data_ml['Target'] = (data_ml['Close'].shift(-1) > data_ml['Close']).astype(int)
        # Clean up NaNs from shifting
        data_ml = data_ml.dropna()
        
        logger.info(f"Features engineered: {len(data_ml.columns)} total columns")
        logger.info(f"Valid data rows: {len(data_ml)}")
        
        return data_ml
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

# Section containing train-test-forecast split functions

def create_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Create chronological train-test split (legacy function)
    
    Args:
        data: DataFrame with features
        train_ratio: Percentage of data for training (default: 0.8)
    
    Returns:
        tuple: (train_data, test_data) DataFrames
    """
    try:
        train_size = int(len(data) * train_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        logger.info(f"Train-Test Split created:")
        logger.info(f"  Training: {len(train_data)} days ({train_ratio*100:.0f}%)")
        logger.info(f"  Testing: {len(test_data)} days ({(1-train_ratio)*100:.0f}%)")
        
        return train_data, test_data
    
    except Exception as e:
        logger.error(f"Error in train-test split: {str(e)}")
        raise


def create_train_test_forecast_split(data: pd.DataFrame, 
                                    train_ratio: float = 0.65, 
                                    test_ratio: float = 0.20,
                                    forecast_ratio: float = 0.15) -> tuple:
    """
    Create chronological 3-way split: Train | Test | Forecast
    
    CRITICAL: This is the CORRECT data pipeline
    - Train: Historical data for model training
    - Test: Held-out data for model evaluation (NO future information)
    - Forecast: Future unseen periods for pure predictions
    
    Args:
        data: DataFrame with features or raw prices
        train_ratio: Percentage for training (default: 0.65)
        test_ratio: Percentage for testing (default: 0.20)
        forecast_ratio: Percentage for future forecasting (default: 0.15)
        
    Returns:
        tuple: (train_data, test_data, forecast_data, split_info) DataFrames
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
                'date_range': f"{train_data.index[0].date()} to {train_data.index[-1].date()}"
            },
            'test': {
                'samples': len(test_data),
                'percentage': test_ratio * 100,
                'date_range': f"{test_data.index[0].date()} to {test_data.index[-1].date()}"
            },
            'forecast': {
                'samples': len(forecast_data),
                'percentage': forecast_ratio * 100,
                'date_range': f"{forecast_data.index[0].date()} to {forecast_data.index[-1].date()}"
            }
        }
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ 3-WAY TRAIN-TEST-FORECAST SPLIT (CORRECT PIPELINE)")
        logger.info(f"{'='*70}")
        logger.info(f"Training:   {len(train_data):5d} days ({train_ratio*100:5.1f}%) | {split_info['train']['date_range']}")
        logger.info(f"Testing:    {len(test_data):5d} days ({test_ratio*100:5.1f}%) | {split_info['test']['date_range']}")
        logger.info(f"Forecast:   {len(forecast_data):5d} days ({forecast_ratio*100:5.1f}%) | {split_info['forecast']['date_range']}")
        logger.info(f"{'='*70}\n")
        
        return train_data, test_data, forecast_data, split_info
    
    except Exception as e:
        logger.error(f"Error in train-test-forecast split: {str(e)}")
        raise


def create_walk_forward_batches(data: pd.DataFrame, window_size: int = 252, step_size: int = 1):
    """
    Create walk-forward validation batches for time series evaluation
    
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

# Section containing data retrieval functions for dashboard

def get_latest_price_data(ticker: str = 'GC=F', days: int = 365) -> pd.DataFrame:
    """
    Get latest price data for dashboard display
    
    Args:
        ticker: Stock ticker
        days: Number of days to retrieve
    
    Returns:
        pd.DataFrame: Price data with OHLCV
    """
    try:
        df = yf.download(ticker, period=f'{days}d', progress=False)
        # Flatten MultiIndex columns if present
        df = _flatten_columns(df)
        logger.info(f"Retrieved {len(df)} days of data for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving price data: {str(e)}")
        return None

def get_intraday_data(ticker: str = 'GC=F', period: str = '30d', interval: str = '1d') -> pd.DataFrame:
    """
    Get intraday or specific interval data
    
    Args:
        ticker: Stock ticker
        period: Period to retrieve
        interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)
    
    Returns:
        pd.DataFrame: Price data at specified interval
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # Flatten MultiIndex columns if present
        df = _flatten_columns(df)
        logger.info(f"Retrieved {len(df)} {interval} candles for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving intraday data: {str(e)}")
        return None

# Section containing metrics calculation functions

def calculate_metrics(y_true, y_pred, model_name: str, ticker: str = 'GC=F') -> dict:
    """
    Calculate performance metrics
    Based on notebook cell 7 functionality
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        ticker: Stock ticker
    
    Returns:
        dict: Performance metrics (RMSE, MAE, MAPE, R2, etc.)
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    try:
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
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

# Section containing data summary functions

def get_data_summary(df: pd.DataFrame, ticker: str = 'GC=F') -> dict:
    """
    Get comprehensive data summary
    
    Args:
        df: DataFrame with Close prices
        ticker: Stock ticker
    
    Returns:
        dict: Data summary statistics
    """
    try:
        if 'Close' in df.columns:
            close_data = df['Close']
        else:
            close_data = df.iloc[:, 0]
        
        summary = {
            'ticker': ticker,
            'data_points': len(df),
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
            'current_price': float(close_data.iloc[-1]),
            'min_price': float(close_data.min()),
            'max_price': float(close_data.max()),
            'avg_price': float(close_data.mean()),
            'std_dev': float(close_data.std()),
        }
        
        if len(close_data) > 1:
            summary['price_change'] = float(close_data.iloc[-1] - close_data.iloc[-2])
            summary['price_change_pct'] = float((summary['price_change'] / close_data.iloc[-2]) * 100)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        raise

# Section containing data freshness check functions

def check_data_freshness(ticker: str = 'GC=F', max_age_days: int = 7) -> dict:
    """
    Check if the data is fresh enough for real-time insights
    
    Args:
        ticker: Stock ticker
        max_age_days: Maximum age of data before prompting for refresh
    
    Returns:
        dict: Data freshness status including last update date and recommendations
    """
    try:
        # Fetch the latest data
        df = yf.download(ticker, period='7d', progress=False)
        df = _flatten_columns(df)
        
        if df is None or len(df) == 0:
            return {
                'ticker': ticker,
                'is_fresh': False,
                'last_update': None,
                'days_since_update': None,
                'message': 'Unable to fetch data',
                'recommendation': 'Please check your internet connection and try again.'
            }
        
        # Get the last date in the data
        last_date = df.index[-1]
        current_date = datetime.now()
        days_since_update = (current_date - last_date).days
        
        # Determine freshness status
        is_fresh = days_since_update <= max_age_days
        
        if is_fresh:
            message = f"Data is up-to-date (last update: {last_date.strftime('%Y-%m-%d')})"
            recommendation = "Models are ready for real-time predictions"
        else:
            message = f"Data is {days_since_update} days old (last update: {last_date.strftime('%Y-%m-%d')})"
            recommendation = "Retrain models with latest data for accurate real-time insights"
        
        return {
            'ticker': ticker,
            'is_fresh': is_fresh,
            'last_update': last_date,
            'days_since_update': days_since_update,
            'current_price': float(df['Close'].iloc[-1]) if 'Close' in df.columns else float(df.iloc[-1, 0]),
            'message': message,
            'recommendation': recommendation
        }
    
    except Exception as e:
        logger.error(f"Error checking data freshness: {str(e)}")
        return {
            'ticker': ticker,
            'is_fresh': False,
            'last_update': None,
            'days_since_update': None,
            'message': f'Error: {str(e)}',
            'recommendation': 'Unable to verify data freshness'
        }


# Section containing forecast generation functions

def generate_forecast_future_periods(model, forecast_data: pd.DataFrame, 
                                    periods: int = 20, model_type: str = 'arima',
                                    max_periods: int = 30) -> pd.DataFrame:
    """
    Generate forecasts for FUTURE periods (periods NOT in training or test data)
    
    CRITICAL: Use only after training on TRAIN set and evaluating on TEST set
    This function generates predictions for the FORECAST period (future data)
    
    ⚠️ MAXIMUM FORECAST HORIZON: 30 days
    Longer forecasts have exponentially increasing uncertainty and lower accuracy
    
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
        
        logger.info(f"✓ Generated {len(forecast_df)} future forecasts")
        logger.info(f"  Period: {forecast_df['Date'].iloc[0].date()} to {forecast_df['Date'].iloc[-1].date()}")
        logger.info(f"  Forecast Range: {forecast_df['Forecast'].min():.4f} to {forecast_df['Forecast'].max():.4f}")
        
        return forecast_df
    
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise


def evaluate_model_on_test_set(y_true, y_pred, model_name: str, ticker: str = 'GC=F') -> dict:
    """
    CORRECT evaluation: evaluate model ONLY on the TEST set
    Never evaluate on training data or future forecast data
    
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
        
        metrics = calculate_metrics(y_true, y_pred, model_name, ticker)
        
        logger.info(f"✓ Test Set Performance:")
        logger.info(f"  RMSE:  {metrics['RMSE']:.4f}")
        logger.info(f"  MAE:   {metrics['MAE']:.4f}")
        logger.info(f"  MAPE:  {metrics['MAPE']:.2f}%")
        logger.info(f"  R²:    {metrics['R2 Score']:.4f}")
        logger.info(f"  DA:    {metrics['Directional Accuracy (%)']:.1f}%")
        logger.info(f"{'='*70}\n")
        
        # Always append metrics to the global results_list
        global results_list
        results_list.append(metrics)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

