"""
Data Processing Utilities for Gold Price Forecasting

Handles data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import yfinance as yf

logger = logging.getLogger(__name__)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns from yfinance data.
    
    Args:
        df: DataFrame potentially with MultiIndex columns
    
    Returns:
        DataFrame with flattened columns
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        logger.info(f"Flattened MultiIndex columns: {list(df.columns)}")
    return df


def extract_series(series_or_df: Any, default_col: str = 'Close') -> pd.Series:
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
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col
        else:
            return series_or_df.iloc[:, 0]
    return series_or_df


def load_data(ticker: str = 'GC=F', period: str = '5y', cutoff_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load gold price data from Yahoo Finance with gold data defaults.
    
    Args:
        ticker: Stock ticker (default: GC=F for Gold Futures)
        period: Period to download (default: 5 years)
        cutoff_date: Date to cut off data (format: YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: Clean dataframe with OHLCV data
    """
    try:
        logger.info(f"Loading {ticker} data from yfinance (period: {period})")
        
        # Download data
        raw = yf.download(ticker, period=period, progress=False)
        
        # Flatten MultiIndex columns if present
        raw = flatten_columns(raw)
        
        # Apply cut-off date if provided
        if cutoff_date:
            cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
            raw = raw.loc[raw.index <= cutoff]
        
        # Handle missing values (forward fill then drop)
        missing_before = raw.isnull().sum().sum()
        raw = raw.ffill().dropna()
        missing_after = raw.isnull().sum().sum()
        
        logger.info(f"Data loaded: {len(raw)} trading days")
        logger.info(f"Missing values - Before: {missing_before}, After: {missing_after}")
        
        # Log gold data statistics
        if ticker == 'GC=F' and len(raw) > 0:
            close_prices = raw['Close']
            mean_price = close_prices.mean()
            median_price = close_prices.median()
            std_dev = close_prices.std()
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * 100
            logger.info(f"Gold data statistics - Mean: ${mean_price:.2f}, Median: ${median_price:.2f}, StdDev: ${std_dev:.2f}, Volatility: {volatility:.2f}%")
        
        return raw
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame, strategy: str = 'ffill') -> pd.DataFrame:
    """
    Clean data by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        strategy: Missing value strategy ('ffill', 'bfill', 'mean', 'drop')
    
    Returns:
        Cleaned DataFrame
    """
    try:
        # Handle missing values
        if strategy == 'ffill':
            df = df.ffill()
        elif strategy == 'bfill':
            df = df.bfill()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'drop':
            df = df.dropna()
        
        # Handle outliers (optional)
        # This could be extended with more sophisticated outlier detection
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise


def get_latest_data(ticker: str = 'GC=F', days: int = 365) -> pd.DataFrame:
    """
    Get latest price data for dashboard display.
    
    Args:
        ticker: Stock ticker
        days: Number of days to retrieve
    
    Returns:
        pd.DataFrame: Price data with OHLCV
    """
    try:
        df = yf.download(ticker, period=f'{days}d', progress=False)
        df = flatten_columns(df)
        logger.info(f"Retrieved {len(df)} days of data for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving price data: {str(e)}")
        return None


def get_intraday_data(ticker: str = 'GC=F', period: str = '30d', interval: str = '1d') -> pd.DataFrame:
    """
    Get intraday or specific interval data.
    
    Args:
        ticker: Stock ticker
        period: Period to retrieve
        interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)
    
    Returns:
        pd.DataFrame: Price data at specified interval
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df = flatten_columns(df)
        logger.info(f"Retrieved {len(df)} {interval} candles for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving intraday data: {str(e)}")
        return None


def check_data_freshness(ticker: str = 'GC=F', max_age_days: int = 7) -> Dict[str, Any]:
    """
    Check if the data is fresh enough for real-time insights.
    
    Args:
        ticker: Stock ticker
        max_age_days: Maximum age of data before prompting for refresh
    
    Returns:
        dict: Data freshness status including last update date and recommendations
    """
    try:
        # Fetch the latest data
        df = yf.download(ticker, period='7d', progress=False)
        df = flatten_columns(df)
        
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


def get_data_summary(df: pd.DataFrame, ticker: str = 'GC=F') -> Dict[str, Any]:
    """
    Get comprehensive data summary.
    
    Args:
        df: DataFrame with OHLCV data
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