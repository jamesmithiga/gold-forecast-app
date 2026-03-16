"""
Data Service for Data Operations

Encapsulates all data-related business logic including loading, processing, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from pydantic_models.schemas import DataSummary, DataFreshness
from utils.data_processing import (
    load_data, clean_data, get_latest_data, 
    get_intraday_data, check_data_freshness, get_data_summary
)

logger = logging.getLogger(__name__)


class DataService:
    """Service for data operations and management"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_data(self, ticker: str = 'GC=F', period: str = '5y',
                 cutoff_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (e.g., '5y', '1y', '6m')
            cutoff_date: Cutoff date in YYYY-MM-DD format
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data for {ticker} (period: {period})")
            
            # Load data
            data = load_data(ticker, period, cutoff_date)
            
            if data is None:
                raise ValueError(f"Failed to load data for {ticker}")
            
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame but got {type(data)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame, strategy: str = 'ffill') -> pd.DataFrame:
        """
        Clean data using specified strategy.
        
        Args:
            data: Input DataFrame
            strategy: Missing value strategy ('ffill', 'bfill', 'mean', 'drop')
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            logger.info(f"Cleaning data with strategy: {strategy}")
            
            cleaned_data = clean_data(data, strategy)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def get_latest_data(self, ticker: str = 'GC=F', days: int = 365) -> pd.DataFrame:
        """
        Get latest data for dashboard display.
        
        Args:
            ticker: Stock ticker
            days: Number of days to retrieve
        
        Returns:
            pd.DataFrame: Latest price data
        """
        try:
            logger.info(f"Getting latest data for {ticker} (last {days} days)")
            
            data = get_latest_data(ticker, days)
            
            if data is None:
                raise ValueError(f"Failed to get latest data for {ticker}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            raise
    
    def get_intraday_data(self, ticker: str = 'GC=F', period: str = '30d', 
                         interval: str = '1d') -> pd.DataFrame:
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
            logger.info(f"Getting intraday data for {ticker} (period: {period}, interval: {interval})")
            
            data = get_intraday_data(ticker, period, interval)
            
            if data is None:
                raise ValueError(f"Failed to get intraday data for {ticker}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting intraday data: {str(e)}")
            raise
    
    def check_data_freshness(self, ticker: str = 'GC=F', max_age_days: int = 7) -> Dict[str, Any]:
        """
        Check if the data is fresh enough for real-time insights.
        
        Args:
            ticker: Stock ticker
            max_age_days: Maximum age of data before prompting for refresh
        
        Returns:
            dict: Data freshness status
        """
        try:
            logger.info(f"Checking data freshness for {ticker} (max age: {max_age_days} days)")
            
            freshness = check_data_freshness(ticker, max_age_days)
            
            return freshness
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {str(e)}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame, ticker: str = 'GC=F') -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Stock ticker
        
        Returns:
            dict: Data summary statistics
        """
        try:
            logger.info(f"Generating data summary for {ticker}")
            
            summary = get_data_summary(data, ticker)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that data contains required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            logger.info(f"Validating data for required columns: {required_columns}")
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
    
    def get_commodity_data(self, commodity: str = "Gold") -> pd.DataFrame:
        """
        Get data for Gold only.
        
        Args:
            commodity: Commodity name (ignored, always Gold)
        
        Returns:
            pd.DataFrame: Data for Gold
        """
        try:
            ticker = "GC=F"
            logger.info(f"Getting data for Gold ({ticker})")
            
            data = self.get_latest_data(ticker)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting commodity data: {str(e)}")
            raise
    
    def get_multiple_commodities_data(self, commodities: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for Gold only (commodities parameter ignored).
        
        Args:
            commodities: List of commodity names (ignored, always returns Gold)
        
        Returns:
            dict: Dictionary with Gold data
        """
        try:
            logger.info(f"Getting data for Gold")
            
            data_dict = {}
            
            try:
                data = self.get_commodity_data("Gold")
                data_dict["Gold"] = data
            except Exception as e:
                logger.warning(f"Failed to get data for Gold: {str(e)}")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error getting commodity data: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        """
        Save data to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            format: File format ('csv', 'excel', 'json')
        
        Returns:
            str: Path to saved file
        """
        try:
            logger.info(f"Saving data to {filename} (format: {format})")
            
            filepath = f"{self.data_dir}/{filename}"
            
            if format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'excel':
                data.to_excel(filepath, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_saved_data(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load saved data from file.
        
        Args:
            filename: Input filename
            format: File format ('csv', 'excel', 'json')
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {filename} (format: {format})")
            
            filepath = f"{self.data_dir}/{filename}"
            
            if format == 'csv':
                data = pd.read_csv(filepath)
            elif format == 'excel':
                data = pd.read_excel(filepath)
            elif format == 'json':
                data = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise