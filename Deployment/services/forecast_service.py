"""
Forecast Service for Prediction and Analysis

Encapsulates all forecasting-related business logic including prediction generation and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from pydantic_models.schemas import Forecast, ForecastSummary
from utils.feature_engineering import generate_forecast_future_periods
from utils.data_processing import get_latest_data

logger = logging.getLogger(__name__)


class ForecastService:
    """Service for forecast generation and analysis"""
    
    def __init__(self, model_service: 'ModelService', data_service: 'DataService'):
        self.model_service = model_service
        self.data_service = data_service
        self.default_forecast_periods = 20
        self.max_forecast_periods = 30
    
    def generate_forecast(self, model_type: str, ticker: str = 'GC=F', 
                         periods: int = 20, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecast for a specific model and ticker.
        
        Args:
            model_type: Type of model to use for prediction
            ticker: Stock ticker symbol
            periods: Number of periods to predict
            confidence_level: Confidence level for prediction bands
        
        Returns:
            dict: Forecast results
        """
        try:
            logger.info(f"Generating forecast for {ticker} using {model_type} model")
            
            # Get latest data
            data = self.data_service.get_latest_data(ticker)
            
            # Engineer features
            from utils.feature_engineering import engineer_features
            features = engineer_features(data)
            
            # Get forecast data
            from utils.feature_engineering import create_train_test_forecast_split
            _, _, forecast_data, _ = create_train_test_forecast_split(features)
            
            # Generate forecasts
            from utils.feature_engineering import generate_forecast_future_periods
            forecast_df = generate_forecast_future_periods(
                model_type=model_type, forecast_data=forecast_data, 
                periods=periods, max_periods=self.max_forecast_periods
            )
            
            # Calculate summary statistics
            current_price = float(data['Close'].iloc[-1]) if 'Close' in data.columns else float(data.iloc[-1, 0])
            forecast_mean = forecast_df['Forecast'].mean()
            forecast_high = forecast_df['Forecast'].max()
            forecast_low = forecast_df['Forecast'].min()
            volatility = forecast_df['Forecast'].std()
            
            forecast_summary = {
                'current_price': current_price,
                'forecast_mean': forecast_mean,
                'forecast_high': forecast_high,
                'forecast_low': forecast_low,
                'volatility': volatility,
                'price_change': forecast_mean - current_price,
                'price_change_pct': ((forecast_mean - current_price) / current_price * 100) if current_price != 0 else 0
            }
            
            return {
                'ticker': ticker,
                'model_type': model_type,
                'forecast_data': forecast_df.to_dict(orient='records'),
                'forecast_summary': forecast_summary,
                'confidence_level': confidence_level,
                'periods': periods,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def compare_forecasts(self, model_types: List[str], ticker: str = 'GC=F', 
                         periods: int = 20) -> List[Dict[str, Any]]:
        """
        Compare forecasts from multiple models.
        
        Args:
            model_types: List of model types to compare
            ticker: Stock ticker symbol
            periods: Number of periods to predict
        
        Returns:
            list: Comparison results for each model
        """
        try:
            logger.info(f"Comparing forecasts for {ticker}: {', '.join(model_types)}")
            
            results = []
            
            for model_type in model_types:
                try:
                    result = self.generate_forecast(model_type, ticker, periods)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to generate forecast for {model_type}: {str(e)}")
                    results.append({
                        'model_type': model_type,
                        'ticker': ticker,
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing forecasts: {str(e)}")
            raise
    
    def analyze_forecast(self, forecast_data: List[Dict[str, Any]], 
                        current_price: float) -> Dict[str, Any]:
        """
        Analyze forecast data and provide insights.
        
        Args:
            forecast_data: List of forecast data points
            current_price: Current price for comparison
        
        Returns:
            dict: Analysis results
        """
        try:
            logger.info("Analyzing forecast data")
            
            if not forecast_data:
                raise ValueError("No forecast data provided")
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            
            # Calculate statistics
            forecast_mean = forecast_df['Forecast'].mean()
            forecast_high = forecast_df['Forecast'].max()
            forecast_low = forecast_df['Forecast'].min()
            volatility = forecast_df['Forecast'].std()
            
            # Price change analysis
            price_change = forecast_mean - current_price
            price_change_pct = (price_change / current_price * 100) if current_price != 0 else 0
            
            # Volatility analysis
            high_volatility = volatility > (current_price * 0.1)
            large_increase = forecast_high > (current_price * 1.3)
            large_decrease = forecast_low < (current_price * 0.7)
            
            # Trend analysis
            trend = 'up' if price_change > 0 else ('down' if price_change < 0 else 'neutral')
            
            analysis = {
                'current_price': current_price,
                'forecast_mean': forecast_mean,
                'forecast_high': forecast_high,
                'forecast_low': forecast_low,
                'volatility': volatility,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'trend': trend,
                'high_volatility': high_volatility,
                'large_increase': large_increase,
                'large_decrease': large_decrease,
                'confidence': self._calculate_confidence(forecast_df),
                'recommendations': self._generate_recommendations(analysis)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing forecast: {str(e)}")
            raise
    
    def _calculate_confidence(self, forecast_df: pd.DataFrame) -> str:
        """
        Calculate confidence level based on forecast characteristics.
        
        Args:
            forecast_df: Forecast DataFrame
        
        Returns:
            str: Confidence level
        """
        try:
            # Calculate volatility
            volatility = forecast_df['Forecast'].std()
            
            # Calculate range
            price_range = forecast_df['Forecast'].max() - forecast_df['Forecast'].min()
            
            # Calculate confidence based on volatility and range
            if volatility < 1 or price_range / forecast_df['Forecast'].mean() < 0.05:
                return 'high'
            elif volatility < 5 or price_range / forecast_df['Forecast'].mean() < 0.15:
                return 'medium'
            else:
                return 'low'
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 'unknown'
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on forecast analysis.
        
        Args:
            analysis: Analysis results
        
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        try:
            # High volatility warning
            if analysis.get('high_volatility'):
                recommendations.append('⚠️ High volatility detected - consider risk management strategies')
            
            # Large price movement warnings
            if analysis.get('large_increase'):
                recommendations.append('⚠️ Large price increase predicted - consider profit-taking strategies')
            
            if analysis.get('large_decrease'):
                recommendations.append('⚠️ Large price decrease predicted - consider hedging strategies')
            
            # Trend-based recommendations
            trend = analysis.get('trend')
            if trend == 'up':
                recommendations.append('📈 Uptrend detected - consider holding or adding to positions')
            elif trend == 'down':
                recommendations.append('📉 Downtrend detected - consider reducing exposure')
            else:
                recommendations.append('↔️ Neutral trend - consider waiting for clearer signals')
            
            # Confidence-based recommendations
            confidence = analysis.get('confidence')
            if confidence == 'high':
                recommendations.append('✅ High confidence in forecast - consider acting on signals')
            elif confidence == 'medium':
                recommendations.append('⚠️ Medium confidence - consider confirming with other indicators')
            else:
                recommendations.append('⚠️ Low confidence - consider waiting for more data')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [f'Error generating recommendations: {str(e)}']
    
    def get_forecast_metrics(self, forecast_data: List[Dict[str, Any]], 
                           actual_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get forecast metrics and performance indicators.
        
        Args:
            forecast_data: List of forecast data points
            actual_data: Optional actual data for comparison
        
        Returns:
            dict: Forecast metrics
        """
        try:
            logger.info("Calculating forecast metrics")
            
            if not forecast_data:
                raise ValueError("No forecast data provided")
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            
            metrics = {
                'mean_forecast': forecast_df['Forecast'].mean(),
                'median_forecast': forecast_df['Forecast'].median(),
                'std_dev': forecast_df['Forecast'].std(),
                'min_forecast': forecast_df['Forecast'].min(),
                'max_forecast': forecast_df['Forecast'].max(),
                'range': forecast_df['Forecast'].max() - forecast_df['Forecast'].min(),
                'volatility': forecast_df['Forecast'].std() / forecast_df['Forecast'].mean() if forecast_df['Forecast'].mean() != 0 else 0
            }
            
            # If actual data is provided, calculate additional metrics
            if actual_data is not None:
                # Get actual values for forecast period
                actual_values = actual_data['Close'].iloc[-len(forecast_df):].values
                forecast_values = forecast_df['Forecast'].values
                
                from utils.metrics import calculate_metrics
                metrics['metrics'] = calculate_metrics(actual_values, forecast_values, 
                                                      'forecast', actual_data.index[-1])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {str(e)}")
            raise
    
    def generate_confidence_bands(self, forecast_data: List[Dict[str, Any]], 
                                 confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Generate confidence bands for forecast.
        
        Args:
            forecast_data: List of forecast data points
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            pd.DataFrame: Forecast with confidence bands
        """
        try:
            logger.info(f"Generating confidence bands (confidence level: {confidence_level})")
            
            if not forecast_data:
                raise ValueError("No forecast data provided")
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            
            # Calculate standard error
            std_error = forecast_df['Forecast'].std()
            
            # Calculate confidence interval
            z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
            margin_of_error = z_score * std_error
            
            forecast_df['Lower_Bound'] = forecast_df['Forecast'] - margin_of_error
            forecast_df['Upper_Bound'] = forecast_df['Forecast'] + margin_of_error
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating confidence bands: {str(e)}")
            raise
    
    def get_forecast_trend_analysis(self, forecast_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze forecast trends and patterns.
        
        Args:
            forecast_data: List of forecast data points
        
        Returns:
            dict: Trend analysis results
        """
        try:
            logger.info("Analyzing forecast trends")
            
            if not forecast_data:
                raise ValueError("No forecast data provided")
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            
            # Calculate daily changes
            forecast_df['Change'] = forecast_df['Forecast'].diff().fillna(0)
            forecast_df['Change_Pct'] = forecast_df['Change'] / forecast_df['Forecast'].shift(1) * 100
            
            # Calculate trend statistics
            total_change = forecast_df['Forecast'].iloc[-1] - forecast_df['Forecast'].iloc[0]
            avg_daily_change = forecast_df['Change'].mean()
            avg_daily_change_pct = forecast_df['Change_Pct'].mean()
            
            # Identify trend direction
            if total_change > 0:
                trend_direction = 'up'
            elif total_change < 0:
                trend_direction = 'down'
            else:
                trend_direction = 'neutral'
            
            # Calculate volatility
            volatility = forecast_df['Change'].std()
            
            trend_analysis = {
                'trend_direction': trend_direction,
                'total_change': total_change,
                'avg_daily_change': avg_daily_change,
                'avg_daily_change_pct': avg_daily_change_pct,
                'volatility': volatility,
                'max_daily_change': forecast_df['Change'].max(),
                'min_daily_change': forecast_df['Change'].min(),
                'days_with_positive_change': (forecast_df['Change'] > 0).sum(),
                'days_with_negative_change': (forecast_df['Change'] < 0).sum()
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing forecast trends: {str(e)}")
            raise