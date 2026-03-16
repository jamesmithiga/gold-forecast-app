"""
Integration tests for Streamlit components

Tests component interactions and workflows
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    data = pd.DataFrame({
        'Open': close_prices + np.random.randn(100),
        'High': close_prices + abs(np.random.randn(100) * 2),
        'Low': close_prices - abs(np.random.randn(100) * 2),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data"""
    dates = pd.date_range(start='2023-04-10', periods=30, freq='D')
    forecast_values = 110 + np.cumsum(np.random.randn(30) * 1)
    
    data = pd.DataFrame({
        'forecast': forecast_values,
        'upper': forecast_values + 5,
        'lower': forecast_values - 5
    }, index=dates)
    
    return data


class TestChartControlIntegration:
    """Test integration between chart and control components"""
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    def test_commodity_selector_with_chart(self, mock_caption, mock_selectbox, sample_ohlcv_data):
        """Test commodity selector feeding into chart"""
        from components.controls import create_commodity_selector
        from components.charts import create_candlestick_chart
        
        mock_selectbox.return_value = "Gold"
        
        commodity, symbol = create_commodity_selector()
        
        # Use selected commodity to create chart
        fig = create_candlestick_chart(sample_ohlcv_data, title=f"{commodity} Prices")
        
        assert fig is not None
        assert commodity == "Gold"
        assert symbol == "GC=F"
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_date_range_with_data_filtering(self, mock_columns, mock_radio, sample_ohlcv_data):
        """Test date range selector with data filtering"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "3M"
        
        start_date, end_date = create_date_range_selector()
        
        # Filter data using selected dates
        filtered_df = sample_ohlcv_data[
            (sample_ohlcv_data.index >= start_date) & 
            (sample_ohlcv_data.index <= end_date)
        ]
        
        assert len(filtered_df) > 0
        assert filtered_df.index.min() >= start_date
        assert filtered_df.index.max() <= end_date


class TestLayoutControlIntegration:
    """Test integration between layout and control components"""
    
    @patch('streamlit.multiselect')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.plotly_chart')
    def test_model_selector_with_performance_charts(self, mock_plotly, mock_subheader, 
                                                     mock_columns, mock_multiselect):
        """Test model selector feeding into performance charts"""
        from components.controls import create_model_selector
        from components.layouts import create_performance_charts
        
        models = {
            'ARIMA': 'arima',
            'Prophet': 'prophet',
            'LSTM': 'lstm'
        }
        
        mock_multiselect.return_value = ['ARIMA', 'Prophet']
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        selected_models = create_model_selector(models)
        
        # Create metrics for selected models
        metrics_df = pd.DataFrame({
            'Model': selected_models,
            'RMSE': [5.2, 4.8],
            'MAE': [3.1, 2.9],
            'R²': [0.85, 0.88],
            'Accuracy': [75.5, 78.2]
        })
        
        create_performance_charts(metrics_df, {})
        
        assert len(selected_models) == 2
        assert mock_plotly.call_count >= 4


class TestDataFlowIntegration:
    """Test complete data flow through components"""
    
    def test_data_load_filter_chart_flow(self, sample_ohlcv_data):
        """Test complete flow: load -> filter -> chart"""
        from components.charts import create_candlestick_chart
        
        # Simulate data loading
        df = sample_ohlcv_data.copy()
        
        # Simulate filtering
        filtered_df = df.tail(50)
        
        # Create chart
        fig = create_candlestick_chart(filtered_df)
        
        assert fig is not None
        assert len(filtered_df) == 50
    
    def test_forecast_confidence_bands_flow(self, sample_forecast_data):
        """Test forecast with confidence bands"""
        from components.utils import create_confidence_bands
        from components.charts import create_forecast_chart
        
        # Create confidence bands
        forecast_with_bands = create_confidence_bands(sample_forecast_data.copy(), 95)
        
        assert forecast_with_bands is not None
        assert 'upper' in forecast_with_bands.columns
        assert 'lower' in forecast_with_bands.columns
        
        # Create chart with bands
        historical = pd.DataFrame({
            'Close': np.linspace(100, 110, 30)
        }, index=pd.date_range('2023-03-10', periods=30, freq='D'))
        
        fig = create_forecast_chart(historical, forecast_with_bands)
        
        assert fig is not None


class TestErrorHandlingIntegration:
    """Test error handling across components"""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        from components.charts import create_candlestick_chart
        
        # Missing required columns
        invalid_df = pd.DataFrame({
            'Price': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError):
            create_candlestick_chart(invalid_df)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        from components.charts import create_candlestick_chart
        
        empty_df = pd.DataFrame({
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': []
        })
        
        fig = create_candlestick_chart(empty_df)
        
        assert fig is not None
    
    def test_missing_forecast_column(self):
        """Test handling of missing forecast column"""
        from components.utils import create_confidence_bands
        
        invalid_forecast = pd.DataFrame({
            'value': [110, 111, 112]
        })
        
        result = create_confidence_bands(invalid_forecast, 95)
        
        assert result is None


class TestDataConsistencyIntegration:
    """Test data consistency across components"""
    
    def test_date_index_consistency(self, sample_ohlcv_data):
        """Test date index consistency"""
        from components.charts import create_candlestick_chart, create_volume_chart
        
        # Create multiple charts with same data
        fig1 = create_candlestick_chart(sample_ohlcv_data)
        fig2 = create_volume_chart(sample_ohlcv_data)
        
        # Both should have same date range
        assert len(fig1.data[0].x) == len(fig2.data[0].x)
    
    def test_metric_calculation_consistency(self, sample_ohlcv_data):
        """Test metric calculation consistency"""
        from components.layouts import create_metrics_dashboard
        
        # Calculate metrics
        metrics = {
            'mean_price': sample_ohlcv_data['Close'].mean(),
            'median_price': sample_ohlcv_data['Close'].median(),
            'std_dev': sample_ohlcv_data['Close'].std(),
            'returns': sample_ohlcv_data['Close'].pct_change().dropna().tolist()
        }
        
        # Verify metrics are consistent
        assert metrics['mean_price'] > 0
        assert metrics['median_price'] > 0
        assert metrics['std_dev'] > 0
        assert len(metrics['returns']) > 0


class TestPerformanceIntegration:
    """Test performance with large datasets"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        from components.charts import create_candlestick_chart
        
        # Create large dataset (1 year of daily data)
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(365) * 2)
        
        large_df = pd.DataFrame({
            'Open': close_prices + np.random.randn(365),
            'High': close_prices + abs(np.random.randn(365) * 2),
            'Low': close_prices - abs(np.random.randn(365) * 2),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 365)
        }, index=dates)
        
        fig = create_candlestick_chart(large_df)
        
        assert fig is not None
        assert len(fig.data[0].x) == 365
    
    def test_multiple_model_comparison_performance(self):
        """Test performance with multiple models"""
        from components.charts import create_comparison_chart
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        historical = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 2)
        }, index=dates)
        
        # Create 10 model forecasts
        models = {
            f'Model_{i}': pd.DataFrame({
                'forecast': 110 + np.cumsum(np.random.randn(30) * 1)
            }, index=pd.date_range('2023-04-10', periods=30, freq='D'))
            for i in range(10)
        }
        
        fig = create_comparison_chart(historical, models)
        
        assert fig is not None
        assert len(fig.data) == 11  # Historical + 10 models


class TestComponentChaining:
    """Test chaining multiple components"""
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_full_workflow_chain(self, mock_columns, mock_radio, mock_caption, 
                                  mock_selectbox, sample_ohlcv_data):
        """Test complete workflow chain"""
        from components.controls import (
            create_commodity_selector,
            create_date_range_selector,
            create_model_selector
        )
        from components.charts import create_candlestick_chart
        
        # Step 1: Select commodity
        mock_selectbox.return_value = "Gold"
        commodity, symbol = create_commodity_selector()
        
        # Step 2: Select date range
        mock_radio.return_value = "6M"
        start_date, end_date = create_date_range_selector()
        
        # Step 3: Filter data
        filtered_df = sample_ohlcv_data[
            (sample_ohlcv_data.index >= start_date) & 
            (sample_ohlcv_data.index <= end_date)
        ]
        
        # Step 4: Create chart
        fig = create_candlestick_chart(filtered_df, title=f"{commodity} Prices")
        
        assert commodity == "Gold"
        assert symbol == "GC=F"
        assert fig is not None
        assert len(filtered_df) > 0


class TestComponentReusability:
    """Test component reusability across different contexts"""
    
    def test_chart_reuse_different_data(self, sample_ohlcv_data):
        """Test reusing chart component with different data"""
        from components.charts import create_candlestick_chart
        
        # Use same component with different data
        fig1 = create_candlestick_chart(sample_ohlcv_data, title="Gold")
        fig2 = create_candlestick_chart(sample_ohlcv_data.tail(50), title="Gold (Recent)")
        
        assert fig1 is not None
        assert fig2 is not None
        assert len(fig1.data[0].x) != len(fig2.data[0].x)
    
    @patch('streamlit.multiselect')
    def test_model_selector_reuse(self, mock_multiselect):
        """Test reusing model selector with different model sets"""
        from components.controls import create_model_selector
        
        models1 = {'ARIMA': 'arima', 'Prophet': 'prophet'}
        models2 = {'LSTM': 'lstm', 'RandomForest': 'rf', 'XGBoost': 'xgb'}
        
        mock_multiselect.side_effect = [['ARIMA'], ['LSTM', 'RandomForest']]
        
        selected1 = create_model_selector(models1)
        selected2 = create_model_selector(models2)
        
        assert selected1 == ['ARIMA']
        assert len(selected2) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
