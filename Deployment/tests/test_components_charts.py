"""
Unit tests for chart components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.charts import (
    create_candlestick_chart,
    create_volume_chart,
    create_forecast_chart,
    create_comparison_chart,
    create_radar_chart
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
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


class TestCandlestickChart:
    """Test candlestick chart creation"""
    
    def test_candlestick_chart_creation(self, sample_ohlcv_data):
        """Test basic candlestick chart creation"""
        fig = create_candlestick_chart(sample_ohlcv_data)
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == 'candlestick'
    
    def test_candlestick_chart_title(self, sample_ohlcv_data):
        """Test candlestick chart with custom title"""
        custom_title = "Gold Prices"
        fig = create_candlestick_chart(sample_ohlcv_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_candlestick_chart_layout(self, sample_ohlcv_data):
        """Test candlestick chart layout properties"""
        fig = create_candlestick_chart(sample_ohlcv_data)
        
        assert fig.layout.height == 600
        assert fig.layout.xaxis.rangeslider.visible == False
        assert fig.layout.hovermode == 'x unified'
    
    def test_candlestick_chart_with_empty_data(self):
        """Test candlestick chart with empty data"""
        empty_df = pd.DataFrame({
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': []
        })
        
        fig = create_candlestick_chart(empty_df)
        assert fig is not None


class TestVolumeChart:
    """Test volume chart creation"""
    
    def test_volume_chart_creation(self, sample_ohlcv_data):
        """Test basic volume chart creation"""
        fig = create_volume_chart(sample_ohlcv_data)
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == 'bar'
    
    def test_volume_chart_colors(self, sample_ohlcv_data):
        """Test volume chart color coding"""
        fig = create_volume_chart(sample_ohlcv_data)
        
        # Check that colors are assigned
        colors = fig.data[0].marker.color
        assert len(colors) == len(sample_ohlcv_data)
        assert all(c in ['green', 'red'] for c in colors)
    
    def test_volume_chart_title(self, sample_ohlcv_data):
        """Test volume chart with custom title"""
        custom_title = "Trading Volume Analysis"
        fig = create_volume_chart(sample_ohlcv_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title


class TestForecastChart:
    """Test forecast chart creation"""
    
    def test_forecast_chart_creation(self, sample_ohlcv_data, sample_forecast_data):
        """Test basic forecast chart creation"""
        fig = create_forecast_chart(sample_ohlcv_data, sample_forecast_data)
        
        assert fig is not None
        assert len(fig.data) >= 2  # Historical + Forecast
    
    def test_forecast_chart_with_confidence_bands(self, sample_ohlcv_data, sample_forecast_data):
        """Test forecast chart with confidence bands"""
        fig = create_forecast_chart(sample_ohlcv_data, sample_forecast_data)
        
        # Should have historical, forecast, upper, lower traces
        assert len(fig.data) >= 4
    
    def test_forecast_chart_without_confidence_bands(self, sample_ohlcv_data):
        """Test forecast chart without confidence bands"""
        forecast_data = pd.DataFrame({
            'forecast': [110, 111, 112, 113, 114]
        }, index=pd.date_range('2023-04-10', periods=5, freq='D'))
        
        fig = create_forecast_chart(sample_ohlcv_data, forecast_data)
        
        assert fig is not None
        assert len(fig.data) >= 2


class TestComparisonChart:
    """Test model comparison chart creation"""
    
    def test_comparison_chart_creation(self, sample_ohlcv_data):
        """Test basic comparison chart creation"""
        models = {
            'ARIMA': pd.DataFrame({
                'forecast': [110, 111, 112]
            }, index=pd.date_range('2023-04-10', periods=3, freq='D')),
            'Prophet': pd.DataFrame({
                'forecast': [109, 110, 111]
            }, index=pd.date_range('2023-04-10', periods=3, freq='D'))
        }
        
        fig = create_comparison_chart(sample_ohlcv_data, models)
        
        assert fig is not None
        assert len(fig.data) >= 3  # Historical + 2 models
    
    def test_comparison_chart_multiple_models(self, sample_ohlcv_data):
        """Test comparison chart with multiple models"""
        models = {
            f'Model_{i}': pd.DataFrame({
                'forecast': [110 + i, 111 + i, 112 + i]
            }, index=pd.date_range('2023-04-10', periods=3, freq='D'))
            for i in range(5)
        }
        
        fig = create_comparison_chart(sample_ohlcv_data, models)
        
        assert len(fig.data) == 6  # Historical + 5 models


class TestRadarChart:
    """Test radar chart creation"""
    
    def test_radar_chart_creation(self):
        """Test basic radar chart creation"""
        metrics_df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'RMSE': [5.2, 4.8, 3.9],
            'MAE': [3.1, 2.9, 2.5],
            'R²': [0.85, 0.88, 0.92]
        })
        
        fig = create_radar_chart(metrics_df)
        
        assert fig is not None
        assert len(fig.data) == 3  # One trace per model
    
    def test_radar_chart_normalization(self):
        """Test radar chart metric normalization"""
        metrics_df = pd.DataFrame({
            'Model': ['Model_A', 'Model_B'],
            'Metric1': [10, 20],
            'Metric2': [100, 200]
        })
        
        fig = create_radar_chart(metrics_df)
        
        assert fig is not None
        # Check that values are normalized (should be between 0-100)
        for trace in fig.data:
            assert all(0 <= r <= 100 for r in trace.r)


class TestChartEdgeCases:
    """Test edge cases for chart components"""
    
    def test_chart_with_nan_values(self, sample_ohlcv_data):
        """Test chart creation with NaN values"""
        data_with_nan = sample_ohlcv_data.copy()
        data_with_nan.iloc[0:5, 0] = np.nan
        
        fig = create_candlestick_chart(data_with_nan)
        assert fig is not None
    
    def test_chart_with_single_row(self):
        """Test chart with single row of data"""
        single_row = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1))
        
        fig = create_candlestick_chart(single_row)
        assert fig is not None
    
    def test_chart_with_duplicate_dates(self):
        """Test chart with duplicate dates"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        dates = dates.append(dates[:2])  # Add duplicates
        
        data = pd.DataFrame({
            'Open': np.random.randn(7) + 100,
            'High': np.random.randn(7) + 102,
            'Low': np.random.randn(7) + 99,
            'Close': np.random.randn(7) + 101,
            'Volume': np.random.randint(1000000, 5000000, 7)
        }, index=dates)
        
        fig = create_candlestick_chart(data)
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
