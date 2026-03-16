"""
Unit tests for layout components
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsDashboard:
    """Test metrics dashboard component"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_metrics_dashboard_creation(self, mock_metric, mock_columns, mock_markdown):
        """Test basic metrics dashboard creation"""
        from components.layouts import create_metrics_dashboard
        
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        metrics = {
            'mean_price': 1950.50,
            'median_price': 1945.00,
            'std_dev': 25.50,
            'returns': [0.01, -0.02, 0.015, -0.005]
        }
        
        create_metrics_dashboard(metrics)
        
        mock_markdown.assert_called()
        assert mock_metric.call_count >= 4
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_metrics_dashboard_with_missing_keys(self, mock_metric, mock_columns, mock_markdown):
        """Test metrics dashboard with missing keys"""
        from components.layouts import create_metrics_dashboard
        
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        metrics = {
            'mean_price': 1950.50
            # Missing other keys
        }
        
        create_metrics_dashboard(metrics)
        
        mock_markdown.assert_called()


class TestAdvancedStatistics:
    """Test advanced statistics component"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        df = pd.DataFrame({
            'Open': prices + np.random.randn(100),
            'High': prices + abs(np.random.randn(100) * 2),
            'Low': prices - abs(np.random.randn(100) * 2),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        returns = df['Close'].pct_change().dropna()
        
        return df, returns
    
    @patch('streamlit.tabs')
    @patch('streamlit.columns')
    @patch('streamlit.write')
    @patch('streamlit.plotly_chart')
    def test_advanced_statistics_creation(self, mock_plotly, mock_write, mock_columns, mock_tabs):
        """Test advanced statistics creation"""
        from components.layouts import create_advanced_statistics
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        df = pd.DataFrame({
            'Open': prices + np.random.randn(100),
            'High': prices + abs(np.random.randn(100) * 2),
            'Low': prices - abs(np.random.randn(100) * 2),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        returns = df['Close'].pct_change().dropna()
        
        # Mock tabs to return 4 context managers
        mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        create_advanced_statistics(df, returns)
        
        mock_tabs.assert_called_once()


class TestModelComparisonTable:
    """Test model comparison table component"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_model_comparison_table_creation(self, mock_dataframe, mock_markdown):
        """Test model comparison table creation"""
        from components.layouts import create_model_comparison_table
        
        df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'RMSE': [5.2, 4.8, 3.9],
            'MAE': [3.1, 2.9, 2.5],
            'R²': [0.85, 0.88, 0.92]
        })
        
        metrics = {'RMSE': 'Root Mean Square Error', 'MAE': 'Mean Absolute Error'}
        
        create_model_comparison_table(df, metrics)
        
        mock_markdown.assert_called()
        mock_dataframe.assert_called_once()
    
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_model_comparison_table_formatting(self, mock_dataframe, mock_markdown):
        """Test model comparison table formatting"""
        from components.layouts import create_model_comparison_table
        
        df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet'],
            'RMSE': [5.234567, 4.876543],
            'MAE': [3.123456, 2.987654]
        })
        
        metrics = {'RMSE': 'Error', 'MAE': 'Error'}
        
        create_model_comparison_table(df, metrics)
        
        # Check that dataframe was called
        mock_dataframe.assert_called_once()


class TestPerformanceCharts:
    """Test performance charts component"""
    
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.plotly_chart')
    def test_performance_charts_creation(self, mock_plotly, mock_subheader, mock_columns):
        """Test performance charts creation"""
        from components.layouts import create_performance_charts
        
        df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'RMSE': [5.2, 4.8, 3.9],
            'MAE': [3.1, 2.9, 2.5],
            'R²': [0.85, 0.88, 0.92],
            'Accuracy': [75.5, 78.2, 82.1]
        })
        
        metrics = {}
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        create_performance_charts(df, metrics)
        
        mock_subheader.assert_called()
        assert mock_plotly.call_count >= 4


class TestModelDetails:
    """Test model details component"""
    
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.write')
    def test_model_details_creation(self, mock_write, mock_markdown, mock_columns):
        """Test model details creation"""
        from components.layouts import create_model_details
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        model_info = {
            'type': 'Time Series',
            'pros': ['Fast', 'Interpretable'],
            'cons': ['Limited to linear trends'],
            'best_for': 'Stationary data'
        }
        
        create_model_details('ARIMA', model_info)
        
        mock_markdown.assert_called()
        assert mock_write.call_count >= 5


class TestMetricsReference:
    """Test metrics reference component"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    @patch('streamlit.write')
    def test_metrics_reference_creation(self, mock_write, mock_expander, mock_markdown):
        """Test metrics reference creation"""
        from components.layouts import create_metrics_reference
        
        mock_expander.return_value = MagicMock()
        
        metrics = {
            'RMSE': {
                'description': 'Root Mean Square Error',
                'range': '0 to ∞',
                'unit': 'Price units'
            },
            'MAE': {
                'description': 'Mean Absolute Error',
                'range': '0 to ∞',
                'unit': 'Price units'
            }
        }
        
        create_metrics_reference(metrics)
        
        mock_markdown.assert_called()
        assert mock_expander.call_count == 2


class TestColorLegend:
    """Test color legend component"""
    
    @patch('streamlit.markdown')
    def test_color_legend_creation(self, mock_markdown):
        """Test color legend creation"""
        from components.layouts import create_color_legend
        
        create_color_legend()
        
        mock_markdown.assert_called_once()
        
        # Check that the markdown contains expected content
        call_args = mock_markdown.call_args[0][0]
        assert 'Training Data' in call_args
        assert 'Test Data' in call_args
        assert 'Forecast' in call_args


class TestLayoutsEdgeCases:
    """Test edge cases for layout components"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_metrics_dashboard_with_empty_returns(self, mock_metric, mock_columns, mock_markdown):
        """Test metrics dashboard with empty returns"""
        from components.layouts import create_metrics_dashboard
        
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        metrics = {
            'mean_price': 1950.50,
            'median_price': 1945.00,
            'std_dev': 25.50,
            'returns': []
        }
        
        create_metrics_dashboard(metrics)
        
        mock_markdown.assert_called()
    
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_model_comparison_table_with_special_characters(self, mock_dataframe, mock_markdown):
        """Test model comparison table with special characters"""
        from components.layouts import create_model_comparison_table
        
        df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet™', 'LSTM®'],
            'RMSE': [5.2, 4.8, 3.9],
            'MAE': [3.1, 2.9, 2.5]
        })
        
        metrics = {}
        
        create_model_comparison_table(df, metrics)
        
        mock_dataframe.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
