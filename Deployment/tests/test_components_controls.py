"""
Unit tests for control components
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCommoditySelector:
    """Test commodity selector component"""
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    def test_commodity_selector_default(self, mock_caption, mock_selectbox):
        """Test commodity selector with default value"""
        from components.controls import create_commodity_selector
        
        mock_selectbox.return_value = "Gold"
        
        commodity, symbol = create_commodity_selector()
        
        assert commodity == "Gold"
        assert symbol == "GC=F"
        mock_selectbox.assert_called_once()
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    def test_commodity_selector_crude_oil(self, mock_caption, mock_selectbox):
        """Test commodity selector with Crude Oil"""
        from components.controls import create_commodity_selector
        
        mock_selectbox.return_value = "Crude Oil"
        
        commodity, symbol = create_commodity_selector()
        
        assert commodity == "Crude Oil"
        assert symbol == "CL=F"
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    def test_commodity_selector_all_commodities(self, mock_caption, mock_selectbox):
        """Test all commodity options"""
        from components.controls import create_commodity_selector
        
        commodities = {
            "Crude Oil": "CL=F",
            "Gold": "GC=F",
            "Natural Gas": "NG=F",
            "Copper": "HG=F",
            "Soybeans": "ZS=F"
        }
        
        for commodity, expected_symbol in commodities.items():
            mock_selectbox.return_value = commodity
            
            selected_commodity, symbol = create_commodity_selector()
            
            assert selected_commodity == commodity
            assert symbol == expected_symbol


class TestDateRangeSelector:
    """Test date range selector component"""
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_date_range_1month(self, mock_columns, mock_radio):
        """Test 1 month quick select"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "1M"
        
        start_date, end_date = create_date_range_selector()
        
        # Check that dates are approximately 30 days apart
        delta = (end_date - start_date).days
        assert 28 <= delta <= 32
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_date_range_3months(self, mock_columns, mock_radio):
        """Test 3 months quick select"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "3M"
        
        start_date, end_date = create_date_range_selector()
        
        # Check that dates are approximately 90 days apart
        delta = (end_date - start_date).days
        assert 88 <= delta <= 92
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_date_range_6months(self, mock_columns, mock_radio):
        """Test 6 months quick select"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "6M"
        
        start_date, end_date = create_date_range_selector()
        
        # Check that dates are approximately 180 days apart
        delta = (end_date - start_date).days
        assert 178 <= delta <= 182
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    @patch('streamlit.date_input')
    def test_date_range_custom(self, mock_date_input, mock_columns, mock_radio):
        """Test custom date range"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "Custom"
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        custom_start = datetime(2023, 1, 1).date()
        custom_end = datetime(2023, 6, 1).date()
        
        mock_date_input.side_effect = [custom_start, custom_end]
        
        start_date, end_date = create_date_range_selector()
        
        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)


class TestModelSelector:
    """Test model selector component"""
    
    @patch('streamlit.multiselect')
    def test_model_selector_default(self, mock_multiselect):
        """Test model selector with default models"""
        from components.controls import create_model_selector
        
        models = {
            'ARIMA': 'arima',
            'Prophet': 'prophet',
            'LSTM': 'lstm',
            'RandomForest': 'rf'
        }
        
        mock_multiselect.return_value = ['ARIMA', 'Prophet', 'LSTM']
        
        selected = create_model_selector(models)
        
        assert len(selected) == 3
        assert 'ARIMA' in selected
    
    @patch('streamlit.multiselect')
    def test_model_selector_custom_default(self, mock_multiselect):
        """Test model selector with custom default"""
        from components.controls import create_model_selector
        
        models = {
            'ARIMA': 'arima',
            'Prophet': 'prophet',
            'LSTM': 'lstm'
        }
        
        default_models = ['Prophet']
        mock_multiselect.return_value = ['Prophet']
        
        selected = create_model_selector(models, default_models=default_models)
        
        assert selected == ['Prophet']
    
    @patch('streamlit.multiselect')
    def test_model_selector_empty(self, mock_multiselect):
        """Test model selector with no selection"""
        from components.controls import create_model_selector
        
        models = {'ARIMA': 'arima', 'Prophet': 'prophet'}
        mock_multiselect.return_value = []
        
        selected = create_model_selector(models)
        
        assert selected == []


class TestForecastConfig:
    """Test forecast configuration component"""
    
    @patch('streamlit.slider')
    @patch('streamlit.checkbox')
    @patch('streamlit.selectbox')
    def test_forecast_config_defaults(self, mock_selectbox, mock_checkbox, mock_slider):
        """Test forecast config with default values"""
        from components.controls import create_forecast_config
        
        mock_slider.side_effect = [30, 180, 95]  # forecast_days, historical_days, confidence_level
        mock_checkbox.side_effect = [True, True]  # show_historical, show_confidence
        mock_selectbox.return_value = 'plotly_white'
        
        config = create_forecast_config()
        
        assert config['forecast_days'] == 30
        assert config['historical_days'] == 180
        assert config['confidence_level'] == 95
        assert config['show_historical'] == True
        assert config['show_confidence'] == True
        assert config['chart_theme'] == 'plotly_white'
    
    @patch('streamlit.slider')
    @patch('streamlit.checkbox')
    @patch('streamlit.selectbox')
    def test_forecast_config_custom_values(self, mock_selectbox, mock_checkbox, mock_slider):
        """Test forecast config with custom values"""
        from components.controls import create_forecast_config
        
        mock_slider.side_effect = [60, 365, 99]
        mock_checkbox.side_effect = [False, False]
        mock_selectbox.return_value = 'plotly_dark'
        
        config = create_forecast_config()
        
        assert config['forecast_days'] == 60
        assert config['historical_days'] == 365
        assert config['confidence_level'] == 99
        assert config['show_historical'] == False
        assert config['show_confidence'] == False
        assert config['chart_theme'] == 'plotly_dark'


class TestPerformanceFilters:
    """Test performance filter component"""
    
    @patch('streamlit.slider')
    @patch('streamlit.markdown')
    def test_performance_filters_defaults(self, mock_markdown, mock_slider):
        """Test performance filters with default values"""
        from components.controls import create_performance_filters
        
        mock_slider.side_effect = [70.0, 50.0]  # min_accuracy, max_rmse
        
        filters = create_performance_filters()
        
        assert filters['min_accuracy'] == 70.0
        assert filters['max_rmse'] == 50.0
    
    @patch('streamlit.slider')
    @patch('streamlit.markdown')
    def test_performance_filters_custom_values(self, mock_markdown, mock_slider):
        """Test performance filters with custom values"""
        from components.controls import create_performance_filters
        
        mock_slider.side_effect = [85.0, 25.0]
        
        filters = create_performance_filters()
        
        assert filters['min_accuracy'] == 85.0
        assert filters['max_rmse'] == 25.0


class TestDataFreshnessCheck:
    """Test data freshness check component"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.success')
    def test_data_freshness_fresh(self, mock_success, mock_markdown):
        """Test data freshness check with fresh data"""
        from components.controls import create_data_freshness_check
        
        # Mock the check_data_freshness function
        with patch('components.controls.check_data_freshness') as mock_check:
            mock_check.return_value = {
                'is_fresh': True,
                'last_update': datetime.now(),
                'current_price': 1950.50
            }
            
            create_data_freshness_check('GC=F')
            
            mock_success.assert_called()
    
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    def test_data_freshness_stale(self, mock_error, mock_warning, mock_markdown):
        """Test data freshness check with stale data"""
        from components.controls import create_data_freshness_check
        
        with patch('components.controls.check_data_freshness') as mock_check:
            mock_check.return_value = {
                'is_fresh': False,
                'message': 'Data is 10 days old',
                'recommendation': 'Please retrain models'
            }
            
            create_data_freshness_check('GC=F')
            
            mock_warning.assert_called()
            mock_error.assert_called()


class TestControlsEdgeCases:
    """Test edge cases for control components"""
    
    @patch('streamlit.selectbox')
    @patch('streamlit.caption')
    def test_commodity_selector_session_state(self, mock_caption, mock_selectbox):
        """Test commodity selector with session state"""
        from components.controls import create_commodity_selector
        
        mock_selectbox.return_value = "Copper"
        
        commodity, symbol = create_commodity_selector(default_commodity="Copper")
        
        assert commodity == "Copper"
        assert symbol == "HG=F"
    
    @patch('streamlit.radio')
    @patch('streamlit.columns')
    def test_date_range_returns_timestamps(self, mock_columns, mock_radio):
        """Test that date range returns pandas Timestamps"""
        from components.controls import create_date_range_selector
        
        mock_radio.return_value = "1M"
        
        start_date, end_date = create_date_range_selector()
        
        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)
        assert start_date < end_date


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
