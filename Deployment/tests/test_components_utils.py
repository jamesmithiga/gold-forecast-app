"""
Unit tests for utility components
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFooter:
    """Test footer component"""
    
    @patch('streamlit.markdown')
    def test_footer_default(self, mock_markdown):
        """Test footer with default values"""
        from components.utils import create_footer
        
        create_footer()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'James Mithiga' in call_args
        assert '58200' in call_args
    
    @patch('streamlit.markdown')
    def test_footer_custom(self, mock_markdown):
        """Test footer with custom values"""
        from components.utils import create_footer
        
        create_footer(author="John Doe", adm_no="12345")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'John Doe' in call_args
        assert '12345' in call_args


class TestWarningBanner:
    """Test warning banner component"""
    
    @patch('streamlit.markdown')
    def test_warning_banner_default(self, mock_markdown):
        """Test warning banner with default severity"""
        from components.utils import create_warning_banner
        
        create_warning_banner("Test warning")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'Test warning' in call_args
        assert '#ff6b6b' in call_args  # Default warning color
    
    @patch('streamlit.markdown')
    def test_warning_banner_error(self, mock_markdown):
        """Test warning banner with error severity"""
        from components.utils import create_warning_banner
        
        create_warning_banner("Test error", severity="error")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'Test error' in call_args
        assert '#ff4444' in call_args  # Error color
    
    @patch('streamlit.markdown')
    def test_warning_banner_info(self, mock_markdown):
        """Test warning banner with info severity"""
        from components.utils import create_warning_banner
        
        create_warning_banner("Test info", severity="info")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'Test info' in call_args
        assert '#ff9800' in call_args  # Info color


class TestSuccessBanner:
    """Test success banner component"""
    
    @patch('streamlit.markdown')
    def test_success_banner_creation(self, mock_markdown):
        """Test success banner creation"""
        from components.utils import create_success_banner
        
        create_success_banner("Operation successful")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert '#4CAF50' in call_args  # Success color


class TestInfoBanner:
    """Test info banner component"""
    
    @patch('streamlit.markdown')
    def test_info_banner_creation(self, mock_markdown):
        """Test info banner creation"""
        from components.utils import create_info_banner
        
        create_info_banner("Information message")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert '#2196F3' in call_args  # Info color


class TestExportSection:
    """Test export section component"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Value': np.random.randn(10),
            'Category': ['A', 'B'] * 5
        })
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.radio')
    @patch('streamlit.checkbox')
    @patch('streamlit.divider')
    @patch('streamlit.download_button')
    @patch('streamlit.info')
    def test_export_section_csv(self, mock_info, mock_download, mock_divider, 
                                 mock_checkbox, mock_radio, mock_columns, mock_markdown, 
                                 sample_dataframe):
        """Test export section with CSV format"""
        from components.utils import create_export_section
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_radio.return_value = "CSV"
        mock_checkbox.side_effect = [True, True]
        
        create_export_section(sample_dataframe)
        
        mock_markdown.assert_called()
        mock_download.assert_called_once()
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.radio')
    @patch('streamlit.checkbox')
    @patch('streamlit.divider')
    @patch('streamlit.download_button')
    @patch('streamlit.info')
    def test_export_section_json(self, mock_info, mock_download, mock_divider,
                                  mock_checkbox, mock_radio, mock_columns, mock_markdown,
                                  sample_dataframe):
        """Test export section with JSON format"""
        from components.utils import create_export_section
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_radio.return_value = "JSON"
        mock_checkbox.side_effect = [True, True]
        
        create_export_section(sample_dataframe)
        
        mock_download.assert_called_once()


class TestDataTable:
    """Test data table component"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing"""
        dates = pd.date_range('2023-01-01', periods=50)
        return pd.DataFrame({
            'Date': dates,
            'Close': 100 + np.cumsum(np.random.randn(50)),
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.slider')
    @patch('streamlit.selectbox')
    @patch('streamlit.dataframe')
    def test_data_table_creation(self, mock_dataframe, mock_selectbox, mock_slider,
                                  mock_columns, mock_markdown, sample_dataframe):
        """Test data table creation"""
        from components.utils import create_data_table
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 20
        mock_selectbox.return_value = "Date"
        
        create_data_table(sample_dataframe)
        
        mock_markdown.assert_called()
        mock_dataframe.assert_called_once()


class TestModelDescription:
    """Test model description component"""
    
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.write')
    def test_model_description_creation(self, mock_write, mock_markdown, mock_columns):
        """Test model description creation"""
        from components.utils import create_model_description
        
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        pros = ['Fast', 'Interpretable']
        cons = ['Limited to linear trends']
        
        create_model_description('ARIMA', 'Time Series', pros, cons, 'Stationary data')
        
        mock_markdown.assert_called()
        assert mock_write.call_count >= 5


class TestColorLegend:
    """Test color legend component"""
    
    @patch('streamlit.markdown')
    def test_color_legend_creation(self, mock_markdown):
        """Test color legend creation"""
        from components.utils import create_color_legend
        
        create_color_legend()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'Training Data' in call_args
        assert 'Test Data' in call_args
        assert 'Forecast' in call_args


class TestVolatilityWarning:
    """Test volatility warning component"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe"""
        dates = pd.date_range('2023-01-01', periods=100)
        return pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 2)
        }, index=dates)
    
    @patch('streamlit.warning')
    def test_volatility_warning_high(self, mock_warning, sample_dataframe):
        """Test volatility warning with high volatility"""
        from components.utils import create_volatility_warning
        
        # Create high volatility data
        high_vol_df = sample_dataframe.copy()
        high_vol_df['Close'] = 100 + np.cumsum(np.random.randn(100) * 10)
        
        create_volatility_warning(high_vol_df, threshold=0.05)
        
        # Warning should be called if volatility is high
        if high_vol_df['Close'].pct_change().std() > 0.05:
            mock_warning.assert_called()
    
    @patch('streamlit.warning')
    def test_volatility_warning_low(self, mock_warning, sample_dataframe):
        """Test volatility warning with low volatility"""
        from components.utils import create_volatility_warning
        
        create_volatility_warning(sample_dataframe, threshold=0.5)
        
        # Warning should not be called for low volatility
        if sample_dataframe['Close'].pct_change().std() <= 0.5:
            mock_warning.assert_not_called()


class TestPriceMovementWarning:
    """Test price movement warning component"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe"""
        dates = pd.date_range('2023-01-01', periods=30)
        return pd.DataFrame({
            'Close': np.linspace(100, 150, 30)
        }, index=dates)
    
    @patch('streamlit.warning')
    def test_price_movement_warning_increase(self, mock_warning, sample_dataframe):
        """Test price movement warning with large increase"""
        from components.utils import create_price_movement_warning
        
        current_price = 100
        create_price_movement_warning(sample_dataframe, current_price, threshold=0.2)
        
        # Should warn about large increase
        if sample_dataframe['Close'].max() > current_price * 1.2:
            mock_warning.assert_called()
    
    @patch('streamlit.warning')
    def test_price_movement_warning_decrease(self, mock_warning):
        """Test price movement warning with large decrease"""
        from components.utils import create_price_movement_warning
        
        dates = pd.date_range('2023-01-01', periods=30)
        df = pd.DataFrame({
            'Close': np.linspace(150, 100, 30)
        }, index=dates)
        
        current_price = 150
        create_price_movement_warning(df, current_price, threshold=0.2)
        
        # Should warn about large decrease
        if df['Close'].min() < current_price * 0.8:
            mock_warning.assert_called()


class TestConfidenceBands:
    """Test confidence bands component"""
    
    def test_confidence_bands_creation(self):
        """Test confidence bands creation"""
        from components.utils import create_confidence_bands
        
        forecast = pd.DataFrame({
            'forecast': [110, 111, 112, 113, 114]
        }, index=pd.date_range('2023-04-10', periods=5, freq='D'))
        
        result = create_confidence_bands(forecast, confidence_level=95)
        
        assert result is not None
        assert 'upper' in result.columns
        assert 'lower' in result.columns
        assert all(result['lower'] < result['forecast'])
        assert all(result['upper'] > result['forecast'])
    
    def test_confidence_bands_without_forecast_column(self):
        """Test confidence bands without forecast column"""
        from components.utils import create_confidence_bands
        
        forecast = pd.DataFrame({
            'value': [110, 111, 112, 113, 114]
        }, index=pd.date_range('2023-04-10', periods=5, freq='D'))
        
        result = create_confidence_bands(forecast, confidence_level=95)
        
        assert result is None
    
    def test_confidence_bands_different_confidence_levels(self):
        """Test confidence bands with different confidence levels"""
        from components.utils import create_confidence_bands
        
        forecast = pd.DataFrame({
            'forecast': [110, 111, 112, 113, 114]
        }, index=pd.date_range('2023-04-10', periods=5, freq='D'))
        
        result_80 = create_confidence_bands(forecast.copy(), confidence_level=80)
        result_99 = create_confidence_bands(forecast.copy(), confidence_level=99)
        
        # 99% confidence should have wider bands than 80%
        assert (result_99['upper'] - result_99['lower']).mean() > (result_80['upper'] - result_80['lower']).mean()


class TestUtilsEdgeCases:
    """Test edge cases for utility components"""
    
    @patch('streamlit.warning')
    def test_volatility_warning_with_none(self, mock_warning):
        """Test volatility warning with None dataframe"""
        from components.utils import create_volatility_warning
        
        create_volatility_warning(None)
        
        mock_warning.assert_not_called()
    
    @patch('streamlit.warning')
    def test_volatility_warning_with_empty_dataframe(self, mock_warning):
        """Test volatility warning with empty dataframe"""
        from components.utils import create_volatility_warning
        
        empty_df = pd.DataFrame({'Close': []})
        
        create_volatility_warning(empty_df)
        
        mock_warning.assert_not_called()
    
    @patch('streamlit.warning')
    def test_price_movement_warning_with_none(self, mock_warning):
        """Test price movement warning with None dataframe"""
        from components.utils import create_price_movement_warning
        
        create_price_movement_warning(None, 100)
        
        mock_warning.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
