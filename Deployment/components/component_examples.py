"""
Component Examples and Demo

This module demonstrates how to use all reusable Streamlit components.
Run with: streamlit run component_examples.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import all components
from components.charts import (
    create_candlestick_chart,
    create_volume_chart,
    create_forecast_chart,
    create_comparison_chart,
    create_radar_chart
)

from components.controls import (
    create_commodity_selector,
    create_date_range_selector,
    create_model_selector,
    create_forecast_config,
    create_performance_filters
)

from components.layouts import (
    create_metrics_dashboard,
    create_model_comparison_table,
    create_performance_charts,
    create_model_details,
    create_metrics_reference,
    create_color_legend
)

from components.utils import (
    create_footer,
    create_warning_banner,
    create_success_banner,
    create_info_banner,
    create_export_section,
    create_data_table,
    create_volatility_warning,
    create_price_movement_warning,
    create_confidence_bands
)


def generate_sample_ohlcv_data(days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(days) * 2)
    
    data = pd.DataFrame({
        'Open': close_prices + np.random.randn(days) * 0.5,
        'High': close_prices + abs(np.random.randn(days) * 2),
        'Low': close_prices - abs(np.random.randn(days) * 2),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    return data


def generate_sample_forecast(days: int = 30) -> pd.DataFrame:
    """Generate sample forecast data"""
    dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
    forecast_values = 110 + np.cumsum(np.random.randn(days) * 1)
    
    data = pd.DataFrame({
        'forecast': forecast_values,
        'upper': forecast_values + 5,
        'lower': forecast_values - 5
    }, index=dates)
    
    return data


def main():
    """Main demo application"""
    st.set_page_config(page_title="Component Examples", layout="wide")
    
    st.title("🎨 Streamlit Components Demo")
    st.markdown("Comprehensive examples of all reusable Streamlit components")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a component category:",
        ["Chart Components", "Control Components", "Layout Components", "Utility Components"]
    )
    
    # Generate sample data
    ohlcv_data = generate_sample_ohlcv_data(100)
    forecast_data = generate_sample_forecast(30)
    
    if page == "Chart Components":
        demo_chart_components(ohlcv_data, forecast_data)
    elif page == "Control Components":
        demo_control_components()
    elif page == "Layout Components":
        demo_layout_components(ohlcv_data)
    elif page == "Utility Components":
        demo_utility_components(ohlcv_data)
    
    # Footer
    st.divider()
    create_footer()


def demo_chart_components(ohlcv_data: pd.DataFrame, forecast_data: pd.DataFrame):
    """Demonstrate chart components"""
    st.header("📊 Chart Components")
    
    # Candlestick Chart
    st.subheader("1. Candlestick Chart")
    st.markdown("Displays OHLC data with volume analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_candlestick_chart(ohlcv_data, title="Gold Prices - Candlestick View")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info("""
        **Features:**
        - Green candles: Price up
        - Red candles: Price down
        - Unified hover mode
        - 600px height
        """)
    
    st.divider()
    
    # Volume Chart
    st.subheader("2. Volume Chart")
    st.markdown("Shows trading volume with color-coded bars")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_volume_chart(ohlcv_data, title="Trading Volume Analysis")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info("""
        **Features:**
        - Green bars: Up days
        - Red bars: Down days
        - Formatted hover info
        - Compact 300px height
        """)
    
    st.divider()
    
    # Forecast Chart
    st.subheader("3. Forecast Chart")
    st.markdown("Historical data with forecast and confidence bands")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_forecast_chart(ohlcv_data, forecast_data, title="30-Day Price Forecast")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info("""
        **Features:**
        - Blue: Historical
        - Red: Forecast
        - Shaded: Confidence bands
        - Legend at top-left
        """)
    
    st.divider()
    
    # Comparison Chart
    st.subheader("4. Model Comparison Chart")
    st.markdown("Compare multiple model forecasts")
    
    models = {
        'ARIMA': generate_sample_forecast(30),
        'Prophet': generate_sample_forecast(30),
        'LSTM': generate_sample_forecast(30)
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_comparison_chart(ohlcv_data, models, title="Model Forecast Comparison")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info("""
        **Features:**
        - Black: Historical
        - Dashed: Forecasts
        - Different colors per model
        - Markers on points
        """)
    
    st.divider()
    
    # Radar Chart
    st.subheader("5. Radar Chart")
    st.markdown("Model performance comparison")
    
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet', 'LSTM'],
        'RMSE': [5.2, 4.8, 3.9],
        'MAE': [3.1, 2.9, 2.5],
        'R²': [0.85, 0.88, 0.92]
    })
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_radar_chart(metrics_df, title="Model Performance Radar")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info("""
        **Features:**
        - Normalized metrics
        - One trace per model
        - Filled areas
        - 0-100 scale
        """)


def demo_control_components():
    """Demonstrate control components"""
    st.header("🎮 Control Components")
    
    # Commodity Selector
    st.subheader("1. Commodity Selector")
    st.markdown("Select from predefined commodities")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        commodity, symbol = create_commodity_selector()
        st.success(f"Selected: {commodity} ({symbol})")
    with col2:
        st.info("""
        **Supported:**
        - Crude Oil
        - Gold
        - Natural Gas
        - Copper
        - Soybeans
        """)
    
    st.divider()
    
    # Date Range Selector
    st.subheader("2. Date Range Selector")
    st.markdown("Quick period selection with custom option")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        start_date, end_date = create_date_range_selector()
        st.success(f"Range: {start_date.date()} to {end_date.date()}")
    with col2:
        st.info("""
        **Options:**
        - 1M (30 days)
        - 3M (90 days)
        - 6M (180 days)
        - Custom range
        """)
    
    st.divider()
    
    # Model Selector
    st.subheader("3. Model Selector")
    st.markdown("Multi-select model comparison")
    
    models = {
        'ARIMA': 'arima',
        'Prophet': 'prophet',
        'LSTM': 'lstm',
        'RandomForest': 'rf'
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_models = create_model_selector(models)
        st.success(f"Selected: {', '.join(selected_models)}")
    with col2:
        st.info("""
        **Features:**
        - Multi-select
        - Custom defaults
        - Help text
        """)
    
    st.divider()
    
    # Forecast Config
    st.subheader("4. Forecast Configuration")
    st.markdown("Configure forecast parameters")
    
    config = create_forecast_config()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast Days", config['forecast_days'])
    with col2:
        st.metric("Historical Days", config['historical_days'])
    with col3:
        st.metric("Confidence Level", f"{config['confidence_level']}%")
    
    st.divider()
    
    # Performance Filters
    st.subheader("5. Performance Filters")
    st.markdown("Filter models by performance metrics")
    
    filters = create_performance_filters()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Min Accuracy", f"{filters['min_accuracy']:.1f}%")
    with col2:
        st.metric("Max RMSE", f"${filters['max_rmse']:.2f}")


def demo_layout_components(ohlcv_data: pd.DataFrame):
    """Demonstrate layout components"""
    st.header("📐 Layout Components")
    
    # Metrics Dashboard
    st.subheader("1. Metrics Dashboard")
    st.markdown("Summary statistics display")
    
    metrics = {
        'mean_price': ohlcv_data['Close'].mean(),
        'median_price': ohlcv_data['Close'].median(),
        'std_dev': ohlcv_data['Close'].std(),
        'returns': ohlcv_data['Close'].pct_change().dropna().tolist()
    }
    
    create_metrics_dashboard(metrics)
    
    st.divider()
    
    # Model Comparison Table
    st.subheader("2. Model Comparison Table")
    st.markdown("Formatted metrics table")
    
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet', 'LSTM'],
        'RMSE': [5.234, 4.876, 3.912],
        'MAE': [3.123, 2.987, 2.456],
        'R²': [0.8523, 0.8812, 0.9234]
    })
    
    metric_info = {
        'RMSE': 'Root Mean Square Error',
        'MAE': 'Mean Absolute Error',
        'R²': 'R-squared Score'
    }
    
    create_model_comparison_table(metrics_df, metric_info)
    
    st.divider()
    
    # Performance Charts
    st.subheader("3. Performance Charts")
    st.markdown("Multi-chart performance comparison")
    
    create_performance_charts(metrics_df, metric_info)
    
    st.divider()
    
    # Model Details
    st.subheader("4. Model Details")
    st.markdown("Model information and characteristics")
    
    model_info = {
        'type': 'Time Series',
        'pros': ['Fast training', 'Interpretable', 'Good for stationary data'],
        'cons': ['Limited to linear trends', 'Requires stationarity'],
        'best_for': 'Stationary commodity prices'
    }
    
    create_model_details('ARIMA', model_info)
    
    st.divider()
    
    # Metrics Reference
    st.subheader("5. Metrics Reference")
    st.markdown("Expandable metric definitions")
    
    metrics_ref = {
        'RMSE': {
            'description': 'Root Mean Square Error - measures average prediction error',
            'range': '0 to ∞',
            'unit': 'Price units'
        },
        'MAE': {
            'description': 'Mean Absolute Error - average absolute prediction error',
            'range': '0 to ∞',
            'unit': 'Price units'
        },
        'R²': {
            'description': 'R-squared - proportion of variance explained',
            'range': '0 to 1',
            'unit': 'Dimensionless'
        }
    }
    
    create_metrics_reference(metrics_ref)
    
    st.divider()
    
    # Color Legend
    st.subheader("6. Color Legend")
    st.markdown("Chart color interpretation")
    
    create_color_legend()


def demo_utility_components(ohlcv_data: pd.DataFrame):
    """Demonstrate utility components"""
    st.header("🛠️ Utility Components")
    
    # Banners
    st.subheader("1. Information Banners")
    st.markdown("Display styled messages")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_success_banner("✅ Operation successful!")
    with col2:
        create_info_banner("ℹ️ Information message")
    with col3:
        create_warning_banner("⚠️ Warning message", severity="warning")
    
    st.divider()
    
    # Export Section
    st.subheader("2. Export Section")
    st.markdown("Data export with multiple formats")
    
    create_export_section(ohlcv_data.head(20), title="Sample Data Export")
    
    st.divider()
    
    # Data Table
    st.subheader("3. Data Table")
    st.markdown("Interactive data table with sorting")
    
    create_data_table(ohlcv_data, title="Price Data")
    
    st.divider()
    
    # Volatility Warning
    st.subheader("4. Volatility Warning")
    st.markdown("Automatic volatility detection")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        create_volatility_warning(ohlcv_data, threshold=0.05)
    with col2:
        volatility = ohlcv_data['Close'].pct_change().std()
        st.metric("Current Volatility", f"{volatility*100:.2f}%")
    
    st.divider()
    
    # Price Movement Warning
    st.subheader("5. Price Movement Warning")
    st.markdown("Detect large price movements")
    
    current_price = ohlcv_data['Close'].iloc[-1]
    col1, col2 = st.columns([2, 1])
    with col1:
        create_price_movement_warning(ohlcv_data, current_price, threshold=0.2)
    with col2:
        st.metric("Current Price", f"${current_price:.2f}")
    
    st.divider()
    
    # Confidence Bands
    st.subheader("6. Confidence Bands")
    st.markdown("Generate confidence intervals for forecasts")
    
    forecast = pd.DataFrame({
        'forecast': [110, 111, 112, 113, 114]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    result = create_confidence_bands(forecast, confidence_level=95)
    
    if result is not None:
        st.dataframe(result, use_container_width=True)
        st.success("✅ Confidence bands generated successfully")


if __name__ == "__main__":
    main()
