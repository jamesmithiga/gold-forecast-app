# Streamlit Components Documentation

## Overview

This document provides comprehensive documentation for the reusable Streamlit components used in the Commodity Price Forecasting Dashboard. The components are organized into four main modules: **charts**, **controls**, **layouts**, and **utils**.

---

## Table of Contents

1. [Chart Components](#chart-components)
2. [Control Components](#control-components)
3. [Layout Components](#layout-components)
4. [Utility Components](#utility-components)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Testing](#testing)

---

## Chart Components

Located in: `components/charts.py`

### 1. `create_candlestick_chart()`

Creates a reusable candlestick chart with volume analysis.

**Signature:**
```python
def create_candlestick_chart(
    df: pd.DataFrame, 
    title: str = "Candlestick Chart", 
    show_volume: bool = True
) -> go.Figure
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume)
- `title` (str): Chart title (default: "Candlestick Chart")
- `show_volume` (bool): Whether to include volume analysis (default: True)

**Returns:**
- `go.Figure`: Plotly Figure object

**Example:**
```python
import pandas as pd
from components.charts import create_candlestick_chart

# Assuming df has OHLCV data
fig = create_candlestick_chart(df, title="Gold Prices")
st.plotly_chart(fig, use_container_width=True)
```

**Features:**
- Green candles for price increases
- Red candles for price decreases
- Unified hover mode for better UX
- 600px height for optimal viewing

---

### 2. `create_volume_chart()`

Creates a volume bar chart with color coding based on price movement.

**Signature:**
```python
def create_volume_chart(
    df: pd.DataFrame, 
    title: str = "Trading Volume"
) -> go.Figure
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLCV data
- `title` (str): Chart title (default: "Trading Volume")

**Returns:**
- `go.Figure`: Plotly Figure object

**Example:**
```python
fig = create_volume_chart(df, title="Daily Trading Volume")
st.plotly_chart(fig, use_container_width=True)
```

**Features:**
- Green bars for up days (Close >= Open)
- Red bars for down days (Close < Open)
- Formatted hover information
- 300px height for compact display

---

### 3. `create_forecast_chart()`

Creates a forecast chart with historical data and prediction confidence bands.

**Signature:**
```python
def create_forecast_chart(
    df: pd.DataFrame, 
    forecast: pd.DataFrame, 
    title: str = "Price Forecast"
) -> go.Figure
```

**Parameters:**
- `df` (pd.DataFrame): Historical price data
- `forecast` (pd.DataFrame): Forecast data with columns: 'forecast', 'upper', 'lower'
- `title` (str): Chart title (default: "Price Forecast")

**Returns:**
- `go.Figure`: Plotly Figure object

**Example:**
```python
fig = create_forecast_chart(
    historical_df, 
    forecast_df, 
    title="30-Day Gold Price Forecast"
)
st.plotly_chart(fig, use_container_width=True)
```

**Features:**
- Blue line for historical data
- Red line for forecast
- Shaded confidence bands (upper/lower bounds)
- Legend positioned at top-left

---

### 4. `create_comparison_chart()`

Creates a multi-model comparison chart showing historical data and multiple forecasts.

**Signature:**
```python
def create_comparison_chart(
    df: pd.DataFrame, 
    models: Dict[str, pd.DataFrame], 
    title: str = "Model Comparison"
) -> go.Figure
```

**Parameters:**
- `df` (pd.DataFrame): Historical price data
- `models` (Dict[str, pd.DataFrame]): Dictionary mapping model names to forecast DataFrames
- `title` (str): Chart title (default: "Model Comparison")

**Returns:**
- `go.Figure`: Plotly Figure object

**Example:**
```python
models = {
    'ARIMA': arima_forecast_df,
    'Prophet': prophet_forecast_df,
    'LSTM': lstm_forecast_df
}

fig = create_comparison_chart(historical_df, models)
st.plotly_chart(fig, use_container_width=True)
```

**Features:**
- Black line for historical data
- Dashed lines for model forecasts
- Different colors for each model
- Markers on forecast points

---

### 5. `create_radar_chart()`

Creates a radar chart for model performance comparison.

**Signature:**
```python
def create_radar_chart(
    df: pd.DataFrame, 
    title: str = "Model Performance Radar"
) -> go.Figure
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame with model metrics (columns: Model, metric1, metric2, ...)
- `title` (str): Chart title (default: "Model Performance Radar")

**Returns:**
- `go.Figure`: Plotly Figure object

**Example:**
```python
metrics_df = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet', 'LSTM'],
    'RMSE': [5.2, 4.8, 3.9],
    'MAE': [3.1, 2.9, 2.5],
    'R²': [0.85, 0.88, 0.92]
})

fig = create_radar_chart(metrics_df)
st.plotly_chart(fig, use_container_width=True)
```

**Features:**
- Normalized metrics (0-100 scale)
- One trace per model
- Filled areas for visual comparison
- Radial axis with 0-100 range

---

## Control Components

Located in: `components/controls.py`

### 1. `create_commodity_selector()`

Creates a reusable commodity selector with sidebar controls.

**Signature:**
```python
def create_commodity_selector(
    default_commodity: str = "Gold"
) -> Tuple[str, str]
```

**Parameters:**
- `default_commodity` (str): Default commodity to select (default: "Gold")

**Returns:**
- `Tuple[str, str]`: (selected_commodity, selected_symbol)

**Example:**
```python
commodity, symbol = create_commodity_selector(default_commodity="Gold")
st.write(f"Selected: {commodity} ({symbol})")
```

**Supported Commodities:**
- Crude Oil (CL=F)
- Gold (GC=F)
- Natural Gas (NG=F)
- Copper (HG=F)
- Soybeans (ZS=F)

---

### 2. `create_date_range_selector()`

Creates a date range selector with quick period options.

**Signature:**
```python
def create_date_range_selector() -> Tuple[pd.Timestamp, pd.Timestamp]
```

**Returns:**
- `Tuple[pd.Timestamp, pd.Timestamp]`: (start_date, end_date)

**Example:**
```python
start_date, end_date = create_date_range_selector()
filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
```

**Quick Period Options:**
- 1M: Last 30 days
- 3M: Last 90 days
- 6M: Last 180 days
- Custom: User-defined range

---

### 3. `create_model_selector()`

Creates a multi-select component for model selection.

**Signature:**
```python
def create_model_selector(
    models: Dict[str, str], 
    default_models: List[str] = None
) -> List[str]
```

**Parameters:**
- `models` (Dict[str, str]): Dictionary of model names and keys
- `default_models` (List[str]): Default models to select (default: first 3 models)

**Returns:**
- `List[str]`: List of selected model names

**Example:**
```python
models = {
    'ARIMA': 'arima',
    'Prophet': 'prophet',
    'LSTM': 'lstm',
    'RandomForest': 'rf'
}

selected_models = create_model_selector(models, default_models=['ARIMA', 'Prophet'])
```

---

### 4. `create_forecast_config()`

Creates a forecast configuration panel with multiple parameters.

**Signature:**
```python
def create_forecast_config() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Configuration dictionary with keys:
  - `forecast_days` (int): 1-90 days
  - `historical_days` (int): 30-365 days
  - `confidence_level` (int): 80-99%
  - `show_historical` (bool)
  - `show_confidence` (bool)
  - `chart_theme` (str): plotly_white, plotly_dark, plotly

**Example:**
```python
config = create_forecast_config()

forecast_days = config['forecast_days']
confidence_level = config['confidence_level']
```

---

### 5. `create_performance_filters()`

Creates performance filter controls.

**Signature:**
```python
def create_performance_filters() -> Dict[str, float]
```

**Returns:**
- `Dict[str, float]`: Filter parameters:
  - `min_accuracy` (float): 0-100%
  - `max_rmse` (float): 0-100 USD

**Example:**
```python
filters = create_performance_filters()

filtered_models = models[
    (models['Accuracy'] >= filters['min_accuracy']) &
    (models['RMSE'] <= filters['max_rmse'])
]
```

---

### 6. `create_data_freshness_check()`

Creates a data freshness check component with retrain functionality.

**Signature:**
```python
def create_data_freshness_check(
    ticker: str, 
    max_age_days: int = 7
) -> None
```

**Parameters:**
- `ticker` (str): Commodity symbol (e.g., "GC=F")
- `max_age_days` (int): Maximum age in days for fresh data (default: 7)

**Example:**
```python
create_data_freshness_check('GC=F', max_age_days=7)
```

**Features:**
- Checks data freshness
- Displays current price
- Provides retrain button if data is stale
- Fetches latest data from Yahoo Finance

---

## Layout Components

Located in: `components/layouts.py`

### 1. `create_metrics_dashboard()`

Creates a metrics dashboard with summary statistics.

**Signature:**
```python
def create_metrics_dashboard(metrics: Dict[str, Any]) -> None
```

**Parameters:**
- `metrics` (Dict[str, Any]): Dictionary with keys:
  - `mean_price` (float)
  - `median_price` (float)
  - `std_dev` (float)
  - `returns` (list)

**Example:**
```python
metrics = {
    'mean_price': 1950.50,
    'median_price': 1945.00,
    'std_dev': 25.50,
    'returns': [0.01, -0.02, 0.015]
}

create_metrics_dashboard(metrics)
```

---

### 2. `create_advanced_statistics()`

Creates advanced statistics tabs with multiple analyses.

**Signature:**
```python
def create_advanced_statistics(
    df: pd.DataFrame, 
    returns: pd.Series
) -> None
```

**Parameters:**
- `df` (pd.DataFrame): Price data with OHLC columns
- `returns` (pd.Series): Daily returns

**Tabs:**
1. Returns: Daily returns statistics and distribution
2. Distribution: Price distribution and metrics
3. Correlation: OHLC correlation matrix
4. Trends: Moving averages and volatility analysis

---

### 3. `create_model_comparison_table()`

Creates a formatted model comparison table.

**Signature:**
```python
def create_model_comparison_table(
    df: pd.DataFrame, 
    metrics: Dict[str, Any]
) -> None
```

**Parameters:**
- `df` (pd.DataFrame): Model metrics DataFrame
- `metrics` (Dict[str, Any]): Metric descriptions

---

### 4. `create_performance_charts()`

Creates performance comparison charts in a 2-column layout.

**Signature:**
```python
def create_performance_charts(
    df: pd.DataFrame, 
    metrics: Dict[str, Any]
) -> None
```

**Charts:**
- RMSE Comparison (Lower is Better)
- MAE Comparison (Lower is Better)
- R² Score (Higher is Better)
- Directional Accuracy (Higher is Better)

---

### 5. `create_model_details()`

Creates a model details section with pros and cons.

**Signature:**
```python
def create_model_details(
    model_name: str, 
    model_info: Dict[str, Any]
) -> None
```

**Parameters:**
- `model_name` (str): Name of the model
- `model_info` (Dict[str, Any]): Dictionary with:
  - `type` (str)
  - `pros` (list)
  - `cons` (list)
  - `best_for` (str)

---

### 6. `create_metrics_reference()`

Creates an expandable metrics reference section.

**Signature:**
```python
def create_metrics_reference(
    metrics: Dict[str, Dict[str, str]]
) -> None
```

**Parameters:**
- `metrics` (Dict): Dictionary with metric info:
  - `description` (str)
  - `range` (str)
  - `unit` (str)

---

### 7. `create_color_legend()`

Creates a color legend for chart interpretation.

**Signature:**
```python
def create_color_legend() -> None
```

---

## Utility Components

Located in: `components/utils.py`

### 1. `create_footer()`

Creates a reusable footer component.

**Signature:**
```python
def create_footer(
    author: str = "James Mithiga", 
    adm_no: str = "58200"
) -> None
```

---

### 2. `create_warning_banner()`

Creates a warning banner with customizable severity.

**Signature:**
```python
def create_warning_banner(
    message: str, 
    severity: str = "warning"
) -> None
```

**Severity Levels:**
- "warning": #ff6b6b (Red)
- "error": #ff4444 (Dark Red)
- "info": #ff9800 (Orange)

---

### 3. `create_success_banner()`

Creates a success banner.

**Signature:**
```python
def create_success_banner(message: str) -> None
```

---

### 4. `create_info_banner()`

Creates an info banner.

**Signature:**
```python
def create_info_banner(message: str) -> None
```

---

### 5. `create_export_section()`

Creates a data export section with multiple format options.

**Signature:**
```python
def create_export_section(
    df: pd.DataFrame, 
    title: str = "Export Data"
) -> None
```

**Supported Formats:**
- CSV
- Excel
- JSON
- Parquet

---

### 6. `create_data_table()`

Creates a reusable data table with sorting and filtering.

**Signature:**
```python
def create_data_table(
    df: pd.DataFrame, 
    title: str = "Data Table"
) -> None
```

---

### 7. `create_model_description()`

Creates a model description component.

**Signature:**
```python
def create_model_description(
    model_name: str, 
    model_type: str, 
    pros: List[str], 
    cons: List[str], 
    best_for: str
) -> None
```

---

### 8. `create_volatility_warning()`

Creates a volatility warning component.

**Signature:**
```python
def create_volatility_warning(
    df: pd.DataFrame, 
    threshold: float = 0.2
) -> None
```

---

### 9. `create_price_movement_warning()`

Creates a price movement warning component.

**Signature:**
```python
def create_price_movement_warning(
    df: pd.DataFrame, 
    current_price: float, 
    threshold: float = 0.3
) -> None
```

---

### 10. `create_confidence_bands()`

Creates confidence bands for forecast data.

**Signature:**
```python
def create_confidence_bands(
    forecast: pd.DataFrame, 
    confidence_level: int
) -> Optional[pd.DataFrame]
```

**Returns:**
- `pd.DataFrame`: Forecast with 'upper' and 'lower' columns, or None

---

## Usage Examples

### Example 1: Complete Dashboard Page

```python
import streamlit as st
import pandas as pd
from components.charts import create_candlestick_chart, create_forecast_chart
from components.controls import create_commodity_selector, create_date_range_selector
from components.layouts import create_metrics_dashboard

# Sidebar controls
commodity, symbol = create_commodity_selector()
start_date, end_date = create_date_range_selector()

# Main content
st.title(f"{commodity} Price Analysis")

# Load data
df = load_data(symbol, start_date, end_date)

# Display metrics
metrics = calculate_metrics(df)
create_metrics_dashboard(metrics)

# Display charts
fig = create_candlestick_chart(df)
st.plotly_chart(fig, use_container_width=True)
```

### Example 2: Model Comparison

```python
from components.controls import create_model_selector
from components.charts import create_comparison_chart
from components.layouts import create_performance_charts

# Select models
models_dict = {'ARIMA': 'arima', 'Prophet': 'prophet', 'LSTM': 'lstm'}
selected_models = create_model_selector(models_dict)

# Get forecasts
forecasts = {model: get_forecast(model) for model in selected_models}

# Display comparison
fig = create_comparison_chart(historical_df, forecasts)
st.plotly_chart(fig, use_container_width=True)

# Display performance metrics
metrics_df = get_model_metrics(selected_models)
create_performance_charts(metrics_df, {})
```

---

## Best Practices

### 1. Data Validation
Always validate input data before passing to components:
```python
if df is not None and len(df) > 0:
    fig = create_candlestick_chart(df)
```

### 2. Error Handling
Use try-except blocks for robust error handling:
```python
try:
    fig = create_candlestick_chart(df)
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error creating chart: {str(e)}")
```

### 3. Performance Optimization
Cache expensive operations:
```python
@st.cache_data
def load_data(symbol, start_date, end_date):
    return fetch_data(symbol, start_date, end_date)
```

### 4. Responsive Design
Use `use_container_width=True` for charts:
```python
st.plotly_chart(fig, use_container_width=True)
```

### 5. Consistent Styling
Maintain consistent styling across components by using the same theme and color schemes.

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_components_charts.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=components
```

### Test Coverage

- **test_components_charts.py**: 50+ test cases for chart components
- **test_components_controls.py**: 40+ test cases for control components
- **test_components_layouts.py**: 35+ test cases for layout components
- **test_components_utils.py**: 45+ test cases for utility components

### Test Categories

1. **Unit Tests**: Test individual component functions
2. **Integration Tests**: Test component interactions
3. **Edge Case Tests**: Test boundary conditions and error handling
4. **Mock Tests**: Test Streamlit interactions using mocks

---

## Troubleshooting

### Issue: Chart not displaying
**Solution**: Ensure data has required columns and use `st.plotly_chart(fig, use_container_width=True)`

### Issue: Component not responding to input
**Solution**: Check that Streamlit session state is properly initialized

### Issue: Performance issues with large datasets
**Solution**: Use data filtering and caching with `@st.cache_data`

### Issue: Import errors
**Solution**: Ensure components directory is in Python path and `__init__.py` exists

---

## Contributing

When adding new components:

1. Follow the existing naming convention: `create_<component_name>()`
2. Add comprehensive docstrings with parameters and returns
3. Include type hints for all parameters and returns
4. Add unit tests in the appropriate test file
5. Update this documentation with usage examples

---

## Version History

- **v1.0** (2024): Initial release with 4 component modules
- **v1.1** (2024): Added comprehensive testing suite
- **v1.2** (2024): Enhanced error handling and documentation

---

## License

These components are part of the Commodity Price Forecasting Dashboard project.

---

## Contact

For questions or issues, contact: James Mithiga (Adm No: 58200)
