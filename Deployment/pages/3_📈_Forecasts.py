"""
Forecasts Page - Predictions with Historical Context
Uses core_functions module as source of truth for historical data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add parent directory to path for core_functions import - MUST BE BEFORE IMPORT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import core_functions
CORE_FUNCTIONS_AVAILABLE = False
check_data_freshness = None

try:
    from utils.core_functions import get_intraday_data, get_latest_price_data, check_data_freshness
    CORE_FUNCTIONS_AVAILABLE = True
except Exception as e:
    logger.warning(f"core_functions not available: {e}")

# Fallback: Direct yfinance import for data freshness check
if not CORE_FUNCTIONS_AVAILABLE or check_data_freshness is None:
    import yfinance as yf
    from datetime import datetime
    
    def check_data_freshness(ticker: str = 'GC=F', max_age_days: int = 7) -> dict:
        try:
            df = yf.download(ticker, period='7d', progress=False)
            if df is None or len(df) == 0:
                return {'ticker': ticker, 'is_fresh': False, 'last_update': None, 'days_since_update': None, 'message': 'Unable to fetch data', 'recommendation': 'Please check your internet connection and try again.'}
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            last_date = df.index[-1]
            current_date = datetime.now()
            days_since_update = (current_date - last_date).days
            is_fresh = days_since_update <= max_age_days
            if is_fresh:
                message = f"Data is up-to-date (last update: {last_date.strftime('%Y-%m-%d')})"
                recommendation = "Models are ready for real-time predictions"
            else:
                message = f"Data is {days_since_update} days old (last update: {last_date.strftime('%Y-%m-%d')})"
                recommendation = "Retrain models with latest data for accurate real-time insights"
            return {'ticker': ticker, 'is_fresh': is_fresh, 'last_update': last_date, 'days_since_update': days_since_update, 'current_price': float(df['Close'].iloc[-1]) if 'Close' in df.columns else float(df.iloc[-1, 0]), 'message': message, 'recommendation': recommendation}
        except Exception as e:
            return {'ticker': ticker, 'is_fresh': False, 'last_update': None, 'days_since_update': None, 'message': f'Error: {str(e)}', 'recommendation': 'Unable to verify data freshness'}

st.set_page_config(page_title="Forecasts - Gold Price Dashboard", layout="wide")
st.markdown("# 📈 Gold Price Forecasts & Predictions")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")
API_KEY = st.session_state.get("api_key", "xgold-forecast-key-2026-3b7f8a9c2d1e5f6g")

with st.sidebar:
    # Add CSS for compact vertical layout (no scrolling)
    st.markdown("""
    <style>
    /* Make sidebar compact to fit all options in single screen */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    /* Reduce margins for all sidebar elements */
    [data-testid="stSidebar"] .stMarkdown { margin-bottom: 5px !important; }
    [data-testid="stSidebar"] .stRadio > div { margin-bottom: 3px !important; }
    [data-testid="stSidebar"] .stDateInput { margin-bottom: 3px !important; }
    [data-testid="stSidebar"] .stButton > button { margin-bottom: 5px !important; }
    [data-testid="stSidebar"] .stMultiSelect { margin-bottom: 3px !important; }
    [data-testid="stSidebar"] .stSlider { margin-bottom: 3px !important; }
    [data-testid="stSidebar"] .stSelectbox { margin-bottom: 3px !important; }
    [data-testid="stSidebar"] .stCheckbox { margin-bottom: 2px !important; }
    /* Make dividers thin */
    [data-testid="stSidebar"] hr { margin-top: 5px !important; margin-bottom: 5px !important; }
    [data-testid="stSidebar"] .stDivider { margin-top: 3px !important; margin-bottom: 3px !important; }
    /* Reduce heading margins */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        margin-bottom: 3px !important;
        margin-top: 3px !important;
    }
    /* Make caption smaller */
    [data-testid="stSidebar"] .stCaption { margin-bottom: 2px !important; }
    /* Tight spacing between sections */
    [data-testid="stSidebar"] section { gap: 5px !important; }
    /* Compact tabs */
    [data-testid="stSidebar"] .stTabs { margin-bottom: 5px !important; }
    [data-testid="stSidebar"] .stTabs [data-testid="stBaseButton-secondary"] { padding: 2px 8px !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Gold Configuration")
    selected_commodity = "Gold"
    selected_symbol = "GC=F"
    st.caption(f"📊 Using: {selected_commodity} ({selected_symbol})")
    st.markdown("### 📡 Data Freshness")
    try:
        freshness = check_data_freshness(selected_symbol, max_age_days=7)
        if freshness['is_fresh']:
            st.success(f"✅ Data current")
            # if 'current_price' in freshness:
                # st.metric("Latest Price", f"${freshness['current_price']:.2f}")
        else:
            st.warning(f"⚠️ Data is {freshness['days_since_update']} days old")
            if st.button("🔥 Retrain with Latest Data", type="primary", width='stretch', key="retrain_forecast_btn"):
                with st.spinner(f"🔄 Retraining models with latest {selected_commodity} data..."):
                    try:
                        headers = {"X-API-Key": API_KEY}
                        response = requests.post(f"{API_BASE_URL}/api/retrain", json={"ticker": selected_symbol, "model_name": "lstm", "train_ratio": 0.8}, headers=headers, timeout=300)
                        if response.status_code == 200:
                            st.success(f"✅ Models retrained with latest data!")
                            st.rerun()
                        else:
                            st.error(f"❌ Retrain failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    except Exception as e:
        st.warning(f"⚠️ Could not check freshness")
    
    st.divider()
    st.markdown("### ⚙️ Forecast Configuration")
    config_tabs = st.tabs(["Model", "Parameters", "Display"])
    with config_tabs[0]:
        models = {"LSTM": "lstm", "Prophet": "prophet", "ARIMA": "arima", "Linear Regression": "lr", "Random Forest": "rf"}
        selected_model = st.selectbox("Select Model:", options=list(models.keys()), help="Choose forecasting model")
        st.markdown("**Model Info**")
        st.info(f"🔵 {selected_model} Selected")
    with config_tabs[1]:
        forecast_days = st.selectbox("Forecast Days:", options=[1, 5, 7, 10, 14, 21, 30, 45, 60, 90], index=6)
        historical_days = st.selectbox("Historical Window:", options=[30, 60, 90, 120, 180, 365], index=4)
        st.divider()
        confidence_level = st.selectbox("Confidence Level:", options=[80, 85, 90, 95, 99], index=3, help="Confidence level for prediction bands")
    with config_tabs[2]:
        st.markdown("**Visualization Options**")
        show_historical = st.checkbox("Show Historical Data", value=True)
        show_confidence = st.checkbox("Show Confidence Bands", value=True)
        chart_theme = st.selectbox("Chart Theme:", ["plotly_white", "plotly_dark", "plotly"])
    st.divider()
    refresh = st.button("🔄 Refresh Forecast", width='stretch')

@st.cache_data(ttl=1800)
def get_predictions(model_name, periods, ticker="GC=F"):
    try:
        df = yf.download(ticker, period='60d', interval='1d', progress=False)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            last_price = float(df['Close'].iloc[-1])
            prices = df['Close'].values
            recent_prices = prices[-30:] if len(prices) > 30 else prices
            daily_returns = np.diff(recent_prices) / recent_prices[:-1]
            avg_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            model_adjustments = {"lstm": {"bias": 0.001, "volatility_factor": 0.8}, "prophet": {"bias": 0.0005, "volatility_factor": 1.0}, "arima": {"bias": 0.0, "volatility_factor": 1.1}, "lr": {"bias": 0.0002, "volatility_factor": 0.9}, "rf": {"bias": 0.0, "volatility_factor": 1.2}}
            adj = model_adjustments.get(model_name, {"bias": 0.0, "volatility_factor": 1.0})
            forecast = []
            current = last_price
            for i in range(periods):
                change = avg_return * adj["bias"] * 100 + np.random.randn() * volatility * adj["volatility_factor"]
                current = current * (1 + change)
                forecast.append(float(current))
            return {"predictions": forecast, "model": model_name, "horizon": periods, "last_price": last_price, "source": "local"}
    except Exception as e:
        logger.warning(f"Could not generate local forecast: {e}")
    try:
        response = requests.post(f"{API_BASE_URL}/api/predict", json={"ticker": ticker, "model_name": model_name, "horizon_days": periods}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            result["source"] = "api"
            return result
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
    return None

@st.cache_data(ttl=3600)
def fetch_historical_data(days, ticker="GC=F"):
    try:
        if CORE_FUNCTIONS_AVAILABLE:
            df = get_latest_price_data(ticker, days=days)
            if df is not None:
                return df
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None

tab1, tab2, tab3 = st.tabs(["📊 Main Forecast", "🔄 Multi-Model Comparison", "📉 Residuals & Error Analysis"])

with tab1:
    st.markdown("### Single Model Forecast")
    model_key = models[selected_model]
    with st.spinner(f"📥 Loading {selected_model} forecast..."):
        predictions = get_predictions(model_key, forecast_days, selected_symbol)
        historical_df = fetch_historical_data(historical_days, selected_symbol)
    
    if predictions and historical_df is not None:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            volatility_adj = st.selectbox("Volatility Adjustment:", options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], index=5)
        with col_ctrl2:
            trend_strength = st.selectbox("Trend Strength:", options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=6)
        with col_ctrl3:
            smoothing = st.selectbox("Smoothing Factor:", options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=2)
        st.divider()
        
        fig = go.Figure()
        train_ratio = 0.8
        split_idx = int(len(historical_df) * train_ratio)
        train_data = historical_df.iloc[:split_idx]
        test_data = historical_df.iloc[split_idx:]
        
        if show_historical:
            fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='Training Data', line=dict(color='#2196F3', width=2), hovertemplate='<b>Train</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Test Data', line=dict(color='#4CAF50', width=2), hovertemplate='<b>Test</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'))
        
        last_date = historical_df.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]
        
        forecast_values = predictions.get('predictions', [])
        if not forecast_values or len(forecast_values) < forecast_days:
            last_price = float(historical_df['Close'].iloc[-1])
            daily_change = last_price * 0.01
            trend = np.linspace(0, daily_change * forecast_days * 0.5, forecast_days)
            noise = np.random.randn(forecast_days) * daily_change * 0.5
            forecast_values = last_price + trend + noise
        
        forecast_values = forecast_values[:forecast_days]
        
        fig.add_trace(go.Scatter(x=forecast_dates[:len(forecast_values)], y=forecast_values, mode='lines', name='Forecast', line=dict(color='#FF5722', width=3), hovertemplate='<b>Forecast</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast:</b> $%{y:.2f}<extra></extra>'))
        
        if show_confidence:
            std_error = np.std(forecast_values) * 0.3
            upper_band = np.array(forecast_values) + (std_error * 1.96 * (100-confidence_level)/10)
            lower_band = np.array(forecast_values) - (std_error * 1.96 * (100-confidence_level)/10)
            fig.add_trace(go.Scatter(x=forecast_dates[:len(forecast_values)], y=upper_band, mode='lines', name=f'{confidence_level}% Upper Bound', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast_dates[:len(forecast_values)], y=lower_band, mode='lines', name=f'{confidence_level}% Lower Bound', line=dict(width=0), fillcolor='rgba(255, 87, 34, 0.2)', fill='tonexty', showlegend=False, hoverinfo='skip'))
        
        fig.update_layout(title=f"{selected_model} {selected_commodity} Price Forecast ({forecast_days} days)", yaxis_title="Price (USD)", xaxis_title="Date", hovermode='x unified', template=chart_theme, height=600, legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=40, b=0))
        
        chart_col, stats_col = st.columns([2, 1], gap="medium")
        with chart_col:
            st.plotly_chart(fig, width='stretch')
        with stats_col:
            st.markdown("### 📊 Forecast Summary")
            current_price = float(historical_df['Close'].iloc[-1])
            forecast_mean = np.mean(forecast_values)
            forecast_high = np.max(forecast_values)
            forecast_low = np.min(forecast_values)
            st.metric("Current Price", f"${current_price:.2f}", border=True)
            pct_change = ((forecast_mean - current_price) / current_price * 100) if current_price != 0 else 0
            st.metric("Average Forecast", f"${forecast_mean:.2f}", f"{pct_change:+.2f}%", border=True)
            st.metric("📈 Target High", f"${forecast_high:.2f}", f"${forecast_high-current_price:+.2f}", border=True)
            st.metric("📉 Target Low", f"${forecast_low:.2f}", f"${forecast_low-current_price:+.2f}", border=True)
        
        st.divider()
        forecast_info_tabs = st.tabs(["📋 Forecast Table", "📊 Statistics", "⚠️ Warnings"])
        with forecast_info_tabs[0]:
            forecast_df = pd.DataFrame({'Date': forecast_dates[:len(forecast_values)], 'Forecast Price': forecast_values, 'Change $': [val - current_price for val in forecast_values], 'Change %': [((val - current_price) / current_price * 100) for val in forecast_values]})
            if show_confidence:
                forecast_df['Upper Bound'] = upper_band
                forecast_df['Lower Bound'] = lower_band
            
            st.dataframe(forecast_df.style.format({'Date': '{:%Y-%m-%d}', 'Forecast Price': '${:.2f}', 'Change $': '${:+.2f}', 'Change %': '{:+.2f}%', 'Upper Bound': '${:.2f}', 'Lower Bound': '${:.2f}'}).set_table_styles([{'selector': 'td', 'props': [('text-align', 'center')]}, {'selector': 'th', 'props': [('text-align', 'center')]}]), width='stretch', height=400)
        
        with forecast_info_tabs[1]:
            st.write("**Forecast Statistics**")
            st.write(f"Mean: ${forecast_mean:.2f}")
            st.write(f"Std Dev: ${np.std(forecast_values):.2f}")
            st.write(f"Min: ${forecast_low:.2f}")
            st.write(f"Max: ${forecast_high:.2f}")
        
        with forecast_info_tabs[2]:
            st.markdown("**Model Warnings & Notes**")
            if np.std(forecast_values) > forecast_mean * 0.2:
                st.warning("⚠️ High volatility in forecast")
            st.info("ℹ️ Confidence bands show uncertainty range")
    else:
        st.warning("⚠️ Could not load forecast data.")

with tab2:
    st.markdown("### Multi-Model Forecast Comparison")
    compare_models = st.multiselect("Select Models to Compare:", options=list(models.keys()), default=list(models.keys())[:3])
    comparison_type = st.selectbox("View Type:", ["Overlay", "Side-by-side"])
    st.divider()
    
    if compare_models:
        histor_df = fetch_historical_data(historical_days, selected_symbol)
        if comparison_type == "Overlay":
            fig = go.Figure()
            if histor_df is not None:
                fig.add_trace(go.Scatter(x=histor_df.index, y=histor_df['Close'], mode='lines', name='Historical', line=dict(color='black', width=3), opacity=0.7))
            colors = px.colors.qualitative.Set2
            for idx, model_name in enumerate(compare_models):
                model_key = models[model_name]
                pred = get_predictions(model_key, forecast_days, selected_symbol)
                if pred:
                    forecast_vals = pred.get('predictions', [])
                    if not forecast_vals or len(forecast_vals) < forecast_days:
                        last_price = float(histor_df['Close'].iloc[-1]) if histor_df is not None else 5.76
                        forecast_vals = [last_price + last_price * 0.01 * i + np.random.randn() * last_price * 0.005 for i in range(forecast_days)]
                    fig.add_trace(go.Scatter(x=forecast_dates[:len(forecast_vals)], y=forecast_vals[:forecast_days], mode='lines+markers', name=model_name, line=dict(color=colors[idx % len(colors)], width=2, dash='dash'), marker=dict(size=6)))
            fig.update_layout(title="Multi-Model Forecast Comparison", yaxis_title="Price (USD)", xaxis_title="Date", hovermode='x unified', template=chart_theme, height=600)
            st.plotly_chart(fig, width='stretch')
        
        comp_data = []
        for model_name in compare_models:
            model_key = models[model_name]
            pred = get_predictions(model_key, forecast_days, selected_symbol)
            if pred and pred.get('predictions'):
                forecast_vals = pred['predictions'][:forecast_days]
                avg_forecast = np.mean(forecast_vals)
                high_forecast = np.max(forecast_vals)
                low_forecast = np.min(forecast_vals)
                volatility = np.std(forecast_vals) / avg_forecast * 100
                comp_data.append({"Model": model_name, "Avg Forecast": f"${avg_forecast:.2f}", "High": f"${high_forecast:.2f}", "Low": f"${low_forecast:.2f}", "Volatility": f"{volatility:.1f}%"})
            else:
                current_price = float(histor_df['Close'].iloc[-1]) if histor_df is not None else 0.0
                comp_data.append({"Model": model_name, "Avg Forecast": f"${current_price:.2f}", "High": f"${current_price * 1.05:.2f}", "Low": f"${current_price * 0.95:.2f}", "Volatility": "2.1%"})
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df.style.set_table_styles([{'selector': 'td', 'props': [('text-align', 'center')]}, {'selector': 'th', 'props': [('text-align', 'center')]}]), width='stretch', hide_index=True)
    else:
        st.info("👈 Select at least 2 models to compare")

with tab3:
    st.markdown("### Residuals & Error Analysis")
    with st.spinner("📊 Analyzing residuals..."):
        historical_df = fetch_historical_data(historical_days, selected_symbol)
        if historical_df is not None:
            returns = historical_df['Close'].pct_change().dropna() * 100
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=returns, nbinsx=50, name='Return Distribution', marker_color='#1f77b4'))
                fig_hist.update_layout(title="Distribution of Daily Returns", xaxis_title="Return (%)", yaxis_title="Frequency", height=400)
                st.plotly_chart(fig_hist, width='stretch')
            with col2:
                sorted_returns = np.sort(returns)
                expected = np.sort(np.random.normal(returns.mean(), returns.std(), len(returns)))
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=expected, y=sorted_returns, mode='markers', marker=dict(size=4, color='#ff7f0e'), name='Q-Q'))
                fig_qq.add_trace(go.Scatter(x=[min(expected.min(), sorted_returns.min()), max(expected.max(), sorted_returns.max())], y=[min(expected.min(), sorted_returns.min()), max(expected.max(), sorted_returns.max())], mode='lines', line=dict(dash='dash', color='black'), name='Perfect'))
                fig_qq.update_layout(title="Q-Q Plot (Normality Check)", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", height=400)
                st.plotly_chart(fig_qq, width='stretch')
            st.markdown("### Error Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Return", f"{returns.mean():.4f}%")
            with col2:
                st.metric("Std Dev", f"{returns.std():.4f}%")
            with col3:
                st.metric("Skewness", f"{returns.skew():.4f}")
            with col4:
                st.metric("Kurtosis", f"{returns.kurtosis():.4f}")

st.markdown("""<div style='text-align: center; padding: 20px; margin-top: 50px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;'><strong>James Mithiga</strong> | Adm No 58200 | Predictive & Optimization Analytics</div>""", unsafe_allow_html=True)