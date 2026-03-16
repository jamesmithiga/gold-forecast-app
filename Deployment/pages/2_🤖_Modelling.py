"""
Modelling Page - Models, Metrics, and Performance Comparison
Uses core_functions module as source of truth for data preparation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add parent directory to path for core_functions import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import core_functions
CORE_FUNCTIONS_AVAILABLE = False
check_data_freshness = None

try:
    from utils.core_functions import get_data_summary, load_and_prepare_data, calculate_metrics, check_data_freshness
    CORE_FUNCTIONS_AVAILABLE = True
except Exception as e:
    logger.warning(f"core_functions not available: {e}")

# Fallback: Direct yfinance import for data freshness check
if not CORE_FUNCTIONS_AVAILABLE or check_data_freshness is None:
    import yfinance as yf
    from datetime import datetime
    
    def check_data_freshness(ticker: str = 'GC=F', max_age_days: int = 7) -> dict:
        """
        Fallback data freshness check using direct yfinance
        """
        try:
            df = yf.download(ticker, period='7d', progress=False)
            if df is None or len(df) == 0:
                return {
                    'ticker': ticker,
                    'is_fresh': False,
                    'last_update': None,
                    'days_since_update': None,
                    'message': 'Unable to fetch data',
                    'recommendation': 'Please check your internet connection and try again.'
                }
            
            # Handle MultiIndex columns
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
            
            return {
                'ticker': ticker,
                'is_fresh': is_fresh,
                'last_update': last_date,
                'days_since_update': days_since_update,
                'current_price': float(df['Close'].iloc[-1]) if 'Close' in df.columns else float(df.iloc[-1, 0]),
                'message': message,
                'recommendation': recommendation
            }
        except Exception as e:
            return {
                'ticker': ticker,
                'is_fresh': False,
                'last_update': None,
                'days_since_update': None,
                'message': f'Error: {str(e)}',
                'recommendation': 'Unable to verify data freshness'
            }

st.set_page_config(page_title="Modelling - Gold Price Dashboard", layout="wide")

st.markdown("<h1 style='font-size: 36px; font-weight: bold;'>🤖 Gold Price Modelling & Performance Metrics</h1>", unsafe_allow_html=True)

# Configuration
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")
API_KEY = st.session_state.get("api_key", "xgold-forecast-key-2026-3b7f8a9c2d1e5f6g")

# Available models (only models trained in the notebook)
AVAILABLE_MODELS = {
    "LSTM": "lstm",
    "Prophet": "prophet",
    "ARIMA": "arima",
    "SARIMA": "sarima",
    "Linear Regression": "lr",
    "Random Forest": "rf"
}

# Metrics definitions
METRICS_INFO = {
    "RMSE": {
        "description": "Root Mean Squared Error - lower is better",
        "range": "0 to ∞",
        "unit": "$"
    },
    "MAE": {
        "description": "Mean Absolute Error - lower is better",
        "range": "0 to ∞",
        "unit": "$"
    },
    "MAPE": {
        "description": "Mean Absolute Percentage Error - lower is better",
        "range": "0% to 100%",
        "unit": "%"
    },
    "R²": {
        "description": "Coefficient of Determination - higher is better",
        "range": "-∞ to 1",
        "unit": ""
    },
    "Directional Accuracy": {
        "description": "Percentage of correctly predicted price movements",
        "range": "0% to 100%",
        "unit": "%"
    }
}

@st.cache_data(ttl=3600)
def fetch_model_comparison(ticker: str = "GC=F"):
    """Fetch comparison data - prioritize local metrics.json"""
    # First try to load from local metrics.json (fast, no network needed)
    try:
        import json
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            ticker_data = data.get(ticker, {})
            if ticker_data:
                logger.info(f"Loaded comparison data from local metrics.json")
                return ticker_data
    except Exception as e:
        logger.error(f"Error loading local metrics: {str(e)}")
    
    # Fallback to API
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/comparison",
            params={"ticker": ticker},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching comparison: {str(e)}")
        return None

def fetch_model_metrics(model_name: str, ticker: str = "GC=F"):
    """Fetch metrics for specific model and commodity - prioritize local metrics.json"""
    # First try to load from local metrics.json (fast, no network needed)
    local_metrics = _load_metrics_from_file(ticker, model_name)
    if local_metrics:
        logger.info(f"Loaded metrics from local metrics.json for {model_name}")
        return local_metrics
    
    # Fallback to API if local file not available
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/metrics/{model_name}",
            params={"ticker": ticker},
            timeout=10  # Reduced timeout for faster fallback
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch metrics for {model_name} on {ticker}: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        return None

def _load_metrics_from_file(ticker: str, model_name: str):
    """Load metrics from metrics.json file as fallback"""
    import json
    try:
        # Map model names to metrics.json keys
        model_key_map = {
            "lstm": "lstm",
            "prophet": "prophet",
            "arima": "arima",
            "sarima": "sarima",
            "lr": "linearregression",
            "rf": "randomforest"
        }
        # Get the correct key for metrics.json
        model_key = model_key_map.get(model_name.lower(), model_name.lower())
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics.json"),
            os.path.join(os.path.dirname(__file__), "..", "metrics.json"),
            "metrics.json"
        ]
        
        metrics_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metrics_path = path
                break
        
        if metrics_path is None:
            logger.error(f"Metrics file not found in any of: {possible_paths}")
            return None
            
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        ticker_data = data.get(ticker, {})
        
        if model_key in ticker_data:
            m = ticker_data[model_key]
            return {
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "mape": m.get("mape"),
                "r2": m.get("r2"),
                "directional_accuracy": m.get("directional_accuracy")
            }
        return None
    except Exception as e:
        logger.error(f"Error loading metrics from file: {str(e)}")
        return None

# Sidebar controls
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
    
    # st.markdown("### 🔧 Gold Configuration")
    
    # Gold configuration
    selected_commodity = "Gold"
    selected_symbol = "GC=F"
    selected_symbols = ["GC=F"]
    
    st.caption(f"📊 Using: {selected_commodity} ({selected_symbol})")
    
    # ============================================================
    # DATA FRESHNESS CHECK - Prompt for retraining with latest data
    # ============================================================
    st.markdown("### 📡 Data Freshness Check")
    
    # Check data freshness (using fallback if core_functions not available)
    try:
        freshness = check_data_freshness(selected_symbol, max_age_days=7)
        
        if freshness['is_fresh']:
            last_date = freshness.get('last_update')
            if last_date:
                date_str = last_date.strftime('%Y-%m-%d') if isinstance(last_date, datetime) else str(last_date)
                st.success(f"✅ Data is up-to-date as at {date_str}")
            else:
                st.success(f"✅ Data is up-to-date")
            # if 'current_price' in freshness:
                # st.metric("Current Price", f"${freshness['current_price']:.2f}")
        else:
            st.warning(f"⚠️ {freshness['message']}")
            st.error(f"🔄 {freshness['recommendation']}")
            
            # Prominent retrain button with latest data
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); 
                        padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <strong>🚀 Retrain Now for Real-Time Insights!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            col_retrain_now1, col_retrain_now2 = st.columns([2, 1])
            with col_retrain_now1:
                retrain_with_latest = st.button(
                    "🔥 Retrain with Latest Yahoo Finance Data",
                    type="primary",
                    width='stretch',
                    key="retrain_latest_btn"
                )
            with col_retrain_now2:
                st.write("")
                st.write("")
                st.caption("Fetches latest data and retrains models")
            
            # Handle retrain with latest data
            if retrain_with_latest:
                with st.spinner(f"🔄 Fetching latest {selected_commodity} data from Yahoo Finance and retraining models..."):
                    try:
                        # Fetch latest data from Yahoo Finance (backend automatically does this)
                        headers = {"X-API-Key": API_KEY}
                        response = requests.post(
                            f"{API_BASE_URL}/api/retrain",
                            json={
                                "ticker": selected_symbol,
                                "model_name": "lstm",
                                "train_ratio": 0.8
                            },
                            headers=headers,
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Models retrained with latest {selected_commodity} data!")
                            st.markdown("**Training Results:**")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("RMSE", f"${result['metrics']['rmse']:.2f}")
                            with col2:
                                st.metric("MAE", f"${result['metrics']['mae']:.2f}")
                            with col3:
                                st.metric("MAPE", f"{result['metrics']['mape']:.2f}%")
                            with col4:
                                st.metric("R²", f"{result['metrics']['r2']:.4f}")
                            with col5:
                                st.metric("Directional Acc", f"{result['metrics']['directional_accuracy']:.1f}%")
                            
                            st.info(f"📊 Training samples: {result['train_samples']} | Test samples: {result['test_samples']}")
                        else:
                            st.error(f"❌ Retrain failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    except Exception as e:
        st.warning(f"⚠️ Could not check data freshness: {str(e)}")
    
    st.markdown("---")  # Add visual separator
    
    st.markdown("### 🔧 Model Selection")
    
    # Create sidebar tabs for better organization
    sidebar_tabs = st.tabs(["Models", "Filters", "Settings"])
    
    with sidebar_tabs[0]:
        selected_models = st.multiselect(
            "Select Models to Compare:",
            options=list(AVAILABLE_MODELS.keys()),
            default=list(AVAILABLE_MODELS.keys())[:3],
            help="Choose multiple models to compare"
        )
    
    with sidebar_tabs[1]:
        # Filter by metric
        st.markdown("**Performance Filter**")
        min_accuracy = st.slider(
            "Minimum Accuracy (%)",
            0.0, 100.0, 70.0, 5.0,
            help="Filter models by minimum directional accuracy"
        )
        
        max_rmse = st.slider(
            "Maximum RMSE ($)",
            0.0, 100.0, 50.0, 5.0,
            help="Filter models by maximum RMSE error"
        )
    
    with sidebar_tabs[2]:
        metric_view = st.radio(
            "View Type:",
            options=["Comparison", "Detailed"],
            help="Switch between comparison and detailed views"
        )
        
        st.markdown("**Display Options**")
        show_radar = st.checkbox("Show Radar Chart", value=True)
        show_heatmap = st.checkbox("Show Heatmap", value=True)
    
    st.divider()
    
    # ============================================================
    # MODEL RETRAINING SECTION - Uses sidebar selection
    # ============================================================
    st.markdown("### 🔄 Model Retraining")
    
    # Show selected models from sidebar
    if selected_models:
        st.markdown(f"**Selected Models:** {', '.join(selected_models)}")
    else:
        st.markdown("⚠️ No models selected - Please select models from sidebar")
    
    # Two-column layout with proper spacing and alignment
    col1, col2 = st.columns(2, gap="medium")
    
    # with col1:
        # st.markdown("**Train : Test**")
        # st.caption("Default: 80:20")
    
    with col2:
        st.markdown(" ")
        st.markdown(" ")
    
    # Action button - full width
    retrain_button = st.button(
        "🚀 Retrain Selected Models", 
        type="primary", 
        width='stretch',
        key="main_retrain_btn",
        disabled=not selected_models
    )
    
    # Handle retrain button click - retrain all selected models
    if retrain_button and selected_models:
        for model_name in selected_models:
            with st.spinner(f"Retraining {model_name} for {selected_commodity}..."):
                try:
                    headers = {"X-API-Key": API_KEY}
                    response = requests.post(
                        f"{API_BASE_URL}/api/retrain",
                        json={
                            "ticker": selected_symbol,
                            "model_name": AVAILABLE_MODELS[model_name],
                            "train_ratio": 0.8
                        },
                        headers=headers,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        api_name_map = {
                            "LSTM": "LSTM",
                            "Prophet": "Prophet",
                            "ARIMA": "ARIMA",
                            "SARIMA": "SARIMA",
                            "Linear Regression": "LinearRegression",
                            "Random Forest": "RandomForest"
                        }
                        expected_api_model = api_name_map.get(model_name, model_name)
                        
                        # Try new format first (metrics_by_model), then fall back to old format
                        trained_metrics = {}
                        if 'metrics_by_model' in result and expected_api_model in result.get('metrics_by_model', {}):
                            trained_metrics = result.get('metrics_by_model', {})
                        elif 'metrics' in result:
                            # Check if metrics is already keyed by model name or is the raw metrics
                            if isinstance(result.get('metrics'), dict) and expected_api_model in result.get('metrics', {}):
                                trained_metrics = result.get('metrics', {})
                            else:
                                # Raw metrics format - wrap it with model name
                                trained_metrics = {expected_api_model: result.get('metrics', {})}
                        
                        if expected_api_model in trained_metrics:
                            st.success(f"✅ {model_name} model retrained successfully!")
                        else:
                            failed = result.get("models_failed", "unknown") if isinstance(result, dict) else "unknown"
                            st.warning(f"⚠️ {model_name} was not retrained (models_failed={failed}). Check backend logs for details.")
                    else:
                        st.error(f"❌ {model_name} retrain failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error retraining {model_name}: {str(e)}")
    
    refresh = st.button("🔄 Refresh Data", width='stretch')
    if refresh:
        st.cache_data.clear()
        st.rerun()

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Metrics Comparison", "🎯 Model Details"])

with tab1:
    st.markdown("### Model Performance Summary")
    with st.spinner("📊 Loading model metrics from local data..."):
        models_data = []
        for model_name in selected_models:
            model_key = AVAILABLE_MODELS[model_name].lower()
            metrics = fetch_model_metrics(model_key, selected_symbol)
            if metrics:
                models_data.append({
                    "Model": model_name,
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                    "MAPE": metrics.get("mape", 0),
                    "R²": metrics.get("r2", 0),
                    "Accuracy": metrics.get("directional_accuracy", 0)
                })

        if models_data:
            df_models = pd.DataFrame(models_data)
            valid_models = [m for m in selected_models if m in df_models['Model'].values]
            if valid_models:
                cols = st.columns(len(valid_models))
                for idx, (col, model) in enumerate(zip(cols, valid_models)):
                    with col:
                        model_data = df_models[df_models["Model"] == model].iloc[0]
                        # Handle null/NaN values for display - Focus on error metrics (RMSE, MAPE)
                        rmse_val = model_data['RMSE'] if pd.notna(model_data['RMSE']) and model_data['RMSE'] is not None else "N/A"
                        mape_val = model_data['MAPE'] if pd.notna(model_data['MAPE']) and model_data['MAPE'] is not None else "N/A"
                        if isinstance(rmse_val, (int, float)):
                            rmse_display = f"${rmse_val:.2f}"
                        else:
                            rmse_display = rmse_val
                        if isinstance(mape_val, (int, float)):
                            mape_display = f"{mape_val:.2f}%"
                        else:
                            mape_display = mape_val
                        st.metric(
                            model,
                            f"{rmse_display}",
                            f"MAPE: {mape_display}"
                        )
            st.markdown("### Detailed Metrics Table")
            display_df = df_models.copy()
            display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A")
            display_df["MAE"] = display_df["MAE"].apply(lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A")
            display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "N/A")
            display_df["R²"] = display_df["R²"].apply(lambda x: f"{x:.4f}" if pd.notna(x) and x is not None else "N/A")
            display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "N/A")
            st.dataframe(display_df.style.set_table_styles([
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ]), width='stretch', hide_index=True)
        else:
            st.warning("⚠️ No model data available")

with tab2:
    st.markdown("### Metrics Comparison Charts")
    
    if models_data:
        df_models = pd.DataFrame(models_data)
        
        # Split error metrics and accuracy metrics
        error_col, accuracy_col = st.columns(2)
        
        with error_col:
            st.subheader("❌ Error Metrics")
            
            # RMSE Comparison
            fig_rmse = px.bar(
                df_models,
                x="Model",
                y="RMSE",
                title="RMSE (Lower is Better)",
                color="RMSE",
                color_continuous_scale="Reds"
            )
            fig_rmse.update_layout(height=350)
            st.plotly_chart(fig_rmse, width='stretch')
            
            # MAE Comparison
            fig_mae = px.bar(
                df_models,
                x="Model",
                y="MAE",
                title="MAE (Lower is Better)",
                color="MAE",
                color_continuous_scale="Oranges"
            )
            fig_mae.update_layout(height=350)
            st.plotly_chart(fig_mae, width='stretch')
        
        with accuracy_col:
            st.subheader("📊 Error & Fit Metrics")
            
            # R² Comparison
            fig_r2 = px.bar(
                df_models,
                x="Model",
                y="R²",
                title="R² Score (Higher is Better)",
                color="R²",
                color_continuous_scale="Greens"
            )
            fig_r2.update_layout(height=350)
            st.plotly_chart(fig_r2, width='stretch')
            
            # MAPE Comparison (focus on error metric)
            fig_mape = px.bar(
                df_models,
                x="Model",
                y="MAPE",
                title="MAPE % (Lower is Better)",
                color="MAPE",
                color_continuous_scale="Purples"
            )
            fig_mape.update_layout(height=350)
            st.plotly_chart(fig_mape, width='stretch')
        
        # Radar chart
        st.markdown("### Model Performance Radar Chart")
        
        # Normalize metrics for radar - Focus on error metrics (RMSE, MAPE)
        df_normalized = df_models.copy()
        df_normalized["RMSE"] = 100 - (df_normalized["RMSE"] / df_normalized["RMSE"].max() * 100)
        df_normalized["MAE"] = 100 - (df_normalized["MAE"] / df_normalized["MAE"].max() * 100)
        df_normalized["MAPE"] = 100 - (df_normalized["MAPE"] / df_normalized["MAPE"].max() * 100)
        df_normalized["R²"] = df_normalized["R²"] * 100
        
        fig_radar = go.Figure()
        
        for idx, row in df_normalized.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row["RMSE"], row["MAE"], row["MAPE"], row["R²"]],
                theta=["RMSE Fit", "MAE Fit", "MAPE Fit", "R² Score"],
                fill='toself',
                name=row["Model"]
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=500
        )
        st.plotly_chart(fig_radar, width='stretch')

with tab3:
    st.markdown("### Model Information & Specifications")
    
    # Model descriptions
    model_descriptions = {
        "LSTM": {
            "type": "Deep Learning - Recurrent Neural Network",
            "pros": ["Excellent for complex patterns", "Captures long-term dependencies"],
            "cons": ["Requires more data", "Computationally intensive"],
            "best_for": "Volatile markets with complex patterns"
        },
        "Prophet": {
            "type": "Statistical - Additive Model",
            "pros": ["Handles seasonal patterns", "Works with limited data"],
            "cons": ["May miss sudden changes", "Less flexible"],
            "best_for": "Regular seasonal patterns"
        },
        "ARIMA": {
            "type": "Statistical - AutoRegressive Integrated Moving Average",
            "pros": ["Theoretically sound", "Good for stationary data"],
            "cons": ["Requires stationarity", "Less flexible"],
            "best_for": "Traditional time series"
        },
        "SARIMA": {
            "type": "Statistical - Seasonal ARIMA",
            "pros": ["Handles seasonal patterns", "Captures periodic trends"],
            "cons": ["More complex tuning", "Computationally intensive"],
            "best_for": "Time series with strong seasonal components"
        },
        "SARIMAX": {
            "type": "Statistical - Seasonal ARIMA with Exogenous variables",
            "pros": ["Handles seasonality", "Can include external factors"],
            "cons": ["Requires more data", "Complex to configure"],
            "best_for": "Seasonal data with external regressors"
        },
        "Exponential Smoothing": {
            "type": "Statistical - Weighted Average",
            "pros": ["Simple and interpretable", "Fast to compute"],
            "cons": ["Assumes linear trends", "Limited complexity"],
            "best_for": "Simple trends and patterns"
        },
        "Gradient Boosting": {
            "type": "Machine Learning - Ensemble",
            "pros": ["Handles non-linearity", "Robust to outliers"],
            "cons": ["Risk of overfitting", "Less interpretable"],
            "best_for": "Complex multivariate relationships"
        }
    }
    
    selected_detail = st.selectbox(
        "Select Model for Details:",
        options=selected_models
    )
    
    if selected_detail in model_descriptions:
        info = model_descriptions[selected_detail]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {selected_detail}")
            st.write(f"**Type:** {info['type']}")
            st.write("**Advantages:**")
            for pro in info['pros']:
                st.write(f"✅ {pro}")
        
        with col2:
            st.write("")
            st.write("")
            st.write("**Disadvantages:**")
            for con in info['cons']:
                st.write(f"❌ {con}")
            st.write(f"\n**Best For:** {info['best_for']}")
    
    # Metrics reference
    st.markdown("### 📚 Metrics Reference")
    
    for metric_name, metric_info in METRICS_INFO.items():
        with st.expander(f"📖 {metric_name}"):
            st.write(f"**Description:** {metric_info['description']}")
            st.write(f"**Range:** {metric_info['range']}")
            if metric_info['unit']:
                st.write(f"**Unit:** {metric_info['unit']}")

# Add footer
st.markdown("""
<div style='text-align: center; padding: 20px; margin-top: 50px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;'>
    <strong>James Mithiga</strong> | Adm No 58200 | Predictive & Optimization Analytics
</div>
""", unsafe_allow_html=True)
