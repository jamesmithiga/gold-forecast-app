"""
Reusable Streamlit control components
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


def create_commodity_selector(default_commodity: str = "Gold") -> Tuple[str, str]:
    """
    Create a reusable commodity selector with sidebar controls
    Gold only version - returns Gold and GC=F
    
    Args:
        default_commodity: Default commodity to select (ignored, always Gold)
    
    Returns:
        Tuple of (selected_commodity, selected_symbol)
    """
    selected_commodity = "Gold"
    selected_symbol = "GC=F"
    st.caption(f"📊 Using: {selected_commodity} ({selected_symbol})")
    
    return selected_commodity, selected_symbol


def create_date_range_selector() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Create a reusable date range selector with quick period options
    
    Returns:
        Tuple of (start_date, end_date)
    """
    quick_period = st.selectbox(
        "Quick Select:",
        options=["1M", "3M", "6M", "Custom"],
        index=1,
        help="Quick date range selection"
    )
    
    if quick_period == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=180),
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now(),
                max_value=datetime.now()
            )
    else:
        period_days = {
            "1M": 30,
            "3M": 90,
            "6M": 180
        }
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days.get(quick_period, 180))
    
    # Convert date to datetime for comparison with dataframe index
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    return start_date, end_date


def create_model_selector(models: Dict[str, str], default_models: List[str] = None) -> List[str]:
    """
    Create a reusable model selector
    
    Args:
        models: Dictionary of model names and keys
        default_models: Default models to select
    
    Returns:
        List of selected model names
    """
    if default_models is None:
        default_models = list(models.keys())[:3]
    
    selected_models = st.multiselect(
        "Select Models to Compare:",
        options=list(models.keys()),
        default=default_models,
        help="Choose multiple models to compare"
    )
    
    return selected_models


def create_forecast_config() -> Dict[str, Any]:
    """
    Create a reusable forecast configuration panel
    
    Returns:
        Dictionary of forecast configuration parameters
    """
    config = {}
    
    # Forecast periods with selectbox
    config["forecast_days"] = st.selectbox(
        "Forecast Days:",
        options=[1, 5, 7, 10, 14, 21, 30, 45, 60, 90],
        index=6,
        help="Number of days to forecast ahead"
    )
    
    # Historical data window
    config["historical_days"] = st.selectbox(
        "Historical Window:",
        options=[30, 60, 90, 120, 180, 365],
        index=4,
        help="Number of historical days to display"
    )
    
    # Confidence interval slider
    config["confidence_level"] = st.slider(
        "Confidence Level:",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help="Confidence level for prediction bands"
    )
    
    # Visualization options
    config["show_historical"] = st.checkbox("Show Historical Data", value=True)
    config["show_confidence"] = st.checkbox("Show Confidence Bands", value=True)
    
    config["chart_theme"] = st.selectbox(
        "Chart Theme:",
        ["plotly_white", "plotly_dark", "plotly"]
    )
    
    return config


def create_performance_filters() -> Dict[str, float]:
    """
    Create reusable performance filters
    
    Returns:
        Dictionary of filter parameters
    """
    filters = {}
    
    # Filter by metric
    st.markdown("**Performance Filter**")
    filters["min_accuracy"] = st.selectbox(
        "Minimum Accuracy (%)",
        options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        index=7,
        help="Filter models by minimum directional accuracy"
    )
    
    filters["max_rmse"] = st.selectbox(
        "Maximum RMSE ($)",
        options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        index=5,
        help="Filter models by maximum RMSE error"
    )
    
    return filters


def create_data_freshness_check(ticker: str, max_age_days: int = 7) -> None:
    """
    Create a reusable data freshness check component
    
    Args:
        ticker: Commodity symbol to check
        max_age_days: Maximum age in days for fresh data
    """
    st.markdown("### 📡 Data Freshness Check")
    
    try:
        freshness = check_data_freshness(ticker, max_age_days=max_age_days)
        
        if freshness['is_fresh']:
            last_date = freshness.get('last_update')
            if last_date:
                date_str = last_date.strftime('%Y-%m-%d') if isinstance(last_date, datetime) else str(last_date)
                st.success(f"✅ Data is up-to-date as at {date_str}")
            else:
                st.success("✅ Data is up-to-date")
            if 'current_price' in freshness:
                st.metric("Current Price", f"${freshness['current_price']:.2f}")
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
                with st.spinner(f"🔄 Fetching latest {ticker} data from Yahoo Finance and retraining models..."):
                    try:
                        # Fetch latest data from Yahoo Finance (backend automatically does this)
                        headers = {"X-API-Key": API_KEY}
                        response = requests.post(
                            f"{API_BASE_URL}/api/retrain",
                            json={
                                "ticker": ticker,
                                "model_name": "lstm",
                                "train_ratio": 0.8
                            },
                            headers=headers,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Models retrained with latest {ticker} data!")
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