"""
Data Page - Historical Gold Price Data with Candlestick Charts
Uses core_functions module as single source of truth for data fetching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add parent directory to path for core_functions import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.core_functions import get_intraday_data
    CORE_FUNCTIONS_AVAILABLE = True
except ImportError:
    CORE_FUNCTIONS_AVAILABLE = False
    logger.warning("core_functions module not available; falling back to direct yfinance calls")

st.set_page_config(page_title="📊 Gold Price Forecasting Dashboard", layout="wide")

st.markdown("# 📊 Historical Data")

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
    </style>
    """, unsafe_allow_html=True)
    
    # st.title("📊 Historical Data")
    
    # Gold configuration
    selected_commodity = "Gold"
    selected_symbol = "GC=F"
    selected_stock = "Gold Futures"
    ticker = "GC=F"
    
    st.caption(f"📊 Using: {selected_commodity} ({selected_symbol})")
    
    st.markdown("---")
    
    # Default settings
    interval = "1d"
    show_volume = True
    show_avg = True
    show_stats = True
    
    # Quick period selector at top level
    quick_period = st.selectbox(
        "Quick Select:",
        options=["1M", "3M", "6M", "Custom"],
        index=1,
        help="Quick date range selection"
    )
    
    # Date range selector
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
    
    # Refresh button
    st.divider()
    refresh = st.button("🔄 Refresh Data", width='stretch')

# Validation
if start_date >= end_date:
    st.error("❌ Start date must be before end date")
    st.stop()

# Load data with caching - Using core_functions as source of truth
@st.cache_data(ttl=3600)
def load_stock_data(ticker, start, end, interval):
    """
    Load stock data using core_functions module (from notebook)
    Falls back to direct yfinance if core_functions not available
    """
    try:
        # Calculate period from date range
        days_diff = (end - start).days
        period_str = f"{days_diff}d" if days_diff > 0 else "30d"
        
        # Try to use core_functions first
        if CORE_FUNCTIONS_AVAILABLE:
            logger.info(f"Loading {ticker} data via core_functions (period: {period_str}, interval: {interval})")
            data = get_intraday_data(ticker, period=period_str, interval=interval)
            if data is not None:
                # Filter to requested date range
                data = data.loc[(data.index >= start) & (data.index <= end)]
                return data
        
        # Fallback to direct yfinance
        logger.info(f"Falling back to direct yfinance for {ticker}")
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False
        )
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
with st.spinner("📥 Loading data..."):
    df = load_stock_data(ticker, start_date, end_date, interval)

if df is not None and len(df) > 0:
    # Main content area with tabs - Only Data and Export
    data_tabs = st.tabs(["📋 Data", "📤 Export"])
    
    with data_tabs[0]:
        st.markdown("### 📋 Data Table")
        
        # Format the dataframe for display
        display_df = df.copy()
        
        # Reset index to make Date a column, then format
        display_df = display_df.reset_index()
        
        # Format Date column to datetime format (YYYY-MM-DD)
        display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Format OHLCV columns to strictly 2 decimal places (show .00)
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        
        # Table view options
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.selectbox("Rows to display:", options=[10, 20, 30, 50, 100], index=1)
        with col2:
            sort_by = st.selectbox("Sort by:", ["Date", "Close", "Volume"])
        
        # Sort and display table
        display_df = display_df.sort_values(sort_by, ascending=False).head(num_rows)
        
        st.dataframe(
            display_df.style.set_table_styles([
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ]),
            width='stretch',
            height=400
        )
    
    with data_tabs[1]:
        st.markdown("### 📤 Export Data")
        
        # Export format selection
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            export_format = st.selectbox(
                "Select Export Format:",
                ["CSV", "Excel", "JSON"]
            )
        with export_col2:
            filename = st.text_input("Filename:", value="gold_price_data")
        
        st.divider()
        
        # Create formatted dataframe for export preview
        export_preview_df = df.copy()
        export_preview_df = export_preview_df.reset_index()
        export_preview_df['Date'] = pd.to_datetime(export_preview_df['Date']).dt.strftime('%Y-%m-%d')
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_columns:
            if col in export_preview_df.columns:
                export_preview_df[col] = export_preview_df[col].apply(lambda x: f"{x:.2f}")
        
        # Data preview before export
        st.markdown("#### Data Preview")
        preview_rows = st.selectbox("Preview rows:", options=[5, 10, 20, 30, 50], index=1)
        st.dataframe(export_preview_df.head(preview_rows).style.set_table_styles([
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ]),
            height=200)
        
        st.divider()
        
        # Export buttons
        if export_format == "CSV":
            csv = df.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Gold Price Data')
            st.download_button(
                label="📥 Download as Excel",
                data=buffer,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            json_str = df.to_json(orient='index', date_format='iso')
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json"
            )
        
        # Data info
        st.divider()
        st.markdown("#### Data Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Total Records", len(df))
        with info_col2:
            st.metric("Date Range Start", df.index.min().strftime('%Y-%m-%d'))
        with info_col3:
            st.metric("Date Range End", df.index.max().strftime('%Y-%m-%d'))

else:
    st.warning("⚠️ No data available for the selected date range. Try different dates.")

# Add footer
st.markdown("""
<div style='text-align: center; padding: 20px; margin-top: 50px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;'>
    <strong>James Mithiga</strong> | Adm No 58200 | Predictive & Optimization Analytics
</div>
""", unsafe_allow_html=True)
