"""
Data Page - Historical Gold Price Data with Candlestick Charts
Uses core_functions module as single source of truth for data fetching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from io import BytesIO
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add parent directory to path for core_functions import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.core_functions import get_intraday_data
    CORE_FUNCTIONS_AVAILABLE = True
except ImportError:
    CORE_FUNCTIONS_AVAILABLE = False
    logger.warning("core_functions module not available; falling back to direct yfinance calls")

st.set_page_config(page_title="Data - Gold Price Dashboard", layout="wide")

st.markdown("# 📊 Gold Price Forecasting Dashboard")

# Sidebar controls
with st.sidebar:
    st.title("📊 Gold Price Forecasting Dashboard")
    
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
    # Main content area with tabs
    data_tabs = st.tabs(["📈 Charts", "📊 Analytics"])
    
    with data_tabs[0]:
        st.markdown("### Price Movement & Volume Analysis")
        
        # Key statistics - 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = float(df['Close'].iloc[-1])
        previous_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}")
        
        with col2:
            high_val = float(df['High'].max())
            low_val = float(df['Low'].min())
            st.metric("High (Period)", f"${high_val:.2f}", f"+{high_val - low_val:.2f}")
        
        with col3:
            st.metric("Low (Period)", f"${low_val:.2f}", f"Volume: {float(df['Volume'].iloc[-1])/1e6:.1f}M")
        
        with col4:
            avg_volume = float(df['Volume'].mean())
            st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M", f"Change: {price_change_pct:.2f}%")
        
        st.divider()
        
        # 2-Column layout: Candlestick chart and Volume analysis
        chart_col, vol_col = st.columns([2, 1], gap="medium")
        
        with chart_col:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            
            # Add moving averages if checkbox enabled
            if show_avg:
                ma20 = df['Close'].rolling(window=20).mean()
                ma50 = df['Close'].rolling(window=50).mean()
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=ma20,
                    mode='lines', name='MA20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=ma50,
                    mode='lines', name='MA50',
                    line=dict(color='purple', width=1)
                ))
            
            fig.update_layout(
                title=f"{selected_stock} ({ticker}) - Candlestick Chart",
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                hovermode='x unified',
                template='plotly_white',
                height=600,
                xaxis_rangeslider_visible=False,
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with vol_col:
            st.markdown("**Chart Settings**")
            ma_period = st.selectbox("MA Period", options=[5, 10, 15, 20, 30, 50, 100], index=3)
            trend_conf = st.selectbox("Display Confidence", options=[25, 50, 75, 90, 100], index=2)
            st.info(f"📊 Showing {len(df)} data points")
            
            st.divider()
            
            st.markdown("**Price Analysis**")
            st.write(f"**Range:** ${float(df['Low'].min()):.2f} - ${float(df['High'].max()):.2f}")
            st.write(f"**Average:** ${float(df['Close'].mean()):.2f}")
            st.write(f"**Volatility:** {float(df['Close'].std()):.2f}")
        
        st.divider()
        
        # Volume analysis with 2-column layout
        st.markdown("### 📊 Trading Volume Analysis")
        
        vol_col1, vol_col2 = st.columns([2, 1])
        
        with vol_col1:
            colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                      for i in range(len(df))]
            
            fig_volume = go.Figure()
            
            fig_volume.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
            ))
            
            fig_volume.update_layout(
                title="Trading Volume",
                yaxis_title="Volume",
                xaxis_title="Date",
                template='plotly_white',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_volume, width='stretch')
        
        with vol_col2:
            st.markdown("**Volume Metrics**")
            vol_info = {
                "Mean": f"{float(df['Volume'].mean()):,.0f}",
                "Median": f"{float(df['Volume'].median()):,.0f}",
                "Max": f"{float(df['Volume'].max()):,.0f}",
                "Min": f"{float(df['Volume'].min()):,.0f}"
            }
            for label, value in vol_info.items():
                st.write(f"**{label}:** {value}")
    
    with data_tabs[1]:
        st.markdown("### 📊 Summary Statistics & Analysis")
        
        # Statistics in 2x2 grid
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Mean Price", f"${float(df['Close'].mean()):.2f}")
        with stat_col2:
            st.metric("Median Price", f"${float(df['Close'].median()):.2f}")
        with stat_col3:
            st.metric("Std Dev", f"${float(df['Close'].std()):.2f}")
        with stat_col4:
            returns = df['Close'].pct_change().dropna()
            st.metric("Volatility", f"{float(returns.std())*100:.2f}%")
        
        st.divider()
        
        # Advanced statistics tabs
        adv_tabs = st.tabs(["Returns", "Distribution", "Correlation", "Trends"])
        
        with adv_tabs[0]:
            # Returns analysis
            returns = df['Close'].pct_change().dropna() * 100
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Daily Returns Statistics**")
                st.write(f"Mean: {returns.mean():.4f}%")
                st.write(f"Median: {returns.median():.4f}%")
                st.write(f"Std Dev: {returns.std():.4f}%")
                st.write(f"Min: {returns.min():.4f}%")
                st.write(f"Max: {returns.max():.4f}%")
            
            with col2:
                fig_ret = px.histogram(returns, nbins=50, title="Return Distribution")
                fig_ret.update_layout(height=400)
                st.plotly_chart(fig_ret, width='stretch')
        
        with adv_tabs[1]:
            # Price distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.histogram(df['Close'], nbins=50, title="Price Distribution")
                fig_dist.update_layout(height=400)
                st.plotly_chart(fig_dist, width='stretch')
            
            with col2:
                st.write("**Distribution Metrics**")
                st.write(f"Skewness: {returns.skew():.4f}")
                st.write(f"Kurtosis: {returns.kurtosis():.4f}")
                st.write(f"Range: ${float(df['Close'].max()) - float(df['Close'].min()):.2f}")
                st.write(f"IQR: ${float(df['Close'].quantile(0.75)) - float(df['Close'].quantile(0.25)):.2f}")
        
        with adv_tabs[2]:
            # Correlation analysis
            st.write("**OHLC Correlation Matrix**")
            corr_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in df.columns for col in corr_cols):
                corr_matrix = df[corr_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, width='stretch')
        
        with adv_tabs[3]:
            # Price trends with indicators
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trend Analysis**")
                window = st.selectbox("Moving Average Window", options=[5, 10, 15, 20, 30, 50, 100], index=3)
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                
                ma = df['Close'].rolling(window=window).mean()
                fig_trend.add_trace(go.Scatter(x=df.index, y=ma, name=f'MA{window}'))
                
                fig_trend.update_layout(height=400, title="Price Trend")
                st.plotly_chart(fig_trend, width='stretch')
            
            with col2:
                st.write("**Volatility Analysis**")
                vol_window = st.selectbox("Volatility Window", options=[5, 10, 15, 20, 30, 50, 100], index=3)
                
                rolling_vol = df['Close'].pct_change().rolling(vol_window).std() * 100
                fig_vol = px.line(y=rolling_vol, title="Rolling Volatility (%)")
                fig_vol.update_layout(height=400)
                st.plotly_chart(fig_vol, width='stretch')

else:
    st.warning("⚠️ No data available for the selected date range. Try different dates.")

# Add footer
st.markdown("""
<div style='text-align: center; padding: 20px; margin-top: 50px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;'>
    <strong>James Mithiga</strong> | Adm No 58200 | Predictive & Optimization Analytics
</div>
""", unsafe_allow_html=True)
