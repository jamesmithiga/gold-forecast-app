"""
Reusable Streamlit layout components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any


def create_metrics_dashboard(metrics: Dict[str, Any]) -> None:
    """
    Create a reusable metrics dashboard
    
    Args:
        metrics: Dictionary of metric names and values
    """
    st.markdown("### 📊 Summary Statistics & Analysis")
    
    # Statistics in 2x2 grid
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Mean Price", f"${metrics.get('mean_price', 3858.46):.2f}")
    with stat_col2:
        st.metric("Median Price", f"${metrics.get('median_price', 3643.70):.2f}")
    with stat_col3:
        st.metric("Std Dev", f"${metrics.get('std_dev', 653.11):.2f}")
    with stat_col4:
        returns = metrics.get('returns', [])
        volatility = float(np.std(returns))*100 if returns else 1.66
        st.metric("Volatility", f"{volatility:.2f}%")
    
    st.divider()


def create_advanced_statistics(df: pd.DataFrame, returns: pd.Series) -> None:
    """
    Create advanced statistics tabs
    
    Args:
        df: DataFrame with price data
        returns: Series with return data
    """
    adv_tabs = st.tabs(["Returns", "Distribution", "Correlation", "Trends"])
    
    with adv_tabs[0]:
        # Returns analysis
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
            window = st.slider("Moving Average Window", 5, 100, 20)
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            
            ma = df['Close'].rolling(window=window).mean()
            fig_trend.add_trace(go.Scatter(x=df.index, y=ma, name=f'MA{window}'))
            
            fig_trend.update_layout(height=400, title="Price Trend")
            st.plotly_chart(fig_trend, width='stretch')
        
        with col2:
            st.write("**Volatility Analysis**")
            vol_window = st.slider("Volatility Window", 5, 100, 20)
            
            rolling_vol = df['Close'].pct_change().rolling(vol_window).std() * 100
            fig_vol = px.line(y=rolling_vol, title="Rolling Volatility (%)")
            fig_vol.update_layout(height=400)
            st.plotly_chart(fig_vol, width='stretch')


def create_model_comparison_table(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """
    Create a model comparison table
    
    Args:
        df: DataFrame with model metrics
        metrics: Dictionary of metric names and descriptions
    """
    st.markdown("### Detailed Metrics Table")
    
    display_df = df.copy()
    for metric in df.columns:
        if metric in metrics:
            display_df[metric] = display_df[metric].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, width='stretch', hide_index=True)


def create_performance_charts(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """
    Create performance comparison charts
    
    Args:
        df: DataFrame with model metrics
        metrics: Dictionary of metric names and descriptions
    """
    error_col, fit_col = st.columns(2)
    
    with error_col:
        st.subheader("❌ Error Metrics")
        
        # RMSE Comparison
        fig_rmse = px.bar(
            df,
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
            df,
            x="Model",
            y="MAE",
            title="MAE (Lower is Better)",
            color="MAE",
            color_continuous_scale="Oranges"
        )
        fig_mae.update_layout(height=350)
        st.plotly_chart(fig_mae, width='stretch')
    
    with fit_col:
        st.subheader("📊 Error & Fit Metrics")
        
        # MAPE Comparison (focus on error metric)
        fig_mape = px.bar(
            df,
            x="Model",
            y="MAPE",
            title="MAPE % (Lower is Better)",
            color="MAPE",
            color_continuous_scale="Purples"
        )
        fig_mape.update_layout(height=350)
        st.plotly_chart(fig_mape, width='stretch')
        
        # R² Comparison
        fig_r2 = px.bar(
            df,
            x="Model",
            y="R²",
            title="R² Score (Higher is Better)",
            color="R²",
            color_continuous_scale="Greens"
        )
        fig_r2.update_layout(height=350)
        st.plotly_chart(fig_r2, width='stretch')


def create_model_details(model_name: str, model_info: Dict[str, Any]) -> None:
    """
    Create model details section
    
    Args:
        model_name: Name of the model
        model_info: Dictionary with model information
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {model_name}")
        st.write(f"**Type:** {model_info.get('type', 'Unknown')}")
        st.write("**Advantages:**")
        for pro in model_info.get('pros', []):
            st.write(f"✅ {pro}")
    
    with col2:
        st.write("")
        st.write("")
        st.write("**Disadvantages:**")
        for con in model_info.get('cons', []):
            st.write(f"❌ {con}")
        st.write(f"\n**Best For:** {model_info.get('best_for', 'Unknown')}")


def create_metrics_reference(metrics: Dict[str, Dict[str, str]]) -> None:
    """
    Create metrics reference section
    
    Args:
        metrics: Dictionary of metric names and descriptions
    """
    st.markdown("### 📚 Metrics Reference")
    
    for metric_name, metric_info in metrics.items():
        with st.expander(f"📖 {metric_name}"):
            st.write(f"**Description:** {metric_info['description']}")
            st.write(f"**Range:** {metric_info['range']}")
            if metric_info['unit']:
                st.write(f"**Unit:** {metric_info['unit']}")


def create_color_legend() -> None:
    """
    Create a color legend for charts
    """
    st.markdown("""
    <div style='padding: 10px; background: #f0f2f6; border-radius: 8px; margin-bottom: 15px;'>
        <b>📌 Color Legend:</b><br>
        <span style='color: #2196F3;•</span> <b>Training Data</b> (80% of historical)<br>
        <span style='color: #4CAF50;•</span> <b>Test Data</b> (20% of historical)<br>
        <span style='color: #FF5722;•</span> <b>Forecast</b> (Predicted values)
    </div>
    """, unsafe_allow_html=True)