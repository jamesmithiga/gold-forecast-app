"""
Reusable Streamlit chart components

This module provides reusable chart components for the Streamlit dashboard.
All functions return Plotly Figure objects for consistent visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logger = logging.getLogger(__name__)


def create_candlestick_chart(df: pd.DataFrame, title: str = "Candlestick Chart", show_volume: bool = True) -> go.Figure:
    """
    Create a reusable candlestick chart with volume analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        title (str): Chart title. Defaults to "Candlestick Chart"
        show_volume (bool): Whether to include volume analysis. Defaults to True
    
    Returns:
        go.Figure: Plotly Figure object
        
    Raises:
        ValueError: If required columns are missing from DataFrame
        
    Example:
        >>> fig = create_candlestick_chart(df, title="Gold Prices")
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    try:
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided to create_candlestick_chart")
            return go.Figure().add_annotation(text="No data available")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            xaxis_rangeslider_visible=False,
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {str(e)}")
        raise


def create_volume_chart(df: pd.DataFrame, title: str = "Trading Volume") -> go.Figure:
    """
    Create a volume bar chart with color coding based on price movement.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        title (str): Chart title. Defaults to "Trading Volume"
    
    Returns:
        go.Figure: Plotly Figure object
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> fig = create_volume_chart(df, title="Daily Volume")
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    try:
        required_cols = ['Open', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided to create_volume_chart")
            return go.Figure().add_annotation(text="No data available")
        
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                  for i in range(len(df))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Volume",
            xaxis_title="Date",
            template='plotly_white',
            height=300,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating volume chart: {str(e)}")
        raise


def create_forecast_chart(df: pd.DataFrame, forecast: pd.DataFrame, title: str = "Price Forecast") -> go.Figure:
    """
    Create a forecast chart with historical data and prediction confidence bands.
    
    Args:
        df (pd.DataFrame): Historical price data with 'Close' column
        forecast (pd.DataFrame): Forecast data with 'forecast' column and optional 'upper', 'lower' columns
        title (str): Chart title. Defaults to "Price Forecast"
    
    Returns:
        go.Figure: Plotly Figure object
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> fig = create_forecast_chart(historical_df, forecast_df)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    try:
        if df.empty or forecast.empty:
            logger.warning("Empty DataFrame provided to create_forecast_chart")
            return go.Figure().add_annotation(text="No data available")
        
        if 'Close' not in df.columns:
            raise ValueError("Historical data must contain 'Close' column")
        if 'forecast' not in forecast.columns:
            raise ValueError("Forecast data must contain 'forecast' column")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='#2196F3', width=2)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF5722', width=3)
        ))
        
        # Confidence bands
        if 'upper' in forecast.columns and 'lower' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast['lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(255, 87, 34, 0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating forecast chart: {str(e)}")
        raise


def create_comparison_chart(df: pd.DataFrame, models: Dict[str, pd.DataFrame], title: str = "Model Comparison") -> go.Figure:
    """
    Create a multi-model comparison chart
    
    Args:
        df: Historical data
        models: Dictionary of model names and their forecast data
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='black', width=3),
        opacity=0.7
    ))
    
    # Add model forecasts
    colors = px.colors.qualitative.Set2
    for idx, (model_name, model_data) in enumerate(models.items()):
        fig.add_trace(go.Scatter(
            x=model_data.index,
            y=model_data['forecast'],
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_metrics_table(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """
    Display metrics in a formatted table
    
    Args:
        df: DataFrame with model metrics
        metrics: Dictionary of metric names and descriptions
    """
    display_df = df.copy()
    
    for metric in df.columns:
        if metric in metrics:
            display_df[metric] = display_df[metric].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, width='stretch', hide_index=True)


def create_radar_chart(df: pd.DataFrame, title: str = "Model Performance Radar") -> go.Figure:
    """
    Create a radar chart for model comparison
    
    Args:
        df: DataFrame with model metrics
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Normalize metrics for radar
    df_normalized = df.copy()
    for col in df.columns:
        if col != 'Model':
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min()) * 100
    
    for idx, row in df_normalized.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[1:].tolist(),
            theta=list(df_normalized.columns[1:]),
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=title,
        height=500
    )
    
    return fig