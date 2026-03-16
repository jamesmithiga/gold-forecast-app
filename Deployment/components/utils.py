"""
Reusable Streamlit utility components
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from io import BytesIO


def create_footer(author: str = "James Mithiga", adm_no: str = "58200") -> None:
    """
    Create a reusable footer component
    
    Args:
        author: Author name
        adm_no: Admission number
    """
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; margin-top: 50px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;'>
        <strong>{author}</strong> | Adm No {adm_no} | Predictive & Optimization Analytics
    </div>
    """, unsafe_allow_html=True)


def create_warning_banner(message: str, severity: str = "warning") -> None:
    """
    Create a reusable warning banner
    
    Args:
        message: Warning message
        severity: Severity level (warning, error, info)
    """
    severity_colors = {
        "warning": "#ff6b6b",
        "error": "#ff4444",
        "info": "#ff9800"
    }
    
    color = severity_colors.get(severity, "#ff6b6b")
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color} 0%, {color} 100%); 
                padding: 12px; border-radius: 8px; margin: 8px 0;'>
        <strong>{message}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_success_banner(message: str) -> None:
    """
    Create a reusable success banner
    
    Args:
        message: Success message
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                padding: 12px; border-radius: 8px; margin: 8px 0;'>
        <strong>{message}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_info_banner(message: str) -> None:
    """
    Create a reusable info banner
    
    Args:
        message: Info message
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                padding: 12px; border-radius: 8px; margin: 8px 0;'>
        <strong>{message}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_export_section(df: pd.DataFrame, title: str = "Export Data") -> None:
    """
    Create a reusable export section
    
    Args:
        df: DataFrame to export
        title: Section title
    """
    st.markdown(f"### 💾 {title}")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.markdown("**Format Selection**")
        export_format = st.selectbox(
            "Select export format:",
            options=["CSV", "Excel", "JSON", "Parquet"],
            label_visibility="collapsed"
        )
    
    with export_col2:
        st.markdown("**Options**")
        include_headers = st.checkbox("Include headers", value=True)
        include_index = st.checkbox("Include index", value=True)
    
    st.divider()
    
    # Create export data
    export_df = df.reset_index()
    
    if export_format == "CSV":
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{title.replace(' ', '_')}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    elif export_format == "Excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Excel",
            data=buffer,
            file_name=f"{title.replace(' ', '_')}.xlsx",
            mime="application/vnd.ms-excel",
            width='stretch'
        )
    
    elif export_format == "JSON":
        json_data = export_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"{title.replace(' ', '_')}.json",
            mime="application/json",
            width='stretch'
        )
    
    st.info("✅ Use the buttons above to download your data in the selected format")


def create_data_table(df: pd.DataFrame, title: str = "Data Table") -> None:
    """
    Create a reusable data table
    
    Args:
        df: DataFrame to display
        title: Table title
    """
    st.markdown(f"### 📋 {title}")
    
    # Table view options
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.selectbox("Rows to display:", options=[10, 20, 30, 50, 100], index=1)
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date", "Close", "Volume"])
    
    # Display table
    display_df = df.reset_index().sort_values(sort_by, ascending=False).head(num_rows)
    
    st.dataframe(
        display_df,
        width='stretch',
        height=400
    )


def create_model_description(model_name: str, model_type: str, pros: List[str], cons: List[str], best_for: str) -> None:
    """
    Create a reusable model description component
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        pros: List of advantages
        cons: List of disadvantages
        best_for: Best use cases
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {model_name}")
        st.write(f"**Type:** {model_type}")
        st.write("**Advantages:**")
        for pro in pros:
            st.write(f"✅ {pro}")
    
    with col2:
        st.write("")
        st.write("")
        st.write("**Disadvantages:**")
        for con in cons:
            st.write(f"❌ {con}")
        st.write(f"\n**Best For:** {best_for}")


def create_color_legend() -> None:
    """
    Create a reusable color legend
    """
    st.markdown("""
    <div style='padding: 10px; background: #f0f2f6; border-radius: 8px; margin-bottom: 15px;'>
        <b>📌 Color Legend:</b><br>
        <span style='color: #2196F3;•</span> <b>Training Data</b> (80% of historical)<br>
        <span style='color: #4CAF50;•</span> <b>Test Data</b> (20% of historical)<br>
        <span style='color: #FF5722;•</span> <b>Forecast</b> (Predicted values)
    </div>
    """, unsafe_allow_html=True)


def create_volatility_warning(df: pd.DataFrame, threshold: float = 0.2) -> None:
    """
    Create a volatility warning component
    
    Args:
        df: DataFrame with price data
        threshold: Volatility threshold for warning
    """
    if df is not None and len(df) > 0:
        volatility = df['Close'].pct_change().std()
        if volatility > threshold:
            st.warning("⚠️ High volatility detected")


def create_price_movement_warning(df: pd.DataFrame, current_price: float, threshold: float = 0.3) -> None:
    """
    Create price movement warning component
    
    Args:
        df: DataFrame with price data
        current_price: Current price
        threshold: Price movement threshold
    """
    if df is not None and len(df) > 0:
        forecast_high = df['Close'].max()
        forecast_low = df['Close'].min()
        
        if forecast_high > current_price * (1 + threshold):
            st.warning("⚠️ Large price increase predicted")
        if forecast_low < current_price * (1 - threshold):
            st.warning("⚠️ Large price decrease predicted")


def create_confidence_bands(forecast: pd.DataFrame, confidence_level: int) -> Optional[pd.DataFrame]:
    """
    Create confidence bands for forecast
    
    Args:
        forecast: Forecast DataFrame
        confidence_level: Confidence level percentage
    
    Returns:
        DataFrame with confidence bands or None
    """
    if 'forecast' in forecast.columns:
        std_error = np.std(forecast['forecast']) * 0.3
        upper_band = forecast['forecast'] + (std_error * 1.96 * (100-confidence_level)/10)
        lower_band = forecast['forecast'] - (std_error * 1.96 * (100-confidence_level)/10)
        
        forecast['upper'] = upper_band
        forecast['lower'] = lower_band
        
        return forecast
    
    return None