"""
Dashboard Entry Point - Commodity Price Forecasting
"""

import streamlit as st

# Custom CSS to increase sidebar page title fonts
st.markdown("""
<style>
    /* Base sidebar styling */
    .stSidebar {
        font-size: 22px !important;
    }
    
    /* Target any list items in sidebar */
    .stSidebar li {
        font-size: 26px !important;
        font-weight: 900 !important;
    }
    
    /* Target any links in sidebar */
    .stSidebar a {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    /* Most aggressive: target everything in sidebar nav */
    [data-testid="stSidebar"] * {
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Redirect to Data Explorer
st.switch_page("pages/1_📊_Data.py")
