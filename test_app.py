import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Test WPI Dashboard")
st.write("Testing basic Streamlit deployment...")

# Test if CSV loads
try:
    df = pd.read_csv('wpi_10_commodities.csv')
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Simple test chart
    if not df.empty:
        st.subheader("Test Chart")
        fig = px.bar(df.head(), x='COMM_NAME', y='COMM_WT')
        st.plotly_chart(fig)
        
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.write("Available files:")
    import os
    st.write(os.listdir('.'))