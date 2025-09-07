import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="India WPI Interactive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and process data
@st.cache_data
def load_and_process_data():
    """Load and process the WPI data"""
    df = pd.read_csv('wpi_10_commodities.csv')
    
    # Get year columns
    year_cols = [col for col in df.columns if 'INDEX' in col]
    
    # Melt to long format
    df_long = pd.melt(df, 
                      id_vars=['COMM_NAME', 'COMM_CODE', 'COMM_WT'],
                      value_vars=year_cols,
                      var_name='Year_Code',
                      value_name='WPI_Index')
    
    # Extract year from INDEX2013 format
    df_long['Year'] = df_long['Year_Code'].str.extract(r'(\d{4})').astype(int)
    
    return df, df_long

# Load data
df, df_long = load_and_process_data()

# Header
st.title("🇮🇳 India WPI Interactive Dashboard (2013-2024)")
st.markdown("### Explore Wholesale Price Index trends across commodities and sectors")
st.markdown("---")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")

# Commodity selection
selected_commodities = st.sidebar.multiselect(
    "Select Commodities:",
    options=sorted(df['COMM_NAME'].unique()),
    default=['All commodities', 'Paddy', 'Wheat', 'Gram'],
    help="Choose one or more commodities to analyze"
)

# Year range selection
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=2013,
    max_value=2024,
    value=(2013, 2024),
    step=1,
    help="Adjust the time period for analysis"
)

# Check if commodities are selected
if not selected_commodities:
    st.warning("Please select at least one commodity from the sidebar.")
    st.stop()

# Filter data
filtered_df = df_long[
    (df_long['COMM_NAME'].isin(selected_commodities)) &
    (df_long['Year'] >= year_range[0]) &
    (df_long['Year'] <= year_range[1])
]

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    # Chart 1: Line Chart - Trend over time
    st.subheader("📈 WPI Trends Over Time")
    
    if not filtered_df.empty:
        line_fig = px.line(filtered_df, 
                           x='Year', 
                           y='WPI_Index',
                           color='COMM_NAME',
                           markers=True,
                           height=400)
        
        line_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_size=12,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(line_fig, use_container_width=True)
    else:
        st.error("No data available for selected filters.")

with col2:
    # Chart 2: Bar Chart - Price Change
    st.subheader("📊 Price Change (%)")
    
    start_year, end_year = year_range[0], year_range[1]
    bar_data = []
    
    for comm in selected_commodities:
        comm_data = filtered_df[filtered_df['COMM_NAME'] == comm]
        if not comm_data.empty:
            start_val = comm_data[comm_data['Year'] == start_year]['WPI_Index']
            end_val = comm_data[comm_data['Year'] == end_year]['WPI_Index']
            
            if len(start_val) > 0 and len(end_val) > 0:
                change = ((end_val.iloc[0] / start_val.iloc[0]) - 1) * 100
                bar_data.append({
                    'Commodity': comm,
                    'Change_Percent': change
                })
    
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bar_fig = px.bar(bar_df,
                         x='Commodity',
                         y='Change_Percent',
                         color='Change_Percent',
                         color_continuous_scale='RdYlBu_r',
                         height=400)
        
        bar_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_size=12,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.error("No data available for price change calculation.")

# Second row
col3, col4 = st.columns(2)

with col3:
    # Chart 3: Sector Comparison
    st.subheader("🏭 Sector-wise Comparison")
    
    sectors = ['All commodities', 'I    PRIMARY ARTICLES', '(A).  FOOD ARTICLES']
    sector_data = df_long[
        (df_long['COMM_NAME'].isin(sectors)) &
        (df_long['Year'] >= year_range[0]) &
        (df_long['Year'] <= year_range[1])
    ]
    
    if not sector_data.empty:
        sector_fig = px.line(sector_data,
                             x='Year',
                             y='WPI_Index',
                             color='COMM_NAME',
                             markers=True,
                             height=400)
        
        sector_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_size=12,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(sector_fig, use_container_width=True)
    else:
        st.error("No sector data available.")

with col4:
    # Chart 4: Volatility
    st.subheader("📉 Price Volatility")
    
    volatility_data = []
    for comm in selected_commodities:
        comm_data = filtered_df[filtered_df['COMM_NAME'] == comm]['WPI_Index']
        if len(comm_data) > 1:
            volatility_data.append({
                'Commodity': comm,
                'Volatility': comm_data.std()
            })
    
    if volatility_data:
        vol_df = pd.DataFrame(volatility_data)
        vol_fig = px.bar(vol_df,
                         x='Volatility',
                         y='Commodity',
                         orientation='h',
                         color='Volatility',
                         color_continuous_scale='Reds',
                         height=400)
        
        vol_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_size=12
        )
        
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.error("No volatility data available.")

# Summary Statistics
st.markdown("---")
st.subheader("📋 Key Statistics")

if not filtered_df.empty:
    overall_data = filtered_df[filtered_df['COMM_NAME'] == 'All commodities']
    if not overall_data.empty:
        start_idx = overall_data[overall_data['Year'] == year_range[0]]['WPI_Index']
        end_idx = overall_data[overall_data['Year'] == year_range[1]]['WPI_Index']
        
        if len(start_idx) > 0 and len(end_idx) > 0:
            total_change = ((end_idx.iloc[0] / start_idx.iloc[0]) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Overall Change",
                    value=f"{total_change:.1f}%",
                    delta=f"{total_change:.1f}% vs base year"
                )
            
            with col2:
                st.metric(
                    label="Commodities Selected",
                    value=len(selected_commodities)
                )
            
            with col3:
                st.metric(
                    label="Time Period",
                    value=f"{year_range[1] - year_range[0] + 1} years"
                )
        else:
            st.warning("No summary data available for selected range.")
    else:
        st.info("Select 'All commodities' to see overall statistics.")
else:
    st.warning("No data available for selected commodities and time range.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>📊 Data Source: Ministry of Commerce & Industry, Government of India</p>
    <p>Built with Streamlit & Plotly | 🚀 <a href='https://github.com/yourusername/wpi-dashboard' target='_blank'>View Code on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard:**
- Interactive WPI analysis tool
- Real-time chart updates
- Historical trend comparison
- Volatility analysis

**How to use:**
1. Select commodities of interest
2. Adjust the year range
3. Explore different visualizations
4. Check key statistics below
""")