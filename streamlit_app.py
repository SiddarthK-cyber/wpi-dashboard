import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="India WPI Comprehensive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_all_data():
    """Load all WPI datasets"""
    try:
        # Load main monthly data
        monthly_data = pd.read_csv('wpi_monthly_data.csv')
        
        # Load seasonality analysis
        seasonality_data = pd.read_csv('wpi_seasonality_analysis.csv')
        
        # Load commodity summary
        summary_data = pd.read_csv('wpi_commodity_summary.csv')
        
        # Load existing annual data for comparison
        annual_df = pd.read_csv('wpi_10_commodities.csv')
        year_cols = [col for col in annual_df.columns if 'INDEX' in col]
        annual_long = pd.melt(annual_df, 
                             id_vars=['COMM_NAME', 'COMM_CODE', 'COMM_WT'],
                             value_vars=year_cols,
                             var_name='Year_Code',
                             value_name='WPI_Index')
        annual_long['Year'] = annual_long['Year_Code'].str.extract(r'(\d{4})').astype(int)
        
        return monthly_data, seasonality_data, summary_data, annual_long
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load all datasets
monthly_data, seasonality_data, summary_data, annual_data = load_all_data()

if monthly_data is None:
    st.stop()

# App Header
st.title("🇮🇳 India WPI Comprehensive Dashboard")
st.markdown("### Advanced Analytics with Seasonality Insights")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📊 Dashboard Controls")
page = st.sidebar.selectbox(
    "Select Analysis View:",
    ["📈 Price Trends", "🔄 Seasonality Analysis", "📊 Comparative Analysis", "📋 Summary Statistics", "📊 Individual Commodity Charts"]
)

# Commodity selection
available_commodities = sorted(monthly_data['COMM_NAME'].unique())
selected_commodities = st.sidebar.multiselect(
    "Select Commodities:",
    options=available_commodities,
    default=available_commodities[:3] if len(available_commodities) >= 3 else available_commodities,
    help="Choose commodities for analysis"
)

# Year range
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=int(monthly_data['Year'].min()),
    max_value=int(monthly_data['Year'].max()),
    value=(int(monthly_data['Year'].min()), int(monthly_data['Year'].max())),
    help="Adjust the time period for analysis"
)

# Filter data based on selections
if not selected_commodities:
    st.warning("Please select at least one commodity from the sidebar.")
    st.stop()

filtered_monthly = monthly_data[
    (monthly_data['COMM_NAME'].isin(selected_commodities)) &
    (monthly_data['Year'] >= year_range[0]) &
    (monthly_data['Year'] <= year_range[1])
]

# PAGE 1: PRICE TRENDS
if page == "📈 Price Trends":
    st.header("📈 Price Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Price Trends
        st.subheader("Monthly Price Trends")
        
        if not filtered_monthly.empty:
            fig_monthly = px.line(filtered_monthly, 
                                x='Date', 
                                y='WPI_Index',
                                color='COMM_NAME',
                                markers=True,
                                height=400,
                                title="Monthly WPI Index Trends")
            
            fig_monthly.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Date",
                yaxis_title="WPI Index",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
    with col2:
        # Year-over-Year Growth
        st.subheader("Year-over-Year Growth Rates")
        
        growth_data = []
        for commodity in selected_commodities:
            comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity]
            
            for year in comm_data['Year'].unique():
                if year > comm_data['Year'].min():
                    current_year_avg = comm_data[comm_data['Year'] == year]['WPI_Index'].mean()
                    prev_year_avg = comm_data[comm_data['Year'] == year-1]['WPI_Index'].mean()
                    
                    if prev_year_avg > 0:
                        growth_rate = ((current_year_avg / prev_year_avg) - 1) * 100
                        growth_data.append({
                            'COMM_NAME': commodity,
                            'Year': year,
                            'Growth_Rate': growth_rate
                        })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            fig_growth = px.bar(growth_df,
                              x='Year',
                              y='Growth_Rate',
                              color='COMM_NAME',
                              title="Annual Growth Rates (%)",
                              height=400)
            
            fig_growth.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig_growth, use_container_width=True)
    
    # Volatility Analysis
    st.subheader("📉 Price Volatility Analysis")
    
    volatility_data = []
    for commodity in selected_commodities:
        comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity]
        if len(comm_data) > 1:
            volatility = comm_data['WPI_Index'].std()
            cv = (volatility / comm_data['WPI_Index'].mean()) * 100
            volatility_data.append({
                'Commodity': commodity,
                'Standard_Deviation': volatility,
                'Coefficient_of_Variation': cv
            })
    
    if volatility_data:
        vol_df = pd.DataFrame(volatility_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vol = px.bar(vol_df,
                            x='Coefficient_of_Variation',
                            y='Commodity',
                            orientation='h',
                            title="Price Volatility (Coefficient of Variation %)",
                            color='Coefficient_of_Variation',
                            color_continuous_scale='Reds')
            
            fig_vol.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Display volatility table
            st.dataframe(vol_df.round(2), use_container_width=True)

# PAGE 2: SEASONALITY ANALYSIS
elif page == "🔄 Seasonality Analysis":
    st.header("🔄 Seasonal Pattern Analysis")
    
    if seasonality_data is not None:
        # Filter seasonality data
        filtered_seasonality = seasonality_data[
            seasonality_data['COMM_NAME'].isin(selected_commodities)
        ]
        
        if not filtered_seasonality.empty:
            # Seasonal patterns heatmap
            st.subheader("🌡️ Seasonal Price Patterns")
            
            # Create pivot table for heatmap
            pivot_data = filtered_seasonality.pivot(
                index='COMM_NAME', 
                columns='Month_Name', 
                values='Seasonality_Index'
            )
            
            # Reorder columns by month
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_data = pivot_data.reindex(columns=month_order)
            
            fig_heatmap = px.imshow(pivot_data,
                                   labels=dict(x="Month", y="Commodity", color="Seasonality Index"),
                                   title="Seasonal Price Index (100 = Annual Average)",
                                   aspect="auto",
                                   color_continuous_scale="RdYlBu_r")
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Seasonal patterns by commodity
            st.subheader("📊 Monthly Average Patterns")
            
            for commodity in selected_commodities:
                comm_seasonality = filtered_seasonality[
                    filtered_seasonality['COMM_NAME'] == commodity
                ]
                
                if not comm_seasonality.empty:
                    st.markdown(f"**{commodity}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Monthly averages
                        fig_seasonal = px.line(comm_seasonality,
                                             x='Month_Name',
                                             y='Avg_Index',
                                             title=f"{commodity} - Monthly Average Index",
                                             markers=True)
                        
                        fig_seasonal.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis_title="Month",
                            yaxis_title="Average WPI Index",
                            height=300
                        )
                        
                        st.plotly_chart(fig_seasonal, use_container_width=True)
                    
                    with col2:
                        # Key insights
                        peak_month = comm_seasonality.loc[
                            comm_seasonality['Avg_Index'].idxmax(), 'Month_Name'
                        ]
                        low_month = comm_seasonality.loc[
                            comm_seasonality['Avg_Index'].idxmin(), 'Month_Name'
                        ]
                        
                        peak_index = comm_seasonality['Avg_Index'].max()
                        low_index = comm_seasonality['Avg_Index'].min()
                        seasonal_range = peak_index - low_index
                        
                        st.metric("Peak Season", peak_month, f"Index: {peak_index:.1f}")
                        st.metric("Low Season", low_month, f"Index: {low_index:.1f}")
                        st.metric("Seasonal Range", f"{seasonal_range:.1f}", "Index Points")
            
            # Seasonality summary table
            st.subheader("📋 Seasonality Summary Table")
            
            # Show high and low seasons for each commodity
            season_summary = []
            for commodity in selected_commodities:
                comm_data = filtered_seasonality[filtered_seasonality['COMM_NAME'] == commodity]
                if not comm_data.empty:
                    peak_row = comm_data.loc[comm_data['Avg_Index'].idxmax()]
                    low_row = comm_data.loc[comm_data['Avg_Index'].idxmin()]
                    
                    season_summary.append({
                        'Commodity': commodity,
                        'Peak_Month': peak_row['Month_Name'],
                        'Peak_Index': round(peak_row['Avg_Index'], 1),
                        'Low_Month': low_row['Month_Name'],
                        'Low_Index': round(low_row['Avg_Index'], 1),
                        'Seasonal_Range': round(peak_row['Avg_Index'] - low_row['Avg_Index'], 1),
                        'Avg_Volatility': round(comm_data['Coefficient_of_Variation'].mean(), 1)
                    })
            
            if season_summary:
                season_df = pd.DataFrame(season_summary)
                st.dataframe(season_df, use_container_width=True)

# PAGE 3: COMPARATIVE ANALYSIS  
elif page == "📊 Comparative Analysis":
    st.header("📊 Comparative Analysis")
    
    if len(selected_commodities) >= 2:
        # Price correlation analysis
        st.subheader("🔗 Price Correlation Analysis")
        
        # Create correlation matrix
        correlation_data = filtered_monthly.pivot(
            index=['Year', 'Month'], 
            columns='COMM_NAME', 
            values='WPI_Index'
        )
        
        correlation_matrix = correlation_data.corr()
        
        fig_corr = px.imshow(correlation_matrix,
                            labels=dict(color="Correlation"),
                            title="Price Correlation Matrix",
                            aspect="auto",
                            color_continuous_scale="RdBu")
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Normalized price comparison
        st.subheader("📈 Normalized Price Trends (Base Year = 100)")
        
        normalized_data = []
        for commodity in selected_commodities:
            comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity].copy()
            if not comm_data.empty:
                base_value = comm_data['WPI_Index'].iloc[0]  # First value as base
                comm_data['Normalized_Index'] = (comm_data['WPI_Index'] / base_value) * 100
                normalized_data.append(comm_data)
        
        if normalized_data:
            normalized_df = pd.concat(normalized_data)
            
            fig_normalized = px.line(normalized_df,
                                   x='Date',
                                   y='Normalized_Index',
                                   color='COMM_NAME',
                                   markers=True,
                                   title="Normalized Price Trends (Starting Point = 100)",
                                   height=500)
            
            fig_normalized.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Date",
                yaxis_title="Normalized Index",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_normalized, use_container_width=True)
    else:
        st.warning("Please select at least 2 commodities for comparative analysis.")

# PAGE 4: SUMMARY STATISTICS
elif page == "📋 Summary Statistics":
    st.header("📋 Comprehensive Summary Statistics")
    
    if summary_data is not None:
        # Filter summary data
        filtered_summary = summary_data[
            summary_data['COMM_NAME'].isin(selected_commodities)
        ]
        
        # Key metrics cards
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_volatility = filtered_summary['Volatility_Percent'].mean()
            st.metric("Average Volatility", f"{avg_volatility:.1f}%")
        
        with col2:
            avg_growth = filtered_summary['Long_Term_Trend_Percent'].mean()
            st.metric("Average Growth", f"{avg_growth:.1f}%")
        
        with col3:
            total_commodities = len(filtered_summary)
            st.metric("Commodities Analyzed", total_commodities)
        
        with col4:
            data_points = filtered_summary['Data_Points'].sum()
            st.metric("Total Data Points", data_points)
        
        # Detailed summary table
        st.subheader("📊 Detailed Statistics Table")
        
        # Display formatted summary table
        display_summary = filtered_summary.copy()
        display_summary = display_summary.round(2)
        
        st.dataframe(display_summary, use_container_width=True)
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performers")
            
            # Highest growth
            st.write("**Highest Growth:**")
            top_growth = filtered_summary.nlargest(3, 'Long_Term_Trend_Percent')[
                ['COMM_NAME', 'Long_Term_Trend_Percent']
            ]
            st.dataframe(top_growth, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Risk Indicators")
            
            # Most volatile
            st.write("**Most Volatile:**")
            most_volatile = filtered_summary.nlargest(3, 'Volatility_Percent')[
                ['COMM_NAME', 'Volatility_Percent']
            ]
            st.dataframe(most_volatile, use_container_width=True)

# PAGE 5: INDIVIDUAL COMMODITY CHARTS
elif page == "📊 Individual Commodity Charts":
    st.header("📊 Individual Commodity Comprehensive Charts")
    
    # Function to create individual comprehensive charts
    def create_individual_chart(commodity):
        """Create comprehensive chart for individual commodity"""
        import numpy as np
        
        comm_monthly = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity]
        comm_seasonality = seasonality_data[seasonality_data['COMM_NAME'] == commodity] if seasonality_data is not None else pd.DataFrame()
        
        if comm_monthly.empty:
            st.warning(f"No data available for {commodity}")
            return
        
        # Create subplot with 4 charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{commodity} - Monthly Price Trends',
                f'{commodity} - Seasonal Pattern',
                f'{commodity} - Price Distribution',
                f'{commodity} - Monthly Volatility'
            ),
            specs=[[{"secondary_y": False}, {"type": "polar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Chart 1: Monthly Price Trends with moving average
        fig.add_trace(
            go.Scatter(
                x=comm_monthly['Date'],
                y=comm_monthly['WPI_Index'],
                mode='lines+markers',
                name='WPI Index',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Add moving average if enough data
        if len(comm_monthly) >= 6:
            comm_monthly_sorted = comm_monthly.sort_values('Date')
            ma_values = comm_monthly_sorted['WPI_Index'].rolling(window=6, center=True).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=comm_monthly_sorted['Date'],
                    y=ma_values,
                    mode='lines',
                    name='6-Month Moving Average',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=1
            )
        
        # Chart 2: Seasonal Pattern (Polar chart)
        if not comm_seasonality.empty:
            fig.add_trace(
                go.Scatterpolar(
                    r=comm_seasonality['Avg_Index'],
                    theta=comm_seasonality['Month_Name'],
                    fill='toself',
                    name='Seasonal Pattern',
                    line_color='green',
                    fillcolor='rgba(0,255,0,0.2)'
                ),
                row=1, col=2
            )
        
        # Chart 3: Price Distribution
        fig.add_trace(
            go.Histogram(
                x=comm_monthly['WPI_Index'],
                name='Price Distribution',
                nbinsx=8,
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Chart 4: Monthly Volatility (by month across years)
        monthly_vol = []
        for month in range(1, 13):
            month_data = comm_monthly[comm_monthly['Month'] == month]['WPI_Index']
            if len(month_data) > 1:
                volatility = month_data.std()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_vol.append({
                    'Month': month_names[month-1],
                    'Volatility': volatility
                })
        
        if monthly_vol:
            vol_df = pd.DataFrame(monthly_vol)
            fig.add_trace(
                go.Bar(
                    x=vol_df['Month'],
                    y=vol_df['Volatility'],
                    name='Monthly Volatility',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Comprehensive Analysis: {commodity}",
            showlegend=False,
            height=800,
            width=1200
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="WPI Index", row=1, col=1)
        fig.update_xaxes(title_text="WPI Index Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = comm_monthly['WPI_Index'].mean()
            st.metric("Average Price Index", f"{avg_price:.1f}")
        
        with col2:
            price_range = comm_monthly['WPI_Index'].max() - comm_monthly['WPI_Index'].min()
            st.metric("Price Range", f"{price_range:.1f}")
        
        with col3:
            volatility = (comm_monthly['WPI_Index'].std() / comm_monthly['WPI_Index'].mean()) * 100
            st.metric("Volatility (%)", f"{volatility:.1f}%")
    
    # Commodity selection for individual charts
    st.subheader("Select Commodity for Detailed Analysis")
    
    chart_commodity = st.selectbox(
        "Choose commodity:",
        options=available_commodities,
        help="Select a commodity to view comprehensive analysis"
    )
    
    if chart_commodity:
        create_individual_chart(chart_commodity)
        
        # Add seasonal insights if available
        if seasonality_data is not None:
            comm_seasonality = seasonality_data[seasonality_data['COMM_NAME'] == chart_commodity]
            
            if not comm_seasonality.empty:
                st.subheader("🌟 Seasonal Insights")
                
                peak_month = comm_seasonality.loc[comm_seasonality['Avg_Index'].idxmax()]
                low_month = comm_seasonality.loc[comm_seasonality['Avg_Index'].idxmin()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Peak Season:** {peak_month['Month_Name']}
                    - Average Index: {peak_month['Avg_Index']:.1f}
                    - Pattern: {peak_month['Season_Pattern']}
                    """)
                
                with col2:
                    st.info(f"""
                    **Low Season:** {low_month['Month_Name']}
                    - Average Index: {low_month['Avg_Index']:.1f}
                    - Pattern: {low_month['Season_Pattern']}
                    """)
    
    # Comparison charts section
    st.markdown("---")
    st.subheader("📊 Pre-built Comparison Charts")
    
    comparison_chart = st.selectbox(
        "Select comparison chart:",
        [
            "Multi-Commodity Price Comparison",
            "Seasonal Heatmap (All Commodities)",
            "Volatility Comparison",
            "Correlation Matrix",
            "Normalized Trends"
        ]
    )
    
    if comparison_chart == "Multi-Commodity Price Comparison":
        fig = px.line(
            filtered_monthly,
            x='Date',
            y='WPI_Index',
            color='COMM_NAME',
            title='Multi-Commodity Price Comparison',
            markers=True,
            height=600
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_chart == "Seasonal Heatmap (All Commodities)" and seasonality_data is not None:
        pivot_seasonal = seasonality_data[seasonality_data['COMM_NAME'].isin(selected_commodities)].pivot(
            index='COMM_NAME',
            columns='Month_Name',
            values='Avg_Index'
        )
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_seasonal = pivot_seasonal.reindex(columns=month_order)
        
        fig = px.imshow(
            pivot_seasonal,
            labels=dict(x="Month", y="Commodity", color="Average WPI Index"),
            title="Seasonal Price Patterns - Selected Commodities",
            color_continuous_scale="RdYlBu_r",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_chart == "Volatility Comparison":
        volatility_data = []
        for commodity in selected_commodities:
            comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity]
            if len(comm_data) > 1:
                vol = (comm_data['WPI_Index'].std() / comm_data['WPI_Index'].mean()) * 100
                volatility_data.append({'Commodity': commodity, 'Volatility': vol})
        
        if volatility_data:
            vol_df = pd.DataFrame(volatility_data)
            fig = px.bar(
                vol_df,
                x='Commodity',
                y='Volatility',
                title='Price Volatility Comparison (%)',
                color='Volatility',
                color_continuous_scale='Reds',
                height=500
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_chart == "Correlation Matrix":
        pivot_data = filtered_monthly.pivot(
            index=['Year', 'Month'],
            columns='COMM_NAME',
            values='WPI_Index'
        )
        
        if len(pivot_data.columns) > 1:
            correlation_matrix = pivot_data.corr()
            fig = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                title="Price Correlation Matrix",
                color_continuous_scale="RdBu",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_chart == "Normalized Trends":
        normalized_data = []
        for commodity in selected_commodities:
            comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity].copy()
            if not comm_data.empty:
                base_value = comm_data['WPI_Index'].iloc[0]
                comm_data['Normalized_Index'] = (comm_data['WPI_Index'] / base_value) * 100
                normalized_data.append(comm_data)
        
        if normalized_data:
            normalized_df = pd.concat(normalized_data)
            fig = px.line(
                normalized_df,
                x='Date',
                y='Normalized_Index',
                color='COMM_NAME',
                title='Normalized Price Trends (Starting Point = 100)',
                markers=True,
                height=600
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>📊 <strong>India WPI Comprehensive Dashboard</strong></p>
    <p>Data Source: Ministry of Commerce & Industry, Government of India</p>
    <p>Features: Price Trends | Seasonality Analysis | Comparative Analytics | Statistical Summary</p>
</div>
""", unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dashboard Features:**
- 📈 Real-time price trend analysis
- 🔄 Comprehensive seasonality patterns
- 📊 Multi-commodity comparisons
- 📋 Advanced statistical insights

**Data Coverage:**
- Time Period: {monthly_data['Year'].min()}-{monthly_data['Year'].max()}
- Commodities: {len(available_commodities)}
- Monthly Records: {len(monthly_data)}
- Update Frequency: Monthly

**How to Use:**
1. Select commodities of interest
2. Adjust the year range
3. Choose analysis view from dropdown
4. Explore interactive visualizations
""")