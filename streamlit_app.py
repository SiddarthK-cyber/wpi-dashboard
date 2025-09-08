import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math

# Configure Streamlit page
st.set_page_config(
    page_title="India WPI Comprehensive Dashboard - 805 Clean Commodities",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_clean_data():
    """Load cleaned WPI datasets"""
    try:
        # Load cleaned monthly data (805 individual commodities)
        monthly_data = pd.read_csv('wpi_commodities_cleaned.csv')
        
        # Load cleaned summary
        summary_data = pd.read_csv('wpi_commodities_summary_cleaned.csv')
        
        return monthly_data, summary_data
        
    except FileNotFoundError as e:
        st.error(f"Cleaned data files not found: {e}")
        st.info("Please run 'python clean_commodity_data.py' first to create cleaned data files.")
        
        # Fallback to original data if cleaned files don't exist
        try:
            monthly_data = pd.read_csv('wpi_all_commodities_monthly.csv')
            summary_data = pd.read_csv('wpi_all_commodities_summary.csv')
            return monthly_data, summary_data
        except:
            return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load all datasets
monthly_data, summary_data = load_clean_data()

if monthly_data is None:
    st.stop()

# App Header
st.title("🇮🇳 India WPI Comprehensive Dashboard")
st.markdown(f"### **{monthly_data['COMM_NAME'].nunique()} Individual Commodities** | **{len(monthly_data):,} Data Points** | **{monthly_data['Year'].min()}-{monthly_data['Year'].max()}**")
st.markdown("---")

# Global search at the top
st.markdown("### 🔍 Quick Search (Global)")
global_search = st.text_input(
    "Global Search",
    placeholder="🔍 Search any commodity across all 805 items... (e.g., 'Rice', 'Steel', 'Cotton')",
    help="Search works across all pages and features",
    label_visibility="collapsed"
)

# Filter commodities based on global search
available_commodities = sorted(monthly_data['COMM_NAME'].unique())
if global_search:
    available_commodities = [comm for comm in available_commodities 
                           if global_search.lower() in comm.lower()]

# Sidebar Navigation
st.sidebar.title("📊 Dashboard Controls")
page = st.sidebar.selectbox(
    "Select Analysis View:",
    [
        "📑 All Commodities Individual Charts",
        "🔍 Single Commodity Analysis",
        "📊 Multi-Commodity Comparison", 
        "🏆 Top Performers & Rankings",
        "📋 Statistics Overview"
    ]
)

# Sidebar filters
st.sidebar.markdown("### 🎛️ Filters")

# Year range
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=int(monthly_data['Year'].min()),
    max_value=int(monthly_data['Year'].max()),
    value=(2020, int(monthly_data['Year'].max())),
    help="Adjust the time period for analysis"
)

# Filter data based on selections
filtered_monthly = monthly_data[
    (monthly_data['COMM_NAME'].isin(available_commodities)) &
    (monthly_data['Year'] >= year_range[0]) &
    (monthly_data['Year'] <= year_range[1])
]

# PAGE 1: ALL COMMODITIES INDIVIDUAL CHARTS (Task 4)
if page == "📑 All Commodities Individual Charts":
    st.header("📑 All Commodities - Individual Charts Display")
    
    # Show total count
    st.info(f"📊 Displaying individual charts for **{len(available_commodities)} commodities** {'(filtered)' if global_search else '(all)'}")
    
    # Chart type selection for all charts
    col1, col2 = st.columns([1, 1])
    with col1:
        chart_type_all = st.selectbox(
            "📊 Chart Type for All:",
            [
                "Line Chart",
                "Area Chart", 
                "Bar Chart",
                "Heatmap Chart",  # Replaced Candlestick (Task 2)
                "Box Plot",
                "Scatter Plot"
            ],
            help="Select visualization type for all individual charts"
        )
    
    with col2:
        charts_per_row = st.selectbox(
            "Charts per Row:",
            [1, 2, 3, 4],
            index=1,
            help="Number of charts to display per row"
        )
    
    # Pagination for better performance
    items_per_page = 20
    total_pages = math.ceil(len(available_commodities) / items_per_page)
    
    if total_pages > 1:
        page_num = st.selectbox(
            f"Page ({total_pages} pages total):",
            range(1, total_pages + 1),
            help=f"Navigate through {len(available_commodities)} commodities"
        )
        start_idx = (page_num - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(available_commodities))
        page_commodities = available_commodities[start_idx:end_idx]
    else:
        page_commodities = available_commodities
        page_num = 1
    
    st.markdown(f"**Showing {len(page_commodities)} commodities** (Page {page_num} of {total_pages})")
    
    # Display individual charts for all commodities
    for i in range(0, len(page_commodities), charts_per_row):
        cols = st.columns(charts_per_row)
        
        for j in range(charts_per_row):
            if i + j < len(page_commodities):
                commodity = page_commodities[i + j]
                comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == commodity].copy()
                
                if not comm_data.empty:
                    comm_data = comm_data.sort_values(['Year', 'Month'])
                    
                    with cols[j]:
                        # Create individual chart
                        if chart_type_all == "Line Chart":
                            fig = px.line(
                                comm_data,
                                x='Date',
                                y='WPI_Index',
                                title=f'{commodity}',
                                height=300
                            )
                            
                        elif chart_type_all == "Area Chart":
                            fig = px.area(
                                comm_data,
                                x='Date',
                                y='WPI_Index',
                                title=f'{commodity}',
                                height=300
                            )
                            
                        elif chart_type_all == "Bar Chart":
                            # Show yearly averages for bars to avoid clutter
                            yearly_avg = comm_data.groupby('Year')['WPI_Index'].mean().reset_index()
                            fig = px.bar(
                                yearly_avg,
                                x='Year',
                                y='WPI_Index',
                                title=f'{commodity}',
                                height=300
                            )
                            
                        elif chart_type_all == "Heatmap Chart":  # Replaced Candlestick
                            # Create month vs year heatmap
                            pivot_data = comm_data.pivot_table(
                                index='Month',
                                columns='Year',
                                values='WPI_Index',
                                aggfunc='mean'
                            )
                            
                            if not pivot_data.empty:
                                fig = px.imshow(
                                    pivot_data,
                                    labels=dict(x="Year", y="Month", color="WPI Index"),
                                    title=f'{commodity}',
                                    aspect="auto",
                                    height=300
                                )
                            else:
                                # Fallback to line chart
                                fig = px.line(comm_data, x='Date', y='WPI_Index', title=f'{commodity}', height=300)
                            
                        elif chart_type_all == "Box Plot":
                            fig = px.box(
                                comm_data,
                                x='Year',
                                y='WPI_Index',
                                title=f'{commodity}',
                                height=300
                            )
                            
                        elif chart_type_all == "Scatter Plot":
                            fig = px.scatter(
                                comm_data,
                                x='Date',
                                y='WPI_Index',
                                color='Month',
                                title=f'{commodity}',
                                height=300
                            )
                        
                        # Enhanced styling for better text visibility (Task 4 from previous)
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(
                                family="Arial, sans-serif",
                                size=10,
                                color="black"
                            ),
                            title=dict(
                                font=dict(size=12, color="black"),
                                x=0.5
                            ),
                            xaxis=dict(
                                title=dict(font=dict(size=10, color="black")),
                                tickfont=dict(size=8, color="black"),
                                gridcolor="lightgray"
                            ),
                            yaxis=dict(
                                title=dict(text="WPI Index", font=dict(size=10, color="black")),
                                tickfont=dict(size=8, color="black"),
                                gridcolor="lightgray"
                            ),
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"{commodity}_{i}_{j}")
                        
                        # Show key metrics below each chart
                        avg_price = comm_data['WPI_Index'].mean()
                        volatility = (comm_data['WPI_Index'].std() / avg_price) * 100
                        growth = ((comm_data['WPI_Index'].iloc[-1] / comm_data['WPI_Index'].iloc[0]) - 1) * 100 if len(comm_data) > 1 else 0
                        
                        st.caption(f"📊 Avg: {avg_price:.1f} | 📈 Vol: {volatility:.1f}% | 🚀 Growth: {growth:.1f}%")

# PAGE 2: SINGLE COMMODITY ANALYSIS
elif page == "🔍 Single Commodity Analysis":
    st.header("🔍 Single Commodity Deep Analysis")
    
    # Commodity selection with better search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_commodity = st.selectbox(
            "Select Commodity for Deep Analysis:",
            options=available_commodities,
            help=f"Choose from {len(available_commodities)} available commodities"
        )
    
    with col2:
        # Chart type selection (Task 2 - Removed Candlestick, added Heatmap)
        chart_type = st.selectbox(
            "📊 Chart Type:",
            [
                "Line Chart",
                "Area Chart", 
                "Bar Chart",
                "Heatmap Chart",  # More relevant than Candlestick
                "Box Plot",
                "Scatter Plot"
            ],
            help="Select visualization type"
        )
    
    if selected_commodity:
        comm_data = filtered_monthly[filtered_monthly['COMM_NAME'] == selected_commodity].copy()
        
        if not comm_data.empty:
            # Sort by date for proper visualization
            comm_data = comm_data.sort_values(['Year', 'Month'])
            
            # Create individual chart based on selected type
            st.subheader(f"📈 {selected_commodity} - Deep Analysis")
            
            # Enhanced chart creation with better text visibility
            if chart_type == "Line Chart":
                fig = px.line(
                    comm_data,
                    x='Date',
                    y='WPI_Index',
                    title=f'{selected_commodity} - Monthly Price Trends',
                    markers=True,
                    height=500
                )
                
            elif chart_type == "Area Chart":
                fig = px.area(
                    comm_data,
                    x='Date',
                    y='WPI_Index',
                    title=f'{selected_commodity} - Price Trend (Area)',
                    height=500
                )
                
            elif chart_type == "Bar Chart":
                fig = px.bar(
                    comm_data,
                    x='Date',
                    y='WPI_Index',
                    title=f'{selected_commodity} - Monthly Prices (Bars)',
                    height=500
                )
                
            elif chart_type == "Heatmap Chart":  # Replaced Candlestick (Task 2)
                # Create month vs year heatmap
                pivot_data = comm_data.pivot_table(
                    index='Month',
                    columns='Year',
                    values='WPI_Index',
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x="Year", y="Month", color="WPI Index"),
                        title=f'{selected_commodity} - Seasonal Price Heatmap',
                        aspect="auto",
                        height=500,
                        color_continuous_scale="RdYlBu_r"
                    )
                    
                    # Add month labels
                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    fig.update_yaxis(
                        tickmode='array',
                        tickvals=list(range(1, 13)),
                        ticktext=month_labels
                    )
                else:
                    # Fallback to line chart
                    fig = px.line(comm_data, x='Date', y='WPI_Index', 
                                title=f'{selected_commodity} - Price Trends', height=500)
                
            elif chart_type == "Box Plot":
                fig = px.box(
                    comm_data,
                    x='Year',
                    y='WPI_Index',
                    title=f'{selected_commodity} - Annual Price Distribution',
                    height=500
                )
                
            elif chart_type == "Scatter Plot":
                fig = px.scatter(
                    comm_data,
                    x='Date',
                    y='WPI_Index',
                    color='Month',
                    size='WPI_Index',
                    title=f'{selected_commodity} - Price Scatter (Sized by Value)',
                    height=500
                )
            
            # Enhanced styling for better text visibility
            if chart_type != "Heatmap Chart":
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color="black"
                    ),
                    title=dict(
                        font=dict(size=18, color="black"),
                        x=0.5
                    ),
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black")),
                        tickfont=dict(size=12, color="black"),
                        gridcolor="lightgray"
                    ),
                    yaxis=dict(
                        title=dict(text="WPI Index", font=dict(size=14, color="black")),
                        tickfont=dict(size=12, color="black"),
                        gridcolor="lightgray"
                    )
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price = comm_data['WPI_Index'].mean()
                st.metric("Average Index", f"{avg_price:.1f}")
            
            with col2:
                volatility = (comm_data['WPI_Index'].std() / avg_price) * 100
                st.metric("Volatility (%)", f"{volatility:.1f}%")
            
            with col3:
                price_change = comm_data['WPI_Index'].iloc[-1] - comm_data['WPI_Index'].iloc[0]
                st.metric("Total Change", f"{price_change:.1f}", delta=f"{price_change:.1f}")
            
            with col4:
                growth_rate = ((comm_data['WPI_Index'].iloc[-1] / comm_data['WPI_Index'].iloc[0]) - 1) * 100
                st.metric("Growth Rate (%)", f"{growth_rate:.1f}%", delta=f"{growth_rate:.1f}%")
                
        else:
            st.warning(f"No data available for {selected_commodity} in the selected time range.")

# PAGE 3: MULTI-COMMODITY COMPARISON 
elif page == "📊 Multi-Commodity Comparison":
    st.header("📊 Multi-Commodity Comparison")
    
    st.subheader("🔍 Select Commodities to Compare")
    
    # Multi-select for comparison
    selected_commodities = st.multiselect(
        "Select Commodities for Comparison:",
        options=available_commodities,
        default=available_commodities[:5] if len(available_commodities) >= 5 else available_commodities[:3],
        help=f"Choose multiple commodities from {len(available_commodities)} available options"
    )
    
    if len(selected_commodities) > 0:
        # Filter data for selected commodities
        comparison_data = filtered_monthly[filtered_monthly['COMM_NAME'].isin(selected_commodities)]
        
        if not comparison_data.empty:
            # Chart type selection for comparison
            comparison_chart_type = st.selectbox(
                "📊 Comparison Chart Type:",
                [
                    "Line Chart (Multi-line)",
                    "Normalized Trends (Base=100)",
                    "Area Chart (Stacked)",
                    "Box Plot Comparison",
                    "Correlation Heatmap"
                ]
            )
            
            # Create very big chart for comparison (Task 2 requirement)
            if comparison_chart_type == "Line Chart (Multi-line)":
                fig = px.line(
                    comparison_data,
                    x='Date',
                    y='WPI_Index',
                    color='COMM_NAME',
                    title=f'Multi-Commodity Price Comparison ({len(selected_commodities)} Items)',
                    markers=True,
                    height=700,  # Very big chart
                    width=1200
                )
                
            elif comparison_chart_type == "Normalized Trends (Base=100)":
                # Normalize to base 100
                normalized_data = []
                for commodity in selected_commodities:
                    comm_data = comparison_data[comparison_data['COMM_NAME'] == commodity].copy()
                    if not comm_data.empty:
                        comm_data = comm_data.sort_values(['Year', 'Month'])
                        base_value = comm_data['WPI_Index'].iloc[0]
                        comm_data['Normalized_Index'] = (comm_data['WPI_Index'] / base_value) * 100
                        normalized_data.append(comm_data)
                
                if normalized_data:
                    norm_df = pd.concat(normalized_data)
                    fig = px.line(
                        norm_df,
                        x='Date',
                        y='Normalized_Index',
                        color='COMM_NAME',
                        title=f'Normalized Price Trends - Base=100 ({len(selected_commodities)} Items)',
                        markers=True,
                        height=700,
                        width=1200
                    )
                    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                                 annotation_text="Baseline (100)")
            
            elif comparison_chart_type == "Area Chart (Stacked)":
                fig = px.area(
                    comparison_data,
                    x='Date',
                    y='WPI_Index',
                    color='COMM_NAME',
                    title=f'Stacked Area Chart - Price Trends ({len(selected_commodities)} Items)',
                    height=700,
                    width=1200
                )
                
            elif comparison_chart_type == "Box Plot Comparison":
                fig = px.box(
                    comparison_data,
                    x='COMM_NAME',
                    y='WPI_Index',
                    title=f'Price Distribution Comparison ({len(selected_commodities)} Items)',
                    height=700,
                    width=1200
                )
                fig.update_xaxes(tickangle=45)
                
            elif comparison_chart_type == "Correlation Heatmap":
                # Create correlation matrix
                pivot_data = comparison_data.pivot_table(
                    index=['Year', 'Month'],
                    columns='COMM_NAME',
                    values='WPI_Index',
                    aggfunc='mean'
                )
                correlation_matrix = pivot_data.corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(color="Correlation"),
                    title=f'Price Correlation Matrix ({len(selected_commodities)} Items)',
                    color_continuous_scale="RdBu",
                    height=700,
                    width=700
                )
            
            # Enhanced styling for very big charts
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(
                    family="Arial, sans-serif",
                    size=16,  # Larger font for big charts
                    color="black"
                ),
                title=dict(
                    font=dict(size=20, color="black"),
                    x=0.5
                ),
                xaxis=dict(
                    title=dict(font=dict(size=16, color="black")),
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title=dict(font=dict(size=16, color="black")),
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray"
                ),
                legend=dict(
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison statistics
            st.subheader("📊 Comparison Statistics")
            
            comparison_stats = []
            for commodity in selected_commodities:
                comm_data = comparison_data[comparison_data['COMM_NAME'] == commodity]
                if not comm_data.empty:
                    comparison_stats.append({
                        'Commodity': commodity,
                        'Average Index': round(comm_data['WPI_Index'].mean(), 2),
                        'Volatility (%)': round((comm_data['WPI_Index'].std() / comm_data['WPI_Index'].mean()) * 100, 2),
                        'Min Value': round(comm_data['WPI_Index'].min(), 2),
                        'Max Value': round(comm_data['WPI_Index'].max(), 2),
                        'Latest Value': round(comm_data['WPI_Index'].iloc[-1], 2)
                    })
            
            if comparison_stats:
                stats_df = pd.DataFrame(comparison_stats)
                st.dataframe(stats_df, use_container_width=True, height=300)
        
        else:
            st.warning("No data available for the selected commodities in the chosen time range.")
    
    else:
        st.info("Please select at least one commodity for comparison.")

# PAGE 4: TOP PERFORMERS & RANKINGS
elif page == "🏆 Top Performers & Rankings":
    st.header("🏆 Top Performers & Rankings")
    
    if summary_data is not None:
        # Filter summary data based on current selections
        filtered_summary = summary_data[summary_data['COMM_NAME'].isin(available_commodities)]
        
        # Top performers tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Highest Growth", "📈 Most Volatile", "💪 Best Performers", "📊 Complete Rankings"])
        
        with tab1:
            st.subheader("🚀 Top 20 Highest Growth Commodities")
            top_growth = filtered_summary.nlargest(20, 'Growth_Percent')[
                ['COMM_NAME', 'Growth_Percent', 'Avg_Index', 'Volatility_Percent']
            ].reset_index(drop=True)
            top_growth.index += 1
            
            # Visualization
            fig = px.bar(
                top_growth,
                x='COMM_NAME',
                y='Growth_Percent',
                color='Growth_Percent',
                title='Top 20 Commodities by Growth Rate',
                color_continuous_scale='Greens',
                height=500
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(top_growth, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Top 20 Most Volatile Commodities")
            most_volatile = filtered_summary.nlargest(20, 'Volatility_Percent')[
                ['COMM_NAME', 'Volatility_Percent', 'Avg_Index', 'Growth_Percent']
            ].reset_index(drop=True)
            most_volatile.index += 1
            
            # Visualization
            fig = px.bar(
                most_volatile,
                x='COMM_NAME',
                y='Volatility_Percent',
                color='Volatility_Percent',
                title='Top 20 Most Volatile Commodities',
                color_continuous_scale='Reds',
                height=500
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(most_volatile, use_container_width=True)
        
        with tab3:
            st.subheader("💪 Best Overall Performers (High Growth + Low Volatility)")
            # Calculate performance score (high growth, low volatility)
            filtered_summary['Performance_Score'] = (
                filtered_summary['Growth_Percent'] - (filtered_summary['Volatility_Percent'] * 0.5)
            )
            
            best_performers = filtered_summary.nlargest(20, 'Performance_Score')[
                ['COMM_NAME', 'Performance_Score', 'Growth_Percent', 'Volatility_Percent', 'Avg_Index']
            ].reset_index(drop=True)
            best_performers.index += 1
            
            # Scatter plot
            fig = px.scatter(
                filtered_summary,
                x='Volatility_Percent',
                y='Growth_Percent',
                size='Avg_Index',
                color='Performance_Score',
                hover_name='COMM_NAME',
                title='Growth vs Volatility Analysis',
                color_continuous_scale='Viridis',
                height=500
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=filtered_summary['Volatility_Percent'].mean(), line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(best_performers, use_container_width=True)
        
        with tab4:
            st.subheader("📊 Complete Rankings")
            
            ranking_metric = st.selectbox(
                "Select Ranking Metric:",
                ["Growth_Percent", "Volatility_Percent", "Avg_Index", "Price_Range"]
            )
            
            ranking_order = st.radio(
                "Ranking Order:",
                ["Highest to Lowest", "Lowest to Highest"]
            )
            
            ascending = ranking_order == "Lowest to Highest"
            complete_rankings = filtered_summary.sort_values(ranking_metric, ascending=ascending).reset_index(drop=True)
            complete_rankings.index += 1
            
            # Show all available commodities
            st.dataframe(complete_rankings, use_container_width=True, height=600)

# PAGE 5: STATISTICS OVERVIEW
elif page == "📋 Statistics Overview":
    st.header("📋 Comprehensive Statistics Overview")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Individual Commodities", f"{len(available_commodities):,}")
    
    with col2:
        st.metric("Total Data Points", f"{len(filtered_monthly):,}")
    
    with col3:
        if summary_data is not None:
            avg_growth = summary_data[summary_data['COMM_NAME'].isin(available_commodities)]['Growth_Percent'].mean()
            st.metric("Average Growth", f"{avg_growth:.1f}%")
    
    with col4:
        if summary_data is not None:
            avg_volatility = summary_data[summary_data['COMM_NAME'].isin(available_commodities)]['Volatility_Percent'].mean()
            st.metric("Average Volatility", f"{avg_volatility:.1f}%")
    
    # Distribution analysis
    st.subheader("📊 Distribution Analysis")
    
    if summary_data is not None:
        filtered_summary = summary_data[summary_data['COMM_NAME'].isin(available_commodities)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                filtered_summary,
                x='Growth_Percent',
                nbins=30,
                title='Growth Rate Distribution',
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_hist2 = px.histogram(
                filtered_summary,
                x='Volatility_Percent',
                nbins=30,
                title='Volatility Distribution',
                height=400
            )
            st.plotly_chart(fig_hist2, use_container_width=True)
        
        # Detailed statistics table
        st.subheader("📋 Detailed Statistics Table")
        
        display_summary = filtered_summary.copy()
        
        st.dataframe(
            display_summary.round(2),
            use_container_width=True,
            height=600,
            column_config={
                "COMM_NAME": st.column_config.TextColumn("Commodity", width="medium"),
                "Growth_Percent": st.column_config.NumberColumn("Growth (%)", format="%.1f"),
                "Volatility_Percent": st.column_config.NumberColumn("Volatility (%)", format="%.1f"),
                "Avg_Index": st.column_config.NumberColumn("Average Index", format="%.1f"),
            }
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 16px;'>
    <p>📊 <strong>India WPI Comprehensive Dashboard - 805 Individual Commodities</strong></p>
    <p>Data Source: Ministry of Commerce & Industry, Government of India</p>
    <p>Features: All Individual Charts | Deep Analysis | Multi-Commodity Comparison | Performance Rankings | Statistics</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar information
st.sidebar.markdown("---")
st.sidebar.info(f"""
**📊 Dashboard Statistics:**
- **Individual Commodities**: {monthly_data['COMM_NAME'].nunique():,}
- **Currently Filtered**: {len(available_commodities):,}
- **Data Points**: {len(monthly_data):,}
- **Time Range**: {monthly_data['Year'].min()}-{monthly_data['Year'].max()}

**🎯 Key Features:**
✅ All 805 commodities individual charts
✅ Clean names (removed a., a1., etc.)
✅ Alphabetical ordering  
✅ Global search functionality
✅ Heatmap charts (replaced candlestick)
✅ Automatic display of all items
✅ Enhanced text visibility

**📈 Available Charts:**
- Line, Area, Bar Charts
- Heatmap (seasonal patterns)
- Box Plot, Scatter Plot
- Correlation Analysis
- Normalized Trends
""")