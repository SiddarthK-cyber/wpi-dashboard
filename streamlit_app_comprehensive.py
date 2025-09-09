import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Configure Streamlit page
st.set_page_config(
    page_title="India WPI Comprehensive Dashboard - 869 Commodities",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Analytics 4 Integration
components.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-E859EJSCKX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-E859EJSCKX');
</script>
""", height=0)

# Load data with caching
@st.cache_data
def load_comprehensive_data():
    """Load all comprehensive WPI datasets"""
    try:
        # Load comprehensive monthly data (869 commodities)
        monthly_data = pd.read_csv('wpi_all_commodities_monthly.csv')
        
        # Load commodity summary
        summary_data = pd.read_csv('wpi_all_commodities_summary.csv')
        
        # Load commodity categories
        categories_data = pd.read_csv('commodity_categories.csv')
        
        return monthly_data, summary_data, categories_data
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please run 'python process_all_commodities.py' first to extract all commodity data.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load all datasets
monthly_data, summary_data, categories_data = load_comprehensive_data()

if monthly_data is None:
    st.stop()

# App Header
st.title("🇮🇳 India WPI Comprehensive Dashboard")
st.markdown(f"### **{monthly_data['COMM_NAME'].nunique()} Commodities** | **{len(monthly_data):,} Data Points** | **{monthly_data['Year'].min()}-{monthly_data['Year'].max()}**")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📊 Dashboard Controls")
page = st.sidebar.selectbox(
    "Select Analysis View:",
    [
        "🔍 Individual Commodity Analysis",
        "📊 Multi-Commodity Comparison", 
        "📈 Category-wise Analysis",
        "🏆 Top Performers & Rankings",
        "📋 Comprehensive Statistics"
    ]
)

# Sidebar filters
st.sidebar.markdown("### 🎛️ Filters")

# Category filter
if categories_data is not None:
    available_categories = sorted(categories_data['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Categories:",
        options=available_categories,
        default=available_categories[:3] if len(available_categories) >= 3 else available_categories,
        help="Filter commodities by category"
    )
    
    # Filter commodities by category
    filtered_categories = categories_data[categories_data['Category'].isin(selected_categories)]
    available_commodities = sorted(filtered_categories['COMM_NAME'].unique())
else:
    available_commodities = sorted(monthly_data['COMM_NAME'].unique())

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

# PAGE 1: INDIVIDUAL COMMODITY ANALYSIS
if page == "🔍 Individual Commodity Analysis":
    st.header("🔍 Individual Commodity Analysis")
    
    # Commodity selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        commodity_search = st.text_input(
            "🔍 Search Commodity:",
            placeholder="Type commodity name to search...",
            help="Start typing to search through 869 commodities"
        )
        
        # Filter commodities based on search
        if commodity_search:
            filtered_commodities = [comm for comm in available_commodities 
                                   if commodity_search.lower() in comm.lower()]
        else:
            filtered_commodities = available_commodities[:50]  # Show first 50 by default
        
        selected_commodity = st.selectbox(
            "Select Commodity:",
            options=filtered_commodities,
            help=f"Choose from {len(available_commodities)} available commodities"
        )
    
    with col2:
        # Chart type selection (Task 3)
        chart_type = st.selectbox(
            "📊 Chart Type:",
            [
                "Line Chart",
                "Area Chart", 
                "Bar Chart",
                "Candlestick Chart",
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
            
            # Create individual chart based on selected type (Task 1 & 3)
            st.subheader(f"📈 {selected_commodity} - Price Trend Analysis")
            
            # Enhanced chart creation with better text visibility (Task 4)
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
                
            elif chart_type == "Candlestick Chart":
                # Create quarterly candlestick data
                quarterly_data = comm_data.groupby(['Year']).agg({
                    'WPI_Index': ['min', 'max', 'first', 'last']
                }).round(2)
                quarterly_data.columns = ['Low', 'High', 'Open', 'Close']
                quarterly_data = quarterly_data.reset_index()
                
                fig = go.Figure(data=go.Candlestick(
                    x=quarterly_data['Year'],
                    open=quarterly_data['Open'],
                    high=quarterly_data['High'],
                    low=quarterly_data['Low'],
                    close=quarterly_data['Close'],
                    name=selected_commodity
                ))
                fig.update_layout(
                    title=f'{selected_commodity} - Annual Candlestick Chart',
                    height=500
                )
                
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
            
            # Enhanced styling for better text visibility (Task 4)
            if chart_type != "Candlestick Chart":
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

# PAGE 2: MULTI-COMMODITY COMPARISON 
elif page == "📊 Multi-Commodity Comparison":
    st.header("📊 Multi-Commodity Comparison")
    
    # Task 2: Very big chart for comparing multiple items
    st.subheader("🔍 Select Commodities to Compare")
    
    # Search and multi-select interface
    search_term = st.text_input(
        "🔍 Search Commodities:",
        placeholder="Type to search through 869 commodities...",
        help="Search and select multiple commodities for comparison"
    )
    
    # Filter commodities based on search
    if search_term:
        searchable_commodities = [comm for comm in available_commodities 
                                 if search_term.lower() in comm.lower()]
    else:
        searchable_commodities = available_commodities[:100]  # Show first 100
    
    # Multi-select for comparison
    selected_commodities = st.multiselect(
        "Select Commodities for Comparison:",
        options=searchable_commodities,
        default=searchable_commodities[:5] if len(searchable_commodities) >= 5 else searchable_commodities[:3],
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
            
            # Task 2: Create very big chart for comparison
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
            
            # Enhanced styling for very big charts (Task 4)
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

# PAGE 3: CATEGORY-WISE ANALYSIS
elif page == "📈 Category-wise Analysis":
    st.header("📈 Category-wise Analysis")
    
    if categories_data is not None:
        # Category overview
        st.subheader("🏷️ Category Distribution")
        
        category_counts = categories_data['Category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Commodity Distribution by Category",
                height=400
            )
            fig_pie.update_traces(textinfo='label+percent+value')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Number of Commodities by Category",
                height=400
            )
            fig_bar.update_layout(
                xaxis_tickangle=45,
                xaxis_title="Category",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Category-wise performance
        st.subheader("📊 Category Performance Analysis")
        
        selected_category = st.selectbox(
            "Select Category for Detailed Analysis:",
            options=available_categories,
            help="Choose a category to analyze its commodities"
        )
        
        if selected_category:
            category_commodities = categories_data[categories_data['Category'] == selected_category]['COMM_NAME'].tolist()
            category_data = filtered_monthly[filtered_monthly['COMM_NAME'].isin(category_commodities)]
            
            if not category_data.empty:
                # Average category performance
                category_avg = category_data.groupby(['Year', 'Month', 'Date'])['WPI_Index'].mean().reset_index()
                category_avg['COMM_NAME'] = f"{selected_category} (Average)"
                
                fig = px.line(
                    category_avg,
                    x='Date',
                    y='WPI_Index',
                    title=f'{selected_category} - Average Category Performance',
                    height=500,
                    line_shape='spline'
                )
                
                # Enhanced styling
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=14, color="black"),
                    title=dict(font=dict(size=18, color="black"), x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top performers in category
                st.subheader(f"🏆 Top Performers in {selected_category}")
                
                category_summary = []
                for commodity in category_commodities:
                    comm_data = category_data[category_data['COMM_NAME'] == commodity]
                    if len(comm_data) > 1:
                        avg_index = comm_data['WPI_Index'].mean()
                        volatility = (comm_data['WPI_Index'].std() / avg_index) * 100
                        growth = ((comm_data['WPI_Index'].iloc[-1] / comm_data['WPI_Index'].iloc[0]) - 1) * 100
                        
                        category_summary.append({
                            'Commodity': commodity,
                            'Avg Index': round(avg_index, 2),
                            'Volatility (%)': round(volatility, 2),
                            'Growth (%)': round(growth, 2)
                        })
                
                if category_summary:
                    category_df = pd.DataFrame(category_summary)
                    category_df = category_df.sort_values('Growth (%)', ascending=False)
                    st.dataframe(category_df, use_container_width=True, height=400)

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
            
            # Show top 50 for performance
            st.dataframe(complete_rankings.head(50), use_container_width=True, height=600)

# PAGE 5: COMPREHENSIVE STATISTICS
elif page == "📋 Comprehensive Statistics":
    st.header("📋 Comprehensive Statistics")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Commodities", f"{len(available_commodities):,}")
    
    with col2:
        st.metric("Total Data Points", f"{len(filtered_monthly):,}")
    
    with col3:
        avg_growth = summary_data[summary_data['COMM_NAME'].isin(available_commodities)]['Growth_Percent'].mean()
        st.metric("Average Growth", f"{avg_growth:.1f}%")
    
    with col4:
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
        st.subheader("📋 Detailed Statistics")
        
        # Search functionality for statistics
        stat_search = st.text_input(
            "🔍 Search in Statistics:",
            placeholder="Search commodity in statistics table..."
        )
        
        display_summary = filtered_summary.copy()
        if stat_search:
            display_summary = display_summary[
                display_summary['COMM_NAME'].str.contains(stat_search, case=False, na=False)
            ]
        
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
    <p>📊 <strong>India WPI Comprehensive Dashboard - 869 Commodities</strong></p>
    <p>Data Source: Ministry of Commerce & Industry, Government of India</p>
    <p>Features: Individual Analysis | Multi-Commodity Comparison | Category Analysis | Performance Rankings | Comprehensive Statistics</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar information
st.sidebar.markdown("---")
st.sidebar.info(f"""
**📊 Dashboard Statistics:**
- **Total Commodities**: {monthly_data['COMM_NAME'].nunique():,}
- **Filtered Items**: {len(available_commodities):,}
- **Data Points**: {len(monthly_data):,}
- **Time Range**: {monthly_data['Year'].min()}-{monthly_data['Year'].max()}

**🎯 Key Features:**
- Individual commodity charts with 6 chart types
- Multi-commodity comparison (very large charts)
- Category-wise analysis
- Performance rankings
- Comprehensive search functionality

**📈 Chart Types Available:**
- Line, Area, Bar, Candlestick
- Box Plot, Scatter Plot
- Correlation Heatmaps
- Normalized Trends
""")