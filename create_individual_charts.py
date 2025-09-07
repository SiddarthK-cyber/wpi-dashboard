import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

def create_individual_commodity_charts():
    """Create comprehensive individual charts for each commodity with multiple chart types"""
    
    # Load data
    monthly_data = pd.read_csv('wpi_monthly_data.csv')
    seasonality_data = pd.read_csv('wpi_seasonality_analysis.csv')
    summary_data = pd.read_csv('wpi_commodity_summary.csv')
    
    # Create charts directory if it doesn't exist
    os.makedirs('individual_charts', exist_ok=True)
    
    print("Creating individual commodity charts...")
    
    for commodity in monthly_data['COMM_NAME'].unique():
        print(f"Creating charts for {commodity}...")
        
        # Filter data for this commodity
        comm_monthly = monthly_data[monthly_data['COMM_NAME'] == commodity]
        comm_seasonality = seasonality_data[seasonality_data['COMM_NAME'] == commodity]
        comm_summary = summary_data[summary_data['COMM_NAME'] == commodity]
        
        # Create subplot with multiple chart types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{commodity} - Monthly Price Trends',
                f'{commodity} - Seasonal Pattern',
                f'{commodity} - Price Distribution',
                f'{commodity} - Year-over-Year Growth'
            ),
            specs=[[{"secondary_y": True}, {"type": "polar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Chart 1: Monthly Price Trends (Line + Candlestick style)
        fig.add_trace(
            go.Scatter(
                x=comm_monthly['Date'],
                y=comm_monthly['WPI_Index'],
                mode='lines+markers',
                name='WPI Index',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add trend line
        if len(comm_monthly) > 1:
            z = pd.to_datetime(comm_monthly['Date']).map(pd.Timestamp.timestamp)
            slope, intercept = np.polyfit(z, comm_monthly['WPI_Index'], 1)
            trend_line = slope * z + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=comm_monthly['Date'],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=1
            )
        
        # Chart 2: Seasonal Pattern (Polar/Radar Chart)
        if not comm_seasonality.empty:
            fig.add_trace(
                go.Scatterpolar(
                    r=comm_seasonality['Avg_Index'],
                    theta=comm_seasonality['Month_Name'],
                    fill='toself',
                    name='Seasonal Pattern',
                    line_color='green'
                ),
                row=1, col=2
            )
        
        # Chart 3: Price Distribution (Histogram)
        fig.add_trace(
            go.Histogram(
                x=comm_monthly['WPI_Index'],
                name='Price Distribution',
                nbinsx=10,
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Chart 4: Year-over-Year Growth
        if len(comm_monthly) > 12:  # Need at least 2 years of data
            growth_data = []
            years = sorted(comm_monthly['Year'].unique())
            
            for year in years[1:]:  # Skip first year
                current_year_avg = comm_monthly[comm_monthly['Year'] == year]['WPI_Index'].mean()
                prev_year_avg = comm_monthly[comm_monthly['Year'] == year-1]['WPI_Index'].mean()
                
                if prev_year_avg > 0:
                    growth = ((current_year_avg / prev_year_avg) - 1) * 100
                    growth_data.append({'Year': year, 'Growth': growth})
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                colors = ['red' if x < 0 else 'green' for x in growth_df['Growth']]
                
                fig.add_trace(
                    go.Bar(
                        x=growth_df['Year'],
                        y=growth_df['Growth'],
                        name='YoY Growth %',
                        marker_color=colors
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
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="WPI Index", row=1, col=1)
        fig.update_xaxes(title_text="WPI Index", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=2)
        fig.update_yaxes(title_text="Growth %", row=2, col=2)
        
        # Save as HTML and PNG
        html_file = f'individual_charts/{commodity.lower().replace(" ", "_")}_comprehensive.html'
        png_file = f'individual_charts/{commodity.lower().replace(" ", "_")}_comprehensive.png'
        
        fig.write_html(html_file)
        fig.write_image(png_file, width=1200, height=800)
        
        print(f"  Saved: {html_file}")
        print(f"  Saved: {png_file}")

def create_comparison_charts():
    """Create comparison charts across commodities"""
    print("Creating comparison charts...")
    
    monthly_data = pd.read_csv('wpi_monthly_data.csv')
    seasonality_data = pd.read_csv('wpi_seasonality_analysis.csv')
    
    # 1. Multi-commodity comparison
    fig_comparison = px.line(
        monthly_data,
        x='Date',
        y='WPI_Index',
        color='COMM_NAME',
        title='Multi-Commodity Price Comparison',
        markers=True,
        width=1200,
        height=600
    )
    
    fig_comparison.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_size=12
    )
    
    fig_comparison.write_html('individual_charts/multi_commodity_comparison.html')
    fig_comparison.write_image('individual_charts/multi_commodity_comparison.png')
    
    # 2. Seasonal heatmap
    pivot_seasonal = seasonality_data.pivot(
        index='COMM_NAME',
        columns='Month_Name',
        values='Avg_Index'
    )
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_seasonal = pivot_seasonal.reindex(columns=month_order)
    
    fig_heatmap = px.imshow(
        pivot_seasonal,
        labels=dict(x="Month", y="Commodity", color="WPI Index"),
        title="Seasonal Price Patterns - All Commodities",
        color_continuous_scale="RdYlBu_r",
        width=1000,
        height=500
    )
    
    fig_heatmap.write_html('individual_charts/seasonal_heatmap_all.html')
    fig_heatmap.write_image('individual_charts/seasonal_heatmap_all.png')
    
    # 3. Volatility comparison
    summary_data = pd.read_csv('wpi_commodity_summary.csv')
    
    fig_volatility = px.bar(
        summary_data,
        x='COMM_NAME',
        y='Volatility_Percent',
        title='Price Volatility Comparison',
        color='Volatility_Percent',
        color_continuous_scale='Reds',
        width=800,
        height=500
    )
    
    fig_volatility.update_layout(
        xaxis_title="Commodity",
        yaxis_title="Volatility (%)",
        plot_bgcolor='white'
    )
    
    fig_volatility.write_html('individual_charts/volatility_comparison.html')
    fig_volatility.write_image('individual_charts/volatility_comparison.png')
    
    print("Comparison charts created!")

def create_advanced_analytics_charts():
    """Create advanced analytics charts"""
    print("Creating advanced analytics charts...")
    
    monthly_data = pd.read_csv('wpi_monthly_data.csv')
    
    # Create correlation matrix
    pivot_data = monthly_data.pivot(
        index=['Year', 'Month'],
        columns='COMM_NAME',
        values='WPI_Index'
    )
    
    correlation_matrix = pivot_data.corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        labels=dict(color="Correlation"),
        title="Price Correlation Matrix",
        color_continuous_scale="RdBu",
        width=700,
        height=600
    )
    
    fig_corr.write_html('individual_charts/correlation_matrix.html')
    fig_corr.write_image('individual_charts/correlation_matrix.png')
    
    # Create normalized trends
    normalized_data = []
    for commodity in monthly_data['COMM_NAME'].unique():
        comm_data = monthly_data[monthly_data['COMM_NAME'] == commodity].copy()
        base_value = comm_data['WPI_Index'].iloc[0]
        comm_data['Normalized_Index'] = (comm_data['WPI_Index'] / base_value) * 100
        normalized_data.append(comm_data)
    
    normalized_df = pd.concat(normalized_data)
    
    fig_normalized = px.line(
        normalized_df,
        x='Date',
        y='Normalized_Index',
        color='COMM_NAME',
        title='Normalized Price Trends (Base = 100)',
        markers=True,
        width=1200,
        height=600
    )
    
    fig_normalized.update_layout(
        plot_bgcolor='white',
        yaxis_title="Normalized Index",
        xaxis_title="Date"
    )
    
    fig_normalized.write_html('individual_charts/normalized_trends.html')
    fig_normalized.write_image('individual_charts/normalized_trends.png')
    
    print("Advanced analytics charts created!")

def main():
    """Main function to create all individual charts"""
    try:
        import numpy as np
        
        # Create all chart types
        create_individual_commodity_charts()
        create_comparison_charts()
        create_advanced_analytics_charts()
        
        print("\n" + "="*50)
        print("INDIVIDUAL CHARTS CREATION SUMMARY")
        print("="*50)
        print("Charts created in 'individual_charts/' directory:")
        
        charts_dir = 'individual_charts'
        if os.path.exists(charts_dir):
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.html')]
            for i, file in enumerate(chart_files, 1):
                print(f"{i:2d}. {file}")
        
        print(f"\nTotal: {len(chart_files)} interactive HTML charts")
        print(f"Plus: {len(chart_files)} PNG images")
        print("="*50)
        
    except ImportError:
        print("Error: numpy is required for trend analysis")
        print("Run: pip install numpy")

if __name__ == "__main__":
    main()