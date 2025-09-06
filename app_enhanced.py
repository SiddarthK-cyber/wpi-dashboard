import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# Load and process data
def load_and_process_data():
    """Load and process both yearly and monthly WPI data"""
    
    # Load yearly data
    df_yearly = pd.read_csv('wpi_10_commodities.csv')
    year_cols = [col for col in df_yearly.columns if 'INDEX' in col]
    
    # Melt yearly data to long format
    df_yearly_long = pd.melt(df_yearly, 
                            id_vars=['COMM_NAME', 'COMM_CODE', 'COMM_WT'],
                            value_vars=year_cols,
                            var_name='Year_Code',
                            value_name='WPI_Index')
    
    # Extract year from INDEX2013 format
    df_yearly_long['Year'] = df_yearly_long['Year_Code'].str.extract(r'(\d{4})').astype(int)
    df_yearly_long['Data_Type'] = 'Yearly'
    df_yearly_long['Date_Full'] = pd.to_datetime(df_yearly_long['Year'].astype(str) + '-01-01')
    
    # Load monthly data (for 4 key commodities)
    try:
        df_monthly = pd.read_csv('wpi_monthly_data.csv')
        df_monthly['Data_Type'] = 'Monthly'
        df_monthly['Date_Full'] = pd.to_datetime(df_monthly['Date'])
        
        # Combine both datasets
        # For monthly data, we need to match the structure
        df_monthly_formatted = df_monthly[['COMM_NAME', 'Year', 'WPI_Index', 'Data_Type', 'Date_Full', 'Date']].copy()
        
        return df_yearly, df_yearly_long, df_monthly_formatted
    except FileNotFoundError:
        print("Monthly data file not found, using yearly data only")
        return df_yearly, df_yearly_long, pd.DataFrame()

# Load data
df_yearly, df_yearly_long, df_monthly = load_and_process_data()

# Get available commodities for monthly data
monthly_commodities = df_monthly['COMM_NAME'].unique().tolist() if not df_monthly.empty else []

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Enhanced WPI Interactive Dashboard (2013-2024)", 
                style={
                    'textAlign': 'center', 
                    'color': '#2c3e50',
                    'marginBottom': '20px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.P("Explore Wholesale Price Index trends with yearly and monthly granularity", 
               style={
                   'textAlign': 'center', 
                   'color': '#7f8c8d',
                   'fontSize': '16px',
                   'marginBottom': '30px'
               })
    ]),
    
    # Controls Row 1: Data Type and Commodities
    html.Div([
        html.Div([
            html.Label("Data Granularity:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.RadioItems(
                id='data-type-radio',
                options=[
                    {'label': ' Yearly (2013-2024) - All Commodities', 'value': 'Yearly'},
                    {'label': ' Monthly (2020-2024) - Key Commodities Only', 'value': 'Monthly'}
                ],
                value='Yearly',
                style={'marginBottom': '15px'},
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Commodities:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='commodity-dropdown',
                multi=True,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'marginBottom': '25px'}),
    
    # Controls Row 2: Time Range  
    html.Div([
        html.Div([
            html.Label(id='time-range-label', 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.RangeSlider(
                id='time-slider',
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '70%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Chart Update:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Checklist(
                id='chart-options',
                options=[
                    {'label': ' Show Trend Line', 'value': 'trend'},
                    {'label': ' Show Data Points', 'value': 'markers'}
                ],
                value=['markers'],
                style={'marginTop': '10px'}
            )
        ], style={'width': '25%', 'float': 'right', 'display': 'inline-block', 'textAlign': 'center'})
    ], style={'marginBottom': '30px'}),
    
    # Charts Grid
    html.Div([
        # Top row
        html.Div([
            html.Div([
                dcc.Graph(id='main-trend-chart')
            ], style={'width': '60%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='comparison-chart')
            ], style={'width': '38%', 'float': 'right', 'display': 'inline-block'})
        ]),
        
        # Bottom row
        html.Div([
            html.Div([
                dcc.Graph(id='volatility-analysis')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='seasonal-pattern')
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ])
    ]),
    
    # Enhanced Summary Stats
    html.Div(id='enhanced-summary', 
             style={
                 'marginTop': '30px', 
                 'padding': '20px', 
                 'backgroundColor': '#f8f9fa',
                 'borderRadius': '10px',
                 'border': '1px solid #e9ecef'
             }),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P([
            "📊 Enhanced Dashboard with Monthly/Yearly Toggle | ",
            "Data Source: Ministry of Commerce & Industry, Government of India | ",
            html.A("View Code", 
                   href="https://github.com/yourusername/wpi-dashboard", 
                   target="_blank")
        ], style={
            'textAlign': 'center', 
            'color': '#95a5a6',
            'marginTop': '20px',
            'fontSize': '14px'
        })
    ])
], style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0 auto',
    'maxWidth': '1400px',
    'padding': '20px'
})

# Callback to update commodity dropdown based on data type
@callback(
    [Output('commodity-dropdown', 'options'),
     Output('commodity-dropdown', 'value'),
     Output('time-slider', 'min'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'), 
     Output('time-slider', 'value'),
     Output('time-range-label', 'children')],
    [Input('data-type-radio', 'value')]
)
def update_controls(data_type):
    if data_type == 'Monthly':
        # Monthly data options
        options = [{'label': comm, 'value': comm} for comm in monthly_commodities]
        default_value = monthly_commodities[:3] if len(monthly_commodities) >= 3 else monthly_commodities
        
        # Time slider for months (2020-2024)
        min_val, max_val = 2020, 2024
        marks = {year: str(year) for year in range(2020, 2025)}
        default_time = [2020, 2024]
        label = "Year Range (Monthly Data):"
        
    else:
        # Yearly data options  
        options = [{'label': comm, 'value': comm} for comm in sorted(df_yearly['COMM_NAME'].unique())]
        default_value = ['All commodities', 'Paddy', 'Wheat', 'Gram']
        
        # Time slider for years (2013-2024)
        min_val, max_val = 2013, 2024
        marks = {year: str(year) for year in range(2013, 2025, 2)}
        default_time = [2013, 2024]
        label = "Year Range:"
    
    return options, default_value, min_val, max_val, marks, default_time, label

# Main callback for updating charts
@callback(
    [Output('main-trend-chart', 'figure'),
     Output('comparison-chart', 'figure'),
     Output('volatility-analysis', 'figure'),
     Output('seasonal-pattern', 'figure'),
     Output('enhanced-summary', 'children')],
    [Input('commodity-dropdown', 'value'),
     Input('time-slider', 'value'),
     Input('data-type-radio', 'value'),
     Input('chart-options', 'value')]
)
def update_enhanced_charts(selected_commodities, time_range, data_type, chart_options):
    if not selected_commodities:
        # Return empty charts if no commodities selected
        empty_fig = go.Figure().add_annotation(
            text="Please select commodities to view charts",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, "Please select commodities to see statistics."
    
    # Determine chart styling
    show_markers = 'markers' in chart_options
    show_trend = 'trend' in chart_options
    
    # Filter data based on type
    if data_type == 'Monthly':
        filtered_data = df_monthly[
            (df_monthly['COMM_NAME'].isin(selected_commodities)) &
            (df_monthly['Year'] >= time_range[0]) &
            (df_monthly['Year'] <= time_range[1])
        ].copy()
        
        x_axis = 'Date_Full'
        x_title = 'Month'
        
    else:
        filtered_data = df_yearly_long[
            (df_yearly_long['COMM_NAME'].isin(selected_commodities)) &
            (df_yearly_long['Year'] >= time_range[0]) &
            (df_yearly_long['Year'] <= time_range[1])
        ].copy()
        
        x_axis = 'Year'
        x_title = 'Year'
    
    if filtered_data.empty:
        empty_fig = go.Figure().add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, "No data available for selected filters."
    
    # Chart 1: Main Trend Chart
    main_fig = px.line(filtered_data, 
                       x=x_axis, 
                       y='WPI_Index',
                       color='COMM_NAME',
                       title=f'WPI Trends - {data_type} Data',
                       markers=show_markers)
    
    main_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_size=12,
        title_font_size=16,
        xaxis_title=x_title,
        yaxis_title='WPI Index',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    # Chart 2: Price Change Comparison
    if data_type == 'Monthly':
        # Compare first and last month
        start_data = filtered_data.groupby('COMM_NAME')['WPI_Index'].first()
        end_data = filtered_data.groupby('COMM_NAME')['WPI_Index'].last()
    else:
        # Compare first and last year  
        start_data = filtered_data[filtered_data['Year'] == time_range[0]].set_index('COMM_NAME')['WPI_Index']
        end_data = filtered_data[filtered_data['Year'] == time_range[1]].set_index('COMM_NAME')['WPI_Index']
    
    change_data = []
    for comm in selected_commodities:
        if comm in start_data.index and comm in end_data.index:
            change_pct = ((end_data[comm] / start_data[comm]) - 1) * 100
            change_data.append({'Commodity': comm, 'Change_Percent': change_pct})
    
    if change_data:
        change_df = pd.DataFrame(change_data)
        comp_fig = px.bar(change_df,
                         x='Change_Percent',
                         y='Commodity',
                         orientation='h',
                         title=f'Price Change: {time_range[0]} vs {time_range[1]} (%)',
                         color='Change_Percent',
                         color_continuous_scale='RdYlBu_r')
    else:
        comp_fig = go.Figure().add_annotation(text="No comparison data available",
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    comp_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_size=12,
        title_font_size=16,
        height=400
    )
    
    # Chart 3: Volatility Analysis
    volatility_data = []
    for comm in selected_commodities:
        comm_data = filtered_data[filtered_data['COMM_NAME'] == comm]['WPI_Index']
        if len(comm_data) > 1:
            volatility_data.append({
                'Commodity': comm,
                'Std_Deviation': comm_data.std(),
                'Coefficient_of_Variation': (comm_data.std() / comm_data.mean()) * 100
            })
    
    if volatility_data:
        vol_df = pd.DataFrame(volatility_data)
        vol_fig = px.bar(vol_df,
                        x='Commodity',
                        y='Coefficient_of_Variation',
                        title='Price Volatility (Coefficient of Variation %)',
                        color='Coefficient_of_Variation',
                        color_continuous_scale='Reds')
    else:
        vol_fig = go.Figure().add_annotation(text="No volatility data available",
                                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    vol_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_size=12,
        title_font_size=16,
        height=400
    )
    
    # Chart 4: Seasonal/Time Pattern Analysis
    if data_type == 'Monthly':
        # Monthly seasonal pattern
        monthly_pattern = filtered_data.copy()
        monthly_pattern['Month'] = monthly_pattern['Date_Full'].dt.month
        monthly_avg = monthly_pattern.groupby(['COMM_NAME', 'Month'])['WPI_Index'].mean().reset_index()
        
        seasonal_fig = px.line(monthly_avg,
                              x='Month',
                              y='WPI_Index',
                              color='COMM_NAME',
                              title='Average Monthly Seasonal Pattern',
                              markers=True)
        seasonal_fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
    else:
        # Yearly growth rate
        growth_data = []
        for comm in selected_commodities:
            comm_data = filtered_data[filtered_data['COMM_NAME'] == comm].sort_values('Year')
            if len(comm_data) > 1:
                for i in range(1, len(comm_data)):
                    prev_val = comm_data.iloc[i-1]['WPI_Index']
                    curr_val = comm_data.iloc[i]['WPI_Index']
                    growth_rate = ((curr_val / prev_val) - 1) * 100
                    growth_data.append({
                        'COMM_NAME': comm,
                        'Year': comm_data.iloc[i]['Year'],
                        'Growth_Rate': growth_rate
                    })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            seasonal_fig = px.bar(growth_df,
                                 x='Year',
                                 y='Growth_Rate',
                                 color='COMM_NAME',
                                 title='Year-over-Year Growth Rate (%)',
                                 barmode='group')
        else:
            seasonal_fig = go.Figure().add_annotation(text="No growth data available",
                                                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    seasonal_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_size=12,
        title_font_size=16,
        height=400
    )
    
    # Enhanced Summary Statistics
    summary_stats = []
    
    if not filtered_data.empty:
        for comm in selected_commodities:
            comm_data = filtered_data[filtered_data['COMM_NAME'] == comm]
            if not comm_data.empty:
                latest_value = comm_data['WPI_Index'].iloc[-1]
                first_value = comm_data['WPI_Index'].iloc[0]
                total_change = ((latest_value / first_value) - 1) * 100
                avg_value = comm_data['WPI_Index'].mean()
                volatility = comm_data['WPI_Index'].std()
                
                summary_stats.append({
                    'commodity': comm,
                    'latest': latest_value,
                    'change': total_change,
                    'average': avg_value,
                    'volatility': volatility
                })
    
    if summary_stats:
        summary_cards = []
        for stat in summary_stats[:4]:  # Show max 4 commodities
            card = html.Div([
                html.H4(stat['commodity'], style={'color': '#2c3e50', 'margin': '0 0 10px 0'}),
                html.P(f"Latest: {stat['latest']:.1f}", style={'margin': '5px 0', 'fontSize': '14px'}),
                html.P(f"Change: {stat['change']:+.1f}%", 
                      style={'margin': '5px 0', 'fontSize': '14px', 
                            'color': '#e74c3c' if stat['change'] > 0 else '#27ae60'}),
                html.P(f"Avg: {stat['average']:.1f}", style={'margin': '5px 0', 'fontSize': '14px'}),
                html.P(f"Volatility: {stat['volatility']:.1f}", style={'margin': '5px 0', 'fontSize': '14px'})
            ], style={
                'width': f'{100/len(summary_stats[:4]):.1f}%',
                'display': 'inline-block',
                'textAlign': 'center',
                'padding': '15px',
                'border': '1px solid #dee2e6',
                'borderRadius': '8px',
                'margin': '0 1% 0 0',
                'backgroundColor': 'white'
            })
            summary_cards.append(card)
        
        enhanced_summary = html.Div([
            html.H3(f"Summary Statistics - {data_type} Data", 
                   style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div(summary_cards, style={'textAlign': 'center'})
        ])
    else:
        enhanced_summary = html.P("No summary statistics available.")
    
    return main_fig, comp_fig, vol_fig, seasonal_fig, enhanced_summary

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)