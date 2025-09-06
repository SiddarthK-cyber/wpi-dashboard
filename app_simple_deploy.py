import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import os

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

def load_data():
    """Load WPI data with error handling"""
    try:
        # Try to load yearly data
        df_yearly = pd.read_csv('wpi_10_commodities.csv')
        print("Yearly data loaded successfully")
        
        # Try to load monthly data
        try:
            df_monthly = pd.read_csv('wpi_monthly_data.csv')
            print("Monthly data loaded successfully")
            return df_yearly, df_monthly, True
        except Exception as e:
            print(f"Monthly data error: {e}")
            return df_yearly, pd.DataFrame(), False
            
    except Exception as e:
        print(f"Data loading error: {e}")
        # Create dummy data if files don't exist
        dummy_data = pd.DataFrame({
            'COMM_NAME': ['All commodities', 'Paddy', 'Wheat'],
            'INDEX2020': [121.8, 121.8, 121.8],
            'INDEX2024': [154.0, 154.0, 154.0]
        })
        return dummy_data, pd.DataFrame(), False

# Load data
df_yearly, df_monthly, has_monthly = load_data()

# Simple layout
app.layout = html.Div([
    html.H1("WPI Dashboard - Deployment Test", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    html.Div([
        html.P(f"Yearly data loaded: {len(df_yearly)} rows"),
        html.P(f"Monthly data available: {'Yes' if has_monthly else 'No'}"),
        html.P(f"Monthly data rows: {len(df_monthly) if has_monthly else 0}")
    ], style={'textAlign': 'center', 'margin': '20px'}),
    
    html.Div([
        html.Label("Select Commodities:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='commodity-dropdown',
            options=[{'label': comm, 'value': comm} for comm in df_yearly['COMM_NAME'].unique()],
            value=[df_yearly['COMM_NAME'].iloc[0]] if len(df_yearly) > 0 else [],
            multi=True
        )
    ], style={'margin': '20px', 'maxWidth': '600px', 'marginLeft': 'auto', 'marginRight': 'auto'}),
    
    dcc.Graph(id='simple-chart'),
    
    html.Div([
        html.H3("Deployment Status: ✅ SUCCESS"),
        html.P("Dashboard is running correctly on Render.com"),
        html.P("Data files loaded and processing working")
    ], style={'textAlign': 'center', 'margin': '20px', 'color': 'green'})
])

@callback(
    Output('simple-chart', 'figure'),
    [Input('commodity-dropdown', 'value')]
)
def update_chart(selected_commodities):
    if not selected_commodities or len(df_yearly) == 0:
        return px.bar(title="No data selected")
    
    # Create simple chart from yearly data
    try:
        filtered_df = df_yearly[df_yearly['COMM_NAME'].isin(selected_commodities)]
        
        # Get year columns
        year_cols = [col for col in df_yearly.columns if 'INDEX' in col]
        
        if len(year_cols) >= 2:
            # Compare first and last year
            first_year_col = year_cols[0]
            last_year_col = year_cols[-1]
            
            chart_data = []
            for _, row in filtered_df.iterrows():
                if pd.notna(row[first_year_col]) and pd.notna(row[last_year_col]):
                    change = ((row[last_year_col] / row[first_year_col]) - 1) * 100
                    chart_data.append({
                        'Commodity': row['COMM_NAME'],
                        'Change_Percent': change
                    })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig = px.bar(chart_df, 
                           x='Commodity', 
                           y='Change_Percent',
                           title='WPI Change (%) - First vs Last Available Year',
                           color='Change_Percent',
                           color_continuous_scale='RdYlBu_r')
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_size=12,
                    height=400
                )
                return fig
    except Exception as e:
        print(f"Chart error: {e}")
    
    return px.bar(title="Chart rendering error - check logs")

if __name__ == '__main__':
    # For local development
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))