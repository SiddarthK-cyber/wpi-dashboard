import pandas as pd
import os
from pathlib import Path

def create_comprehensive_monthly_data():
    """Create comprehensive monthly dataset from existing CSV and add more commodities"""
    print("Loading existing monthly data...")
    
    # Load existing monthly data
    existing_data = pd.read_csv('wpi_monthly_data.csv')
    print(f"Existing data: {len(existing_data)} records, {existing_data['COMM_NAME'].nunique()} commodities")
    
    # Check what commodities we have
    existing_commodities = existing_data['COMM_NAME'].unique()
    print(f"Current commodities: {', '.join(existing_commodities)}")
    
    return existing_data

def compute_seasonality_analysis(monthly_data):
    """Compute comprehensive seasonality analysis"""
    print("Computing seasonality patterns...")
    
    seasonality_results = []
    
    for commodity in monthly_data['COMM_NAME'].unique():
        comm_data = monthly_data[monthly_data['COMM_NAME'] == commodity]
        
        # Calculate monthly averages across all years
        for month in range(1, 13):
            month_data = comm_data[comm_data['Month'] == month]['WPI_Index']
            
            if len(month_data) > 0:
                # Month names
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Calculate statistics
                avg_price = month_data.mean()
                min_price = month_data.min()
                max_price = month_data.max()
                std_price = month_data.std()
                
                # Calculate year-over-year growth for this month
                yoy_growth = []
                years = sorted(comm_data['Year'].unique())
                for i in range(1, len(years)):
                    prev_year_data = comm_data[(comm_data['Year'] == years[i-1]) & (comm_data['Month'] == month)]
                    curr_year_data = comm_data[(comm_data['Year'] == years[i]) & (comm_data['Month'] == month)]
                    
                    if len(prev_year_data) > 0 and len(curr_year_data) > 0:
                        growth = ((curr_year_data['WPI_Index'].iloc[0] / prev_year_data['WPI_Index'].iloc[0]) - 1) * 100
                        yoy_growth.append(growth)
                
                avg_growth = sum(yoy_growth) / len(yoy_growth) if yoy_growth else 0
                
                seasonality_results.append({
                    'COMM_NAME': commodity,
                    'Month': month,
                    'Month_Name': month_names[month-1],
                    'Avg_Index': round(avg_price, 2),
                    'Min_Index': round(min_price, 2),
                    'Max_Index': round(max_price, 2),
                    'Std_Dev': round(std_price, 2),
                    'Avg_YoY_Growth': round(avg_growth, 2),
                    'Data_Points': len(month_data),
                    'Coefficient_of_Variation': round((std_price / avg_price) * 100, 2) if avg_price > 0 else 0
                })
    
    seasonality_df = pd.DataFrame(seasonality_results)
    
    # Calculate seasonality index for each commodity
    for commodity in seasonality_df['COMM_NAME'].unique():
        comm_seasonality = seasonality_df[seasonality_df['COMM_NAME'] == commodity]
        annual_avg = comm_seasonality['Avg_Index'].mean()
        
        # Update seasonality index
        seasonality_df.loc[seasonality_df['COMM_NAME'] == commodity, 'Seasonality_Index'] = \
            (comm_seasonality['Avg_Index'] / annual_avg * 100).round(2)
        
        # Classify seasonal patterns
        seasonality_df.loc[seasonality_df['COMM_NAME'] == commodity, 'Season_Pattern'] = \
            comm_seasonality['Avg_Index'].apply(
                lambda x: 'High Season' if x > annual_avg * 1.05 else 
                         ('Low Season' if x < annual_avg * 0.95 else 'Normal')
            )
    
    return seasonality_df

def create_enhanced_datasets():
    """Create enhanced datasets for the dashboard"""
    
    # Load and process monthly data
    monthly_data = create_comprehensive_monthly_data()
    
    # Compute seasonality analysis
    seasonality_data = compute_seasonality_analysis(monthly_data)
    
    # Save seasonality analysis
    seasonality_data.to_csv('wpi_seasonality_analysis.csv', index=False)
    print(f"Saved seasonality analysis: {len(seasonality_data)} records")
    
    # Create summary statistics
    summary_stats = []
    for commodity in monthly_data['COMM_NAME'].unique():
        comm_data = monthly_data[monthly_data['COMM_NAME'] == commodity]
        
        # Overall statistics
        overall_avg = comm_data['WPI_Index'].mean()
        overall_std = comm_data['WPI_Index'].std()
        price_range = comm_data['WPI_Index'].max() - comm_data['WPI_Index'].min()
        
        # Volatility (coefficient of variation)
        volatility = (overall_std / overall_avg) * 100 if overall_avg > 0 else 0
        
        # Trend calculation (simple linear trend)
        years = comm_data['Year'].unique()
        if len(years) > 1:
            annual_avgs = []
            for year in sorted(years):
                year_avg = comm_data[comm_data['Year'] == year]['WPI_Index'].mean()
                annual_avgs.append(year_avg)
            
            # Simple trend (last year vs first year)
            trend = ((annual_avgs[-1] / annual_avgs[0]) - 1) * 100 if annual_avgs[0] > 0 else 0
        else:
            trend = 0
        
        # Find peak and low seasons
        comm_seasonality = seasonality_data[seasonality_data['COMM_NAME'] == commodity]
        if len(comm_seasonality) > 0:
            peak_month = comm_seasonality.loc[comm_seasonality['Avg_Index'].idxmax(), 'Month_Name']
            low_month = comm_seasonality.loc[comm_seasonality['Avg_Index'].idxmin(), 'Month_Name']
        else:
            peak_month = 'N/A'
            low_month = 'N/A'
        
        summary_stats.append({
            'COMM_NAME': commodity,
            'Overall_Avg_Index': round(overall_avg, 2),
            'Volatility_Percent': round(volatility, 2),
            'Price_Range': round(price_range, 2),
            'Long_Term_Trend_Percent': round(trend, 2),
            'Peak_Season_Month': peak_month,
            'Low_Season_Month': low_month,
            'Data_Points': len(comm_data),
            'Year_Range': f"{comm_data['Year'].min()}-{comm_data['Year'].max()}"
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('wpi_commodity_summary.csv', index=False)
    print(f"Saved commodity summary: {len(summary_df)} commodities")
    
    # Print summary report
    print("\n" + "="*50)
    print("WPI DATA ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Commodities Analyzed: {len(summary_df)}")
    print(f"Total Monthly Records: {len(monthly_data)}")
    print(f"Date Range: {monthly_data['Year'].min()}-{monthly_data['Year'].max()}")
    
    print(f"\nMost Volatile Commodities:")
    top_volatile = summary_df.nlargest(3, 'Volatility_Percent')[['COMM_NAME', 'Volatility_Percent']]
    for _, row in top_volatile.iterrows():
        print(f"  - {row['COMM_NAME']}: {row['Volatility_Percent']:.1f}%")
    
    print(f"\nHighest Growth Commodities:")
    top_growth = summary_df.nlargest(3, 'Long_Term_Trend_Percent')[['COMM_NAME', 'Long_Term_Trend_Percent']]
    for _, row in top_growth.iterrows():
        print(f"  - {row['COMM_NAME']}: {row['Long_Term_Trend_Percent']:.1f}%")
    
    print("\n" + "="*50)
    
    return monthly_data, seasonality_data, summary_df

if __name__ == "__main__":
    monthly, seasonality, summary = create_enhanced_datasets()
    print("\nAll datasets created successfully!")
    print("Files generated:")
    print("  - wpi_seasonality_analysis.csv")
    print("  - wpi_commodity_summary.csv")