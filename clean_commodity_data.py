import pandas as pd
import re

def clean_commodity_names(df):
    """Clean commodity names by removing prefixes and standardizing format"""
    
    def clean_name(name):
        """Clean individual commodity name"""
        if pd.isna(name):
            return name
            
        name = str(name).strip()
        
        # Remove prefixes like "a.", "a1.", "a2.", "(A).", etc.
        prefixes_to_remove = [
            r'^[a-z]\d*\.\s*',  # a., a1., a2., etc.
            r'^\([A-Z]\)\.\s*',  # (A)., (B)., etc.
            r'^\([a-z]\)\.\s*',  # (a)., (b)., etc.
            r'^\d+\.\s*',        # 1., 2., etc.
            r'^\([A-Za-z]\d*\)\.\s*',  # (A1)., (B2)., etc.
        ]
        
        for prefix_pattern in prefixes_to_remove:
            name = re.sub(prefix_pattern, '', name)
        
        # Clean up extra spaces and formatting
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        name = name.strip()
        
        # Capitalize first letter of each word properly
        # But keep existing capitalization for abbreviations
        words = name.split()
        cleaned_words = []
        for word in words:
            if len(word) <= 3 and word.isupper():  # Keep abbreviations like WPI, CPI
                cleaned_words.append(word)
            elif word.startswith('(') and word.endswith(')'):  # Keep parenthetical as is
                cleaned_words.append(word)
            else:
                cleaned_words.append(word.capitalize())
        
        name = ' '.join(cleaned_words)
        
        return name
    
    # Apply cleaning to commodity names
    df['COMM_NAME_ORIGINAL'] = df['COMM_NAME'].copy()  # Keep original for reference
    df['COMM_NAME'] = df['COMM_NAME'].apply(clean_name)
    
    return df

def filter_relevant_commodities(df):
    """Filter out aggregate categories and keep individual commodities"""
    
    # Remove these broad categories that are aggregates
    categories_to_exclude = [
        'All Commodities',
        'Primary Articles', 
        'Food Articles',
        'Food Grains (Cereals+Pulses)',
        'Cereals',
        'Pulses',
        'Manufactured Products',
        'Basic Metals',
        'Non-Ferrous Metals',
        'Textiles',
        'Chemical Products',
        'Paper Products',
        'Rubber Products',
        'Non-Metallic Mineral Products'
    ]
    
    # Filter out rows that are clearly aggregate categories
    def is_individual_commodity(name):
        name_lower = str(name).lower()
        
        # Exclude if it's a broad category
        for category in categories_to_exclude:
            if category.lower() in name_lower:
                return False
                
        # Exclude if name is very generic/broad
        generic_terms = [
            'products', 'articles', 'manufacturing', 'industries',
            'sectors', 'groups', 'categories', 'divisions'
        ]
        
        if any(term in name_lower for term in generic_terms):
            return False
            
        return True
    
    # Apply filtering
    individual_commodities = df[df['COMM_NAME'].apply(is_individual_commodity)].copy()
    
    return individual_commodities

def main():
    """Main function to clean and process commodity data"""
    print("Loading and cleaning commodity data...")
    
    # Load data
    monthly_data = pd.read_csv('wpi_all_commodities_monthly.csv')
    print(f"Original data: {len(monthly_data)} records, {monthly_data['COMM_NAME'].nunique()} commodities")
    
    # Clean commodity names
    monthly_data = clean_commodity_names(monthly_data)
    
    # Filter to individual commodities only
    clean_data = filter_relevant_commodities(monthly_data)
    
    print(f"After cleaning: {len(clean_data)} records, {clean_data['COMM_NAME'].nunique()} individual commodities")
    
    # Remove duplicates if any
    clean_data = clean_data.drop_duplicates(subset=['COMM_NAME', 'Year', 'Month'])
    
    # Sort commodities alphabetically
    clean_data = clean_data.sort_values(['COMM_NAME', 'Year', 'Month'])
    
    # Save cleaned data
    clean_data.to_csv('wpi_commodities_cleaned.csv', index=False)
    
    # Create updated summary
    summary_data = []
    for commodity in sorted(clean_data['COMM_NAME'].unique()):
        comm_data = clean_data[clean_data['COMM_NAME'] == commodity]
        
        if len(comm_data) > 1:
            avg_index = comm_data['WPI_Index'].mean()
            volatility = (comm_data['WPI_Index'].std() / avg_index) * 100 if avg_index > 0 else 0
            
            # Calculate growth
            years = sorted(comm_data['Year'].unique())
            if len(years) > 1:
                first_year_avg = comm_data[comm_data['Year'] == years[0]]['WPI_Index'].mean()
                last_year_avg = comm_data[comm_data['Year'] == years[-1]]['WPI_Index'].mean()
                growth = ((last_year_avg / first_year_avg) - 1) * 100 if first_year_avg > 0 else 0
            else:
                growth = 0
            
            summary_data.append({
                'COMM_NAME': commodity,
                'Avg_Index': round(avg_index, 2),
                'Volatility_Percent': round(volatility, 2),
                'Price_Range': round(comm_data['WPI_Index'].max() - comm_data['WPI_Index'].min(), 2),
                'Growth_Percent': round(growth, 2),
                'Data_Points': len(comm_data),
                'Year_Range': f"{comm_data['Year'].min()}-{comm_data['Year'].max()}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('wpi_commodities_summary_cleaned.csv', index=False)
    
    print(f"\nCleaned commodity names (first 20):")
    for i, commodity in enumerate(sorted(clean_data['COMM_NAME'].unique())[:20]):
        print(f"{i+1:2d}. {commodity}")
    
    print(f"\nFiles created:")
    print(f"- wpi_commodities_cleaned.csv ({len(clean_data)} records)")
    print(f"- wpi_commodities_summary_cleaned.csv ({len(summary_df)} commodities)")
    
    return clean_data, summary_df

if __name__ == "__main__":
    clean_data, summary_data = main()