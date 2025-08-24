import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

class FlightDataPreprocessor:
    """
    Comprehensive flight data preprocessing pipeline for Mumbai/Delhi flight scheduling optimization
    """
    
    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load data from Excel file and combine all sheets"""
        print("Loading flight data from Excel file...")
        
        excel_file = pd.ExcelFile(self.excel_file_path)
        all_sheets_data = []
        
        for sheet_name in excel_file.sheet_names:
            print(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
            df['Time_Period'] = sheet_name
            all_sheets_data.append(df)
        
        # Combine all sheets
        self.raw_data = pd.concat(all_sheets_data, ignore_index=True)
        print(f"Loaded {len(self.raw_data)} rows from {len(excel_file.sheet_names)} sheets")
        return self.raw_data
    
    def clean_basic_structure(self):
        """Clean basic data structure and identify flight records"""
        print("Cleaning basic data structure...")
        
        df = self.raw_data.copy()
        
        # The data structure is: Flight header row followed by daily records
        # Filter for rows that have actual flight data (not headers)
        flight_data = []
        current_flight_number = None
        
        for idx, row in df.iterrows():
            # Check if this is a flight header row (has S.No and Flight Number)
            if pd.notna(row['S.No']) and pd.notna(row['Flight Number']):
                current_flight_number = row['Flight Number']
                continue
            
            # Check if this is a data row (has From, To, and other details)
            if (pd.notna(row['From']) and pd.notna(row['To']) and 
                pd.notna(row['STD']) and pd.notna(row['ATD'])):
                
                # Add flight number to this row
                row_data = row.copy()
                row_data['Flight_Number'] = current_flight_number
                flight_data.append(row_data)
        
        if flight_data:
            cleaned_df = pd.DataFrame(flight_data)
            print(f"Extracted {len(cleaned_df)} valid flight records")
            return cleaned_df
        else:
            print("No valid flight records found")
            return pd.DataFrame()
    
    def standardize_datetime_fields(self, df):
        """Standardize and parse datetime fields"""
        print("Standardizing datetime fields...")
        
        df = df.copy()
        
        # Handle the date column (different formats in different sheets)
        date_col = 'Unnamed: 2' if 'Unnamed: 2' in df.columns else 'Date'
        
        # Parse dates
        df['Flight_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Parse scheduled and actual times
        time_columns = ['STD', 'ATD', 'STA']
        
        for col in time_columns:
            if col in df.columns:
                # Convert time objects to strings for easier processing
                df[f'{col}_parsed'] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time
        
        # Parse ATA (Actual Time of Arrival) - format like "Landed 8:14 AM"
        if 'ATA' in df.columns:
            df['ATA_parsed'] = df['ATA'].apply(self._parse_ata_time)
        
        return df
    
    def _parse_ata_time(self, ata_string):
        """Parse ATA string like 'Landed 8:14 AM' to time object"""
        if pd.isna(ata_string) or not isinstance(ata_string, str):
            return None
        
        # Extract time from strings like "Landed 8:14 AM"
        time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)'
        match = re.search(time_pattern, ata_string, re.IGNORECASE)
        
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            am_pm = match.group(3).upper()
            
            # Convert to 24-hour format
            if am_pm == 'PM' and hours != 12:
                hours += 12
            elif am_pm == 'AM' and hours == 12:
                hours = 0
            
            # Return as time object for consistency
            from datetime import time
            return time(hours, minutes)
        
        return None
    
    def calculate_delays(self, df):
        """Calculate various delay metrics"""
        print("Calculating delay metrics...")
        
        df = df.copy()
        
        # Calculate departure delays (ATD - STD)
        df['Departure_Delay_Minutes'] = self._time_diff_in_minutes(
            df['ATD_parsed'], df['STD_parsed']
        )
        
        # Calculate arrival delays (ATA - STA)
        df['Arrival_Delay_Minutes'] = self._time_diff_in_minutes(
            df['ATA_parsed'], df['STA_parsed']
        )
        
        # Categorize delays
        df['Departure_Delay_Category'] = df['Departure_Delay_Minutes'].apply(self._categorize_delay)
        df['Arrival_Delay_Category'] = df['Arrival_Delay_Minutes'].apply(self._categorize_delay)
        
        # Binary delay indicators
        df['Is_Delayed_Departure'] = df['Departure_Delay_Minutes'] > 15  # >15 min considered delayed
        df['Is_Delayed_Arrival'] = df['Arrival_Delay_Minutes'] > 15
        
        return df
    
    def _time_diff_in_minutes(self, actual_time_series, scheduled_time_series):
        """Calculate time difference in minutes for Series"""
        def calculate_single_diff(actual_time, scheduled_time):
            if pd.isna(actual_time) or pd.isna(scheduled_time):
                return np.nan
            
            try:
                # Convert to datetime for calculation
                from datetime import time, datetime
                
                if isinstance(actual_time, str):
                    actual_dt = datetime.strptime(actual_time, '%H:%M:%S')
                elif isinstance(actual_time, time):
                    actual_dt = datetime.combine(datetime.today(), actual_time)
                else:
                    actual_dt = actual_time
                
                if isinstance(scheduled_time, str):
                    scheduled_dt = datetime.strptime(scheduled_time, '%H:%M:%S')
                elif isinstance(scheduled_time, time):
                    scheduled_dt = datetime.combine(datetime.today(), scheduled_time)
                else:
                    scheduled_dt = scheduled_time
                
                diff = actual_dt - scheduled_dt
                return diff.total_seconds() / 60
            except Exception as e:
                print(f"Error calculating time diff: {e}, actual: {actual_time}, scheduled: {scheduled_time}")
                return np.nan
        
        # Apply to each pair of times
        result = []
        for actual, scheduled in zip(actual_time_series, scheduled_time_series):
            result.append(calculate_single_diff(actual, scheduled))
        return pd.Series(result)
    
    def _categorize_delay(self, delay_minutes):
        """Categorize delays into groups"""
        if pd.isna(delay_minutes):
            return 'Unknown'
        elif delay_minutes <= 0:
            return 'On Time/Early'
        elif delay_minutes <= 15:
            return 'Minor Delay'
        elif delay_minutes <= 60:
            return 'Moderate Delay'
        else:
            return 'Major Delay'
    
    def clean_airport_codes(self, df):
        """Extract and standardize airport codes"""
        print("Cleaning airport codes...")
        
        df = df.copy()
        
        # Extract airport codes from strings like "Mumbai (BOM)"
        df['Origin_Airport'] = df['From'].apply(self._extract_airport_code)
        df['Destination_Airport'] = df['To'].apply(self._extract_airport_code)
        df['Origin_City'] = df['From'].apply(self._extract_city_name)
        df['Destination_City'] = df['To'].apply(self._extract_city_name)
        
        return df
    
    def _extract_airport_code(self, airport_string):
        """Extract airport code from string like 'Mumbai (BOM)'"""
        if pd.isna(airport_string):
            return None
        
        # Remove non-breaking spaces and extract code in parentheses
        clean_string = airport_string.replace('\xa0', ' ').strip()
        match = re.search(r'\(([A-Z]{3})\)', clean_string)
        return match.group(1) if match else None
    
    def _extract_city_name(self, airport_string):
        """Extract city name from string like 'Mumbai (BOM)'"""
        if pd.isna(airport_string):
            return None
        
        # Remove non-breaking spaces and extract city name
        clean_string = airport_string.replace('\xa0', ' ').strip()
        match = re.search(r'^([^(]+)', clean_string)
        return match.group(1).strip() if match else None
    
    def add_time_features(self, df):
        """Add time-based features for analysis"""
        print("Adding time-based features...")
        
        df = df.copy()
        
        # Extract time features from Flight_Date
        df['Year'] = df['Flight_Date'].dt.year
        df['Month'] = df['Flight_Date'].dt.month
        df['Day'] = df['Flight_Date'].dt.day
        df['Weekday'] = df['Flight_Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Weekday_Name'] = df['Flight_Date'].dt.day_name()
        df['Is_Weekend'] = df['Weekday'].isin([5, 6])  # Saturday, Sunday
        
        # Extract hour from scheduled departure time
        df['Scheduled_Departure_Hour'] = pd.to_datetime(
            df['STD_parsed'], format='%H:%M:%S', errors='coerce'
        ).dt.hour
        
        # Extract hour from scheduled arrival time
        df['Scheduled_Arrival_Hour'] = pd.to_datetime(
            df['STA_parsed'], format='%H:%M:%S', errors='coerce'
        ).dt.hour
        
        # Categorize time periods
        df['Time_Period_Category'] = df['Time_Period']
        
        return df
    
    def add_route_features(self, df):
        """Add route-specific features"""
        print("Adding route features...")
        
        df = df.copy()
        
        # Create route identifier
        df['Route'] = df['Origin_Airport'] + '-' + df['Destination_Airport']
        
        # Calculate route statistics
        route_stats = df.groupby('Route').agg({
            'Departure_Delay_Minutes': ['count', 'mean', 'std'],
            'Arrival_Delay_Minutes': ['mean', 'std'],
            'Flight time': 'first'  # Duration should be consistent for same route
        }).round(2)
        
        route_stats.columns = ['Route_Flight_Count', 'Route_Avg_Dep_Delay', 'Route_Std_Dep_Delay',
                              'Route_Avg_Arr_Delay', 'Route_Std_Arr_Delay', 'Route_Duration']
        
        # Merge back to main dataframe
        df = df.merge(route_stats, on='Route', how='left')
        
        # Add route popularity ranking
        route_popularity = df['Route'].value_counts().reset_index()
        route_popularity.columns = ['Route', 'Route_Popularity_Rank']
        route_popularity['Route_Popularity_Rank'] = route_popularity.index + 1
        
        df = df.merge(route_popularity, on='Route', how='left')
        
        return df
    
    def detect_peak_hours(self, df):
        """Detect and mark peak hours based on flight density"""
        print("Detecting peak hours...")
        
        df = df.copy()
        
        # Count flights by hour
        hourly_counts = df.groupby('Scheduled_Departure_Hour').size()
        
        # Define peak hours (top 25% busiest hours)
        peak_threshold = hourly_counts.quantile(0.75)
        peak_hours = hourly_counts[hourly_counts >= peak_threshold].index.tolist()
        
        df['Is_Peak_Hour'] = df['Scheduled_Departure_Hour'].isin(peak_hours)
        
        print(f"Peak hours identified: {peak_hours}")
        
        return df
    
    def process_all(self):
        """Run the complete preprocessing pipeline"""
        import os
        print("Starting complete preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean structure
        df = self.clean_basic_structure()
        
        if df.empty:
            print("No data to process")
            return None
        
        # Apply all preprocessing steps
        df = self.standardize_datetime_fields(df)
        df = self.calculate_delays(df)
        df = self.clean_airport_codes(df)
        df = self.add_time_features(df)
        df = self.add_route_features(df)
        df = self.detect_peak_hours(df)
        
        # Select and reorder relevant columns
        columns_order = [
            'Flight_Number', 'Flight_Date', 'Weekday_Name', 'Is_Weekend',
            'Origin_City', 'Origin_Airport', 'Destination_City', 'Destination_Airport', 'Route',
            'STD_parsed', 'ATD_parsed', 'STA_parsed', 'ATA_parsed',
            'Scheduled_Departure_Hour', 'Scheduled_Arrival_Hour',
            'Departure_Delay_Minutes', 'Arrival_Delay_Minutes',
            'Departure_Delay_Category', 'Arrival_Delay_Category',
            'Is_Delayed_Departure', 'Is_Delayed_Arrival',
            'Is_Peak_Hour', 'Time_Period_Category',
            'Route_Avg_Dep_Delay', 'Route_Avg_Arr_Delay', 'Route_Popularity_Rank',
            'Aircraft', 'Flight time'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]
        
        self.processed_data = df
        
        # Save processed data
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_flight_data.csv")
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        # Generate summary report
        self.generate_summary_report(df)
        
        return df
    
    def generate_summary_report(self, df):
        """Generate a summary report of the processed data"""
        print("\n" + "="*50)
        print("FLIGHT DATA PREPROCESSING SUMMARY REPORT")
        print("="*50)
        
        print(f"\nTotal flights processed: {len(df)}")
        print(f"Date range: {df['Flight_Date'].min()} to {df['Flight_Date'].max()}")
        print(f"Unique routes: {df['Route'].nunique()}")
        print(f"Unique airlines: {df['Flight_Number'].str[:2].nunique()}")
        
        print(f"\nAirports covered:")
        airports = pd.concat([df['Origin_Airport'], df['Destination_Airport']]).unique()
        print(f"  {', '.join(sorted(airports))}")
        
        print(f"\nDelay Statistics:")
        print(f"  Average departure delay: {df['Departure_Delay_Minutes'].mean():.1f} minutes")
        print(f"  Average arrival delay: {df['Arrival_Delay_Minutes'].mean():.1f} minutes")
        print(f"  Flights with departure delays >15min: {df['Is_Delayed_Departure'].sum()} ({df['Is_Delayed_Departure'].mean()*100:.1f}%)")
        print(f"  Flights with arrival delays >15min: {df['Is_Delayed_Arrival'].sum()} ({df['Is_Delayed_Arrival'].mean()*100:.1f}%)")
        
        print(f"\nTop 5 busiest routes:")
        top_routes = df['Route'].value_counts().head()
        for route, count in top_routes.items():
            print(f"  {route}: {count} flights")
        
        print(f"\nPeak hours analysis:")
        peak_hours = df[df['Is_Peak_Hour']]['Scheduled_Departure_Hour'].unique()
        print(f"  Peak departure hours: {sorted(peak_hours)}")
        
        # Save detailed report
        report_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessing_report.txt")
        with open(report_file, 'w') as f:
            f.write("FLIGHT DATA PREPROCESSING DETAILED REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Processing timestamp: {datetime.now()}\n")
            f.write(f"Total flights: {len(df)}\n")
            f.write(f"Date range: {df['Flight_Date'].min()} to {df['Flight_Date'].max()}\n\n")
            
            f.write("DELAY ANALYSIS:\n")
            f.write(df[['Departure_Delay_Minutes', 'Arrival_Delay_Minutes']].describe().to_string())
            f.write("\n\n")
            
            f.write("ROUTE ANALYSIS:\n")
            f.write(df['Route'].value_counts().to_string())
            f.write("\n\n")
            
            f.write("HOURLY DISTRIBUTION:\n")
            f.write(df['Scheduled_Departure_Hour'].value_counts().sort_index().to_string())
            
        print(f"\nDetailed report saved to 'preprocessing_report.txt'")

# Usage example and main execution
if __name__ == "__main__":
    # Initialize preprocessor
    import os
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx")
    preprocessor = FlightDataPreprocessor(data_file)
    
    # Run complete preprocessing pipeline
    processed_df = preprocessor.process_all()
    
    if processed_df is not None:
        print(f"\nPreprocessing completed successfully!")
        print(f"Processed dataset shape: {processed_df.shape}")
        print(f"Output files: processed_flight_data.csv, preprocessing_report.txt")
    else:
        print("Preprocessing failed!")
