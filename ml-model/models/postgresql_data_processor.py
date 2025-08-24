"""
PostgreSQL-optimized Data Processor for Flight Schedule Optimization
Eliminates redundant calculations through efficient database operations
"""

import pandas as pd
import numpy as np
import json
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, text
from database.config import get_db, engine
from database.models import Flight, DataUpload, AnalyticsCache, RoutePerformance, PeakHourAnalysis

class PostgreSQLFlightDataProcessor:
    """
    High-performance flight data processor using PostgreSQL for optimized analytics
    """
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.csv']
        self.required_columns = [
            'Flight Number', 'From', 'To', 'STD', 'ATD', 'STA', 'ATA'
        ]
        self.processed_data = None
        self.eda_results = {}
        self.context_data = {}
        self.db = next(get_db())
    
    def validate_file_format(self, file_path: str) -> dict:
        """Validate uploaded file format and structure"""
        file_path = Path(file_path)
        
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {
                'name': file_path.name,
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'format': file_path.suffix
            }
        }
        
        # Check file extension
        if file_path.suffix not in self.supported_formats:
            validation_result['errors'].append(f"Unsupported file format: {file_path.suffix}")
            return validation_result
        
        # Check file exists
        if not file_path.exists():
            validation_result['errors'].append(f"File not found: {file_path}")
            return validation_result
        
        try:
            # Load file to check structure
            if file_path.suffix == '.xlsx':
                try:
                    # Try with openpyxl engine first
                    excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                    validation_result['file_info']['sheets'] = excel_file.sheet_names
                    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
                except Exception as e:
                    print(f"Failed with openpyxl: {e}")
                    # Fallback to xlrd
                    try:
                        excel_file = pd.ExcelFile(file_path, engine='xlrd')
                        validation_result['file_info']['sheets'] = excel_file.sheet_names
                        df = pd.read_excel(file_path, sheet_name=0, engine='xlrd')
                    except Exception as e2:
                        raise Exception(f"Cannot read Excel file with openpyxl or xlrd: {e2}")
            else:
                # Try different encodings for CSV; if tokenization fails, try Excel fallback
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                token_error = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except pd.errors.ParserError as pe:
                        token_error = pe
                        # try next encoding
                        continue
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    # As a fallback, attempt to read file as Excel in case client mislabelled the file
                    try:
                        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                        df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
                    except Exception:
                        try:
                            excel_file = pd.ExcelFile(file_path, engine='xlrd')
                            df = pd.read_excel(file_path, sheet_name=0, engine='xlrd')
                        except Exception:
                            if token_error:
                                raise Exception(f"Error tokenizing data: {token_error}")
                            raise Exception(f"Cannot read CSV file with encodings {encodings}")
            
            validation_result['file_info']['rows'] = len(df)
            validation_result['file_info']['columns'] = list(df.columns)
            
            # Check for required columns (flexible matching)
            missing_columns = []
            column_mapping = {}
            
            for req_col in self.required_columns:
                found = False
                for actual_col in df.columns:
                    if (req_col.lower().replace(' ', '') in actual_col.lower().replace(' ', '') or
                        actual_col.lower().replace(' ', '') in req_col.lower().replace(' ', '')):
                        column_mapping[req_col] = actual_col
                        found = True
                        break
                
                if not found:
                    missing_columns.append(req_col)
            
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            else:
                validation_result['valid'] = True
                validation_result['column_mapping'] = column_mapping
            
            # Data quality checks
            if validation_result['valid']:
                if len(df) == 0:
                    validation_result['errors'].append("File contains no data rows")
                    validation_result['valid'] = False
                
                null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
                if null_percentage > 0.8:
                    validation_result['warnings'].append(f"High percentage of missing data: {null_percentage*100:.1f}%")
        
        except Exception as e:
            validation_result['errors'].append(f"Error reading file: {str(e)}")
        
        return validation_result
    
    def process_uploaded_file(self, file_path: str, data_source: str = "upload") -> dict:
        """Process uploaded file with PostgreSQL optimizations"""
        print(f"🔄 Processing uploaded file: {file_path}")
        
        start_time = datetime.now()
        result = {
            'success': False,
            'validation': None,
            'eda': None,
            'processed_records': 0,
            'context_data': None,
            'error': None,
            'processing_time': 0
        }
        
        # Create upload record
        upload_record = DataUpload(
            filename=Path(file_path).name,
            file_format=Path(file_path).suffix,
            file_size_bytes=Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            processing_status='pending'
        )
        self.db.add(upload_record)
        self.db.commit()
        
        try:
            # Step 1: Validate file
            validation = self.validate_file_format(file_path)
            result['validation'] = validation
            
            if not validation['valid']:
                upload_record.processing_status = 'failed'
                upload_record.error_details = f"Validation failed: {validation['errors']}"
                self.db.commit()
                result['error'] = f"Validation failed: {validation['errors']}"
                return result
            
            # Step 2: Load and preprocess data
            df = self._load_and_standardize_file(file_path, validation.get('column_mapping', {}))
            processed_df = self._advanced_preprocessing(df)
            
            # Step 3: Perform EDA with database optimization
            eda_results = self._perform_optimized_eda(processed_df)
            result['eda'] = eda_results
            
            # Step 4: Bulk insert to database with conflict resolution
            records_stored = self._bulk_insert_flights(processed_df, data_source, upload_record.id)
            result['processed_records'] = records_stored
            
            # Step 5: Pre-compute analytics for performance
            self._precompute_analytics()
            
            # Step 6: Generate LLM context from database
            context_data = self._generate_database_context()
            result['context_data'] = context_data
            
            # Update upload record
            processing_time = (datetime.now() - start_time).total_seconds()
            upload_record.processing_status = 'success'
            upload_record.total_records = len(df)
            upload_record.valid_records = records_stored
            upload_record.processing_duration_seconds = processing_time
            upload_record.data_quality_score = float(eda_results.get('data_quality_score', 0))
            
            if not processed_df.empty and 'Flight_Date' in processed_df.columns:
                upload_record.date_range_start = processed_df['Flight_Date'].min()
                upload_record.date_range_end = processed_df['Flight_Date'].max()
            
            self.db.commit()
            
            result['success'] = True
            result['processing_time'] = processing_time
            self.processed_data = processed_df
            self.context_data = context_data
            
            print(f"✅ Successfully processed {records_stored} records in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            result['error'] = error_msg
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Rollback any pending transactions
            self.db.rollback()
            
            # Update upload record with error in a new transaction
            try:
                upload_record.processing_status = 'failed'
                upload_record.error_details = error_msg
                upload_record.processing_duration_seconds = result['processing_time']
                self.db.commit()
            except Exception as commit_error:
                print(f"⚠️  Could not update error record: {commit_error}")
                self.db.rollback()
            
            print(f"❌ {error_msg}")
        
        return result
    
    def _load_and_standardize_file(self, file_path: str, column_mapping: dict) -> pd.DataFrame:
        """Load file and standardize column names"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.xlsx':
            try:
                # Try with openpyxl engine first
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                if len(excel_file.sheet_names) > 1:
                    all_data = []
                    for sheet_name in excel_file.sheet_names:
                        sheet_df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                        # Check if this is the complex format and transform it
                        transformed_df = self._transform_complex_excel_format(sheet_df)
                        if transformed_df is not None:
                            transformed_df['Time_Period'] = sheet_name
                            all_data.append(transformed_df)
                        else:
                            sheet_df['Time_Period'] = sheet_name
                            all_data.append(sheet_df)
                    df = pd.concat(all_data, ignore_index=True)
                else:
                    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
                    # Check if this is the complex format and transform it
                    transformed_df = self._transform_complex_excel_format(df)
                    if transformed_df is not None:
                        df = transformed_df
            except Exception as e:
                print(f"Failed with openpyxl: {e}")
                # Fallback to xlrd
                try:
                    excel_file = pd.ExcelFile(file_path, engine='xlrd')
                    if len(excel_file.sheet_names) > 1:
                        all_data = []
                        for sheet_name in excel_file.sheet_names:
                            sheet_df = pd.read_excel(file_path, sheet_name=sheet_name, engine='xlrd')
                            transformed_df = self._transform_complex_excel_format(sheet_df)
                            if transformed_df is not None:
                                transformed_df['Time_Period'] = sheet_name
                                all_data.append(transformed_df)
                            else:
                                sheet_df['Time_Period'] = sheet_name
                                all_data.append(sheet_df)
                        df = pd.concat(all_data, ignore_index=True)
                    else:
                        df = pd.read_excel(file_path, sheet_name=0, engine='xlrd')
                        transformed_df = self._transform_complex_excel_format(df)
                        if transformed_df is not None:
                            df = transformed_df
                except Exception as e2:
                    raise Exception(f"Cannot read Excel file with openpyxl or xlrd: {e2}")
        else:
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise Exception(f"Cannot read CSV file with any encoding: {encodings}")
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Ensure standard column names for processing
        standard_mapping = {
            'From': 'Origin_Airport',
            'To': 'Destination_Airport', 
            'Flight Number': 'Flight_Number',
            'Aircraft': 'Aircraft_Type'
        }
        
        for old_col, new_col in standard_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # --- NEW: propagate flight numbers in grouped format (header row + multiple date rows) ---
        # Detect grouped format: many missing Flight_Number but a few non-null at top of groups
        if 'Flight_Number' in df.columns:
            non_null_ratio = df['Flight_Number'].notna().mean()
            if 0 < non_null_ratio < 0.5:  # likely grouped format
                # Forward fill flight numbers downward until next header
                df['Flight_Number'] = df['Flight_Number'].ffill()
        elif 'Flight Number' in df.columns:  # if rename didn't happen due to mapping
            tmp_ratio = df['Flight Number'].notna().mean()
            if 0 < tmp_ratio < 0.5:
                df['Flight Number'] = df['Flight Number'].ffill()
                df = df.rename(columns={'Flight Number': 'Flight_Number'})

        # Filter out header/group separator rows: rows without a date and without key timing fields
        potential_date_cols = [c for c in df.columns if c.lower() in ('date', 'flight_date') or 'date' in c.lower()]
        date_col = potential_date_cols[0] if potential_date_cols else None
        if date_col:
            # Keep rows that have a date OR (STD/ATD present); then ensure flight number not null
            key_time_cols = [c for c in ['STD','ATD','STA','ATA'] if c in df.columns]
            if key_time_cols:
                mask_valid = df[date_col].notna() & df['Flight_Number'].notna()
                df = df[mask_valid].copy()

        # Trim stray whitespace in airport/city fields if present
        for col in ['Origin_Airport','Destination_Airport']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('\xa0',' ').str.strip()

        # Construct route if possible
        if 'Origin_Airport' in df.columns and 'Destination_Airport' in df.columns:
            if 'Route' not in df.columns:
                df['Route'] = df['Origin_Airport'].fillna('') + ' -> ' + df['Destination_Airport'].fillna('')
        
        return df
    
    def _transform_complex_excel_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform complex Excel format (flight number in header, data in rows) to standard format"""
        try:
            # Check if this is the complex format
            if len(df) < 2:
                return None
            
            # Attempt to detect multiple flight header sections (numbers in first column + hyperlink flight numbers)
            # Strategy: iterate rows; when a row has S.No-like value (int/float) AND a cell containing alnum flight code pattern, set current flight
            flight_sections = []
            current_flight = None
            date_col_idx = None
            # Identify a plausible date column by scanning for datetime objects
            for col_idx in range(min(8, df.shape[1])):  # scan first few columns
                sample_vals = df.iloc[1:15, col_idx]
                if sample_vals.apply(lambda v: hasattr(v, 'year')).sum() >= 3:
                    date_col_idx = col_idx
                    break
            if date_col_idx is None:
                return None

            # Iterate rows
            for i in range(df.shape[0]):
                row = df.iloc[i]
                # Header detection
                possible_flight = None
                for val in row.values:
                    if isinstance(val, str) and len(val) <= 10 and any(c.isalpha() for c in val) and any(c.isdigit() for c in val):
                        # Filter out generic words
                        if re.match(r'^[A-Z0-9]+$', val.replace('-', '')):
                            possible_flight = val.strip()
                            break
                if possible_flight and (pd.isna(row.iloc[date_col_idx]) or not hasattr(row.iloc[date_col_idx], 'year')):
                    current_flight = possible_flight
                    continue

                # Data row with date and current flight
                if current_flight and pd.notna(row.iloc[date_col_idx]) and hasattr(row.iloc[date_col_idx], 'year'):
                    try:
                        flight_sections.append({
                            'Flight Number': current_flight,
                            'Flight_Date': row.iloc[date_col_idx],
                            'From': str(row.iloc[3]).strip() if df.shape[1] > 3 and pd.notna(row.iloc[3]) else '',
                            'To': str(row.iloc[4]).strip() if df.shape[1] > 4 and pd.notna(row.iloc[4]) else '',
                            'Aircraft': str(row.iloc[5]).strip() if df.shape[1] > 5 and pd.notna(row.iloc[5]) else '',
                            'Flight time': str(row.iloc[6]) if df.shape[1] > 6 and pd.notna(row.iloc[6]) else '',
                            'STD': row.iloc[7] if df.shape[1] > 7 and pd.notna(row.iloc[7]) else None,
                            'ATD': row.iloc[8] if df.shape[1] > 8 and pd.notna(row.iloc[8]) else None,
                            'STA': row.iloc[9] if df.shape[1] > 9 and pd.notna(row.iloc[9]) else None,
                            'ATA': str(row.iloc[11]).strip() if df.shape[1] > 11 and pd.notna(row.iloc[11]) else ''
                        })
                    except Exception:
                        continue

            if not flight_sections:
                return None

            print(f"🔄 Transforming complex Excel format detected {len(set([r['Flight Number'] for r in flight_sections]))} flights")
            new_df = pd.DataFrame(flight_sections)
            
            # Clean up airport names (remove extra characters)
            for col in ['From', 'To', 'Aircraft']:
                if col in new_df.columns:
                    new_df[col] = new_df[col].str.replace('\xa0', ' ', regex=False).str.strip()
            
            # Add route column
            new_df['Route'] = new_df['From'] + ' -> ' + new_df['To']
            
            print(f"✅ Transformed {len(new_df)} flight records")
            return new_df
            
        except Exception as e:
            print(f"⚠️  Excel transformation error: {e}")
            return None
    
    def _advanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced preprocessing optimized for PostgreSQL storage"""
        try:
            # Use existing preprocessing pipeline but optimize for database
            from scripts.preprocessing_pipeline import FlightDataPreprocessor
            
            # Create temporary file with proper encoding
            temp_file = '/tmp/temp_processing.csv'
            df.to_csv(temp_file, index=False, encoding='utf-8')
            
            processor = FlightDataPreprocessor(temp_file)
            processor.load_data()
            processed_df = processor.clean_basic_structure()
            
            if not processed_df.empty:
                processed_df = processor.standardize_datetime_fields(processed_df)
                processed_df = processor.calculate_delays(processed_df)
                processed_df = processor.clean_airport_codes(processed_df)
                processed_df = processor.add_time_features(processed_df)
                processed_df = processor.add_route_features(processed_df)
                processed_df = processor.detect_peak_hours(processed_df)
            
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
            
            return processed_df
            
        except Exception as e:
            print(f"⚠️  Preprocessing error: {e}, returning original data")
            return df
    
    def _perform_optimized_eda(self, df: pd.DataFrame) -> dict:
        """Perform EDA with database-optimized calculations"""
        print("🔍 Performing optimized EDA...")
        
        eda_results = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'date_range': None
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.astype(str).to_dict()
            },
            'flight_statistics': {},
            'route_analysis': {},
            'recommendations': []
        }
        
        try:
            # Calculate data quality score
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            duplicate_penalty = df.duplicated().sum() / len(df) * 20 if len(df) > 0 else 0
            missing_penalty = (missing_cells / total_cells) * 30 if total_cells > 0 else 0
            
            eda_results['data_quality_score'] = float(max(0, min(100, 100 - missing_penalty - duplicate_penalty)))
            
            # Basic statistics
            if 'Flight_Number' in df.columns:
                eda_results['flight_statistics']['unique_flights'] = df['Flight_Number'].nunique()
            
            if 'Route' in df.columns:
                route_counts = df['Route'].value_counts()
                eda_results['route_analysis']['unique_routes'] = df['Route'].nunique()
                eda_results['route_analysis']['busiest_routes'] = route_counts.head(5).to_dict()
            
            # Date range analysis
            if 'Flight_Date' in df.columns:
                df['Flight_Date'] = pd.to_datetime(df['Flight_Date'], errors='coerce')
                eda_results['basic_info']['date_range'] = {
                    'start': df['Flight_Date'].min().isoformat() if df['Flight_Date'].min() is not pd.NaT else None,
                    'end': df['Flight_Date'].max().isoformat() if df['Flight_Date'].max() is not pd.NaT else None,
                    'unique_dates': df['Flight_Date'].nunique()
                }
            
        except Exception as e:
            eda_results['error'] = f"EDA error: {str(e)}"
        
        return eda_results
    
    def _bulk_insert_flights(self, df: pd.DataFrame, data_source: str, upload_id: int) -> int:
        """Efficiently bulk insert flight data with conflict resolution"""
        if df.empty:
            return 0
        
        print("💾 Bulk inserting flight data...")
        
        # Prepare data for database
        flight_records = []
        
        for _, row in df.iterrows():
            try:
                # Create flight record
                # Get flight date for combining with times
                flight_date = pd.to_datetime(row.get('Flight_Date')) if pd.notna(row.get('Flight_Date')) else None
                
                # Helper function to combine date and time
                def combine_date_time(date_val, time_val):
                    if date_val is None or pd.isna(time_val):
                        return None
                    try:
                        if hasattr(time_val, 'hour'):  # It's a datetime.time object
                            return pd.Timestamp.combine(date_val.date(), time_val)
                        else:
                            # Try to parse as string or other format
                            parsed_time = pd.to_datetime(str(time_val), format='%H:%M:%S', errors='coerce')
                            if parsed_time is not None:
                                return pd.Timestamp.combine(date_val.date(), parsed_time.time())
                    except:
                        pass
                    return None
                
                # Helper function to parse ATA strings like "Landed 8:14 AM"
                def parse_ata_time(date_val, ata_string):
                    if date_val is None or pd.isna(ata_string):
                        return None
                    try:
                        import re
                        # Extract time from strings like "Landed 8:14 AM"
                        time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)'
                        match = re.search(time_pattern, str(ata_string), re.IGNORECASE)
                        
                        if match:
                            hours = int(match.group(1))
                            minutes = int(match.group(2))
                            am_pm = match.group(3).upper()
                            
                            # Convert to 24-hour format
                            if am_pm == 'PM' and hours != 12:
                                hours += 12
                            elif am_pm == 'AM' and hours == 12:
                                hours = 0
                            
                            # Create time object and combine with date
                            time_obj = pd.Timestamp(f"{hours:02d}:{minutes:02d}:00").time()
                            return pd.Timestamp.combine(date_val.date(), time_obj)
                    except:
                        pass
                    return None
                
                # Helper function to calculate delay in minutes
                def calculate_delay_minutes(scheduled_time, actual_time):
                    if scheduled_time is None or actual_time is None:
                        return None
                    try:
                        delay_delta = actual_time - scheduled_time
                        return int(delay_delta.total_seconds() / 60)
                    except:
                        return None
                
                # Calculate actual times
                scheduled_departure_time = combine_date_time(flight_date, row.get('STD'))
                scheduled_arrival_time = combine_date_time(flight_date, row.get('STA'))
                actual_departure_time = combine_date_time(flight_date, row.get('ATD'))
                actual_arrival_time = parse_ata_time(flight_date, row.get('ATA'))
                
                # Calculate delays
                departure_delay_minutes = calculate_delay_minutes(scheduled_departure_time, actual_departure_time)
                arrival_delay_minutes = calculate_delay_minutes(scheduled_arrival_time, actual_arrival_time)
                
                # Use up to 255 chars for descriptive airport/route names (avoid prior hard truncation)
                flight_data = {
                    'flight_number': str(row.get('Flight_Number', '')),
                    'flight_date': flight_date,
                    'origin_airport': (str(row.get('Origin_Airport', '')) if pd.notna(row.get('Origin_Airport')) else '')[:255],
                    'destination_airport': (str(row.get('Destination_Airport', '')) if pd.notna(row.get('Destination_Airport')) else '')[:255],
                    'route': (str(row.get('Route', '')) if pd.notna(row.get('Route')) else '')[:255],
                    'scheduled_departure_time': scheduled_departure_time,
                    'scheduled_arrival_time': scheduled_arrival_time,
                    'actual_departure_time': actual_departure_time,
                    'actual_arrival_time': actual_arrival_time,
                    'scheduled_departure_hour': int(row.get('STD').hour) if pd.notna(row.get('STD')) and hasattr(row.get('STD'), 'hour') else 0,
                    'departure_delay_minutes': departure_delay_minutes,
                    'arrival_delay_minutes': arrival_delay_minutes,
                    'is_delayed_departure': departure_delay_minutes is not None and departure_delay_minutes > 15,
                    'is_delayed_arrival': arrival_delay_minutes is not None and arrival_delay_minutes > 15,
                    'aircraft': (str(row.get('Aircraft_Type', ''))[:255] if pd.notna(row.get('Aircraft_Type')) else None),
                    'status': 'landed' if actual_arrival_time else 'scheduled',
                    # Ensure data_source fits DB column (String(50)) to avoid truncation errors
                    # Keep data_source truncation conservative to match current DB column length (50)
                    'data_source': (data_source[:50] if isinstance(data_source, str) else str(data_source))
                }
                
                # Only add if required fields are present
                if flight_data['flight_number'] and flight_data['flight_date'] and flight_data['scheduled_departure_time']:
                    flight_records.append(flight_data)
                    
            except Exception as e:
                print(f"⚠️  Error processing row: {e}")
                continue
        
        if not flight_records:
            return 0
        
        try:
            # Use bulk insert with ON CONFLICT handling
            insert_stmt = text("""
                INSERT INTO flights (
                    flight_number, flight_date, origin_airport, destination_airport, route,
                    scheduled_departure_time, scheduled_arrival_time, actual_departure_time, actual_arrival_time,
                    scheduled_departure_hour, departure_delay_minutes, arrival_delay_minutes,
                    is_delayed_departure, is_delayed_arrival, aircraft, status, data_source
                ) VALUES (
                    :flight_number, :flight_date, :origin_airport, :destination_airport, :route,
                    :scheduled_departure_time, :scheduled_arrival_time, :actual_departure_time, :actual_arrival_time,
                    :scheduled_departure_hour, :departure_delay_minutes, :arrival_delay_minutes,
                    :is_delayed_departure, :is_delayed_arrival, :aircraft, :status, :data_source
                )
                ON CONFLICT (flight_number, flight_date, scheduled_departure_time) 
                DO UPDATE SET
                    actual_departure_time = EXCLUDED.actual_departure_time,
                    actual_arrival_time = EXCLUDED.actual_arrival_time,
                    departure_delay_minutes = EXCLUDED.departure_delay_minutes,
                    arrival_delay_minutes = EXCLUDED.arrival_delay_minutes,
                    is_delayed_departure = EXCLUDED.is_delayed_departure,
                    is_delayed_arrival = EXCLUDED.is_delayed_arrival,
                    updated_at = CURRENT_TIMESTAMP
            """)
            
            self.db.execute(insert_stmt, flight_records)
            self.db.commit()
            
            print(f"✅ Bulk inserted {len(flight_records)} flight records")
            return len(flight_records)
            
        except Exception as e:
            print(f"❌ Bulk insert error: {e}")
            self.db.rollback()
            return 0
    
    def _precompute_analytics(self):
        """Pre-compute analytics tables for faster queries"""
        print("⚡ Pre-computing analytics...")
        
        try:
            # Pre-compute route performance
            self._compute_route_performance()
            
            # Pre-compute peak hour analysis
            self._compute_peak_hour_analysis()
            
            # Invalidate existing cache
            self._invalidate_cache()
            
        except Exception as e:
            print(f"⚠️  Analytics pre-computation error: {e}")
    
    def _compute_route_performance(self):
        """Pre-compute route performance metrics"""
        route_performance_query = text("""
            INSERT INTO route_performance (
                route, total_flights, avg_departure_delay, avg_arrival_delay, 
                on_time_percentage, severe_delay_count, peak_hour, peak_hour_flights,
                data_start_date, data_end_date, calculation_date
            )
            SELECT 
                f.route,
                COUNT(*) as total_flights,
                AVG(f.departure_delay_minutes) as avg_departure_delay,
                AVG(f.arrival_delay_minutes) as avg_arrival_delay,
                (COUNT(*) FILTER (WHERE f.departure_delay_minutes <= 15) * 100.0 / COUNT(*)) as on_time_percentage,
                COUNT(*) FILTER (WHERE f.departure_delay_minutes > 60) as severe_delay_count,
                MODE() WITHIN GROUP (ORDER BY f.scheduled_departure_hour) as peak_hour,
                MAX(hourly_counts.flight_count) as peak_hour_flights,
                MIN(f.flight_date) as data_start_date,
                MAX(f.flight_date) as data_end_date,
                CURRENT_TIMESTAMP as calculation_date
            FROM flights f
            LEFT JOIN (
                SELECT route, scheduled_departure_hour, COUNT(*) as flight_count
                FROM flights 
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
                GROUP BY route, scheduled_departure_hour
            ) hourly_counts ON f.route = hourly_counts.route
            WHERE f.created_at >= CURRENT_DATE - INTERVAL '1 day'
            GROUP BY f.route
            ON CONFLICT (route, calculation_date) DO UPDATE SET
                total_flights = EXCLUDED.total_flights,
                avg_departure_delay = EXCLUDED.avg_departure_delay,
                avg_arrival_delay = EXCLUDED.avg_arrival_delay,
                on_time_percentage = EXCLUDED.on_time_percentage,
                severe_delay_count = EXCLUDED.severe_delay_count
        """)
        
        self.db.execute(route_performance_query)
        self.db.commit()
    
    def _compute_peak_hour_analysis(self):
        """Pre-compute peak hour analysis"""
        peak_hour_query = text("""
            INSERT INTO peak_hour_analysis (
                airport, hour, total_flights, departure_flights, arrival_flights,
                avg_delay_minutes, congestion_score, is_peak_hour, analysis_date
            )
            SELECT 
                COALESCE(origin_airport, destination_airport) as airport,
                scheduled_departure_hour as hour,
                COUNT(*) as total_flights,
                COUNT(*) FILTER (WHERE origin_airport IS NOT NULL) as departure_flights,
                COUNT(*) FILTER (WHERE destination_airport IS NOT NULL) as arrival_flights,
                AVG(COALESCE(departure_delay_minutes, arrival_delay_minutes, 0)) as avg_delay_minutes,
                CASE 
                    WHEN COUNT(*) > 0 THEN 
                        AVG(COALESCE(departure_delay_minutes, arrival_delay_minutes, 0)) * LOG(COUNT(*) + 1)
                    ELSE 0 
                END as congestion_score,
                COUNT(*) > 10 as is_peak_hour,
                CURRENT_TIMESTAMP as analysis_date
            FROM flights 
            WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
            GROUP BY COALESCE(origin_airport, destination_airport), scheduled_departure_hour
            ON CONFLICT (airport, hour, analysis_date) DO UPDATE SET
                total_flights = EXCLUDED.total_flights,
                avg_delay_minutes = EXCLUDED.avg_delay_minutes,
                congestion_score = EXCLUDED.congestion_score,
                is_peak_hour = EXCLUDED.is_peak_hour
        """)
        
        self.db.execute(peak_hour_query)
        self.db.commit()
    
    def _invalidate_cache(self):
        """Invalidate analytics cache when new data is added"""
        try:
            self.db.query(AnalyticsCache).filter(
                AnalyticsCache.expires_at < datetime.now()
            ).delete()
            self.db.commit()
        except Exception as e:
            print(f"⚠️  Cache invalidation error: {e}")
    
    def _generate_database_context(self) -> dict:
        """Generate LLM context from database analytics"""
        context = {
            'dataset_summary': {},
            'operational_insights': {},
            'performance_metrics': {}
        }
        
        try:
            # Basic statistics from database
            total_flights = self.db.query(func.count(Flight.id)).scalar()
            unique_routes = self.db.query(func.count(func.distinct(Flight.route))).scalar()
            
            context['dataset_summary'] = {
                'total_flights': total_flights,
                'unique_routes': unique_routes,
                'last_updated': datetime.now().isoformat()
            }
            
            # Peak hours from pre-computed data
            peak_hours = self.db.query(PeakHourAnalysis).filter(
                PeakHourAnalysis.is_peak_hour == True
            ).order_by(PeakHourAnalysis.congestion_score.desc()).limit(5).all()
            
            context['operational_insights']['peak_hours'] = [
                {
                    'hour': ph.hour,
                    'airport': ph.airport,
                    'congestion_score': ph.congestion_score,
                    'total_flights': ph.total_flights
                } for ph in peak_hours
            ]
            
            # Route performance from pre-computed data
            top_routes = self.db.query(RoutePerformance).order_by(
                RoutePerformance.total_flights.desc()
            ).limit(5).all()
            
            context['performance_metrics']['busiest_routes'] = [
                {
                    'route': rp.route,
                    'total_flights': rp.total_flights,
                    'on_time_percentage': rp.on_time_percentage,
                    'avg_delay': rp.avg_departure_delay
                } for rp in top_routes
            ]
            
        except Exception as e:
            context['error'] = f"Context generation error: {str(e)}"
        
        return context
    
    def get_processing_history(self) -> List[dict]:
        """Get processing history from database"""
        uploads = self.db.query(DataUpload).order_by(
            DataUpload.upload_timestamp.desc()
        ).limit(20).all()
        
        return [
            {
                'filename': upload.filename,
                'upload_timestamp': upload.upload_timestamp.isoformat(),
                'processing_status': upload.processing_status,
                'total_records': upload.total_records,
                'valid_records': upload.valid_records,
                'data_quality_score': upload.data_quality_score,
                'processing_duration': upload.processing_duration_seconds
            } for upload in uploads
        ]
    
    def get_current_context(self) -> dict:
        """Get current dataset context from database"""
        return self._generate_database_context()
    
    def get_latest_data(self, limit: int = 1000) -> pd.DataFrame:
        """Get latest flight data from database"""
        query = self.db.query(Flight).order_by(Flight.created_at.desc()).limit(limit)
        
        # Convert to DataFrame
        flights = query.all()
        if not flights:
            return pd.DataFrame()
        
        data = []
        for flight in flights:
            data.append({
                'flight_number': flight.flight_number,
                'flight_date': flight.flight_date,
                'origin_airport': flight.origin_airport,
                'destination_airport': flight.destination_airport,
                'route': flight.route,
                'departure_delay_minutes': flight.departure_delay_minutes,
                'arrival_delay_minutes': flight.arrival_delay_minutes,
                'scheduled_departure_time': flight.scheduled_departure_time,
                'scheduled_arrival_time': flight.scheduled_arrival_time,
                'actual_departure_time': flight.actual_departure_time,
                'actual_arrival_time': flight.actual_arrival_time,
                'scheduled_departure_hour': flight.scheduled_departure_hour,
                'is_delayed_departure': flight.is_delayed_departure,
                'is_delayed_arrival': flight.is_delayed_arrival,
                'aircraft': flight.aircraft,
                'created_at': flight.created_at
            })
        
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    print("🐘 Testing PostgreSQL data processor...")
    
    try:
        from database.config import create_database_if_not_exists, test_connection
        from database.models import create_all_tables
        
        # Test database setup
        create_database_if_not_exists()
        test_connection()
        create_all_tables()
        
        # Test processor
        processor = PostgreSQLFlightDataProcessor()
        print("✅ PostgreSQL data processor initialized successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure PostgreSQL is running and configured correctly")
