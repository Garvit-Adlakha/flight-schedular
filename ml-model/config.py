"""
Configuration settings for ML model training and API
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Data file paths
RAW_DATA_FILE = DATA_DIR / "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
PROCESSED_DATA_FILE = DATA_DIR / "processed_flight_data.csv"

# Model file paths
DELAY_REGRESSOR_MODEL = MODELS_DIR / "delay_regressor.pkl"
DELAY_CLASSIFIER_MODEL = MODELS_DIR / "delay_classifier.pkl"
FEATURE_SCALER_MODEL = MODELS_DIR / "feature_scaler.pkl"
LABEL_ENCODERS_MODEL = MODELS_DIR / "label_encoders.pkl"
FEATURE_COLUMNS_MODEL = MODELS_DIR / "feature_columns.pkl"

# Database configuration (import from database config)
try:
    from database.config import DATABASE_CONFIG, get_database_url
except ImportError:
    # Fallback configuration if database config is not available
    DATABASE_CONFIG = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'flight_schedule_optimization'),
        'username': os.getenv('POSTGRES_USER', 'flight_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'flight_password')
    }
    
    def get_database_url():
        return (
            f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
            f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
        )

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Feature engineering settings
PEAK_HOURS = [8, 9, 10, 17, 18, 19]  # Based on typical airport traffic
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday (0=Monday)

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10

print(f"✅ Configuration loaded successfully")
print(f"   📁 Base directory: {BASE_DIR}")
print(f"   📊 Raw data file: {RAW_DATA_FILE}")
print(f"   🔄 Processed data file: {PROCESSED_DATA_FILE}")
print(f"   🤖 Models directory: {MODELS_DIR}")
