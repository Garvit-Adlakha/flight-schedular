# Flight Schedule Optimization ML Model

## Project Structure

```
ml-model/
├── data/                           # Raw and processed data files
│   ├── 429e6e3f-...-Flight_Data.xlsx  # Original flight data
│   ├── cleaned_flight_data.csv     # Initial cleaned data
│   └── processed_flight_data.csv   # Fully processed dataset
├── scripts/                        # Main processing scripts
│   ├── preprocessing_pipeline.py   # Data cleaning and feature engineering
│   └── flight_analytics.py        # Comprehensive analytics and insights
├── analysis/                       # Analysis outputs and reports
│   ├── flight_analysis_results.xlsx   # Detailed analysis results
│   ├── optimization_insights.txt      # Key insights and recommendations
│   └── preprocessing_report.txt       # Data preprocessing summary
├── models/                         # ML models and optimization algorithms
│   └── (to be created)
├── outputs/                        # Generated visualizations and reports
│   └── flight_analytics_dashboard.png # Analytics dashboard
└── README.md                       # This file
```

## Problem Statement

Flight operations at busy airports (Mumbai, Delhi) face scheduling nightmares due to:
- Capacity limitations and heavy passenger load
- Limited runway capacity for takeoffs and landings
- Schedule disruptions causing cascading effects
- Weather-related disruptions reducing runway capacity

## Dataset

- **Source**: Flight radar data for Mumbai Airport
- **Duration**: 1 week of flight schedules and actual data
- **Coverage**: 785 flights across 59 routes and 22 airlines
- **Time Period**: July 19-25, 2025
- **Airports**: 59 different airports (focus on Mumbai BOM departures)

## Key Findings

### Peak Hours Analysis
- **Peak Hours**: 6 AM and 10 AM (highest congestion scores)
- **Peak Hour Traffic**: 303 flights (38.6% of total traffic)
- **Average Delay**: 32.1 minutes during peak hours

### Route Performance
- **Most Problematic Route**: BOM-IDR (68.8 min average delay)
- **Busiest Route**: BOM-DEL (90 flights)
- **Most Reliable**: BOM-GOI (0% delay rate)

### Delay Statistics
- **Average Departure Delay**: 32.1 minutes
- **Average Arrival Delay**: 3.8 minutes
- **Flights with >15min delays**: 45.7% departure, 15.0% arrival

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing (if needed)
```bash
cd scripts/
python preprocessing_pipeline.py
```

### 3. Start API Server
```bash
cd api/
python start_api.py
```

### 4. Access API
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### 5. Test API Endpoints
```bash
cd api/
python test_api.py
```

## Workflow Implementation Status

✅ **Data Acquisition**: Dynamic file upload (Excel/CSV) with database storage
✅ **Preprocessing**: Automated EDA + ML-ready feature generation  
✅ **Analytics & Heuristics**: Peak hours, cascading delays, optimization
✅ **NLP Layer**: Pattern-based + Ollama LLM with custom context prompting
✅ **Backend API**: Dynamic data processing with validation & model retraining
⏳ **Frontend/Dashboard**: Planned for future implementation

## API Endpoints

### 🔄 Dynamic Data Processing
- `POST /api/upload-data` - **Upload new Excel/CSV files with auto-processing**
- `POST /api/validate-data` - **Validate file structure before upload**
- `GET /api/processing-history` - **Track all data uploads and processing**
- `GET /api/current-context` - **Get current dataset context for LLM**

### 📊 Core Analytics
- `GET /api/peak-hours` - Peak hour summaries
- `GET /api/delay-cascades` - Flight delay cascades analysis
- `POST /api/optimize-route` - Route optimization suggestions
- `POST /api/schedule-impact` - Schedule change impact analysis
- `POST /api/predict-delay` - Delay prediction for flights

### 🤖 NLP & LLM Interface
- `POST /api/nlp-query` - Pattern-based natural language processing
- `POST /api/ollama-query` - **Ollama LLM with custom context prompting**
- `POST /api/ollama-schedule-analysis` - **Advanced schedule change analysis with LLM**
- `POST /api/ollama-optimal-time` - **Optimal time finding using LLM**

### 📈 Data & Performance
- `GET /api/route-performance` - Route performance analytics
- `GET /api/statistics` - Comprehensive flight statistics

## 🦙 Ollama LLM Integration

This system now includes **full Ollama LLM integration** with custom context prompting as specified in the workflow:

### Setup Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service  
ollama serve

# Pull model
ollama pull llama3.2
```

### Example LLM Queries
```bash
# Advanced schedule analysis
curl -X POST http://localhost:8000/api/ollama-schedule-analysis \
  -d '{"flight_id": "AI2509", "current_hour": 6, "new_hour": 8, "route": "BOM-DEL"}'

# Optimal time finding with LLM
curl -X POST http://localhost:8000/api/ollama-optimal-time \
  -d '{"route": "BOM-DEL", "date": "2025-07-26"}'
```

**See `OLLAMA_SETUP.md` for complete setup guide.**

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- openpyxl
- datetime

## Key Optimization Opportunities

1. Redistribute 163 flights from 6 AM peak hour to off-peak hours
2. Focus delay reduction on high-volume routes (BOM-DEL, BOM-BLR)
3. Implement buffer times for high-risk aircraft
4. Schedule maintenance during low-traffic periods (5 AM, 12-1 PM)
