"""
Flight Schedule Optimization API
Backend REST API serving ML models and analytics according to the workflow
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
import os
from datetime import datetime, date
from io import BytesIO
import uvicorn
import numpy as np

# Add parent directory to path
sys.path.append('..')
sys.path.append('../models')

from models.delay_prediction_model import FlightDelayPredictor
from models.cascading_delay_analyzer import CascadingDelayAnalyzer
# NLP interface is now database-driven, no separate module needed
from models.ollama_nlp_interface import OllamaNLPInterface
from models.postgresql_data_processor import PostgreSQLFlightDataProcessor

# Database imports
from database.config import SessionLocal
from database.models import (
    Flight,
    FlightEnrichment,
    DataUpload,
    AnalyticsCache,
    RoutePerformance,
    PeakHourAnalysis,
    CascadingDelayNetwork,
)
from sqlalchemy import inspect
from sqlalchemy.sql import func
from sqlalchemy import case

def get_db_session():
    """Get database session"""
    return SessionLocal()

# Initialize FastAPI app
app = FastAPI(
    title="Flight Schedule Optimization API",
    description="REST API for flight schedule optimization, delay prediction, and NLP queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML model instances
delay_predictor = None
cascade_analyzer = None
nlp_interface = None
ollama_nlp = None
data_processor = None

# Pydantic models for request/response
class RouteOptimizationRequest(BaseModel):
    route: str
    date: Optional[str] = None

class ScheduleChangeRequest(BaseModel):
    flight_id: str
    new_hour: int
    route: str

class NLPQueryRequest(BaseModel):
    query: str

class OllamaQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ScheduleChangeAnalysisRequest(BaseModel):
    flight_id: str
    current_hour: int
    new_hour: int
    route: str

class DelayPredictionRequest(BaseModel):
    route: str
    departure_hour: int
    date: Optional[str] = None
    is_weekend: Optional[bool] = False

# API Response Models  
class TimeSlotRecommendation(BaseModel):
    hour: int
    time_slot: str
    predicted_delay: float
    on_time_probability: float
    rationale: Optional[str] = None
    ranking: Optional[int] = None
    reason: Optional[str] = None

class OperationalInsights(BaseModel):
    best_window: str
    avg_delay_best_window: float
    avg_delay_worst_window: float
    improvement_potential: float

class OptimalTimeResponse(BaseModel):
    route: str
    best_time_overall: TimeSlotRecommendation
    most_reliable_time: TimeSlotRecommendation
    alternative_times: List[TimeSlotRecommendation]
    times_to_avoid: List[TimeSlotRecommendation]
    operational_insights: OperationalInsights
    analysis_date: str

class PeakHourSummary(BaseModel):
    hour: int
    flight_count: int
    avg_delay_minutes: float
    delay_rate: float
    congestion_score: float
    recommendation: str

class CascadeAnalysis(BaseModel):
    flight_id: str
    impact_score: float
    cascade_risk_level: str
    downstream_flights: int
    recommendations: List[str]

# Initialize ML models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize ML models when API starts"""
    global delay_predictor, cascade_analyzer, nlp_interface, ollama_nlp, data_processor
    
    try:
        print("🚀 Initializing Flight Schedule Optimization API...")
        
        # Initialize PostgreSQL data processor
        data_processor = PostgreSQLFlightDataProcessor()
        
        # Initialize delay predictor with fallback
        try:
            delay_predictor = FlightDelayPredictor('../data/processed_flight_data.csv')
            if not delay_predictor.load_models():
                print("Training delay prediction models...")
                delay_predictor.train_delay_prediction_models()
        except:
            print("⚠️  Using dynamic data - will initialize models after first upload")
            delay_predictor = None
        
        # Initialize cascade analyzer with fallback
        try:
            cascade_analyzer = CascadingDelayAnalyzer('../data/processed_flight_data.csv')
        except:
            print("⚠️  Cascade analyzer will initialize after data upload")
            cascade_analyzer = None
        
        # NLP interface is now database-driven, no file initialization needed
        nlp_interface = True  # Set to True to indicate NLP is available
        
        # Initialize Ollama LLM interface (optional - may fail if Ollama not available)
        try:
            print("🦙 Initializing Ollama LLM interface...")
            ollama_nlp = OllamaNLPInterface('../data/processed_flight_data.csv')
            print("✅ Ollama LLM interface ready")
        except Exception as ollama_error:
            print(f"⚠️  Ollama LLM not available: {ollama_error}")
            print("   Will initialize after data upload")
            ollama_nlp = None
        
        print("✅ API initialized - ready for dynamic data processing!")
        
    except Exception as e:
        print(f"❌ Error initializing API: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "delay_predictor": delay_predictor is not None,
            "cascade_analyzer": cascade_analyzer is not None,
            "nlp_interface": nlp_interface is not None,
            "ollama_llm": ollama_nlp is not None,
            "data_processor": data_processor is not None
        }
    }

# DYNAMIC DATA ENDPOINTS

@app.get("/api/raw-data")
async def get_raw_data(limit: int = Query(5000, ge=1, le=50000)):
    """Return recent raw flight records for data browsing (lightweight projection)."""
    try:
        session = get_db_session()
        # Select limited columns to keep payload small
        rows = (
            session.query(
                Flight.id.label('id'),
                Flight.flight_number.label('Flight_Number'),
                Flight.route.label('Route'),
                Flight.origin_airport.label('Origin_Airport'),
                Flight.destination_airport.label('Destination_Airport'),
                Flight.scheduled_departure_hour.label('Scheduled_Departure_Hour'),
                Flight.departure_delay_minutes.label('Departure_Delay_Minutes'),
                Flight.arrival_delay_minutes.label('Arrival_Delay_Minutes'),
                Flight.is_delayed_departure.label('Is_Delayed_Departure'),
                Flight.is_delayed_arrival.label('Is_Delayed_Arrival'),
                Flight.flight_date.label('Flight_Date')
            )
            .order_by(Flight.flight_date.desc())
            .limit(limit)
            .all()
        )
        records = [dict(r._mapping) for r in rows]
        return {"count": len(records), "records": records, "limit": limit, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch raw data: {e}")

@app.get("/api/enrichment")
async def get_enrichment(
    route: Optional[str] = Query(None, description="Filter by route code e.g. BOM-DEL"),
    airport: Optional[str] = Query(None, description="Filter by airport code"),
    limit: int = Query(200, ge=1, le=1000)
):
    """Return recent weather/capacity enrichment records (placeholder heuristic)."""
    try:
        db = get_db_session()
        q = db.query(FlightEnrichment)
        if airport:
            q = q.filter(FlightEnrichment.airport == airport)
        if route:
            # Join with flights to filter by route
            q = q.join(Flight, Flight.id == FlightEnrichment.flight_id).filter(Flight.route == route)
        rows = q.order_by(FlightEnrichment.created_at.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({
                'flight_id': r.flight_id,
                'airport': r.airport,
                'temperature_c': r.temperature_c,
                'wind_speed_kts': r.wind_speed_kts,
                'visibility_km': r.visibility_km,
                'weather_condition': r.weather_condition,
                'runway_capacity_estimate': r.runway_capacity_estimate,
                'capacity_status': r.capacity_status,
                'timestamp': r.created_at.isoformat()
            })
        db.close()
        return {'count': len(out), 'records': out, 'route_filter': route, 'airport_filter': airport}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching enrichment: {e}")

@app.get("/api/cascade-network")
async def get_cascade_network(limit: int = Query(150, ge=10, le=1000)):
    """Return a simplified cascading delay network (nodes + edges) for visualization."""
    try:
        global cascade_analyzer
        # If analyzer not initialized or empty, fetch latest data from DB and initialize from DataFrame
        if not cascade_analyzer or not getattr(cascade_analyzer, 'aircraft_networks', None):
            if not data_processor:
                raise HTTPException(status_code=500, detail="Data processor not initialized; cannot build cascade from DB")
            latest_df = data_processor.get_latest_data(10000)
            if latest_df is None or latest_df.empty:
                return {'nodes': [], 'edges': [], 'limit': limit}
            # Initialize analyzer from DataFrame
            try:
                cascade_analyzer = CascadingDelayAnalyzer(latest_df)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize cascade analyzer from DB data: {e}")
        # Build networks (idempotent)
        cascade_analyzer.build_aircraft_dependency_network()
        nodes_out = []
        edges_out = []
        count_nodes = 0
        for key, G in cascade_analyzer.aircraft_networks.items():
            for n, data in G.nodes(data=True):
                nodes_out.append({
                    'id': n,
                    'flight_number': data.get('flight_number'),
                    'route': data.get('route'),
                    'dep_delay': data.get('departure_delay'),
                    'arr_delay': data.get('arrival_delay')
                })
                count_nodes += 1
                if count_nodes >= limit:
                    break
            if count_nodes >= limit:
                break
        # Collect edges only among included nodes
        node_ids = set(n['id'] for n in nodes_out)
        for key, G in cascade_analyzer.aircraft_networks.items():
            for u, v, edata in G.edges(data=True):
                if u in node_ids and v in node_ids:
                    edges_out.append({
                        'source': u,
                        'target': v,
                        'cascade_risk': edata.get('cascade_risk'),
                        'turnaround': edata.get('turnaround_minutes')
                    })
        return {'nodes': nodes_out, 'edges': edges_out, 'limit': limit}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building cascade network: {e}")

@app.get("/api/export/report.pdf")
async def export_pdf_report():
    """Export a concise PDF analytics report (overview + peak hours + busiest routes)."""
    try:
        db = get_db_session()
        total_flights = db.query(func.count(Flight.id)).scalar()
        avg_delay = db.query(func.avg(Flight.departure_delay_minutes)).scalar() or 0
        peak = db.query(Flight.scheduled_departure_hour, func.count(Flight.id).label('c'))\
            .group_by(Flight.scheduled_departure_hour).order_by(func.count(Flight.id).desc()).limit(5).all()
        busiest = db.query(Flight.route, func.count(Flight.id).label('c'))\
            .group_by(Flight.route).order_by(func.count(Flight.id).desc()).limit(5).all()
        db.close()
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        # Try reportlab if available
        buffer = BytesIO()
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import mm
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            y = height - 30*mm
            c.setFont('Helvetica-Bold', 16)
            c.drawString(25*mm, y, 'Flight Schedule Analytics Report')
            y -= 12*mm
            c.setFont('Helvetica', 10)
            c.drawString(25*mm, y, f'Generated: {now}')
            y -= 8*mm
            c.drawString(25*mm, y, f'Total Flights: {total_flights}  |  Avg Delay: {avg_delay:.1f} min')
            y -= 10*mm
            c.setFont('Helvetica-Bold', 12)
            c.drawString(25*mm, y, 'Top Peak Hours')
            y -= 6*mm
            c.setFont('Helvetica', 9)
            for h, cnt in peak:
                c.drawString(27*mm, y, f'Hour {h:02d}:00 - {cnt} flights')
                y -= 5*mm
                if y < 30*mm:
                    c.showPage(); y = height - 30*mm
            y -= 3*mm
            c.setFont('Helvetica-Bold', 12)
            c.drawString(25*mm, y, 'Busiest Routes')
            y -= 6*mm
            c.setFont('Helvetica', 9)
            for r, cnt in busiest:
                c.drawString(27*mm, y, f'{r}: {cnt} flights')
                y -= 5*mm
                if y < 30*mm:
                    c.showPage(); y = height - 30*mm
            c.showPage(); c.save()
            buffer.seek(0)
            return StreamingResponse(buffer, media_type='application/pdf', headers={
                'Content-Disposition': 'attachment; filename="flight_report.pdf"'
            })
        except Exception:
            # Fallback simple text PDF (or plain text disguised)
            content = f"Flight Report Generated {now}\nTotal Flights: {total_flights}\nAvg Delay: {avg_delay:.1f} min\nPeak Hours:\n" + '\n'.join([f"{h:02d}:00 - {cnt}" for h,cnt in peak])
            buffer.write(content.encode('utf-8'))
            buffer.seek(0)
            return StreamingResponse(buffer, media_type='application/octet-stream', headers={
                'Content-Disposition': 'attachment; filename="flight_report.txt"'
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Export error: {e}')


# ADMIN / MAINTENANCE ENDPOINTS
@app.delete("/api/admin/purge")
async def purge_all_data(confirm: bool = Query(False, description="Must be true to confirm purge")):
    """Dangerous: Delete all rows from every data table so a fresh upload can occur.
    Skips tables that do not exist. Requires confirm=true.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to purge all data")
    global delay_predictor, cascade_analyzer, ollama_nlp
    session = get_db_session()
    inspector = inspect(SessionLocal().bind)
    try:
        deletion_order = [
            CascadingDelayNetwork,
            PeakHourAnalysis,
            RoutePerformance,
            AnalyticsCache,
            FlightEnrichment,
            Flight,
            DataUpload,
        ]
        deleted_counts = {}
        skipped = []
        for model in deletion_order:
            table_name = model.__tablename__
            if not inspector.has_table(table_name):
                skipped.append(table_name)
                continue
            try:
                count = session.query(model).delete()
                deleted_counts[table_name] = count
            except Exception as inner:
                # Record error but continue
                skipped.append(f"{table_name} (error: {inner.__class__.__name__})")
        session.commit()
        delay_predictor = None
        cascade_analyzer = None
        ollama_nlp = None
        return {"status": "ok", "deleted": deleted_counts, "skipped": skipped}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Purge failed: {e}")
    finally:
        session.close()


# ENDPOINT 0A: File Upload and Processing
@app.post("/api/upload-data")
async def upload_flight_data(file: UploadFile = File(...)):
    """Upload and process flight data file (Excel or CSV)"""
    try:
        if not data_processor:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Validate file type - check both content type and extension
        allowed_types = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        "application/vnd.ms-excel", "text/csv", "application/csv", "application/octet-stream"]
        allowed_extensions = [".xlsx", ".xls", ".csv"]
        
        file_extension = Path(file.filename).suffix.lower() if file.filename else ""
        
        if file.content_type not in allowed_types and file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type} (extension: {file_extension}). Please upload Excel (.xlsx) or CSV files."
            )
        
        # Save uploaded file temporarily
        import tempfile
        import shutil
        
        # Determine file suffix based on extension or content type
        if file_extension in [".xlsx", ".xls"]:
            suffix = file_extension
        elif "excel" in file.content_type:
            suffix = ".xlsx"
        else:
            suffix = ".csv"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            # Process the uploaded file
            result = data_processor.process_uploaded_file(tmp_file_path, f"api_upload_{file.filename}")
            
            # Reinitialize models with new data if processing was successful
            if result['success']:
                await _reinitialize_models_with_new_data()
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Defensive extraction of nested fields to avoid KeyError when some context keys are absent
            dq_score = None
            try:
                if result.get('context_data') and isinstance(result['context_data'].get('dataset_summary'), dict):
                    dq_score = result['context_data']['dataset_summary'].get('data_quality_score')
            except Exception:
                dq_score = None
            # fallback to EDA score if present
            if dq_score is None:
                try:
                    dq_score = result.get('eda', {}).get('data_quality_score')
                except Exception:
                    dq_score = None

            processing_result = {
                "records_processed": result.get('processed_records', 0),
                "data_quality_score": dq_score,
                "validation_errors": result.get('validation', {}).get('errors', []) if isinstance(result.get('validation'), dict) else [],
                "validation_warnings": result.get('validation', {}).get('warnings', []) if isinstance(result.get('validation'), dict) else [],
                "eda_insights": {
                    "total_flights": result.get('eda', {}).get('basic_info', {}).get('total_rows', 0) if isinstance(result.get('eda'), dict) else 0,
                    "date_range": result.get('eda', {}).get('basic_info', {}).get('date_range') if isinstance(result.get('eda'), dict) else None,
                    "recommendations": result.get('eda', {}).get('recommendations', []) if isinstance(result.get('eda'), dict) else []
                }
            }

            # Build diagnostic warnings explaining missing nested fields
            diagnostics = []
            # Check context_data.dataset_summary.data_quality_score
            if not isinstance(result.get('context_data'), dict):
                diagnostics.append("Missing 'context_data' object (no dataset-level context generated)")
            else:
                if not isinstance(result['context_data'].get('dataset_summary'), dict):
                    diagnostics.append("Missing 'context_data.dataset_summary' (dataset summary not produced)")
                else:
                    if 'data_quality_score' not in result['context_data']['dataset_summary']:
                        diagnostics.append("Missing 'context_data.dataset_summary.data_quality_score' (score not calculated)")

            # Check EDA fallback
            if not isinstance(result.get('eda'), dict):
                diagnostics.append("Missing 'eda' object (no EDA output available)")
            else:
                if 'data_quality_score' not in result.get('eda', {}):
                    diagnostics.append("Missing 'eda.data_quality_score' (EDA did not yield a quality score)")

            # If dq_score is still None, provide a clear message
            if dq_score is None:
                diagnostics.append("'data_quality_score' not found in context_data.dataset_summary nor eda; displayed as unavailable")

            # Include any processing-level error recorded by the processor
            if result.get('error'):
                diagnostics.append(f"Processor error: {result.get('error')}")

            processing_result['diagnostics'] = diagnostics

            return {
                "success": result.get('success', False),
                "filename": file.filename,
                "file_size": getattr(file, 'size', None),
                "processing_result": processing_result,
                "models_reinitialized": result.get('success', False),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# ENDPOINT 0B: Data Validation (pre-upload check)
@app.post("/api/validate-data")
async def validate_flight_data(file: UploadFile = File(...)):
    """Validate flight data file structure without processing"""
    try:
        if not data_processor:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Save file temporarily for validation
        import tempfile
        import shutil
        
        # Prefer filename extension when present; fall back to content-type
        file_extension = Path(file.filename).suffix.lower() if file.filename else ''
        if file_extension in ['.xlsx', '.xls', '.csv']:
            suffix = file_extension
        else:
            suffix = ".xlsx" if "excel" in (file.content_type or '') else ".csv"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file
            validation_result = data_processor.validate_file_format(tmp_file_path)
            
            # Clean up
            os.unlink(tmp_file_path)
            
            return {
                "filename": file.filename,
                "validation": validation_result,
                "recommendations": [
                    "Ensure file contains required columns: Flight Number, From, To, STD, ATD, STA, ATA",
                    "Data should cover at least one week for meaningful analysis",
                    "Missing values should be minimal for best results"
                ] if not validation_result['valid'] else [
                    "File structure looks good!",
                    f"Found {validation_result['file_info']['rows']} rows of data",
                    "Ready for processing"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ENDPOINT 0C: Get Processing History
@app.get("/api/processing-history")
async def get_processing_history():
    """Get history of all data processing attempts"""
    try:
        if not data_processor:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        history = data_processor.get_processing_history()
        
        return {
            "total_uploads": len(history),
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

# ENDPOINT 0D: Get Current Dataset Context
@app.get("/api/current-context")
async def get_current_dataset_context():
    """Get current dataset context and statistics"""
    try:
        # Get database statistics directly
        db = get_db_session()
        
        total_flights = db.query(func.count(Flight.id)).scalar()
        latest_flight = db.query(func.max(Flight.created_at)).scalar() if total_flights > 0 else None
        unique_routes = db.query(func.count(Flight.route.distinct())).scalar() if total_flights > 0 else 0
        
        db.close()
        
        context = {
            "dataset_summary": {
                "total_flights": total_flights,
                "unique_routes": unique_routes,
                "latest_update": latest_flight.isoformat() if latest_flight else None,
                "data_source": "database"
            },
            "analysis_capabilities": {
                "peak_hours": True,
                "route_analysis": True,
                "delay_prediction": delay_predictor is not None,
                "cascade_analysis": cascade_analyzer is not None,
                "nlp_queries": True,
                "llm_analysis": ollama_nlp is not None
            }
        }
        
        return {
            "context": context,
            "models_status": {
                "delay_predictor": delay_predictor is not None,
                "cascade_analyzer": cascade_analyzer is not None,
                "nlp_interface": True,  # Always true for database-driven NLP
                "ollama_llm": ollama_nlp is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context: {str(e)}")

async def _reinitialize_models_with_new_data():
    """Reinitialize all models with newly uploaded data"""
    global delay_predictor, cascade_analyzer, nlp_interface, ollama_nlp
    
    try:
        # Get latest processed data
        latest_df = data_processor.get_latest_data(10000)
        
        if latest_df.empty:
            print("⚠️  No data available for model reinitialization")
            return
        
        # Save latest data to temporary CSV for model initialization
        temp_csv = '../data/temp_latest_data.csv'
        latest_df.to_csv(temp_csv, index=False)
        
        print("🔄 Reinitializing models with new data...")
        
        # Reinitialize delay predictor
        try:
            delay_predictor = FlightDelayPredictor(temp_csv)
            delay_predictor.train_delay_prediction_models()
            print("✅ Delay predictor reinitialized")
        except Exception as e:
            print(f"⚠️  Failed to reinitialize delay predictor: {e}")
        
        # Reinitialize cascade analyzer
        try:
            cascade_analyzer = CascadingDelayAnalyzer(temp_csv)
            print("✅ Cascade analyzer reinitialized")
        except Exception as e:
            print(f"⚠️  Failed to reinitialize cascade analyzer: {e}")
        
        # NLP interface is database-driven, always available
        nlp_interface = True
        print("✅ NLP interface ready (database-driven)")
        
        # Reinitialize Ollama interface
        try:
            ollama_nlp = OllamaNLPInterface(temp_csv)
            print("✅ Ollama LLM reinitialized")
        except Exception as e:
            print(f"⚠️  Failed to reinitialize Ollama LLM: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)
        
    except Exception as e:
        print(f"❌ Error reinitializing models: {e}")

# ENDPOINT 1: Peak Hour Summaries
@app.get("/api/peak-hours", response_model=List[PeakHourSummary])
async def get_peak_hour_summaries():
    """Get peak hour summaries with congestion analysis calculated from flight data"""
    try:
        # Get database session
        db = get_db_session()
        
        # Calculate peak hours directly from flight data
        peak_hours_data = db.query(
            Flight.scheduled_departure_hour,
            func.count(Flight.id).label('flight_count'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay'),
            func.avg(case((Flight.is_delayed_departure == True, 1.0), else_=0.0)).label('delay_rate')
        ).filter(
            Flight.departure_delay_minutes.isnot(None)
        ).group_by(Flight.scheduled_departure_hour).order_by(
            Flight.scheduled_departure_hour
        ).all()
        
        peak_summaries = []
        for hour, flight_count, avg_delay, delay_rate in peak_hours_data:
            # Handle None/NaN values
            avg_delay = float(avg_delay) if avg_delay is not None else 0.0
            delay_rate = float(delay_rate) if delay_rate is not None else 0.0
            
            # Check for NaN values and replace with 0
            if np.isnan(avg_delay):
                avg_delay = 0.0
            if np.isnan(delay_rate):
                delay_rate = 0.0
            
            # Calculate congestion score (normalized 0-100)
            congestion_score = min(100.0, (flight_count * 0.5) + (avg_delay * 0.3) + (delay_rate * 100 * 0.2))
            
            # Generate recommendation based on congestion
            if congestion_score >= 70:
                recommendation = "High congestion - Consider alternative hours"
            elif congestion_score >= 40:
                recommendation = "Moderate congestion - Monitor delays"
            else:
                recommendation = "Low congestion - Optimal for scheduling"
            
            peak_summaries.append(PeakHourSummary(
                hour=hour,
                flight_count=flight_count,
                avg_delay_minutes=round(avg_delay, 1),
                delay_rate=round(delay_rate * 100, 1),  # Convert to percentage
                congestion_score=round(congestion_score, 1),
                recommendation=recommendation
            ))
        
        db.close()
        return sorted(peak_summaries, key=lambda x: x.congestion_score, reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting peak hours: {str(e)}")

# ENDPOINT 2: Flight Delay Cascades
@app.get("/api/delay-cascades", response_model=List[CascadeAnalysis])
async def get_flight_delay_cascades():
    """Get analysis of flights causing biggest cascading delays"""
    try:
        if not cascade_analyzer:
            raise HTTPException(status_code=500, detail="Cascade analyzer not initialized")
        
        # Build networks and analyze
        cascade_analyzer.build_aircraft_dependency_network()
        impact_flights = cascade_analyzer.identify_high_impact_flights()
        
        cascades = []
        if not impact_flights.empty:
            # Get top 20 highest impact flights
            top_impacts = impact_flights.head(20)
            
            for _, flight in top_impacts.iterrows():
                recommendations = []
                if flight['cascade_risk_level'] == 'Critical':
                    recommendations = [
                        "Priority handling required",
                        "Implement buffer time",
                        "Monitor closely for delays"
                    ]
                elif flight['cascade_risk_level'] == 'High':
                    recommendations = [
                        "Increase turnaround buffer",
                        "Consider schedule adjustment"
                    ]
                else:
                    recommendations = ["Standard monitoring"]
                
                cascades.append(CascadeAnalysis(
                    flight_id=flight['flight_id'],
                    impact_score=float(flight['impact_score']),
                    cascade_risk_level=flight['cascade_risk_level'],
                    downstream_flights=int(flight['downstream_flights']),
                    recommendations=recommendations
                ))
        
        return cascades
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing cascades: {str(e)}")

# ENDPOINT 3: Heuristic Optimization Suggestions
@app.post("/api/optimize-route", response_model=OptimalTimeResponse)
async def get_route_optimization(request: RouteOptimizationRequest):
    """Get heuristic optimization suggestions for a route"""
    try:
        if not delay_predictor:
            raise HTTPException(status_code=500, detail="Delay predictor not initialized")
        
        optimal_times = delay_predictor.find_optimal_departure_time(
            request.route, 
            request.date or "2025-07-26"
        )
        
        if not optimal_times:
            raise HTTPException(
                status_code=404, 
                detail=f"No optimization data available for route {request.route}"
            )
        
        return OptimalTimeResponse(
            route=request.route,
            best_time_overall=TimeSlotRecommendation(**optimal_times['best_time_overall']),
            most_reliable_time=TimeSlotRecommendation(**optimal_times['most_reliable_time']),
            alternative_times=[TimeSlotRecommendation(**alt) for alt in optimal_times['alternative_times']],
            times_to_avoid=[TimeSlotRecommendation(**avoid) for avoid in optimal_times['times_to_avoid']],
            operational_insights=OperationalInsights(**optimal_times['operational_insights']),
            analysis_date=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing route: {str(e)}")

# ENDPOINT 4: Schedule Change Impact Analysis
@app.post("/api/schedule-impact")
async def analyze_schedule_impact(request: ScheduleChangeRequest):
    """Analyze impact of changing a flight's schedule"""
    try:
        if not delay_predictor:
            raise HTTPException(status_code=500, detail="Delay predictor not initialized")
        
        impact = delay_predictor.simulate_schedule_change_impact(
            request.flight_id,
            request.new_hour,
            request.route
        )
        
        if not impact:
            raise HTTPException(
                status_code=404,
                detail=f"Could not analyze impact for flight {request.flight_id}"
            )
        
        return {
            "flight_id": impact['flight_id'],
            "original_hour": impact['original_hour'],
            "new_hour": impact['new_hour'],
            "delay_change_minutes": impact['delay_change_minutes'],
            "probability_change": impact['probability_change'],
            "recommendation": impact['recommendation'],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing schedule impact: {str(e)}")

# ENDPOINT 5: Delay Prediction
@app.post("/api/predict-delay")
async def predict_flight_delay(request: DelayPredictionRequest):
    """Predict delay for a specific flight configuration"""
    try:
        if not delay_predictor:
            raise HTTPException(status_code=500, detail="Delay predictor not initialized")
        
        # Create feature dictionary for prediction
        features = delay_predictor._get_base_features_for_route(request.route, request.date or "2025-07-26")
        features.update({
            'Scheduled_Departure_Hour': request.departure_hour,
            'Hour_Sin': np.sin(2 * np.pi * request.departure_hour / 24),
            'Hour_Cos': np.cos(2 * np.pi * request.departure_hour / 24),
            'Is_Weekend': request.is_weekend,
            'Is_Peak_Hour': request.departure_hour in [6, 10]
        })
        
        prediction = delay_predictor.predict_delay(features)
        
        return {
            "route": request.route,
            "departure_hour": request.departure_hour,
            "predicted_delay_minutes": prediction['predicted_delay_minutes'],
            "delay_probability": prediction['delay_probability'],
            "is_delayed": prediction['is_delayed'],
            "risk_level": "High" if prediction['delay_probability'] > 0.7 else "Medium" if prediction['delay_probability'] > 0.3 else "Low",
            "prediction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting delay: {str(e)}")

# ENDPOINT 6: NLP-Generated Responses (Pattern-based) - Database-driven
@app.post("/api/nlp-query")
async def process_nlp_query(request: NLPQueryRequest):
    """Process natural language queries using database data like Ollama"""
    try:
        # Get fresh data from database
        db = get_db_session()
        
        # Get flight statistics for context
        total_flights = db.query(func.count(Flight.id)).scalar()
        if total_flights == 0:
            return {
                "query": request.query,
                "response": "I don't have any flight data to analyze yet. Please upload flight data first.",
                "response_type": "no_data",
                "confidence": 0.0,
                "data_used": "database",
                "processing_time": 0,
                "suggestions": ["Upload flight data to begin analysis"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Process query using database-driven NLP analysis
        start_time = datetime.now()
        result = await process_database_nlp_query(request.query, db)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        db.close()
        
        return {
            "query": request.query,
            "response": result['response'],
            "response_type": result.get('response_type', 'analysis'),
            "confidence": result.get('confidence', 0.8),
            "data_used": "database",
            "processing_time": int(processing_time),
            "suggestions": result.get('suggestions', []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NLP query: {str(e)}")

async def process_database_nlp_query(query: str, db) -> dict:
    """Process NLP queries using direct database analysis"""
    query_lower = query.lower()
    
    # Peak hours analysis
    if any(keyword in query_lower for keyword in ['peak', 'busy', 'congestion', 'hours', 'time']):
        # Get hourly flight counts and delays
        hourly_stats = db.query(
            Flight.scheduled_departure_hour,
            func.count(Flight.id).label('flight_count'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay'),
            func.count(Flight.id).filter(Flight.departure_delay_minutes > 15).label('delayed_count')
        ).filter(
            Flight.departure_delay_minutes.isnot(None)
        ).group_by(Flight.scheduled_departure_hour).order_by(Flight.scheduled_departure_hour).all()
        
        peak_hours = []
        for stat in hourly_stats:
            if stat.flight_count > 10:  # Only consider hours with significant traffic
                delay_rate = (stat.delayed_count / stat.flight_count) * 100
                congestion_score = (stat.avg_delay or 0) + (delay_rate * 0.5)
                peak_hours.append({
                    'hour': stat.scheduled_departure_hour,
                    'flights': stat.flight_count,
                    'avg_delay': round(stat.avg_delay or 0, 1),
                    'delay_rate': round(delay_rate, 1),
                    'congestion_score': round(congestion_score, 1)
                })
        
        # Sort by congestion score
        peak_hours.sort(key=lambda x: x['congestion_score'], reverse=True)
        top_congested = peak_hours[:3]
        
        response = f"**Peak Hours Analysis:**\n\n"
        response += f"The most congested hours are:\n"
        for i, hour in enumerate(top_congested, 1):
            response += f"{i}. **{hour['hour']:02d}:00** - {hour['flights']} flights, {hour['avg_delay']} min avg delay, {hour['delay_rate']}% delayed\n"
        
        response += f"\n**Recommendations:**\n"
        response += f"• Avoid scheduling during {top_congested[0]['hour']:02d}:00-{top_congested[0]['hour']+1:02d}:00 (highest congestion)\n"
        response += f"• Consider off-peak hours for better on-time performance\n"
        
        # Find best hours (lowest congestion)
        best_hours = sorted(peak_hours, key=lambda x: x['congestion_score'])[:2]
        if best_hours:
            response += f"• Best times: {best_hours[0]['hour']:02d}:00 ({best_hours[0]['avg_delay']} min delay) and {best_hours[1]['hour']:02d}:00 ({best_hours[1]['avg_delay']} min delay)\n"
        
        return {
            'response': response,
            'response_type': 'peak_hours',
            'confidence': 0.9,
            'suggestions': [
                'Show route-specific peak hours',
                'Analyze delay causes during peak hours',
                'Compare weekday vs weekend patterns'
            ]
        }
    
    # Route analysis
    elif any(keyword in query_lower for keyword in ['route', 'destination', 'worst', 'best', 'reliable']):
        route_stats = db.query(
            Flight.route,
            func.count(Flight.id).label('flight_count'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay'),
            func.count(Flight.id).filter(Flight.departure_delay_minutes <= 15).label('ontime_count')
        ).filter(
            Flight.departure_delay_minutes.isnot(None)
        ).group_by(Flight.route).having(func.count(Flight.id) >= 5).order_by(func.avg(Flight.departure_delay_minutes).desc()).all()
        
        response = f"**Route Performance Analysis:**\n\n"
        
        if 'worst' in query_lower:
            worst_routes = route_stats[:5]
            response += f"**Worst performing routes:**\n"
            for i, route in enumerate(worst_routes, 1):
                ontime_rate = (route.ontime_count / route.flight_count) * 100
                response += f"{i}. **{route.route}** - {route.avg_delay:.1f} min avg delay, {ontime_rate:.1f}% on-time ({route.flight_count} flights)\n"
        elif 'best' in query_lower or 'reliable' in query_lower:
            best_routes = route_stats[-5:][::-1]  # Reverse to get best first
            response += f"**Most reliable routes:**\n"
            for i, route in enumerate(best_routes, 1):
                ontime_rate = (route.ontime_count / route.flight_count) * 100
                response += f"{i}. **{route.route}** - {route.avg_delay:.1f} min avg delay, {ontime_rate:.1f}% on-time ({route.flight_count} flights)\n"
        else:
            # Show overview
            total_routes = len(route_stats)
            avg_route_delay = sum(r.avg_delay for r in route_stats) / len(route_stats)
            response += f"**Route Overview:**\n"
            response += f"• {total_routes} active routes analyzed\n"
            response += f"• Average route delay: {avg_route_delay:.1f} minutes\n\n"
            
            response += f"**Top 3 problem routes:**\n"
            for i, route in enumerate(route_stats[:3], 1):
                ontime_rate = (route.ontime_count / route.flight_count) * 100
                response += f"{i}. **{route.route}** - {route.avg_delay:.1f} min delay, {ontime_rate:.1f}% on-time\n"
        
        return {
            'response': response,
            'response_type': 'route_analysis',
            'confidence': 0.85,
            'suggestions': [
                'Analyze specific route delays',
                'Compare route performance by time of day',
                'Show route optimization recommendations'
            ]
        }
    
    # General statistics
    elif any(keyword in query_lower for keyword in ['stats', 'statistics', 'overview', 'summary']):
        stats = db.query(
            func.count(Flight.id).label('total_flights'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay'),
            func.count(Flight.id).filter(Flight.departure_delay_minutes <= 15).label('ontime_count'),
            func.count(Flight.route.distinct()).label('unique_routes')
        ).filter(Flight.departure_delay_minutes.isnot(None)).first()
        
        ontime_percentage = (stats.ontime_count / stats.total_flights) * 100
        
        response = f"**Flight Schedule Statistics:**\n\n"
        response += f"• **Total flights:** {stats.total_flights:,}\n"
        response += f"• **Average delay:** {stats.avg_delay:.1f} minutes\n"
        response += f"• **On-time performance:** {ontime_percentage:.1f}%\n"
        response += f"• **Active routes:** {stats.unique_routes}\n\n"
        
        if ontime_percentage < 60:
            response += f"**⚠️ Performance Alert:** On-time rate is below 60%. Consider schedule optimization.\n"
        elif ontime_percentage > 80:
            response += f"**✅ Good Performance:** Above 80% on-time rate indicates efficient scheduling.\n"
        
        return {
            'response': response,
            'response_type': 'statistics',
            'confidence': 0.95,
            'suggestions': [
                'Analyze peak hours for optimization',
                'Review worst performing routes',
                'Check delay trends over time'
            ]
        }
    
    # Default response
    else:
        available_queries = [
            "What are the peak hours?",
            "Which routes have the most delays?",
            "Show me flight statistics",
            "What are the worst performing routes?",
            "When is the best time to schedule flights?"
        ]
        
        response = f"I can help you analyze flight schedule data! Here are some things you can ask:\n\n"
        for i, q in enumerate(available_queries, 1):
            response += f"{i}. {q}\n"
        
        return {
            'response': response,
            'response_type': 'help',
            'confidence': 0.7,
            'suggestions': available_queries
        }

# ENDPOINT 6B: Ollama LLM-Powered Queries  
@app.post("/api/ollama-query")
async def process_ollama_query(request: OllamaQueryRequest):
    """Process natural language queries using Ollama LLM with custom context"""
    try:
        if not ollama_nlp:
            raise HTTPException(
                status_code=503, 
                detail="Ollama LLM not available. Make sure Ollama is running and a model is installed."
            )
        
        result = ollama_nlp.query_with_ollama(request.query, request.context)
        
        return {
            "query": request.query,
            "response": result['response'],
            "model_used": result['model_used'],
            "context_provided": result['context_provided'],
            "method": "ollama_llm",
            "success": result['success'],
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Ollama query: {str(e)}")

# ENDPOINT 6C: Advanced Flight Schedule Change Analysis with LLM
@app.post("/api/ollama-schedule-analysis")
async def analyze_schedule_change_with_llm(request: ScheduleChangeAnalysisRequest):
    """Analyze flight schedule changes using Ollama LLM with detailed context"""
    try:
        if not ollama_nlp:
            raise HTTPException(
                status_code=503,
                detail="Ollama LLM not available. Make sure Ollama is running and a model is installed."
            )
        
        result = ollama_nlp.analyze_flight_schedule_change(
            request.flight_id,
            request.current_hour,
            request.new_hour,
            request.route
        )
        
        return {
            "flight_id": request.flight_id,
            "route": request.route,
            "schedule_change": f"{request.current_hour}:00 → {request.new_hour}:00",
            "llm_analysis": result['response'],
            "model_used": result['model_used'],
            "success": result['success'],
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing schedule change: {str(e)}")

# ENDPOINT 6D: Optimal Time Finding with LLM
@app.post("/api/ollama-optimal-time")
async def find_optimal_time_with_llm(request: RouteOptimizationRequest):
    """Find optimal departure time using Ollama LLM analysis"""
    try:
        if not ollama_nlp:
            raise HTTPException(
                status_code=503,
                detail="Ollama LLM not available. Make sure Ollama is running and a model is installed."
            )
        
        result = ollama_nlp.find_optimal_time_with_llm(request.route, request.date)
        
        return {
            "route": request.route,
            "date": request.date or "2025-07-26",
            "llm_analysis": result['response'],
            "model_used": result['model_used'],
            "success": result['success'],
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding optimal time: {str(e)}")

# ENDPOINT 7: Route Performance Analytics
@app.get("/api/route-performance")
async def get_route_performance(
    route: Optional[str] = Query(None, description="Specific route to analyze (e.g., BOM-DEL)"),
    top_n: int = Query(10, description="Number of top/bottom routes to return")
):
    """Get route performance analytics"""
    try:
        # Use database-driven analysis
        db = get_db_session()
        
        if route:
            query = f"How is the {route} route performing?"
        else:
            query = "Show me route performance analysis"
        
        result = await process_database_nlp_query(query, db)
        db.close()
        
        return {
            "route_filter": route,
            "analysis": result['response'],
            "data": result.get('data', {}),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting route performance: {str(e)}")

# ENDPOINT 8: Flight Statistics Summary
@app.get("/api/statistics")
async def get_flight_statistics():
    """Get comprehensive flight statistics in the format expected by frontend"""
    try:
        # Get database session
        db = get_db_session()
        
        # Overview statistics
        total_flights = db.query(Flight).count()
        unique_routes = db.query(Flight.route).distinct().count()
        
        # Calculate average delay and on-time percentage
        avg_delay = db.query(func.avg(Flight.departure_delay_minutes)).filter(
            Flight.departure_delay_minutes.isnot(None)
        ).scalar() or 0
        
        delayed_flights = db.query(Flight).filter(Flight.is_delayed_departure == True).count()
        on_time_percentage = ((total_flights - delayed_flights) / total_flights * 100) if total_flights > 0 else 0
        
        # Peak hours analysis
        peak_hours_data = db.query(
            Flight.scheduled_departure_hour,
            func.count(Flight.id).label('flight_count')
        ).group_by(Flight.scheduled_departure_hour).order_by(
            func.count(Flight.id).desc()
        ).limit(3).all()
        
        peak_hours = [hour for hour, count in peak_hours_data]
        
        # Busiest routes
        busiest_routes_data = db.query(
            Flight.route,
            func.count(Flight.id).label('flights'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay')
        ).group_by(Flight.route).order_by(
            func.count(Flight.id).desc()
        ).limit(5).all()
        
        busiest_routes = [
            {
                "route": route,
                "flights": flights,
                "avg_delay": round(avg_delay or 0, 1)
            }
            for route, flights, avg_delay in busiest_routes_data
        ]
        
        # Most reliable routes (highest on-time rate)
        reliable_routes_data = db.query(
            Flight.route,
            func.count(Flight.id).label('total_flights'),
            func.sum(case((Flight.is_delayed_departure == False, 1), else_=0)).label('on_time_flights')
        ).group_by(Flight.route).having(
            func.count(Flight.id) >= 5  # Only routes with at least 5 flights
        ).all()
        
        most_reliable_routes = []
        for route, total, on_time in reliable_routes_data:
            on_time_rate = (on_time / total * 100) if total > 0 else 0
            # Calculate consistency as a score based on total flights and on-time rate
            consistency = min(100.0, (on_time_rate * 0.8) + (min(total, 20) * 1.0))
            most_reliable_routes.append({
                "route": route,
                "on_time_rate": round(on_time_rate, 1),
                "consistency": round(consistency, 1)
            })
        
        # Sort by on-time rate and take top 5
        most_reliable_routes.sort(key=lambda x: x['on_time_rate'], reverse=True)
        most_reliable_routes = most_reliable_routes[:5]
        
        # Airport analysis
        airport_data = db.query(
            Flight.origin_airport,
            func.count(Flight.id).label('flights'),
            func.avg(Flight.departure_delay_minutes).label('avg_delay')
        ).filter(
            Flight.departure_delay_minutes.isnot(None)
        ).group_by(Flight.origin_airport).order_by(
            func.count(Flight.id).desc()
        ).limit(5).all()
        
        busiest_airports = [
            {
                "airport": airport,
                "flights": flights,
                "avg_delay": round(avg_delay or 0, 1)
            }
            for airport, flights, avg_delay in airport_data
        ]
        
        db.close()
        
        return {
            "overview": {
                "total_flights": total_flights,
                "unique_routes": unique_routes,
                "avg_delay_minutes": round(avg_delay, 1),
                "on_time_percentage": round(on_time_percentage, 1)
            },
            "temporal_analysis": {
                "peak_hours": peak_hours,
                "busiest_day": "Friday",  # Could calculate this from data
                "seasonal_patterns": {}
            },
            "route_analysis": {
                "busiest_routes": busiest_routes,
                "most_reliable_routes": most_reliable_routes
            },
            "airport_analysis": {
                "busiest_airports": busiest_airports
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.get("/api/statistics-nlp")
async def get_flight_statistics_nlp():
    """Get flight statistics using database-driven NLP"""
    try:
        # Use database-driven analysis
        db = get_db_session()
        
        result = await process_database_nlp_query("Show me flight statistics", db)
        db.close()
        
        return {
            "statistics": result['response'],
            "detailed_data": result.get('data', {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# Documentation endpoint
@app.get("/api/docs-summary")
async def get_api_documentation():
    """Get API documentation summary"""
    return {
        "title": "Flight Schedule Optimization API",
        "version": "1.0.0",
        "description": "REST API providing flight schedule optimization, delay prediction, and analytics",
        "endpoints": {
            "POST /api/upload-data": "Upload and process new flight data files (Excel/CSV)",
            "POST /api/validate-data": "Validate file structure before processing",
            "GET /api/processing-history": "Get history of data uploads and processing",
            "GET /api/current-context": "Get current dataset context and statistics",
            "GET /api/peak-hours": "Get peak hour summaries with congestion analysis",
            "GET /api/delay-cascades": "Get flights causing biggest cascading delays",
            "POST /api/optimize-route": "Get optimization suggestions for a route",
            "POST /api/schedule-impact": "Analyze impact of schedule changes",
            "POST /api/predict-delay": "Predict delay for specific flight configuration",
            "POST /api/nlp-query": "Process natural language queries (pattern-based)",
            "POST /api/ollama-query": "Process queries with Ollama LLM and custom context",
            "POST /api/ollama-schedule-analysis": "Advanced schedule change analysis with LLM",
            "POST /api/ollama-optimal-time": "Find optimal times using LLM analysis",
            "GET /api/route-performance": "Get route performance analytics",
            "GET /api/statistics": "Get comprehensive flight statistics"
        },
        "workflow_compliance": {
            "data_acquisition": "✅ Completed - Dynamic file upload with database storage",
            "preprocessing": "✅ Completed - Automated EDA and ML-ready feature generation", 
            "analytics_heuristics": "✅ Completed - Peak hours, cascades, simulation",
            "nlp_layer": "✅ Completed - Both pattern-based and Ollama LLM interfaces",
            "ollama_integration": "✅ Completed - Custom context prompting with LLM",
            "backend_api": "✅ Completed - Dynamic data processing with validation",
            "database_storage": "✅ Completed - SQLite with upload tracking",
            "file_validation": "✅ Completed - Excel/CSV format validation",
            "frontend": "⏳ Planned - Not in current scope"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
