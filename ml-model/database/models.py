"""
PostgreSQL Database Models for Flight Schedule Optimization
Optimized schema with proper indexing to eliminate redundant calculations
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, Index, UniqueConstraint
from sqlalchemy.sql import func
from database.config import Base
from datetime import datetime

class Flight(Base):
    """
    Main flights table with optimized structure for analytics
    """
    __tablename__ = "flights"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Flight identification
    flight_number = Column(String(10), nullable=False, index=True)
    flight_date = Column(DateTime, nullable=False, index=True)
    
    # Airport information (allow longer descriptive names to avoid truncation)
    origin_airport = Column(String(255), nullable=False, index=True)
    destination_airport = Column(String(255), nullable=False, index=True)
    route = Column(String(255), nullable=False, index=True)
    
    # Scheduled times
    scheduled_departure_time = Column(DateTime, nullable=False)
    scheduled_arrival_time = Column(DateTime, nullable=False)
    scheduled_departure_hour = Column(Integer, nullable=False, index=True)  # Cached for performance
    
    # Actual times
    actual_departure_time = Column(DateTime, nullable=True)
    actual_arrival_time = Column(DateTime, nullable=True)
    
    # Pre-calculated delays (to avoid redundant calculations)
    departure_delay_minutes = Column(Float, nullable=True, index=True)
    arrival_delay_minutes = Column(Float, nullable=True, index=True)
    
    # Delay flags for quick filtering
    is_delayed_departure = Column(Boolean, default=False, index=True)
    is_delayed_arrival = Column(Boolean, default=False, index=True)
    
    # Additional information
    aircraft = Column(String(20), nullable=True)
    flight_duration_minutes = Column(Integer, nullable=True)
    status = Column(String(20), nullable=True, index=True)
    
    # Data source tracking
    data_source = Column(String(50), nullable=False, default='upload')
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_route_date', route, flight_date),
        Index('idx_origin_date', origin_airport, flight_date),
        Index('idx_departure_hour_route', scheduled_departure_hour, route),
        Index('idx_delay_flags', is_delayed_departure, is_delayed_arrival),
        UniqueConstraint('flight_number', 'flight_date', 'scheduled_departure_time', name='uq_flight_schedule'),
    )

class FlightEnrichment(Base):
    """Weather and capacity enrichment data for flights (separate table to avoid altering core schema)."""
    __tablename__ = "flight_enrichment"

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, index=True)  # FK reference (implicit to keep lightweight)
    airport = Column(String(255), index=True)
    temperature_c = Column(Float, nullable=True)
    wind_speed_kts = Column(Float, nullable=True)
    visibility_km = Column(Float, nullable=True)
    weather_condition = Column(String(50), nullable=True)
    runway_capacity_estimate = Column(Integer, nullable=True)
    capacity_status = Column(String(20), nullable=True, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    data_source = Column(String(30), default='heuristic', index=True)

    __table_args__ = (
        Index('idx_enrichment_flight', flight_id, airport),
        Index('idx_enrichment_airport_capacity', airport, capacity_status),
    )

class DataUpload(Base):
    """
    Track all data upload attempts and processing status
    """
    __tablename__ = "data_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=func.now(), index=True)
    file_format = Column(String(10), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Processing statistics
    total_records = Column(Integer, nullable=False, default=0)
    valid_records = Column(Integer, nullable=False, default=0)
    duplicate_records = Column(Integer, nullable=False, default=0)
    error_records = Column(Integer, nullable=False, default=0)
    
    # Processing status
    processing_status = Column(String(20), nullable=False, default='pending')  # pending, success, failed
    processing_duration_seconds = Column(Float, nullable=True)
    error_details = Column(Text, nullable=True)
    
    # Data quality metrics
    data_quality_score = Column(Float, nullable=True)
    missing_values_percentage = Column(Float, nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)

class AnalyticsCache(Base):
    """
    Cache frequently computed analytics to improve performance
    """
    __tablename__ = "analytics_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    cache_type = Column(String(50), nullable=False, index=True)  # peak_hours, route_performance, etc.
    
    # Cached data
    result_json = Column(Text, nullable=False)  # JSON serialized results
    
    # Cache metadata
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True, index=True)
    data_hash = Column(String(64), nullable=False)  # Hash of source data for invalidation
    
    # Cache statistics
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=func.now())

class RoutePerformance(Base):
    """
    Pre-computed route performance metrics for faster analytics
    """
    __tablename__ = "route_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    route = Column(String(255), nullable=False, index=True)
    calculation_date = Column(DateTime, default=func.now(), index=True)
    
    # Performance metrics
    total_flights = Column(Integer, nullable=False)
    avg_departure_delay = Column(Float, nullable=True)
    avg_arrival_delay = Column(Float, nullable=True)
    on_time_percentage = Column(Float, nullable=True)
    severe_delay_count = Column(Integer, nullable=False, default=0)
    
    # Peak hour analysis
    peak_hour = Column(Integer, nullable=True)
    peak_hour_flights = Column(Integer, nullable=False, default=0)
    off_peak_performance_score = Column(Float, nullable=True)
    
    # Data range
    data_start_date = Column(DateTime, nullable=True)
    data_end_date = Column(DateTime, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('route', 'calculation_date', name='uq_route_calc_date'),
        Index('idx_route_performance', route, calculation_date),
    )

class PeakHourAnalysis(Base):
    """
    Pre-computed peak hour analysis by airport and time
    """
    __tablename__ = "peak_hour_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    airport = Column(String(255), nullable=False, index=True)
    hour = Column(Integer, nullable=False, index=True)
    analysis_date = Column(DateTime, default=func.now(), index=True)
    
    # Flight volume metrics
    total_flights = Column(Integer, nullable=False)
    departure_flights = Column(Integer, nullable=False, default=0)
    arrival_flights = Column(Integer, nullable=False, default=0)
    
    # Congestion metrics
    avg_delay_minutes = Column(Float, nullable=True)
    congestion_score = Column(Float, nullable=True, index=True)
    delay_probability = Column(Float, nullable=True)
    
    # Performance indicators
    is_peak_hour = Column(Boolean, default=False, index=True)
    recommended_for_scheduling = Column(Boolean, default=True, index=True)
    
    __table_args__ = (
        UniqueConstraint('airport', 'hour', 'analysis_date', name='uq_airport_hour_analysis'),
        Index('idx_peak_analysis', airport, hour, is_peak_hour),
    )

class CascadingDelayNetwork(Base):
    """
    Pre-computed cascading delay relationships for network analysis
    """
    __tablename__ = "cascading_delay_network"
    
    id = Column(Integer, primary_key=True, index=True)
    source_flight_id = Column(Integer, nullable=False, index=True)
    target_flight_id = Column(Integer, nullable=False, index=True)
    
    # Relationship metrics
    delay_impact_minutes = Column(Float, nullable=False)
    probability_score = Column(Float, nullable=False, index=True)
    cascade_depth = Column(Integer, nullable=False, default=1)
    
    # Network analysis
    betweenness_centrality = Column(Float, nullable=True)
    degree_centrality = Column(Float, nullable=True)
    
    # Metadata
    analysis_date = Column(DateTime, default=func.now(), index=True)
    data_hash = Column(String(64), nullable=False)
    
    __table_args__ = (
        Index('idx_cascade_network', source_flight_id, target_flight_id),
        Index('idx_cascade_impact', probability_score, delay_impact_minutes),
    )

# Database initialization functions
def create_all_tables():
    """Create all database tables with indexes"""
    from database.config import engine
    Base.metadata.create_all(bind=engine)
    print("✅ All database tables created successfully")

def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    from database.config import engine
    Base.metadata.drop_all(bind=engine)
    print("⚠️  All database tables dropped")

if __name__ == "__main__":
    print("🏗️  Creating database schema...")
    create_all_tables()
