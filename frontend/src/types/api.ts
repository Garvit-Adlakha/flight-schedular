/**
 * TypeScript interfaces for Flight Schedule Optimization API
 */

export interface PeakHourSummary {
  hour: number;
  flight_count: number;
  avg_delay_minutes: number;
  delay_rate: number;
  congestion_score: number;
  recommendation: string;
}

export interface CascadeAnalysis {
  flight_id: string;
  impact_score: number;
  cascade_risk_level: string;
  downstream_flights: number;
  recommendations: string[];
}

export interface TimeSlotRecommendation {
  hour: number;
  time_slot: string;
  predicted_delay: number;
  on_time_probability: number;
  rationale?: string;
  ranking?: number;
  reason?: string;
}

export interface OperationalInsights {
  best_window: string;
  avg_delay_best_window: number;
  avg_delay_worst_window: number;
  improvement_potential: number;
}

export interface RouteOptimization {
  route: string;
  best_time_overall: TimeSlotRecommendation;
  most_reliable_time: TimeSlotRecommendation;
  alternative_times: TimeSlotRecommendation[];
  times_to_avoid: TimeSlotRecommendation[];
  operational_insights: OperationalInsights;
  analysis_date: string;
}

export interface DelayPrediction {
  predicted_delay_minutes: number;
  confidence_score: number;
  risk_level: string;
  contributing_factors: string[];
  recommendations: string[];
}

export interface UploadResult {
  success: boolean;
  filename: string;
  file_size: number;
  processing_result: {
    records_processed: number;
    data_quality_score: number;
    validation_errors: string[];
    validation_warnings: string[];
    eda_insights: {
      total_flights: number;
      date_range: {
        start: string;
        end: string;
      } | null;
      recommendations: string[];
    };
  };
  models_reinitialized: boolean;
  timestamp: string;
}

export interface ValidationResult {
  filename: string;
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    file_info: {
      name: string;
      size: number;
      format: string;
      rows: number;
      columns: string[];
    };
  };
  recommendations: string[];
  timestamp: string;
}

export interface ProcessingHistory {
  total_uploads: number;
  history: Array<{
    filename: string;
    upload_timestamp: string;
    processing_status: string;
    total_records: number;
    valid_records: number;
    data_quality_score: number | null;
    processing_duration: number | null;
  }>;
  timestamp: string;
}

export interface CurrentContext {
  context: {
    dataset_summary: {
      total_flights: number;
      unique_routes: number;
      last_updated: string;
    };
    operational_insights: {
      peak_hours: Array<{
        hour: number;
        airport: string;
        congestion_score: number;
        total_flights: number;
      }>;
    };
    performance_metrics: {
      busiest_routes: Array<{
        route: string;
        total_flights: number;
        on_time_percentage: number;
        avg_delay: number;
      }>;
    };
  };
  database_summary: {
    total_records_in_db: number;
    latest_date: string | null;
    sample_records: number;
  } | null;
  models_status: {
    delay_predictor: boolean;
    cascade_analyzer: boolean;
    nlp_interface: boolean;
    ollama_llm: boolean;
  };
  timestamp: string;
}

export interface NLPResponse {
  query: string;
  response: string;
  response_type: string;
  confidence: number;
  data_used: any;
  processing_time: number;
  suggestions: string[];
}

export interface OllamaResponse {
  query: string;
  response: string;
  model_used?: string;
  context_used: any;
  processing_time?: number;
  confidence?: number;
  success?: boolean;
  timestamp?: string;
}

export interface RoutePerformance {
  route: string;
  statistics: {
    total_flights: number;
    avg_delay: number;
    on_time_percentage: number;
    delay_categories: {
      on_time: number;
      minor_delay: number;
      major_delay: number;
      severe_delay: number;
    };
  };
  peak_performance: {
    best_hour: number;
    worst_hour: number;
    most_reliable_hour: number;
  };
  trends: {
    delay_trend: string;
    volume_trend: string;
    recommendations: string[];
  };
}

export interface FlightStatistics {
  overview: {
    total_flights: number;
    unique_routes: number;
    avg_delay_minutes: number;
    on_time_percentage: number;
  };
  temporal_analysis: {
    peak_hours: number[];
    busiest_day: string;
    seasonal_patterns: any;
  };
  route_analysis: {
    busiest_routes: Array<{
      route: string;
      flights: number;
      avg_delay: number;
    }>;
    most_reliable_routes: Array<{
      route: string;
      on_time_rate: number;
      consistency: number;
    }>;
  };
  airport_analysis: {
    busiest_airports: Array<{
      airport: string;
      flights: number;
      avg_delay: number;
    }>;
  };
}

export interface EnrichmentRecord {
  flight_id: number;
  airport: string;
  temperature_c?: number;
  wind_speed_kts?: number;
  visibility_km?: number;
  weather_condition?: string;
  runway_capacity_estimate?: number;
  capacity_status?: string;
  timestamp: string;
}

export interface EnrichmentResponse {
  count: number;
  records: EnrichmentRecord[];
  route_filter?: string | null;
  airport_filter?: string | null;
}
