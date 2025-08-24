/**
 * API service for Flight Schedule Optimization
 * Handles all communication with the PostgreSQL-optimized backend
 */

import axios from 'axios';
import {
  PeakHourSummary,
  CascadeAnalysis,
  RouteOptimization,
  DelayPrediction,
  UploadResult,
  ValidationResult,
  ProcessingHistory,
  CurrentContext,
  NLPResponse,
  OllamaResponse,
  RoutePerformance,
  FlightStatistics
} from '../types/api';
import { EnrichmentResponse } from '../types/api';

// Configure axios instance
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased to 60 seconds for LLM queries
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Health Check
 */
export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

/**
 * Data Management APIs
 */
export const uploadFlightData = async (file: File): Promise<UploadResult> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post('/api/upload-data', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 1 minute for file upload
  });
  
  return response.data;
};

export const validateFlightData = async (file: File): Promise<ValidationResult> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post('/api/validate-data', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getProcessingHistory = async (): Promise<ProcessingHistory> => {
  const response = await apiClient.get('/api/processing-history');
  return response.data;
};

export const getCurrentContext = async (): Promise<CurrentContext> => {
  const response = await apiClient.get('/api/current-context');
  return response.data;
};

/**
 * Analytics APIs
 */
export const getPeakHours = async (): Promise<PeakHourSummary[]> => {
  const response = await apiClient.get('/api/peak-hours');
  return response.data;
};

export const getDelayCascades = async (): Promise<CascadeAnalysis[]> => {
  const response = await apiClient.get('/api/delay-cascades');
  return response.data;
};

export const optimizeRoute = async (route: string, date?: string): Promise<RouteOptimization> => {
  const response = await apiClient.post('/api/optimize-route', {
    route,
    date,
  });
  return response.data;
};

export const analyzeScheduleImpact = async (flightId: string, newHour: number, route: string) => {
  const response = await apiClient.post('/api/schedule-impact', {
    flight_id: flightId,
    new_hour: newHour,
    route,
  });
  return response.data;
};

export const predictDelay = async (
  route: string,
  departureHour: number,
  date?: string,
  isWeekend?: boolean
): Promise<DelayPrediction> => {
  const response = await apiClient.post('/api/predict-delay', {
    route,
    departure_hour: departureHour,
    date,
    is_weekend: isWeekend,
  });
  return response.data;
};

export const getRoutePerformance = async (route?: string): Promise<RoutePerformance[]> => {
  const params = route ? { route } : {};
  const response = await apiClient.get('/api/route-performance', { params });
  return response.data;
};

export const getFlightStatistics = async (): Promise<FlightStatistics> => {
  const response = await apiClient.get('/api/statistics');
  return response.data;
};

export const getEnrichment = async (route?: string, airport?: string, limit: number = 200): Promise<EnrichmentResponse> => {
  const params: any = { limit };
  if (route) params.route = route;
  if (airport) params.airport = airport;
  const response = await apiClient.get('/api/enrichment', { params });
  return response.data;
};

/**
 * NLP and LLM APIs
 */
export const queryNLP = async (query: string): Promise<NLPResponse> => {
  const response = await apiClient.post('/api/nlp-query', {
    query,
  });
  return response.data;
};

export const queryOllama = async (query: string, context?: any): Promise<OllamaResponse> => {
  const response = await apiClient.post('/api/ollama-query', {
    query,
    context,
  }, {
    timeout: 90000, // 90 seconds specifically for Ollama queries
  });
  return response.data;
};

export const analyzeScheduleWithLLM = async (
  flightId: string,
  currentHour: number,
  newHour: number,
  route: string
): Promise<OllamaResponse> => {
  const response = await apiClient.post('/api/ollama-schedule-analysis', {
    flight_id: flightId,
    current_hour: currentHour,
    new_hour: newHour,
    route,
  });
  return response.data;
};

export const findOptimalTimeWithLLM = async (route: string, date?: string): Promise<OllamaResponse> => {
  const response = await apiClient.post('/api/ollama-optimal-time', {
    route,
    date,
  });
  return response.data;
};

export const getCascadeNetwork = async (limit: number = 150) => {
  const response = await apiClient.get('/api/cascade-network', { params: { limit } });
  return response.data;
};

export const downloadPDFReport = async () => {
  const response = await apiClient.get('/api/export/report.pdf', { responseType: 'blob' });
  const blob = new Blob([response.data], { type: 'application/pdf' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'flight_report.pdf';
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

/**
 * Export API documentation
 */
export const getAPIDocumentation = async () => {
  const response = await apiClient.get('/docs');
  return response.data;
};

/**
 * Utility functions
 */
export const downloadFile = (data: any, filename: string, type: string = 'application/json') => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const exportAnalysisResults = async (analysisType: string) => {
  try {
    let data;
    let filename;
    
    switch (analysisType) {
      case 'peak-hours':
        data = await getPeakHours();
        filename = 'peak-hours-analysis.json';
        break;
      case 'delay-cascades':
        data = await getDelayCascades();
        filename = 'delay-cascades-analysis.json';
        break;
      case 'statistics':
        data = await getFlightStatistics();
        filename = 'flight-statistics.json';
        break;
      default:
        throw new Error('Unknown analysis type');
    }
    
    downloadFile(data, filename);
  } catch (error) {
    console.error('Export failed:', error);
    throw error;
  }
};

export default apiClient;
