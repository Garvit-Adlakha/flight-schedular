/**
 * Custom React hooks for API data fetching with React Query
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as api from '../services/api';

// Query keys for React Query
export const queryKeys = {
  health: ['health'],
  peakHours: ['peak-hours'],
  delayCascades: ['delay-cascades'],
  processingHistory: ['processing-history'],
  currentContext: ['current-context'],
  routePerformance: (route?: string) => ['route-performance', route],
  flightStatistics: ['flight-statistics'],
  routeOptimization: (route: string, date?: string) => ['route-optimization', route, date],
  enrichment: (route?: string, airport?: string) => ['enrichment', route, airport],
};

/**
 * Health Check Hook
 */
export const useHealthCheck = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.healthCheck,
    refetchInterval: 60000, // Check every minute
    staleTime: 30000, // Consider fresh for 30 seconds
  });
};

/**
 * Data Management Hooks
 */
export const useUploadData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.uploadFlightData,
    onSuccess: () => {
      // Invalidate related queries after successful upload
      queryClient.invalidateQueries({ queryKey: queryKeys.currentContext });
      queryClient.invalidateQueries({ queryKey: queryKeys.processingHistory });
      queryClient.invalidateQueries({ queryKey: queryKeys.peakHours });
      queryClient.invalidateQueries({ queryKey: queryKeys.delayCascades });
      queryClient.invalidateQueries({ queryKey: queryKeys.flightStatistics });
    },
  });
};

export const useValidateData = () => {
  return useMutation({
    mutationFn: api.validateFlightData,
  });
};

export const useProcessingHistory = () => {
  return useQuery({
    queryKey: queryKeys.processingHistory,
    queryFn: api.getProcessingHistory,
    staleTime: 30000,
  });
};

export const useCurrentContext = () => {
  return useQuery({
    queryKey: queryKeys.currentContext,
    queryFn: api.getCurrentContext,
    staleTime: 60000,
    refetchInterval: 120000, // Refresh every 2 minutes
  });
};

/**
 * Analytics Hooks
 */
export const usePeakHours = () => {
  return useQuery({
    queryKey: queryKeys.peakHours,
    queryFn: api.getPeakHours,
    staleTime: 300000, // 5 minutes
    refetchInterval: 600000, // Refresh every 10 minutes
  });
};

export const useDelayCascades = () => {
  return useQuery({
    queryKey: queryKeys.delayCascades,
    queryFn: api.getDelayCascades,
    staleTime: 300000,
    refetchInterval: 600000,
  });
};

export const useRoutePerformance = (route?: string) => {
  return useQuery({
    queryKey: queryKeys.routePerformance(route),
    queryFn: () => api.getRoutePerformance(route),
    staleTime: 300000,
    enabled: true, // Always enabled, route is optional
  });
};

export const useFlightStatistics = () => {
  return useQuery({
    queryKey: queryKeys.flightStatistics,
    queryFn: api.getFlightStatistics,
    staleTime: 300000,
    refetchInterval: 600000,
  });
};

export const useEnrichment = (route?: string, airport?: string) => {
  return useQuery({
    queryKey: queryKeys.enrichment(route, airport),
    queryFn: () => api.getEnrichment(route, airport),
    staleTime: 300000,
    enabled: true,
  });
};

export const useRouteOptimization = (route: string, date?: string) => {
  return useQuery({
    queryKey: queryKeys.routeOptimization(route, date),
    queryFn: () => api.optimizeRoute(route, date),
    enabled: !!route, // Only run when route is provided
    staleTime: 300000,
  });
};

/**
 * Prediction and Analysis Hooks
 */
export const usePredictDelay = () => {
  return useMutation({
    mutationFn: ({
      route,
      departureHour,
      date,
      isWeekend,
    }: {
      route: string;
      departureHour: number;
      date?: string;
      isWeekend?: boolean;
    }) => api.predictDelay(route, departureHour, date, isWeekend),
  });
};

export const useAnalyzeScheduleImpact = () => {
  return useMutation({
    mutationFn: ({
      flightId,
      newHour,
      route,
    }: {
      flightId: string;
      newHour: number;
      route: string;
    }) => api.analyzeScheduleImpact(flightId, newHour, route),
  });
};

/**
 * NLP and LLM Hooks
 */
export const useNLPQuery = () => {
  return useMutation({
    mutationFn: api.queryNLP,
  });
};

export const useOllamaQuery = () => {
  return useMutation({
    mutationFn: ({ query, context }: { query: string; context?: any }) =>
      api.queryOllama(query, context),
  });
};

export const useScheduleAnalysisLLM = () => {
  return useMutation({
    mutationFn: ({
      flightId,
      currentHour,
      newHour,
      route,
    }: {
      flightId: string;
      currentHour: number;
      newHour: number;
      route: string;
    }) => api.analyzeScheduleWithLLM(flightId, currentHour, newHour, route),
  });
};

export const useOptimalTimeLLM = () => {
  return useMutation({
    mutationFn: ({ route, date }: { route: string; date?: string }) =>
      api.findOptimalTimeWithLLM(route, date),
  });
};

/**
 * Export Hook
 */
export const useExportAnalysis = () => {
  return useMutation({
    mutationFn: api.exportAnalysisResults,
  });
};

/**
 * Utility Hook for Manual Refetch
 */
export const useRefreshData = () => {
  const queryClient = useQueryClient();
  
  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.peakHours });
    queryClient.invalidateQueries({ queryKey: queryKeys.delayCascades });
    queryClient.invalidateQueries({ queryKey: queryKeys.currentContext });
    queryClient.invalidateQueries({ queryKey: queryKeys.flightStatistics });
  };
  
  const refreshSpecific = (queryKey: string[]) => {
    queryClient.invalidateQueries({ queryKey });
  };
  
  return { refreshAll, refreshSpecific };
};
