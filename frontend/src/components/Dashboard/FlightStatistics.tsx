/**
 * Flight Statistics Overview Component
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { cn } from '../../lib/utils';
import {
  Plane,
  Route,
  Clock,
  TrendingUp,
  TrendingDown,
  CheckCircle,
} from 'lucide-react';
import { useFlightStatistics, useCurrentContext } from '../../hooks/useAPI';

const FlightStatistics: React.FC = () => {
  const { data: statistics, isLoading: statsLoading, error: statsError } = useFlightStatistics();
  const { data: context, isLoading: contextLoading } = useCurrentContext();

  if (statsLoading || contextLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="hover-lift border-border/50 shadow-sm">
            <CardContent className="p-6">
              <div className="space-y-3">
                <div className="animate-pulse bg-muted h-4 w-24 rounded"></div>
                <div className="animate-pulse bg-muted h-8 w-16 rounded"></div>
                <div className="animate-pulse bg-muted h-3 w-20 rounded"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (statsError) {
    return (
      <Alert className="border-destructive">
        <AlertDescription>
          Failed to load flight statistics. Please try refreshing.
        </AlertDescription>
      </Alert>
    );
  }

  if (!statistics && !context) {
    return (
      <Alert>
        <AlertDescription>
          No flight data available. Upload flight data to see statistics.
        </AlertDescription>
      </Alert>
    );
  }

  // Use context data as fallback if statistics not available
  const totalFlights = statistics?.overview?.total_flights || context?.context?.dataset_summary?.total_flights || 0;
  const uniqueRoutes = statistics?.overview?.unique_routes || context?.context?.dataset_summary?.unique_routes || 0;
  const avgDelay = statistics?.overview?.avg_delay_minutes || 0;
  const onTimePercentage = statistics?.overview?.on_time_percentage || 0;

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    subtitle?: string;
    icon: React.ReactNode;
    color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
    trend?: 'up' | 'down' | 'neutral';
  }> = ({ title, value, subtitle, icon, color, trend }) => {


    return (
      <Card className="modern-card-interactive relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-br from-white/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        <CardContent className="p-6 relative z-10">
          <div className="flex items-center justify-between">
            <div className="space-y-2 flex-1">
              <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                {title}
              </p>
              <p className="text-4xl lg:text-5xl font-bold gradient-text">
                {typeof value === 'number' ? value.toLocaleString() : value}
              </p>
              {subtitle && (
                <p className="text-sm text-muted-foreground">
                  {subtitle}
                </p>
              )}
            </div>
            <div className="h-16 w-16 gradient-primary rounded-2xl flex items-center justify-center shadow-lg text-white transform group-hover:scale-110 transition-transform duration-300">
              {icon}
            </div>
          </div>
          {trend && (
            <div className="mt-4 flex items-center gap-1">
              {trend === 'up' && <TrendingUp className="h-4 w-4 text-green-600" />}
              {trend === 'down' && <TrendingDown className="h-4 w-4 text-red-600" />}
              {trend === 'neutral' && <CheckCircle className="h-4 w-4 text-blue-600" />}
              <span className="text-xs text-muted-foreground capitalize">{trend} trend</span>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Main Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Flights"
          value={totalFlights}
          subtitle="in database"
          icon={<Plane className="h-8 w-8" />}
          color="primary"
        />

        <StatCard
          title="Unique Routes"
          value={uniqueRoutes}
          subtitle="active routes"
          icon={<Route className="h-8 w-8" />}
          color="secondary"
        />

        <StatCard
          title="Avg Delay"
          value={`${avgDelay.toFixed(1)} min`}
          subtitle="per flight"
          icon={<Clock className="h-8 w-8" />}
          color={avgDelay > 30 ? 'error' : avgDelay > 15 ? 'warning' : 'success'}
          trend={avgDelay > 30 ? 'up' : avgDelay < 10 ? 'down' : 'neutral'}
        />

        <StatCard
          title="On-Time Rate"
          value={`${onTimePercentage.toFixed(1)}%`}
          subtitle="≤15 min delay"
          icon={<CheckCircle className="h-8 w-8" />}
          color={onTimePercentage > 80 ? 'success' : onTimePercentage > 60 ? 'warning' : 'error'}
          trend={onTimePercentage > 80 ? 'up' : onTimePercentage < 60 ? 'down' : 'neutral'}
        />

      </div>

      {/* Additional Details */}
      {statistics && statistics.temporal_analysis && statistics.route_analysis && statistics.airport_analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Peak Hours */}
          <Card className="hover-lift border-border/50 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2">
                <div className="h-2 w-2 bg-yellow-500 rounded-full"></div>
                Peak Hours
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {statistics.temporal_analysis?.peak_hours?.map((hour) => (
                  <Badge key={hour} variant="outline" className="border-yellow-500 text-yellow-700 bg-yellow-50 hover:bg-yellow-100 transition-colors">
                    {hour}:00
                  </Badge>
                )) || <p className="text-sm text-muted-foreground">No peak hours data available</p>}
              </div>
              {statistics.temporal_analysis?.busiest_day && (
                <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    <span className="font-medium">Busiest day:</span> {statistics.temporal_analysis.busiest_day}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Busiest Routes */}
          <Card className="hover-lift border-border/50 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2">
                <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                Busiest Routes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {statistics.route_analysis?.busiest_routes?.slice(0, 5).map((route, index) => (
                  <div key={index} className="flex justify-between items-center p-3 rounded-lg hover:bg-muted/50 transition-colors">
                    <p className="font-medium text-sm flex-1">
                      {route.route}
                    </p>
                    <div className="flex gap-2 flex-wrap">
                      <Badge variant="outline" className="border-blue-500 text-blue-700 bg-blue-50">
                        {route.flights} flights
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={cn(
                          (route.avg_delay || 0) > 30 ? 'border-red-500 text-red-700 bg-red-50' : 'border-green-500 text-green-700 bg-green-50'
                        )}
                      >
                        {(route.avg_delay || 0).toFixed(1)}m delay
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Most Reliable Routes */}
          <Card>
            <CardHeader>
              <CardTitle>Most Reliable Routes</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {statistics.route_analysis?.most_reliable_routes?.slice(0, 5).map((route, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <p className="font-medium text-sm">
                      {route.route}
                    </p>
                    <div className="flex gap-2">
                      <Badge variant="outline" className="border-green-500 text-green-700">
                        {route.on_time_rate.toFixed(1)}% on-time
                      </Badge>
                      <Badge variant="outline" className="border-blue-500 text-blue-700">
                        {route.consistency.toFixed(1)}% consistent
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Airport Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Busiest Airports</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {statistics.airport_analysis?.busiest_airports?.slice(0, 5).map((airport, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <p className="font-medium text-sm">
                      {airport.airport}
                    </p>
                    <div className="flex gap-2">
                      <Badge variant="outline" className="border-blue-500 text-blue-700">
                        {airport.flights} flights
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={cn(
                          (airport.avg_delay || 0) > 20 ? 'border-red-500 text-red-700' : 'border-green-500 text-green-700'
                        )}
                      >
                        {(airport.avg_delay || 0).toFixed(1)}m
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Context Info */}
      {context && (
        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">
                Last Updated: {context.context?.dataset_summary?.last_updated ? new Date(context.context.dataset_summary.last_updated).toLocaleString() : 'N/A'}
              </Badge>
              {context.database_summary && (
                <Badge variant="outline" className="border-blue-500 text-blue-700">
                  DB Records: {context.database_summary.total_records_in_db?.toLocaleString() || 'N/A'}
                </Badge>
              )}
              <Badge 
                variant="outline" 
                className={cn(
                  context.models_status?.delay_predictor ? 'border-green-500 text-green-700' : 'border-red-500 text-red-700'
                )}
              >
                Delay Predictor: {context.models_status?.delay_predictor ? 'Active' : 'Inactive'}
              </Badge>
              <Badge 
                variant="outline" 
                className={cn(
                  context.models_status?.ollama_llm ? 'border-green-500 text-green-700' : 'border-yellow-500 text-yellow-700'
                )}
              >
                Ollama LLM: {context.models_status?.ollama_llm ? 'Active' : 'Inactive'}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default FlightStatistics;
