/**
 * Peak Hours Congestion Visualization Component
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { cn } from '../../lib/utils';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { usePeakHours } from '../../hooks/useAPI';

const PeakHoursChart: React.FC = () => {
  const { data: peakHours, isLoading, error } = usePeakHours();

  // Color scheme for congestion levels
  const getBarColor = (congestionScore: number) => {
    if (congestionScore >= 80) return '#f44336'; // Red - High congestion
    if (congestionScore >= 60) return '#ff9800'; // Orange - Moderate
    if (congestionScore >= 40) return '#ffeb3b'; // Yellow - Low-moderate
    return '#4caf50'; // Green - Low congestion
  };

  const getRecommendationIcon = (recommendation: string) => {
    if (recommendation.includes('AVOID')) return <AlertTriangle className="h-4 w-4 text-red-600" />;
    if (recommendation.includes('CAUTION')) return <TrendingUp className="h-4 w-4 text-yellow-600" />;
    return <TrendingDown className="h-4 w-4 text-green-600" />;
  };

  const chartData = peakHours?.map(hour => ({
    hour: `${hour.hour}:00`,
    congestion: hour.congestion_score,
    flights: hour.flight_count,
    avgDelay: hour.avg_delay_minutes,
    delayRate: hour.delay_rate,
    recommendation: hour.recommendation,
  })) || [];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card p-3 border border-border rounded-lg shadow-lg">
          <p className="font-semibold text-sm">
            Hour: {label}
          </p>
          <p className="text-sm text-muted-foreground">
            Flights: {data.flights}
          </p>
          <p className="text-sm text-muted-foreground">
            Avg Delay: {data.avgDelay.toFixed(1)} min
          </p>
          <p className="text-sm text-muted-foreground">
            Delay Rate: {data.delayRate.toFixed(1)}%
          </p>
          <p className="text-sm text-muted-foreground">
            Congestion: {data.congestion.toFixed(1)}
          </p>
          <Badge 
            variant="outline" 
            className={cn(
              "mt-2",
              data.recommendation.includes('AVOID') ? 'border-red-500 text-red-700' :
              data.recommendation.includes('CAUTION') ? 'border-yellow-500 text-yellow-700' : 'border-green-500 text-green-700'
            )}
          >
            {data.recommendation}
          </Badge>
        </div>
      );
    }
    return null;
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Peak Hours Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse bg-muted h-96 rounded"></div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Peak Hours Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert className="border-destructive">
            <AlertDescription>
              Failed to load peak hours data. Please try refreshing.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!peakHours || peakHours.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Peak Hours Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              No peak hours data available. Upload flight data to see analysis.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const highCongestionHours = peakHours.filter(h => h.congestion_score >= 80);
  const recommendedHours = peakHours.filter(h => h.recommendation.includes('RECOMMENDED'));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Peak Hours Congestion Analysis</CardTitle>
        <p className="text-sm text-muted-foreground">Flight volume and delay patterns by hour</p>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="mb-6 flex gap-2 flex-wrap">
          <Badge variant="outline" className="border-red-500 text-red-700">
            <AlertTriangle className="mr-1 h-3 w-3" />
            {highCongestionHours.length} High Congestion Hours
          </Badge>
          <Badge variant="outline" className="border-green-500 text-green-700">
            <TrendingDown className="mr-1 h-3 w-3" />
            {recommendedHours.length} Recommended Hours
          </Badge>
          <Badge variant="outline">
            Peak: {Math.max(...peakHours.map(h => h.congestion_score)).toFixed(1)} congestion
          </Badge>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="hour" 
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis yAxisId="left" orientation="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            <Bar
              yAxisId="left"
              dataKey="congestion"
              name="Congestion Score"
              radius={[4, 4, 0, 0]}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.congestion)} />
              ))}
            </Bar>
            
            <Bar
              yAxisId="right"
              dataKey="flights"
              name="Flight Count"
              fill="#1976d2"
              opacity={0.7}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>

        {/* Recommendations */}
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">
            Recommendations
          </h3>
          <div className="space-y-2">
            {peakHours.slice(0, 5).map((hour, index) => (
              <div key={index} className="flex items-center gap-2">
                {getRecommendationIcon(hour.recommendation)}
                <p className="text-sm">
                  <strong>{hour.hour}:00</strong> - {hour.recommendation}
                  {hour.congestion_score >= 80 && (
                    <span className="text-red-600">
                      {' '}(Score: {hour.congestion_score.toFixed(1)})
                    </span>
                  )}
                </p>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PeakHoursChart;
