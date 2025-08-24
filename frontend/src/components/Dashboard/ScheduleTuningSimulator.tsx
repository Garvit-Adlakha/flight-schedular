import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { useRouteOptimization, useAnalyzeScheduleImpact } from '../../hooks/useAPI';
import { Badge } from '../ui/badge';
import { RefreshCw } from 'lucide-react';

const ScheduleTuningSimulator: React.FC = () => {
  const [route, setRoute] = useState('BOM-DEL');
  const [flightId, setFlightId] = useState('');
  const [newHour, setNewHour] = useState(10);
  const { data: optimization, refetch: refetchOptimization, isFetching } = useRouteOptimization(route);
  const scheduleImpact = useAnalyzeScheduleImpact();
  const [result, setResult] = useState<any>(null);

  const runImpact = () => {
    if (!flightId) return;
    scheduleImpact.mutate({ flightId, newHour, route }, { onSuccess: (data)=> setResult(data) });
  };

  return (
    <Card className="border shadow-sm">
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <CardTitle className="text-base">Schedule Tuning Simulator</CardTitle>
        <div className="flex gap-2 flex-wrap items-center">
          <Input value={route} onChange={e=>setRoute(e.target.value.toUpperCase())} placeholder="Route (e.g. BOM-DEL)" className="h-8 w-32" />
          <Button variant="outline" size="sm" onClick={()=> refetchOptimization()} disabled={isFetching} className="gap-1 h-8"><RefreshCw className={`h-4 w-4 ${isFetching?'animate-spin':''}`} />Suggest</Button>
          <Badge variant="secondary" className="text-[10px]">{optimization?.alternative_times?.length || 0} alternatives</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-3 rounded-md border bg-muted/30">
            <h4 className="text-xs font-semibold mb-2">Optimal Times</h4>
            {optimization && (
              <div className="space-y-2 text-xs">
                <div><strong>Best:</strong> {optimization.best_time_overall?.time_slot} ({optimization.best_time_overall?.predicted_delay}m)</div>
                <div><strong>Reliable:</strong> {optimization.most_reliable_time?.time_slot} ({optimization.most_reliable_time?.predicted_delay}m)</div>
                <div className="mt-2">
                  <strong>Alternatives:</strong>
                  <ul className="list-disc ml-4">
                    {optimization.alternative_times?.slice(0,4).map((t:any,i:number)=>(
                      <li key={i}>{t.time_slot} ({t.predicted_delay}m)</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
            {!optimization && <div className="text-muted-foreground text-xs">No optimization data yet</div>}
          </div>
          <div className="p-3 rounded-md border bg-muted/30 flex flex-col gap-3">
            <h4 className="text-xs font-semibold">Simulate Change</h4>
            <Input value={flightId} onChange={e=>setFlightId(e.target.value)} placeholder="Flight ID" className="h-8" />
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-medium">New Hour</label>
              <select
                value={newHour}
                onChange={e=> setNewHour(parseInt(e.target.value))}
                className="h-8 border rounded px-2 text-xs bg-background"
              >
                {Array.from({length:24}).map((_,h)=>(<option key={h} value={h}>{h}:00</option>))}
              </select>
            </div>
            <Button size="sm" onClick={runImpact} disabled={scheduleImpact.isPending || !flightId}>Simulate</Button>
            {scheduleImpact.isPending && <div className="text-xs text-muted-foreground">Analyzing...</div>}
          </div>
          <div className="p-3 rounded-md border bg-muted/30">
            <h4 className="text-xs font-semibold mb-2">Impact Result</h4>
            {result ? (
              <div className="text-xs space-y-1">
                <div><strong>Flight:</strong> {result.flight_id}</div>
                <div><strong>Change:</strong> {result.original_hour}:00 → {result.new_hour}:00</div>
                <div><strong>Delay Δ:</strong> {result.delay_change_minutes} min</div>
                <div><strong>Probability Δ:</strong> {(result.probability_change*100).toFixed(1)}%</div>
                <div><strong>Recommendation:</strong> {result.recommendation}</div>
              </div>
            ) : <div className="text-muted-foreground text-xs">Run a simulation to see impact.</div> }
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ScheduleTuningSimulator;
