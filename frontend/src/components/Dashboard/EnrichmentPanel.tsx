import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { useEnrichment } from '../../hooks/useAPI';
import { RefreshCw } from 'lucide-react';
import { Button } from '../ui/button';

const EnrichmentPanel: React.FC = () => {
  const [route, setRoute] = useState<string>('');
  const [airport, setAirport] = useState<string>('');
  const { data, isLoading, refetch } = useEnrichment(route || undefined, airport || undefined);

  const avgTemp = data?.records?.length ? (data.records.filter(r=> r.temperature_c!=null).reduce((a,r)=> a + (r.temperature_c||0), 0) / data.records.filter(r=> r.temperature_c!=null).length).toFixed(1) : null;
  const capacityDist: Record<string, number> = data?.records?.reduce((acc:Record<string,number>,r:any)=> { const key = r.capacity_status || 'unknown'; acc[key]=(acc[key]||0)+1; return acc; }, {} as Record<string,number>) || {};
  const capBadges: Array<[string, number]> = Object.entries(capacityDist).sort((a:[string,number],b:[string,number])=> b[1]-a[1]);

  const capacityColor = (status?:string) => {
    switch(status){
      case 'critical': return 'bg-red-500/80 text-white';
      case 'high': return 'bg-orange-500/80 text-white';
      case 'normal': return 'bg-green-500/80 text-white';
      case 'low': return 'bg-emerald-500/80 text-white';
      default: return 'bg-slate-500/60 text-white';
    }
  };

  return (
    <Card className="border shadow-sm">
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <CardTitle className="text-base">Weather & Capacity Enrichment</CardTitle>
          <p className="text-xs text-muted-foreground">Heuristic / external data associated with recent flights</p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <Input placeholder="Route (e.g. BOM-DEL)" value={route} onChange={e=>setRoute(e.target.value.toUpperCase())} className="h-8 w-32" />
          <Input placeholder="Airport" value={airport} onChange={e=>setAirport(e.target.value.toUpperCase())} className="h-8 w-28" />
          <Button variant="outline" size="sm" onClick={()=>refetch()} disabled={isLoading} className="gap-1 h-8"><RefreshCw className={`h-4 w-4 ${isLoading?'animate-spin':''}`} /> Refresh</Button>
          <Badge variant="secondary" className="text-[10px]">{data?.count || 0} records</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {data?.records?.length ? (
          <div className="flex flex-wrap gap-2 px-4 py-2 text-[10px] border-b bg-muted/30">
            {avgTemp && <div className="flex items-center gap-1"><span className="font-medium">Avg Temp:</span>{avgTemp}°C</div>}
            {capBadges.map(([k,v]) => (
              <span key={k} className={`px-2 py-0.5 rounded-md capitalize ${capacityColor(k)}`}>{k}: {v}</span>
            ))}
          </div>
        ) : null}
        <div className="overflow-auto max-h-[40vh] text-xs">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-background shadow">
              <tr>
                {['flight_id','airport','temperature_c','wind_speed_kts','visibility_km','weather_condition','runway_capacity_estimate','capacity_status','timestamp'].map(k=> <th key={k} className="text-left px-3 py-2 border-b whitespace-nowrap">{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {data?.records?.map((r,i)=> (
                <tr key={i} className="hover:bg-muted/50">
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.flight_id}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.airport}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.temperature_c ?? ''}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.wind_speed_kts ?? ''}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.visibility_km ?? ''}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.weather_condition ?? ''}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{r.runway_capacity_estimate ?? ''}</td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">
                    {r.capacity_status && (
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium capitalize ${capacityColor(r.capacity_status)}`}>{r.capacity_status}</span>
                    )}
                  </td>
                  <td className="px-3 py-1 border-b whitespace-nowrap">{new Date(r.timestamp).toLocaleString()}</td>
                </tr>
              ))}
              {!isLoading && (data?.records?.length ?? 0) === 0 && (
                <tr><td colSpan={9} className="text-center text-muted-foreground py-6">No enrichment data yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default EnrichmentPanel;
