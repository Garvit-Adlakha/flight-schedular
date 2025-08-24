import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { RefreshCw } from 'lucide-react';
import * as api from '../../services/api';
import { forceSimulation, forceManyBody, forceCenter, forceLink, forceCollide } from 'd3-force';

interface GraphData {
  nodes: Array<{ id: string; flight_number: string; route: string; dep_delay: number; arr_delay: number }>;
  edges: Array<{ source: string; target: string; cascade_risk: string; turnaround: number }>;
}

const riskColor = (risk?: string) => {
  switch (risk) {
    case 'high': return '#dc2626';
    case 'medium': return '#f59e0b';
    case 'low': return '#16a34a';
    default: return '#64748b';
  }
};

const CascadeNetworkGraph: React.FC = () => {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [limit, setLimit] = useState(120);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<{x:number;y:number;content:string}|null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getCascadeNetwork(limit);
      setData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(()=> { fetchData(); }, [fetchData]);

  // Run force simulation when data changes
  useEffect(() => {
    if (!data || !svgRef.current) return;
    const nodes: any[] = data.nodes.map(n => ({ ...n }));
    const nodeById: Record<string, any> = {};
    nodes.forEach(n => { nodeById[n.id] = n; });
    const links: any[] = data.edges.map(e => ({ ...e, source: nodeById[e.source], target: nodeById[e.target] }));
    const sim = forceSimulation(nodes)
      .force('charge', forceManyBody().strength(-40))
      .force('center', forceCenter(600, 190))
      .force('link', forceLink(links).id((d: any) => d.id).distance(60).strength(0.6))
      .force('collide', forceCollide(18));
    let frame = 0;
    const ticked = () => {
      frame++;
      if (frame > 300) sim.stop();
      const svg = svgRef.current;
      if (!svg) return;
      const gLinks = svg.querySelectorAll('[data-link]');
      links.forEach((l, i) => {
        const el = gLinks[i] as SVGLineElement;
        if (!el) return;
        el.setAttribute('x1', String(l.source.x));
        el.setAttribute('y1', String(l.source.y));
        el.setAttribute('x2', String(l.target.x));
        el.setAttribute('y2', String(l.target.y));
      });
      const gNodes = svg.querySelectorAll('[data-node]');
      nodes.forEach((n, i) => {
        const el = gNodes[i] as SVGCircleElement;
        if (!el) return;
        el.setAttribute('cx', String(n.x));
        el.setAttribute('cy', String(n.y));
      });
    };
    sim.on('tick', ticked);
    return () => { sim.stop(); };
  }, [data]);

  return (
    <Card className="border shadow-sm">
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <CardTitle className="text-base">Cascading Delay Network</CardTitle>
        <div className="flex items-center gap-2 flex-wrap">
          <select value={limit} onChange={e=> setLimit(parseInt(e.target.value))} className="h-8 border rounded px-2 text-xs">
            {[60,120,200,300].map(l=> <option key={l} value={l}>{l} nodes</option>)}
          </select>
          <Button size="sm" variant="outline" onClick={fetchData} disabled={loading} className="gap-1 h-8">
            <RefreshCw className={`h-4 w-4 ${loading?'animate-spin':''}`} /> Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {!data && <div className="text-xs text-muted-foreground">Loading network...</div>}
        {data && (
          <div className="relative w-full overflow-auto border rounded-md p-2 bg-muted/30" style={{ height: 400 }}>
            <svg ref={svgRef} width={1200} height={380}>
              {data.edges.map((e, idx) => (
                <line data-link key={idx} stroke={riskColor(e.cascade_risk)} strokeWidth={ e.cascade_risk==='high'?2.2:1 } opacity={0.55} />
              ))}
              {data.nodes.map((n) => (
                <circle
                  data-node
                  key={n.id}
                  r={8}
                  fill={riskColor()}
                  stroke="#fff"
                  strokeWidth={1.3}
                  onMouseEnter={(ev)=> setHover({ x: ev.clientX, y: ev.clientY, content: `${n.flight_number}\n${n.route}\nDep ${n.dep_delay}m Arr ${n.arr_delay}m` })}
                  onMouseLeave={()=> setHover(null)}
                />
              ))}
            </svg>
            {hover && (
              <div className="pointer-events-none absolute z-10 text-[10px] whitespace-pre rounded bg-background/90 backdrop-blur px-2 py-1 border shadow" style={{ left: hover.x - 260, top: hover.y - 120 }}>
                {hover.content}
              </div>
            )}
            <div className="absolute top-2 right-2 text-[10px] bg-background/80 backdrop-blur px-2 py-1 rounded border">Nodes: {data.nodes.length} • Edges: {data.edges.length}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CascadeNetworkGraph;
