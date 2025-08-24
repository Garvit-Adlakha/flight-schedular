import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';
import { Upload, RefreshCw, Filter } from 'lucide-react';
import FileUpload from '../Upload/FileUpload';
import axios from 'axios';
import { useToast } from '../ui/toast';

interface FlightRecord { [key: string]: any }

const DataBrowser: React.FC = () => {
  const { push } = useToast();
  const [data, setData] = useState<FlightRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [pageSize, setPageSize] = useState(100);
  const [page, setPage] = useState(1);
  const [sortKey, setSortKey] = useState<string>('Flight_Date');
  const [sortDir, setSortDir] = useState<'asc'|'desc'>('desc');
  const [routeFilter, setRouteFilter] = useState<string>('');
  const [selected, setSelected] = useState<FlightRecord | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await axios.get((process.env.REACT_APP_API_URL || 'http://localhost:8000') + '/api/raw-data');
      setData(res.data.records || res.data || []);
      push({ type:'success', message:'Dataset refreshed', title:'Reloaded' });
    } catch (e:any) {
      setError(e.message || 'Failed to load data');
      push({ type:'error', message: e.message || 'Failed to load data', title:'Error' });
    } finally { setLoading(false); }
  }, [push]);

  // Load persisted preferences
  useEffect(() => {
    try {
      const raw = localStorage.getItem('dataBrowserPrefs');
      if (raw) {
        const prefs = JSON.parse(raw);
        if (prefs.filter) setFilter(prefs.filter);
        if (prefs.pageSize) setPageSize(prefs.pageSize);
        if (prefs.sortKey) setSortKey(prefs.sortKey);
        if (prefs.sortDir) setSortDir(prefs.sortDir);
        if (prefs.routeFilter) setRouteFilter(prefs.routeFilter);
      }
    } catch {}
    fetchData();
  }, [fetchData]);

  // Persist preferences
  useEffect(() => {
    const prefs = { filter, pageSize, sortKey, sortDir, routeFilter };
    try { localStorage.setItem('dataBrowserPrefs', JSON.stringify(prefs)); } catch {}
  }, [filter, pageSize, sortKey, sortDir, routeFilter]);


  const keys = data[0] ? Object.keys(data[0]) : [];
  const filtered = useMemo(()=>{
    let rows = data;
    if (filter) rows = rows.filter(r => JSON.stringify(r).toLowerCase().includes(filter.toLowerCase()));
    if (routeFilter) rows = rows.filter(r => (r.Route||r.route||'').toLowerCase() === routeFilter.toLowerCase());
    return rows;
  }, [data, filter, routeFilter]);
  const sorted = useMemo(()=>{
    if (!sortKey) return filtered;
    return [...filtered].sort((a,b)=>{
      const av = a[sortKey]; const bv = b[sortKey];
      if (av==null && bv==null) return 0; if (av==null) return 1; if (bv==null) return -1;
      if (av < bv) return sortDir==='asc'?-1:1;
      if (av > bv) return sortDir==='asc'?1:-1;
      return 0;
    });
  }, [filtered, sortKey, sortDir]);
  const uniqueRoutes = useMemo(()=> Array.from(new Set(data.map(r=> r.Route || r.route).filter(Boolean))).slice(0,100), [data]);
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  const pageRows = sorted.slice(start, start + pageSize);

  const toggleSort = (k:string)=> {
    if (sortKey === k) {
      setSortDir(d => d==='asc'?'desc':'asc');
    } else { setSortKey(k); setSortDir('asc'); }
  };

  const exportCSV = ()=> {
    const header = keys.join(',');
    const rows = sorted.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','));
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = 'flights_filtered.csv'; a.click(); URL.revokeObjectURL(url);
    push({ type:'success', title:'Exported', message:`Saved ${sorted.length} rows` });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <CardTitle className="text-lg flex items-center gap-3">Dataset Preview
              {routeFilter && <Badge variant="outline" className="cursor-pointer" onClick={()=> setRouteFilter('')}>{routeFilter} ×</Badge>}
            </CardTitle>
            <p className="text-xs text-muted-foreground">Page {safePage}/{totalPages} • {pageRows.length} rows (Filtered {sorted.length} / Total {data.length}) • Sorted by {sortKey} {sortDir==='asc'?'↑':'↓'}</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Input value={filter} onChange={e=>{setFilter(e.target.value); setPage(1);}} placeholder="Search..." className="h-8 w-48" />
            </div>
            <select value={routeFilter} onChange={e=> {setRouteFilter(e.target.value); setPage(1);}} className="h-8 border rounded px-2 text-xs bg-background">
              <option value="">All Routes</option>
              {uniqueRoutes.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            <Button variant="outline" size="sm" onClick={fetchData} disabled={loading} className="gap-1">
              <RefreshCw className={"h-4 w-4" + (loading? ' animate-spin':'')} /> Refresh
            </Button>
            <Button variant="outline" size="sm" onClick={exportCSV}>Export CSV</Button>
            <Badge variant="secondary" className="text-xs">{data.length} rows</Badge>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {error && <Alert className="m-4"><AlertDescription>{error}</AlertDescription></Alert>}
          <div className="overflow-auto max-h-[55vh] text-xs relative">
            {loading && (
              <div className="absolute inset-0 flex flex-col gap-2 p-6 bg-background/70 backdrop-blur-sm">
                {Array.from({length:8}).map((_,i)=>(<div key={i} className="h-4 bg-muted rounded animate-pulse" />))}
              </div>
            )}
            <table className="w-full border-collapse">
              <thead className="sticky top-0 bg-background shadow">
                <tr>
                  {keys.map(k => <th key={k} onClick={()=> toggleSort(k)} className="text-left px-3 py-2 border-b font-medium whitespace-nowrap cursor-pointer select-none hover:bg-muted/40">{k}{sortKey===k && (sortDir==='asc'?' ↑':' ↓')}</th>)}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row, idx) => (
                  <tr key={start + idx} className="hover:bg-muted/50 cursor-pointer" onClick={()=> setSelected(row)}>
                    {keys.map(k => <td key={k} className="px-3 py-1 border-b whitespace-nowrap">{String(row[k])}</td>)}
                  </tr>
                ))}
                {!loading && pageRows.length === 0 && (
                  <tr><td colSpan={keys.length} className="text-center text-muted-foreground py-10">No rows match current filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="p-3 flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Rows per page:</span>
              <select
                value={pageSize}
                onChange={e=>{setPageSize(parseInt(e.target.value)); setPage(1);}}
                className="h-7 border rounded px-2 bg-background text-xs"
              >
                {[50,100,200,500].map(sz => <option key={sz} value={sz}>{sz}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2 justify-center">
              <Button variant="outline" size="sm" onClick={()=> setPage(1)} disabled={safePage===1}>{'<<'}</Button>
              <Button variant="outline" size="sm" onClick={()=> setPage(p => Math.max(1, p-1))} disabled={safePage===1}>{'<'}</Button>
              <span className="text-xs font-medium px-2">{safePage} / {totalPages}</span>
              <Button variant="outline" size="sm" onClick={()=> setPage(p => Math.min(totalPages, p+1))} disabled={safePage===totalPages}>{'>'}</Button>
              <Button variant="outline" size="sm" onClick={()=> setPage(totalPages)} disabled={safePage===totalPages}>{'>>'}</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2"><Upload className="h-4 w-4" /> Upload / Replace Data</CardTitle>
        </CardHeader>
        <CardContent>
          <FileUpload />
        </CardContent>
      </Card>
      {selected && (
        <div className="fixed top-0 right-0 h-full w-[360px] sm:w-[420px] bg-background border-l shadow-xl z-50 flex flex-col">
          <div className="p-3 flex items-center justify-between border-b">
            <div className="text-sm font-medium">Flight Detail</div>
            <button className="text-xs text-muted-foreground hover:text-foreground" onClick={()=> setSelected(null)}>Close</button>
          </div>
          <div className="flex-1 overflow-auto p-3 space-y-2 text-xs">
            {Object.entries(selected).map(([k,v]) => (
              <div key={k} className="flex justify-between gap-2">
                <span className="font-medium truncate max-w-[40%]" title={k}>{k}</span>
                <span className="text-right flex-1 break-all" title={String(v)}>{String(v)}</span>
              </div>
            ))}
          </div>
          <div className="p-3 border-t flex gap-2">
            <Button size="sm" variant="outline" className="flex-1" onClick={()=> {navigator.clipboard.writeText(JSON.stringify(selected,null,2)); push({type:'success', title:'Copied', message:'Flight JSON copied'});}}>Copy JSON</Button>
            <Button size="sm" className="flex-1" onClick={()=> setSelected(null)}>Done</Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataBrowser;
