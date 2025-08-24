/**
 * Project Analysis Component - Interface for all project expectations
 * Provides specialized interfaces for:
 * 1. Optimal takeoff/landing times (scheduled vs actual analysis)
 * 2. Airport congestion analysis
 * 3. Schedule tuning impact modeling
 * 4. Cascading delay flight isolation
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';
import { 
  Clock, 
  AlertTriangle, 
  Settings, 
  Network, 
  Plane, 
  BarChart3,
  Info,
  Zap
} from 'lucide-react';
import { useOllamaQuery, useCurrentContext } from '../../hooks/useAPI';

interface ProjectAnalysisResult {
  query: string;
  response: string;
  model_used: string;
  success: boolean;
  timestamp: string;
  analysis_type: string;
}

const ProjectAnalysis: React.FC = () => {
  const [activeAnalysis, setActiveAnalysis] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, ProjectAnalysisResult>>({});
  
  // Form states for different analysis types
  const [optimalTimingForm, setOptimalTimingForm] = useState({
    route: 'BOM-DEL',
    date: '2025-07-26'
  });
  
  const [congestionForm, setCongestionForm] = useState({
    date: '2025-07-26'
  });
  
  const [scheduleTuningForm, setScheduleTuningForm] = useState({
    flight_id: 'AI2509',
    current_hour: 6,
    new_hour: 8,
    route: 'BOM-DEL'
  });
  
  const [cascadingForm, setCascadingForm] = useState({
    date: '2025-07-26'
  });

  const ollamaMutation = useOllamaQuery();
  const { data: contextData } = useCurrentContext();

  const analysisTypes = [
    {
      id: 'optimal_timing',
      title: 'Optimal Takeoff/Landing Times',
      description: 'Find best times using scheduled vs actual analysis',
      icon: Clock,
      color: 'bg-blue-500',
      models: ['RandomForestRegressor', 'RandomForestClassifier', 'Ollama LLM'],
    },
    {
      id: 'congestion_analysis',
      title: 'Airport Congestion Analysis',
      description: 'Identify busiest time slots to avoid',
      icon: AlertTriangle,
      color: 'bg-orange-500',
      models: ['Peak Hour Analysis', 'Congestion Scoring', 'Ollama LLM'],
    },
    {
      id: 'schedule_tuning',
      title: 'Schedule Tuning Impact',
      description: 'Model impact of schedule changes on delays',
      icon: Settings,
      color: 'bg-green-500',
      models: ['RandomForestRegressor', 'Impact Simulation', 'Ollama LLM'],
    },
    {
      id: 'cascading_analysis',
      title: 'Cascading Delay Isolation',
      description: 'Isolate flights with biggest cascading impact',
      icon: Network,
      color: 'bg-purple-500',
      models: ['NetworkX Graph Analysis', 'Dependency Networks', 'Ollama LLM'],
    },
  ];

  const executeAnalysis = async (analysisType: string) => {
    setActiveAnalysis(analysisType);
    
    let query = '';
    let specificContext = {};
    
    switch (analysisType) {
      case 'optimal_timing':
        query = `Find the best takeoff and landing times for route ${optimalTimingForm.route} on ${optimalTimingForm.date}. Use scheduled vs actual time analysis to identify optimal departure windows with lowest delay probability.`;
        specificContext = {
          analysis_type: 'optimal_timing',
          target_route: optimalTimingForm.route,
          target_date: optimalTimingForm.date,
          requirements: 'scheduled_vs_actual_analysis'
        };
        break;
        
      case 'congestion_analysis':
        query = `Identify the busiest time slots at Mumbai Airport that should be avoided for flight scheduling on ${congestionForm.date}. Provide specific hours to avoid with alternative recommendations.`;
        specificContext = {
          analysis_type: 'congestion_analysis',
          target_date: congestionForm.date,
          requirements: 'busy_periods_identification'
        };
        break;
        
      case 'schedule_tuning':
        query = `Model the impact of tuning flight schedule time for flight ${scheduleTuningForm.flight_id} on route ${scheduleTuningForm.route} from ${scheduleTuningForm.current_hour}:00 to ${scheduleTuningForm.new_hour}:00. Provide specific delay impact projections.`;
        specificContext = {
          analysis_type: 'schedule_tuning',
          flight_id: scheduleTuningForm.flight_id,
          route: scheduleTuningForm.route,
          current_hour: scheduleTuningForm.current_hour,
          new_hour: scheduleTuningForm.new_hour,
          requirements: 'impact_modeling'
        };
        break;
        
      case 'cascading_analysis':
        query = `Isolate flights that have the biggest cascading impact on schedule delays for ${cascadingForm.date}. Provide ranked list of high-impact flights with specific intervention strategies.`;
        specificContext = {
          analysis_type: 'cascading_analysis',
          target_date: cascadingForm.date,
          requirements: 'cascade_impact_isolation'
        };
        break;
    }
    
    try {
      ollamaMutation.mutate(
        { 
          query, 
          context: { ...contextData?.context, ...specificContext }
        },
        {
          onSuccess: (response) => {
            const result: ProjectAnalysisResult = {
              query,
              response: response?.response || 'No response received',
              model_used: response?.model_used || 'Unknown',
              success: true,
              timestamp: new Date().toISOString(),
              analysis_type: analysisType
            };
            setResults(prev => ({ ...prev, [analysisType]: result }));
            setActiveAnalysis(null);
          },
          onError: (error) => {
            const errorResult: ProjectAnalysisResult = {
              query,
              response: `Error: ${error.message}`,
              model_used: 'Error',
              success: false,
              timestamp: new Date().toISOString(),
              analysis_type: analysisType
            };
            setResults(prev => ({ ...prev, [analysisType]: errorResult }));
            setActiveAnalysis(null);
          },
        }
      );
    } catch (error) {
      console.error('Analysis error:', error);
      setActiveAnalysis(null);
    }
  };

  const renderAnalysisForm = (analysisType: string) => {
    switch (analysisType) {
      case 'optimal_timing':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="route">Route</Label>
              <Input
                id="route"
                value={optimalTimingForm.route}
                onChange={(e) => setOptimalTimingForm(prev => ({ ...prev, route: e.target.value }))}
                placeholder="e.g., BOM-DEL"
              />
            </div>
            <div>
              <Label htmlFor="date">Analysis Date</Label>
              <Input
                id="date"
                type="date"
                value={optimalTimingForm.date}
                onChange={(e) => setOptimalTimingForm(prev => ({ ...prev, date: e.target.value }))}
              />
            </div>
          </div>
        );
        
      case 'congestion_analysis':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="congestion-date">Analysis Date</Label>
              <Input
                id="congestion-date"
                type="date"
                value={congestionForm.date}
                onChange={(e) => setCongestionForm(prev => ({ ...prev, date: e.target.value }))}
              />
            </div>
          </div>
        );
        
      case 'schedule_tuning':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="flight-id">Flight ID</Label>
              <Input
                id="flight-id"
                value={scheduleTuningForm.flight_id}
                onChange={(e) => setScheduleTuningForm(prev => ({ ...prev, flight_id: e.target.value }))}
                placeholder="e.g., AI2509"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="current-hour">Current Hour</Label>
                <Input
                  id="current-hour"
                  type="number"
                  min="0"
                  max="23"
                  value={scheduleTuningForm.current_hour}
                  onChange={(e) => setScheduleTuningForm(prev => ({ ...prev, current_hour: parseInt(e.target.value) }))}
                />
              </div>
              <div>
                <Label htmlFor="new-hour">New Hour</Label>
                <Input
                  id="new-hour"
                  type="number"
                  min="0"
                  max="23"
                  value={scheduleTuningForm.new_hour}
                  onChange={(e) => setScheduleTuningForm(prev => ({ ...prev, new_hour: parseInt(e.target.value) }))}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="tuning-route">Route</Label>
              <Input
                id="tuning-route"
                value={scheduleTuningForm.route}
                onChange={(e) => setScheduleTuningForm(prev => ({ ...prev, route: e.target.value }))}
                placeholder="e.g., BOM-DEL"
              />
            </div>
          </div>
        );
        
      case 'cascading_analysis':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="cascade-date">Analysis Date</Label>
              <Input
                id="cascade-date"
                type="date"
                value={cascadingForm.date}
                onChange={(e) => setCascadingForm(prev => ({ ...prev, date: e.target.value }))}
              />
            </div>
          </div>
        );
        
      default:
        return null;
    }
  };

  const formatResponse = (response: string): string => {
    if (!response) return 'No response available';
    
    // Simple formatting - just clean up the text
    return response
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .trim();
  };
  // Parse new concise backend response formats
  interface ParsedResponse {
    sections: Record<string, string[]>;
    raw: string;
  }

  const parseConciseResponse = (response: string): ParsedResponse => {
    const lines = response.split('\n').map(l => l.trim()).filter(Boolean);
    const sections: Record<string, string[]> = {};
    let current = 'GENERAL';
    const headerRegex = /^(PRIMARY SLOT|SECONDARY SLOT|TERTIARY SLOT|TIMES TO AVOID|PEAK CONGESTION HOURS|RECOMMENDED ALTERNATIVES|SUMMARY|NOTES|SCHEDULE CHANGE|CURRENT|PROPOSED|IMPACT|RECOMMENDATION|TOP CASCADING FLIGHTS|HIGHEST RISK|MONITOR)$/i;
    lines.forEach(line => {
      const headerMatch = line.match(/^([A-Z ]+):/i); // capture token before first colon
      if (headerMatch) {
        const candidate = headerMatch[1].toUpperCase().trim();
        if (headerRegex.test(candidate)) {
          current = candidate; // switch section
          if (!sections[current]) sections[current] = [];
          const remainder = line.substring(headerMatch[0].length).trim();
          if (remainder) sections[current].push(remainder); // inline content after header
          return;
        }
      }
      if (!sections[current]) sections[current] = [];
      sections[current].push(line);
    });
    return { sections, raw: response };
  };

  const renderKeyValueBadges = (text: string) => {
    // Split by | and key=value pairs
    const parts = text.split('|').map(p => p.trim());
    return (
      <div className="flex flex-wrap gap-2 mt-1">
        {parts.slice(1).map((p, i) => (
          <span key={i} className="text-[10px] bg-muted px-2 py-0.5 rounded border">
            {p}
          </span>
        ))}
      </div>
    );
  };

  const renderSection = (title: string, items: string[]) => {
    if (!items || items.length === 0) return null;
    const colorMap: Record<string,string> = {
      'PRIMARY SLOT': 'border-green-500',
      'SECONDARY SLOT': 'border-blue-500',
      'TERTIARY SLOT': 'border-indigo-500',
      'TIMES TO AVOID': 'border-red-500',
      'PEAK CONGESTION HOURS': 'border-red-500',
      'RECOMMENDED ALTERNATIVES': 'border-green-500',
      'SUMMARY': 'border-slate-500',
      'SCHEDULE CHANGE': 'border-amber-500',
      'CURRENT': 'border-slate-400',
      'PROPOSED': 'border-slate-400',
      'IMPACT': 'border-purple-500',
      'RECOMMENDATION': 'border-emerald-500',
      'TOP CASCADING FLIGHTS': 'border-rose-500',
      'HIGHEST RISK': 'border-rose-600',
      'MONITOR': 'border-fuchsia-500',
    };
    const border = colorMap[title] || 'border-gray-300';

    // Specialized rendering
    if (['PRIMARY SLOT','SECONDARY SLOT','TERTIARY SLOT'].includes(title)) {
      const line = items[0] || '';
      return (
        <div key={title} className={`p-3 rounded-lg border ${border} bg-background`}> 
          <div className="text-xs font-semibold tracking-wide text-muted-foreground">{title}</div>
          <div className="font-medium mt-1 text-sm">{line.split('|')[0]}</div>
          {renderKeyValueBadges(line)}
        </div>
      );
    }

    if (title === 'TIMES TO AVOID') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border} bg-red-50/40`}> 
          <div className="text-xs font-semibold text-red-700">{title}</div>
          <ul className="mt-1 space-y-1 text-xs">
            {items.map((l,i)=>(<li key={i} className="flex gap-1"><span className="text-red-500">•</span><span>{l.replace(/^[-*]\s*/,'')}</span></li>))}
          </ul>
        </div>
      );
    }

    if (title === 'PEAK CONGESTION HOURS' || title === 'RECOMMENDED ALTERNATIVES') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border}`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <div className="mt-2 space-y-1 text-xs">
            {items.map((l,i)=>(<div key={i} className="flex items-start gap-2"><span className="text-muted-foreground">{i+1}.</span><span className="font-mono">{l}</span></div>))}
          </div>
        </div>
      );
    }

    if (title === 'TOP CASCADING FLIGHTS') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border}`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <ol className="mt-2 space-y-1 text-xs list-decimal ml-4">
            {items.map((l,i)=>(<li key={i} className="font-mono">{l.replace(/^\d+\.\s*/,'')}</li>))}
          </ol>
        </div>
      );
    }

    if (title === 'MONITOR') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border}`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <div className="mt-1 text-xs space-y-1 font-mono">
            {items.map((l,i)=>(<div key={i}>{l}</div>))}
          </div>
        </div>
      );
    }

    if (title === 'IMPACT') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border}`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <div className="mt-1 text-xs font-mono">{items[0]}</div>
        </div>
      );
    }

    if (title === 'RECOMMENDATION') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border} bg-emerald-50/40`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <div className="mt-1 text-xs font-medium">{items[0]}</div>
        </div>
      );
    }

    if (title === 'SUMMARY' || title === 'HIGHEST RISK' || title === 'SCHEDULE CHANGE' || title === 'CURRENT' || title === 'PROPOSED') {
      return (
        <div key={title} className={`p-3 rounded-lg border ${border}`}> 
          <div className="text-xs font-semibold tracking-wide">{title}</div>
          <div className="mt-1 text-xs font-mono space-y-1">{items.map((l,i)=>(<div key={i}>{l}</div>))}</div>
        </div>
      );
    }


    // Fallback generic section
    return (
      <div key={title} className={`p-3 rounded-lg border ${border}`}> 
        <div className="text-xs font-semibold tracking-wide">{title}</div>
        <div className="mt-1 text-xs space-y-1">{items.map((l,i)=>(<div key={i}>{l}</div>))}</div>
      </div>
    );
  };

  const renderFormattedResponse = (response: string) => {
    if (!response) return <div className="text-muted-foreground">No response available</div>;
    // Specialized schedule tuning JSON renderer
    const tryRenderScheduleTuning = (raw: string) => {
      if (!/SCHEDULE_CHANGE/i.test(raw)) return null;
      const start = raw.indexOf('{');
      const end = raw.lastIndexOf('}');
      if (start === -1 || end === -1) return null;
      let jsonPart = raw.slice(start, end + 1).trim();
      // Remove trailing comments or backticks
      jsonPart = jsonPart.replace(/```/g, '').trim();
      try {
        const data = JSON.parse(jsonPart);
        const schedule = data.SCHEDULE_CHANGE || data.schedule_change;
        const current = data.CURRENT || data.current;
        const proposed = data.PROPOSED || data.proposed;
        const impact = data.IMPACT || data.impact;
        const recommendation = data.RECOMMENDATION || data.recommendation;
        const notes = (data.NOTES || data.notes || '').split(/\.?\s+/).filter(Boolean);
        if (!schedule || !current || !proposed) return null;
        const depDelta = (proposed.avg_dep_delay_min ?? 0) - (current.avg_dep_delay_min ?? 0);
        const arrDelta = (proposed.avg_arr_delay_min ?? 0) - (current.avg_arr_delay_min ?? 0);
        const classify = (val: number) => val < -0.1 ? 'Improved' : val > 0.1 ? 'Worse' : 'Neutral';
        const depClass = classify(depDelta);
        const arrClass = classify(arrDelta);
        const badgeColor = (cls: string) => cls === 'Improved' ? 'bg-green-100 text-green-700 border-green-300' : cls === 'Worse' ? 'bg-red-100 text-red-700 border-red-300' : 'bg-yellow-50 text-yellow-700 border-yellow-300';
        const pctVal = (v: number | undefined) => (typeof v === 'number' ? `${v > 0 ? '+' : ''}${v.toFixed(2)}` : 'NA');
        const impactDepRate = impact?.dep_delay_rate_pct;
        const impactArrRate = impact?.arr_delay_rate_pct;
        return (
          <div className="space-y-4">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
              <div className="font-semibold text-sm">Schedule Change</div>
              <div className="flex items-center gap-2 font-mono text-xs bg-muted px-3 py-1 rounded">
                <span>{schedule.old_schedule || 'Current'}</span>
                <span className="text-muted-foreground">→</span>
                <span className="font-semibold text-blue-600">{schedule.new_schedule || 'Proposed'}</span>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-3">
              <div className="p-3 border rounded-lg bg-background">
                <div className="text-xs font-semibold text-muted-foreground mb-1">CURRENT</div>
                <div className="space-y-1 text-xs font-mono">
                  <div>dep_delay: {current.avg_dep_delay_min?.toFixed(2)}m</div>
                  <div>arr_delay: {current.avg_arr_delay_min?.toFixed(2)}m</div>
                </div>
              </div>
              <div className="p-3 border rounded-lg bg-background">
                <div className="text-xs font-semibold text-muted-foreground mb-1">PROPOSED</div>
                <div className="space-y-1 text-xs font-mono">
                  <div>dep_delay: {proposed.avg_dep_delay_min?.toFixed(2)}m</div>
                  <div>arr_delay: {proposed.avg_arr_delay_min?.toFixed(2)}m</div>
                </div>
              </div>
              <div className="p-3 border rounded-lg bg-background">
                <div className="text-xs font-semibold text-muted-foreground mb-1">IMPACT (Δ)</div>
                <div className="space-y-1 text-xs font-mono">
                  <div className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border ${badgeColor(depClass)}`}>dep_delay Δ {depDelta.toFixed(2)}m ({depClass})</div>
                  <div className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border ${badgeColor(arrClass)}`}>arr_delay Δ {arrDelta.toFixed(2)}m ({arrClass})</div>
                  {impact && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      <span className="text-[10px] bg-muted px-2 py-0.5 rounded border">dep_rate% {pctVal(impactDepRate)}</span>
                      <span className="text-[10px] bg-muted px-2 py-0.5 rounded border">arr_rate% {pctVal(impactArrRate)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
            {recommendation && (
              <div className="p-3 border rounded-lg bg-emerald-50/60">
                <div className="text-xs font-semibold text-emerald-700 mb-1">RECOMMENDATION</div>
                <div className="text-xs leading-relaxed">{recommendation}</div>
              </div>
            )}
            {notes.length > 0 && (
              <div className="p-3 border rounded-lg bg-muted/30">
                <div className="text-xs font-semibold text-muted-foreground mb-1">NOTES</div>
                <ul className="text-xs space-y-1 list-disc ml-4">
                  {notes.slice(0,4).map((n: string, i: number)=>(<li key={i}>{n}</li>))}
                </ul>
              </div>
            )}
          </div>
        );
      } catch (e) {
        return null;
      }
    };
    const scheduleUI = tryRenderScheduleTuning(response);
    if (scheduleUI) return scheduleUI;
    const parsed = parseConciseResponse(formatResponse(response));
    const order = [
      'PRIMARY SLOT','SECONDARY SLOT','TERTIARY SLOT','TIMES TO AVOID',
      'PEAK CONGESTION HOURS','RECOMMENDED ALTERNATIVES','SUMMARY',
      'SCHEDULE CHANGE','CURRENT','PROPOSED','IMPACT','RECOMMENDATION',
      'TOP CASCADING FLIGHTS','HIGHEST RISK','MONITOR','NOTES'
    ];
    const present = order.filter(k => parsed.sections[k]);
    if (present.length === 0 && parsed.raw) {
      return <div className="text-xs whitespace-pre-wrap font-mono">{parsed.raw}</div>;
    }
    return <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">{present.map(k => renderSection(k, parsed.sections[k]))}</div>;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold flex items-center justify-center gap-2">
          <Plane className="h-6 w-6" />
          Flight Schedule Optimization Analysis
        </h2>
        <p className="text-muted-foreground">
          Comprehensive analysis using open-source AI tools for flight data optimization
        </p>
      </div>

      {/* Model Information */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          <strong>Models Used:</strong> RandomForestRegressor & RandomForestClassifier for delay prediction, 
          NetworkX for cascading analysis, PostgreSQL for data processing, and Ollama LLM for natural language insights.
        </AlertDescription>
      </Alert>

      {/* Analysis Types Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {analysisTypes.map((analysis) => {
          const Icon = analysis.icon;
          const result = results[analysis.id];
          const isActive = activeAnalysis === analysis.id;
          
          return (
            <Card key={analysis.id} className="relative transition-transform hover:-translate-y-1 hover:shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${analysis.color} text-white ring-1 ring-white/20 shadow-sm`} title={analysis.title}>
                      <Icon className="h-5 w-5" />
                    </div>
                  <div>
                    <div className="text-lg">{analysis.title}</div>
                    <div className="text-sm text-muted-foreground font-normal">
                      {analysis.description}
                    </div>
                  </div>
                    <div className="ml-auto flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Models</span>
                      <span className="text-[11px] bg-muted px-2 py-0.5 rounded border">{analysis.models.length}</span>
                      {result?.timestamp && (
                        <span className="text-xs text-muted-foreground ml-3" title={`Last run: ${result.timestamp}`}>
                          Last: {new Date(result.timestamp).toLocaleString()}
                        </span>
                      )}
                    </div>
                </CardTitle>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-3 gap-4">
                  {/* Models column */}
                  <div className="md:col-span-1">
                    <Label className="text-xs text-muted-foreground">Models Used</Label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {analysis.models.map((model, idx) => (
                        <Badge key={idx} variant="secondary" className="text-[11px] px-2 py-1" title={model}>
                          {model}
                        </Badge>
                      ))}
                    </div>
                    <div className="text-xs text-muted-foreground mt-3">Tip: Click Start to run the selected analysis.</div>
                  </div>

                  {/* Form + actions column */}
                  <div className="md:col-span-2 space-y-3">
                    {renderAnalysisForm(analysis.id)}
                
                {/* Action Button */}
                <Button
                  onClick={() => executeAnalysis(analysis.id)}
                  disabled={isActive || ollamaMutation.isPending}
                  className="w-full"
                  variant={result?.success ? "secondary" : "default"}
                >
                  {isActive ? (
                    <>
                      <Zap className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : result?.success ? (
                    <>
                      <BarChart3 className="h-4 w-4 mr-2" />
                      Re-run Analysis
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Start Analysis
                    </>
                  )}
                </Button>
                
                {/* Results */}
                {result && (
                  <div className="mt-4 p-4 bg-muted rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <Label className="text-sm font-medium">Analysis Result</Label>
                      <Badge variant={result.success ? "default" : "destructive"}>
                        {result.success ? "Success" : "Error"}
                      </Badge>
                    </div>

                    <div className="text-sm space-y-2">
                      <div>
                        <strong>Model:</strong> {result.model_used}
                      </div>
                      <div>
                        <strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}
                      </div>
                      <div className="mt-3">
                        <strong>Analysis Result:</strong>
                        <div className="mt-2 p-4 bg-background rounded-lg border max-h-80 overflow-y-auto">
                          <div className="prose prose-sm max-w-none">
                            {renderFormattedResponse(result.response)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                </div>
              </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Summary */}
      {Object.keys(results).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {Object.values(results).filter(r => r.analysis_type === 'optimal_timing').length}
                </div>
                <div className="text-sm text-muted-foreground">Timing Analyses</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-600">
                  {Object.values(results).filter(r => r.analysis_type === 'congestion_analysis').length}
                </div>
                <div className="text-sm text-muted-foreground">Congestion Analyses</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {Object.values(results).filter(r => r.analysis_type === 'schedule_tuning').length}
                </div>
                <div className="text-sm text-muted-foreground">Schedule Tuning</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">
                  {Object.values(results).filter(r => r.analysis_type === 'cascading_analysis').length}
                </div>
                <div className="text-sm text-muted-foreground">Cascade Analyses</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ProjectAnalysis;
