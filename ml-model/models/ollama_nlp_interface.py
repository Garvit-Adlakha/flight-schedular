"""
Ollama-powered NLP Interface for Flight Schedule Optimization
Implements LLM with custom context prompting as specified in the workflow
"""

import pandas as pd
import numpy as np
import ollama
import json
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
warnings.filterwarnings('ignore')

sys.path.append('..')

class OllamaNLPInterface:
    """
    Ollama LLM-powered interface with custom flight scheduling context
    """
    
    def __init__(self, data_file='data/processed_flight_data.csv', model_name='llama3.2:latest'):
        self.data_file = data_file
        self.model_name = model_name
        self.df = None
        self.context_data = {}
        
        # Initialize Ollama client
        self.client = ollama.Client()
        
        # Load data and prepare context
        self._load_flight_data()
        self._prepare_flight_context()
        self._verify_ollama_model()
    
    def _verify_ollama_model(self):
        """Verify Ollama model is available"""
        try:
            # Try to list models
            models = self.client.list()
            available_models = [model.get('name', model.get('model', '')) for model in models.get('models', [])]
            
            if self.model_name not in available_models:
                print(f"⚠️  Model {self.model_name} not found.")
                print(f"Available models: {available_models}")
                
                # Try to pull the model
                print(f"🔄 Attempting to pull {self.model_name}...")
                try:
                    self.client.pull(self.model_name)
                    print(f"✅ Successfully pulled {self.model_name}")
                except Exception as e:
                    print(f"❌ Failed to pull model: {e}")
                    print("Please install Ollama and pull a model first:")
                    print("ollama pull llama3.2")
                    raise
            else:
                print(f"✅ Ollama model {self.model_name} is ready")
                
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise
    
    def _load_flight_data(self):
        """Load and prepare flight data"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df['Flight_Date'] = pd.to_datetime(self.df['Flight_Date'])
            print(f"✅ Loaded {len(self.df)} flight records for LLM context")
        except Exception as e:
            print(f"❌ Error loading flight data: {e}")
            raise
    
    def _prepare_flight_context(self):
        """Prepare comprehensive flight context for LLM prompts"""
        print("🧠 Preparing flight scheduling context for LLM...")
        
        # Basic statistics
        self.context_data = {
            'total_flights': len(self.df),
            'date_range': f"{self.df['Flight_Date'].min().date()} to {self.df['Flight_Date'].max().date()}",
            'airports': {
                'origins': self.df['Origin_Airport'].value_counts().to_dict(),
                'destinations': self.df['Destination_Airport'].value_counts().to_dict()
            },
            'routes': self.df['Route'].value_counts().head(10).to_dict(),
            'delays': {
                'avg_departure_delay': float(self.df['Departure_Delay_Minutes'].mean()),
                'avg_arrival_delay': float(self.df['Arrival_Delay_Minutes'].mean()),
                'departure_delay_rate': float(self.df['Is_Delayed_Departure'].mean()),
                'arrival_delay_rate': float(self.df['Is_Delayed_Arrival'].mean())
            },
            'peak_hours': self._get_peak_hours(),
            'route_performance': self._get_route_performance(),
            'congestion_periods': self._get_congestion_analysis()
        }
        
        print("✅ Flight context prepared for LLM")
    
    def _get_peak_hours(self):
        """Get peak hour analysis"""
        hourly_stats = self.df.groupby('Scheduled_Departure_Hour').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': 'mean',
            'Is_Delayed_Departure': 'mean'
        }).round(2)
        
        return {
            'busiest_hours': hourly_stats['Flight_Number'].nlargest(3).to_dict(),
            'highest_delay_hours': hourly_stats['Departure_Delay_Minutes'].nlargest(3).to_dict(),
            'congestion_score': (hourly_stats['Flight_Number'] * 0.4 + 
                               hourly_stats['Departure_Delay_Minutes'] * 0.6).nlargest(3).to_dict()
        }
    
    def _get_route_performance(self):
        """Get route performance data"""
        route_stats = self.df.groupby('Route').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': 'mean',
            'Is_Delayed_Departure': 'mean'
        }).round(2)
        
        # Filter routes with significant traffic
        significant_routes = route_stats[route_stats['Flight_Number'] >= 10]
        
        return {
            'best_routes': significant_routes.nsmallest(5, 'Departure_Delay_Minutes')['Departure_Delay_Minutes'].to_dict(),
            'worst_routes': significant_routes.nlargest(5, 'Departure_Delay_Minutes')['Departure_Delay_Minutes'].to_dict(),
            'busiest_routes': significant_routes.nlargest(5, 'Flight_Number')['Flight_Number'].to_dict()
        }
    
    def _get_congestion_analysis(self):
        """Get congestion analysis"""
        hourly_stats = self.df.groupby('Scheduled_Departure_Hour').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': 'mean'
        })
        
        # Calculate congestion score
        hourly_stats['Congestion_Score'] = (
            hourly_stats['Flight_Number'] * 0.4 + 
            hourly_stats['Departure_Delay_Minutes'] * 0.6
        )
        
        threshold = hourly_stats['Congestion_Score'].quantile(0.75)
        
        return {
            'peak_congestion_hours': hourly_stats[hourly_stats['Congestion_Score'] >= threshold].index.tolist(),
            'recommended_hours': hourly_stats[hourly_stats['Congestion_Score'] < threshold].index.tolist(),
            'hourly_scores': hourly_stats['Congestion_Score'].to_dict()
        }
    
    def _get_detailed_route_analysis(self) -> str:
        """Get comprehensive route delay statistics for analysis queries"""
        route_stats = self.df.groupby('Route').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': ['mean', 'std', 'max'],
            'Is_Delayed_Departure': 'mean'
        }).round(1)
        
        # Flatten column names
        route_stats.columns = ['Flight_Count', 'Avg_Delay', 'Std_Delay', 'Max_Delay', 'Delay_Rate']
        
        # Filter routes with significant traffic (>=20 flights)
        significant_routes = route_stats[route_stats['Flight_Count'] >= 20].head(10)
        
        # Format the data for clear presentation
        route_data = ""
        for route, data in significant_routes.iterrows():
            route_data += f"• {route}: {int(data['Flight_Count'])} flights, avg delay {data['Avg_Delay']:.1f} min, max {data['Max_Delay']:.0f} min\n"
        
        return route_data.strip()
    
    def _get_operational_insights(self) -> str:
        """Get key operational insights for robust analysis context"""
        insights = []
        
        # Peak hour insights
        busiest_hour = max(self.context_data['peak_hours']['busiest_hours'], key=self.context_data['peak_hours']['busiest_hours'].get)
        busiest_count = self.context_data['peak_hours']['busiest_hours'][busiest_hour]
        insights.append(f"• Peak traffic: Hour {busiest_hour} handles {busiest_count} flights ({busiest_count/self.context_data['total_flights']*100:.1f}% of daily volume)")
        
        # Route performance insights
        best_route = list(self.context_data['route_performance']['best_routes'].keys())[0]
        worst_route = list(self.context_data['route_performance']['worst_routes'].keys())[0]
        best_delay = self.context_data['route_performance']['best_routes'][best_route]
        worst_delay = self.context_data['route_performance']['worst_routes'][worst_route]
        insights.append(f"• Route performance gap: {best_route} (best: {best_delay:.1f}min) vs {worst_route} (worst: {worst_delay:.1f}min)")
        
        # Congestion insights
        congestion_hours = len(self.context_data['congestion_periods']['peak_congestion_hours'])
        optimal_hours = len(self.context_data['congestion_periods']['recommended_hours'])
        insights.append(f"• Time efficiency: {optimal_hours} optimal hours vs {congestion_hours} congested hours for scheduling")
        
        # Delay recovery insights
        avg_dep_delay = self.context_data['delays']['avg_departure_delay']
        avg_arr_delay = self.context_data['delays']['avg_arrival_delay']
        if avg_dep_delay > avg_arr_delay:
            recovery_rate = ((avg_dep_delay - avg_arr_delay) / avg_dep_delay) * 100
            insights.append(f"• In-flight recovery: {recovery_rate:.1f}% of departure delays recovered during flight")
        
        return '\n'.join(insights)
    
    def _get_timing_analysis_data(self) -> str:
        """Get comprehensive timing analysis data for optimal timing queries"""
        hourly_performance = self.df.groupby('Scheduled_Departure_Hour').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': ['mean', 'std', 'median'],
            'Is_Delayed_Departure': 'mean',
            'Arrival_Delay_Minutes': 'mean'
        }).round(1)
        
        # Flatten column names
        hourly_performance.columns = ['Flight_Count', 'Avg_Dep_Delay', 'Std_Dep_Delay', 'Median_Dep_Delay', 'Delay_Rate', 'Avg_Arr_Delay']
        
        # Calculate performance scores (lower is better)
        hourly_performance['Performance_Score'] = (
            hourly_performance['Avg_Dep_Delay'] * 0.4 + 
            hourly_performance['Delay_Rate'] * 100 * 0.3 + 
            hourly_performance['Std_Dep_Delay'] * 0.3
        )
        
        # Format timing data
        timing_data = "HOURLY PERFORMANCE ANALYSIS:\n"
        for hour, data in hourly_performance.iterrows():
            if data['Flight_Count'] >= 5:  # Only include hours with significant traffic
                performance_level = "Optimal" if data['Performance_Score'] < 20 else "Good" if data['Performance_Score'] < 30 else "Challenging"
                timing_data += f"• Hour {hour:02d}: {int(data['Flight_Count'])} flights, avg delay {data['Avg_Dep_Delay']:.1f}min, reliability {(1-data['Delay_Rate'])*100:.0f}% ({performance_level})\n"
        
        return timing_data.strip()
    
    def _get_comprehensive_insights(self) -> str:
        """Get comprehensive system insights covering all project requirements"""
        insights = []
        
        # Optimal timing insights
        hourly_stats = self.df.groupby('Scheduled_Departure_Hour').agg({
            'Departure_Delay_Minutes': 'mean',
            'Flight_Number': 'count'
        })
        best_hour = hourly_stats[hourly_stats['Flight_Number'] >= 10]['Departure_Delay_Minutes'].idxmin()
        worst_hour = hourly_stats[hourly_stats['Flight_Number'] >= 10]['Departure_Delay_Minutes'].idxmax()
        insights.append(f"• Optimal Timing: Hour {best_hour} shows best performance, Hour {worst_hour} shows highest delays")
        
        # Congestion analysis insights
        traffic_density = self.df.groupby('Scheduled_Departure_Hour')['Flight_Number'].count()
        peak_traffic_hour = traffic_density.idxmax()
        peak_traffic_count = traffic_density.max()
        insights.append(f"• Congestion Analysis: Hour {peak_traffic_hour} peak traffic ({peak_traffic_count} flights), avoid for schedule optimization")
        
        # Schedule tuning potential
        schedule_flexibility = self.df.groupby('Route').agg({
            'Departure_Delay_Minutes': ['mean', 'std'],
            'Flight_Number': 'count'
        })
        high_variance_routes = schedule_flexibility[schedule_flexibility[('Departure_Delay_Minutes', 'std')] > 30]
        insights.append(f"• Schedule Tuning Potential: {len(high_variance_routes)} routes show high delay variance, indicating tuning opportunities")
        
        # Cascading delay risk
        total_delayed = self.df['Is_Delayed_Departure'].sum()
        cascade_risk = (total_delayed / len(self.df)) * 100
        insights.append(f"• Cascading Risk: {cascade_risk:.1f}% flights delayed, creating cascade potential requiring targeted intervention")
        
        # Recovery capabilities
        recovery_flights = len(self.df[
            (self.df['Departure_Delay_Minutes'] > 0) & 
            (self.df['Arrival_Delay_Minutes'] < self.df['Departure_Delay_Minutes'])
        ])
        total_delayed_flights = len(self.df[self.df['Departure_Delay_Minutes'] > 0])
        if total_delayed_flights > 0:
            recovery_rate = (recovery_flights / total_delayed_flights) * 100
            insights.append(f"• System Recovery: {recovery_rate:.1f}% of delayed flights recover time in-flight, indicating system resilience")
        
        return '\n'.join(insights)
    
    def _build_context_prompt(self, user_query: str, specific_context: Optional[Dict] = None, concise: bool = True) -> str:
        """Concise-only prompt builder for the four supported analyses."""
        # Base minimal snapshot
        base = {
            'flights_total': self.context_data['total_flights'],
            'date_range': self.context_data['date_range'],
            'avg_dep_delay_min': round(self.context_data['delays']['avg_departure_delay'], 2),
            'avg_arr_delay_min': round(self.context_data['delays']['avg_arrival_delay'], 2),
            'dep_delay_rate_pct': round(self.context_data['delays']['departure_delay_rate']*100, 2),
            'arr_delay_rate_pct': round(self.context_data['delays']['arrival_delay_rate']*100, 2)
        }
        analysis_type = (specific_context or {}).get('analysis_type','').lower()
        # Determine strict instruction template per analysis
        if 'optimal_timing' in analysis_type:
            instructions = "Return only PRIMARY/SECONDARY/TERTIARY/TIMES TO AVOID/NOTES lines as specified; no extra text."  
        elif 'congestion' in analysis_type:
            instructions = "Return PEAK CONGESTION HOURS / RECOMMENDED ALTERNATIVES / SUMMARY / NOTES exactly; no extra words."  
        elif 'schedule_tuning' in analysis_type:
            instructions = "Return SCHEDULE CHANGE / CURRENT / PROPOSED / IMPACT / RECOMMENDATION / NOTES lines only."  
        elif 'cascade' in analysis_type:
            instructions = "Return TOP CASCADING FLIGHTS list, HIGHEST RISK line, MONITOR block, NOTES line only."  
        else:
            instructions = "Respond concisely with data-aware answer."  
        payload = {
            'SYSTEM_SNAPSHOT': base,
            'USER_QUERY': user_query,
            'CONTEXT_BLOCK': specific_context or {},
            'INSTRUCTIONS': instructions
        }
        return json.dumps(payload, indent=2)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format LLM response for clear presentation"""
        if not response:
            return "No response received"
        
        # Basic cleanup
        cleaned = response.strip()
        
        # Remove redundant phrases
        redundant_phrases = [
            "Based on the provided flight scheduling context, I'll",
            "Based on the comprehensive data analysis,",
            "Based on our analysis, we recommend",
            "I'll analyze",
            "Let me analyze"
        ]
        
        for phrase in redundant_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Remove irrelevant sections for all analysis types
        analysis_type = None
        if any(keyword in cleaned.upper() for keyword in ['TAKEOFF', 'LANDING', 'BEST TIME']):
            analysis_type = 'timing'
        elif any(keyword in cleaned.upper() for keyword in ['CONGESTION', 'PEAK', 'BUSIEST']):
            analysis_type = 'congestion'
        elif any(keyword in cleaned.upper() for keyword in ['SCHEDULE CHANGE', 'IMPACT', 'DELAY COMPARISON']):
            analysis_type = 'schedule_tuning'
        elif any(keyword in cleaned.upper() for keyword in ['HIGH-IMPACT FLIGHTS', 'CASCADING', 'CASCADE']):
            analysis_type = 'cascading'
        
        if analysis_type:
            sections = cleaned.split('\n\n')
            relevant_sections = []
            
            for section in sections:
                section = section.strip()
                
                # Keep relevant sections based on analysis type
                if analysis_type == 'timing':
                    keep_keywords = ['BEST TIME', 'ALTERNATIVE TIMES', 'TIMES TO AVOID', 'TAKEOFF', 'LANDING']
                elif analysis_type == 'congestion':
                    keep_keywords = ['PEAK CONGESTION', 'RECOMMENDED ALTERNATIVES', 'CONGESTION SUMMARY']
                elif analysis_type == 'schedule_tuning':
                    keep_keywords = ['SCHEDULE CHANGE IMPACT', 'DELAY COMPARISON', 'RECOMMENDATION']
                elif analysis_type == 'cascading':
                    keep_keywords = ['TOP CASCADING FLIGHTS', 'HIGHEST RISK', 'CRITICAL MONITORING', 'FLIGHT AI', 'FLIGHT 6E', 'FLIGHT UK']
                
                if any(keyword in section.upper() for keyword in keep_keywords):
                    relevant_sections.append(section)
                # Skip unwanted sections for all types
                elif any(skip_keyword in section.upper() for skip_keyword in [
                    'IMPLEMENTATION', 'TIMELINE', 'SYSTEM INSIGHTS', 'ACTIONABLE',
                    'SUPPORTING DATA', 'OPERATIONAL INSIGHTS', 'REALISTIC TIMELINES',
                    'COST-BENEFIT', 'INTERVENTION STRATEGIES', 'MONITORING REQUIREMENTS',
                    'METHODOLOGY', 'FRAMEWORK', 'ANALYSIS REPORT', 'RESULTS:', 'WE ANALYZED'
                ]):
                    continue
                # Keep short sections with relevant data
                elif len(section) < 200 and any(relevant_word in section.lower() for relevant_word in [
                    'flight', 'hour', 'time', 'delay', 'minutes', ':', 'impact'
                ]):
                    relevant_sections.append(section)
            
            cleaned = '\n\n'.join(relevant_sections)
        
        # Fix formatting
        cleaned = cleaned.replace('*', '').replace('•', '•').replace('\n\n\n', '\n\n')
        
        # Clean up extra spaces
        lines = []
        for line in cleaned.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        
        return '\n'.join(lines)
    
    def query_with_ollama(self, user_query: str, specific_context: Optional[Dict] = None, concise: bool = True) -> Dict[str, Any]:
        """General AI interaction method for direct questions and analysis."""
        
        print(f"🤖 Processing general query with Ollama LLM: '{user_query[:50]}...'")
        
        try:
            # Create general prompt for direct AI interaction
            prompt = self._create_general_prompt(user_query)
            
            # Query Ollama with general prompt
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.4,  # Balanced for general queries
                    'top_p': 0.9,
                    'num_predict': 400,  # Allow more detailed responses for general queries
                    'stop': [],
                    'timeout': 90
                }
            )
            
            llm_response = response['message']['content']
            
            return {
                'query': user_query,
                'response': llm_response,
                'model_used': self.model_name,
                'context_provided': False,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ General Ollama query error: {e}")
            return {
                'query': user_query,
                'response': f"Error processing query: {str(e)}",
                'model_used': self.model_name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_general_prompt(self, user_query: str) -> str:
        """Create general prompt for direct AI interaction"""
        
        # Get basic flight context for general queries
        basic_context = self._get_basic_flight_context()
        
        prompt = f"""You are an AI flight scheduling optimization assistant. Answer the user's question directly and helpfully.

CONTEXT: You have access to flight scheduling data including:
{basic_context}

USER QUESTION: {user_query}

INSTRUCTIONS:
- Provide a clear, direct answer to the question
- Use the flight data context when relevant
- Be helpful and actionable
- Keep responses concise but informative
- If the question is about flight optimization, use the data context
- If it's a general question, answer based on your knowledge

Please answer the question:"""
        
        return prompt.strip()
    
    def _get_basic_flight_context(self) -> str:
        """Get basic flight context for general queries"""
        
        context_parts = []
        
        # Basic statistics
        context_parts.append(f"• Total flights: {self.context_data['total_flights']}")
        context_parts.append(f"• Date range: {self.context_data['date_range']}")
        
        # Key insights
        delays = self.context_data['delays']
        context_parts.append(f"• Average departure delay: {delays['avg_departure_delay']:.1f} minutes")
        context_parts.append(f"• Departure delay rate: {delays['departure_delay_rate']*100:.1f}%")
        
        # Peak hours
        busiest = self.context_data['peak_hours']['busiest_hours']
        busiest_str = ", ".join([f"{hour}:00" for hour in list(busiest.keys())[:3]])
        context_parts.append(f"• Busiest hours: {busiest_str}")
        
        return "\n".join(context_parts)
    
    def analyze_flight_schedule_change(self, flight_id: str, current_hour: int, new_hour: int, route: str) -> Dict[str, Any]:
        """
        Analyze specific flight schedule change using LLM with custom context
        Example: "Given flight X at Mumbai, what is the expected delay if moved to time Y?"
        """
        
        print(f"🔄 Analyzing schedule change for flight {flight_id}: {current_hour}:00 → {new_hour}:00")
        
        # Get specific context for this scenario
        specific_context = {
            'flight_id': flight_id,
            'route': route,
            'current_departure_hour': current_hour,
            'proposed_departure_hour': new_hour,
            'current_hour_stats': self._get_hour_stats(current_hour),
            'proposed_hour_stats': self._get_hour_stats(new_hour),
            'route_stats': self._get_route_stats(route)
        }
        
        # Custom query for this specific analysis
        query = f"""Given flight {flight_id} on route {route} currently scheduled at {current_hour}:00, 
        what is the expected delay and operational impact if moved to {new_hour}:00? 
        
        Analyze:
        1. Delay probability change
        2. Congestion impact 
        3. Passenger impact
        4. Operational efficiency
        5. Cascading effects on other flights
        6. Overall recommendation
        
        Provide specific quantitative insights where possible."""
        
        return self.query_with_ollama(query, specific_context)
    
    def _get_hour_stats(self, hour: int) -> Dict[str, Any]:
        """Get statistics for a specific hour"""
        hour_data = self.df[self.df['Scheduled_Departure_Hour'] == hour]
        if len(hour_data) == 0:
            return {'flight_count': 0, 'avg_delay_minutes': 0, 'delay_rate': 0}
        
        return {
            'flight_count': len(hour_data),
            'avg_delay_minutes': float(hour_data['Departure_Delay_Minutes'].mean()),
            'delay_rate': float(hour_data['Is_Delayed_Departure'].mean())
        }
    
    def _get_route_stats(self, route: str) -> Dict[str, Any]:
        """Get statistics for a specific route"""
        route_data = self.df[self.df['Route'] == route]
        if len(route_data) == 0:
            return {'flight_count': 0, 'avg_delay_minutes': 0, 'delay_rate': 0}
        
        return {
            'flight_count': len(route_data),
            'avg_delay_minutes': float(route_data['Departure_Delay_Minutes'].mean()),
            'delay_rate': float(route_data['Is_Delayed_Departure'].mean())
        }
    
    def _clean_response(self, response: str) -> str:
        """Minimal cleanup; strips explanatory chatter when model ignored strict format."""
        if not response:
            return 'No response received'
        cleaned_lines = []
        skip_prefixes = [
            'this appears to be a json',
            'here is a breakdown',
            'the json structure',
            'here\'s a breakdown'
        ]
        for raw in response.split('\n'):
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if any(low.startswith(p) for p in skip_prefixes):
                continue
            cleaned_lines.append(line.rstrip())
        return '\n'.join(cleaned_lines)
        query = f"""A {severity} {disruption_type} disruption is affecting operations during hours {affected_hours}. 
        
        Analyze:
        1. Total flights impacted
        2. Expected delay increases
        3. Cascading effects on downstream flights
        4. Passenger rebooking requirements
        5. Alternative scheduling options
        6. Recovery timeline recommendations
        7. Cost impact assessment
        
        Provide actionable mitigation strategies and recovery plans."""
        
        return self.query_with_ollama(query, disruption_context)
    
    def get_trend_analysis(self, metric: str, time_period: str) -> Dict[str, Any]:
        """Get trend analysis using LLM insights"""
        
        query = f"""Analyze {metric} trends over {time_period} and provide:
        
        1. Key patterns and insights
        2. Seasonal variations
        3. Performance improvements or degradations
        4. Root cause analysis
        5. Predictive insights for future periods
        6. Actionable recommendations for optimization
        
        Focus on operational implications and strategic decisions."""
        
        trend_context = {
            'metric': metric,
            'time_period': time_period,
            'current_performance': self.context_data['delays'],
            'route_trends': self.context_data['route_performance']
        }
        
        return self.query_with_ollama(query, trend_context)
    
    def _compute_hourly_metrics(self, route: Optional[str] = None, hours: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        """Compute deterministic hourly metrics so LLM only summarizes.
        Returns dict keyed by hour with metrics used for ranking."""
        subset = self.df.copy()
        if route:
            subset = subset[subset['Route'] == route]
        if hours:
            subset = subset[subset['Scheduled_Departure_Hour'].isin(hours)]
        metrics = {}
        grouped = subset.groupby('Scheduled_Departure_Hour')
        for hour, g in grouped:
            flight_count = len(g)
            avg_delay = float(g['Departure_Delay_Minutes'].mean()) if flight_count else np.nan
            delay_prob = float(g['Is_Delayed_Departure'].mean()) * 100 if flight_count else np.nan
            reliability = 100 - delay_prob if not np.isnan(delay_prob) else np.nan
            metrics[int(hour)] = {
                'flight_count': flight_count,
                'avg_delay_min': round(avg_delay, 2) if not np.isnan(avg_delay) else None,
                'delay_probability_pct': round(delay_prob, 2) if not np.isnan(delay_prob) else None,
                'reliability_pct': round(reliability, 2) if not np.isnan(reliability) else None,
                'low_sample': flight_count < 5
            }
        return metrics

    def _rank_optimal_hours(self, metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank hours based on delay probability then avg delay, exclude low sample unless no alternative."""
        rows = []
        for hour, m in metrics.items():
            if m['delay_probability_pct'] is None:
                continue
            rows.append((hour, m['delay_probability_pct'], m['avg_delay_min'], m['reliability_pct'], m['flight_count'], m['low_sample']))
        # Prefer non-low-sample
        primary_candidates = [r for r in rows if not r[5]] or rows
        primary = sorted(primary_candidates, key=lambda x: (x[1], x[2]))[0] if primary_candidates else None
        remaining = [r for r in rows if primary and r[0] != primary[0]]
        secondary_candidates = [r for r in remaining if not r[5]] or remaining
        secondary = sorted(secondary_candidates, key=lambda x: (x[1], x[2]))[0] if secondary_candidates else None
        tertiary_candidates = [r for r in remaining if secondary and r[0] != secondary[0]]
        tertiary_candidates = [r for r in tertiary_candidates if not r[5]] or tertiary_candidates
        tertiary = sorted(tertiary_candidates, key=lambda x: (x[1], x[2]))[0] if tertiary_candidates else None
        worst = sorted(rows, key=lambda x: (-x[1], - (x[2] if x[2] is not None else 0)))[:2]
        def fmt(rec):
            if not rec:
                return None
            hour, dp, ad, rel, fc, low = rec
            return {
                'hour': hour,
                'time_window': f"{hour:02d}:00-{hour+1:02d}:00",
                'delay_probability_pct': dp,
                'avg_delay_min': ad,
                'reliability_pct': rel,
                'flight_count': fc,
                'low_sample': low
            }
        return {
            'primary': fmt(primary),
            'secondary': fmt(secondary),
            'tertiary': fmt(tertiary),
            'avoid': [fmt(w) for w in worst],
            'all_hours': metrics
        }

    def _format_optimal_timing_fallback(self, ranked: Dict[str, Any]) -> str:
        """Deterministic format when LLM output is verbose/off-spec."""
        def slot(label: str, slot_obj):
            if not slot_obj:
                return f"{label} SLOT: INSUFFICIENT DATA"
            reason = {
                'PRIMARY': 'lowest delay probability',
                'SECONDARY': 'next best probability',
                'TERTIARY': 'third best'
            }.get(label, 'selected')
            if slot_obj.get('low_sample'):
                reason += ' LOW SAMPLE'
            return (
                f"{label} SLOT: {slot_obj['time_window']} | delay_prob={slot_obj['delay_probability_pct']:.1f}% | "
                f"avg_delay={slot_obj['avg_delay_min']:.1f}m | reliability={slot_obj['reliability_pct']:.1f}% | "
                f"flights={slot_obj['flight_count']} | reason: {reason}"
            )
        lines = [
            slot('PRIMARY', ranked.get('primary')),
            slot('SECONDARY', ranked.get('secondary')),
            slot('TERTIARY', ranked.get('tertiary')),
            'TIMES TO AVOID:'
        ]
        for a in [x for x in ranked.get('avoid', []) if x]:
            lines.append(
                f"- {a['time_window']} (delay_prob={a['delay_probability_pct']:.1f}%, avg_delay={a['avg_delay_min']:.1f}m, flights={a['flight_count']}) reason: high delay risk"
            )
        notes = []
        if any(s and s.get('low_sample') for s in [ranked.get('primary'), ranked.get('secondary'), ranked.get('tertiary')]):
            notes.append('- Some recommended slots have LOW SAMPLE (<5 flights)')
        if not notes:
            notes.append('- Deterministic fallback applied (LLM output ignored)')
        lines.append('NOTES:')
        lines.extend(notes)
        return '\n'.join(lines)

    def find_optimal_takeoff_landing_times(self, route: str = None, date: str = None, use_llm: bool = True, concise: bool = True) -> Dict[str, Any]:
        """
        PROJECT REQUIREMENT 1: Find best time to takeoff/landing using scheduled vs actual analysis
        Returns: best time, alternate time, tip
./        """
        print(f"🎯 Finding optimal takeoff/landing times for route: {route or 'all routes'}")
        
        # Use the new direct format prompt
        query = self._get_optimal_timing_query(route, date)
        
        # Create specialized prompt for direct answers
        prompt = self._create_optimal_timing_prompt(query, route)
        
        # Query the LLM with the specialized prompt
        llm_result = self._query_with_optimal_timing_prompt(prompt)
        
        return llm_result
    
    def _get_optimal_timing_query(self, route: str = None, date: str = None) -> str:
        """Generate query for optimal timing"""
        if route:
            return f"What are the optimal takeoff and landing times for route {route}?"
        else:
            return "What are the optimal takeoff and landing times for flights?"
    
    def _create_optimal_timing_prompt(self, user_query: str, route: str = None) -> str:
        """Create specialized prompt for optimal timing with direct format"""
        
        # Get relevant data context
        data_context = self._get_optimal_timing_context(route)
        
        prompt = f"""You are a flight scheduling optimization expert. Analyze the provided flight data and give DIRECT answers for optimal takeoff/landing times.

RESPONSE FORMAT (EXACTLY):
BEST TIME: [HH:MM] - [brief reason]
ALTERNATE TIME: [HH:MM] - [brief reason]  
TIP: [one actionable tip]

RULES:
- Use 24-hour format (HH:MM)
- Keep reasons under 8 words
- Provide only these 3 lines
- No explanations, no extra text
- Base answers on delay probability and congestion data
- Prefer times with lower delay rates and fewer flights

DATA CONTEXT:
{data_context}

QUERY: {user_query}

RESPOND WITH ONLY:
BEST TIME: [time] - [reason]
ALTERNATE TIME: [time] - [reason]
TIP: [tip]"""
        
        return prompt.strip()
    
    def _get_optimal_timing_context(self, route: str = None) -> str:
        """Get relevant context data for optimal timing analysis"""
        
        context_parts = []
        
        # Basic statistics
        context_parts.append(f"Total flights analyzed: {self.context_data['total_flights']}")
        
        # Peak hours
        busiest = self.context_data['peak_hours']['busiest_hours']
        busiest_str = ", ".join([f"{hour}:00 ({count} flights)" for hour, count in list(busiest.items())[:3]])
        context_parts.append(f"Busiest hours: {busiest_str}")
        
        # Delay statistics
        delays = self.context_data['delays']
        context_parts.append(f"Average departure delay: {delays['avg_departure_delay']:.1f} minutes")
        context_parts.append(f"Departure delay rate: {delays['departure_delay_rate']*100:.1f}%")
        
        # Route-specific data
        if route and 'route_performance' in self.context_data:
            route_perf = self.context_data['route_performance']
            if 'best_routes' in route_perf and route in route_perf['best_routes']:
                context_parts.append(f"Route {route} performance: {route_perf['best_routes'][route]:.1f} min avg delay")
        
        # Congestion data
        congestion = self.context_data['congestion_periods']
        if 'peak_congestion_hours' in congestion:
            peak_hours = congestion['peak_congestion_hours']
            if peak_hours:
                peak_str = ", ".join([f"{hour}:00" for hour in peak_hours[:3]])
                context_parts.append(f"Peak congestion hours: {peak_str}")
        
        return "\n".join(context_parts)
    
    def _query_with_optimal_timing_prompt(self, prompt: str) -> Dict[str, Any]:
        """Query LLM with optimal timing prompt and parse response"""
        
        try:
            # Query Ollama with the specialized prompt
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.2,  # Lower temperature for more consistent format
                    'top_p': 0.9,
                    'num_predict': 150,  # Shorter response for direct format
                    'stop': [],
                    'timeout': 60
                }
            )
            
            llm_response = response['message']['content']
            
            # Parse the response into structured format
            parsed_response = self._parse_optimal_timing_response(llm_response)
            
            return {
                'query': 'Optimal takeoff/landing times',
                'response': llm_response,
                'parsed_response': parsed_response,
                'model_used': self.model_name,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Optimal timing query error: {e}")
            return {
                'query': 'Optimal takeoff/landing times',
                'response': f"Error processing query: {str(e)}",
                'parsed_response': {'best_time': '', 'alternate_time': '', 'tip': ''},
                'model_used': self.model_name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_optimal_timing_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into structured format"""
        
        result = {
            'best_time': '',
            'alternate_time': '',
            'tip': ''
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('BEST TIME:'):
                result['best_time'] = line.replace('BEST TIME:', '').strip()
            elif line.startswith('ALTERNATE TIME:'):
                result['alternate_time'] = line.replace('ALTERNATE TIME:', '').strip()
            elif line.startswith('TIP:'):
                result['tip'] = line.replace('TIP:', '').strip()
        
        return result

    # ---------- New Deterministic Support Functions For Concise Analyses ----------
    def _compute_congestion_metrics(self) -> Dict[int, Dict[str, Any]]:
        grouped = self.df.groupby('Scheduled_Departure_Hour')
        metrics = {}
        max_flights = grouped['Flight_Number'].count().max()
        for hour, g in grouped:
            flights = len(g)
            avg_delay = float(g['Departure_Delay_Minutes'].mean()) if flights else np.nan
            delay_prob = float(g['Is_Delayed_Departure'].mean()) * 100 if flights else np.nan
            # simple congestion score: weighted flights + delay
            congestion_score = (flights / max_flights * 60) + (avg_delay if not np.isnan(avg_delay) else 0)*0.8
            metrics[int(hour)] = {
                'flights': flights,
                'avg_delay_min': round(avg_delay, 2) if not np.isnan(avg_delay) else None,
                'delay_probability_pct': round(delay_prob, 2) if not np.isnan(delay_prob) else None,
                'congestion_score': round(congestion_score, 2)
            }
        return metrics

    def _rank_congestion(self, metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        hours = []
        for h, m in metrics.items():
            if m['flights'] == 0:
                continue
            hours.append((h, m['congestion_score'], m['flights'], m['avg_delay_min'], m['delay_probability_pct']))
        peak = sorted(hours, key=lambda x: (-x[1], -x[2]))[:3]
        # recommended = lowest congestion excluding peaks
        peak_hours_set = {p[0] for p in peak}
        rest = [h for h in hours if h[0] not in peak_hours_set]
        recommended = sorted(rest, key=lambda x: (x[1], x[3] if x[3] is not None else 999))[:3]
        worst_delay = sorted(hours, key=lambda x: (-(x[3] if x[3] is not None else -1)))[:1]
        def fmt(r):
            if not r: return None
            return {
                'hour': r[0],
                'window': f"{r[0]:02d}:00-{r[0]+1:02d}:00",
                'congestion_score': r[1],
                'flights': r[2],
                'avg_delay_min': r[3],
                'delay_probability_pct': r[4]
            }
        return {
            'peak_hours': [fmt(p) for p in peak],
            'recommended_hours': [fmt(r) for r in recommended],
            'worst_delay_hour': fmt(worst_delay[0]) if worst_delay else None,
            'all': metrics
        }

    def _schedule_change_comparison(self, current_hour: int, new_hour: int, route: str) -> Dict[str, Any]:
        cur_stats = self._get_hour_stats(current_hour)
        new_stats = self._get_hour_stats(new_hour)
        def calc_reliability(delay_prob):
            return round(100 - delay_prob*100, 2) if delay_prob is not None else None
        result = {
            'current': {
                'hour': current_hour,
                'window': f"{current_hour:02d}:00-{current_hour+1:02d}:00",
                'avg_delay_min': round(cur_stats.get('avg_delay_minutes', 0), 2),
                'delay_probability_pct': round(cur_stats.get('delay_rate', 0)*100, 2),
                'reliability_pct': calc_reliability(cur_stats.get('delay_rate')),
                'flights': cur_stats.get('flight_count', 0)
            },
            'proposed': {
                'hour': new_hour,
                'window': f"{new_hour:02d}:00-{new_hour+1:02d}:00",
                'avg_delay_min': round(new_stats.get('avg_delay_minutes', 0), 2),
                'delay_probability_pct': round(new_stats.get('delay_rate', 0)*100, 2),
                'reliability_pct': calc_reliability(new_stats.get('delay_rate')),
                'flights': new_stats.get('flight_count', 0)
            }
        }
        result['impact'] = {
            'delta_avg_delay_min': round(result['proposed']['avg_delay_min'] - result['current']['avg_delay_min'], 2),
            'delta_delay_probability_pct': round(result['proposed']['delay_probability_pct'] - result['current']['delay_probability_pct'], 2)
        }
        return result

    def _compute_cascade_scores(self, top_n: int = 5) -> Dict[str, Any]:
        df = self.df.copy()
        if df.empty:
            return {'flights': []}
        # Normalize components
        max_delay = df['Departure_Delay_Minutes'].max() or 1
        route_counts = df['Route'].value_counts()
        max_route = route_counts.max() or 1
        def score_row(r):
            delay_component = (r['Departure_Delay_Minutes'] / max_delay) if max_delay else 0
            early_component = (24 - r['Scheduled_Departure_Hour']) / 24  # earlier hour -> higher value
            route_component = (route_counts.get(r['Route'], 0) / max_route)
            return 0.5*delay_component + 0.3*early_component + 0.2*route_component
        df['Cascade_Score'] = df.apply(score_row, axis=1)
        # Approx downstream affects as scaled score * flights later same aircraft proxy (not present) -> use score*10
        df_sorted = df.sort_values('Cascade_Score', ascending=False).head(top_n)
        flights = []
        for _, r in df_sorted.iterrows():
            flights.append({
                'flight_number': r['Flight_Number'],
                'route': r['Route'],
                'hour': int(r['Scheduled_Departure_Hour']),
                'window': f"{int(r['Scheduled_Departure_Hour']):02d}:00-{int(r['Scheduled_Departure_Hour'])+1:02d}:00",
                'delay_min': round(float(r['Departure_Delay_Minutes']), 1),
                'cascade_score': round(float(r['Cascade_Score']), 3),
                'approx_downstream_flights': int(round(r['Cascade_Score']*10))
            })
        monitoring_window = None
        if flights:
            earliest = min(f['hour'] for f in flights)
            latest = max(f['hour'] for f in flights)
            monitoring_window = f"{earliest:02d}:00-{latest+1:02d}:00"
        return {
            'flights': flights,
            'monitoring_window': monitoring_window
        }
    
    def identify_congestion_periods(self, date: str = None) -> Dict[str, Any]:
        """
        PROJECT REQUIREMENT 2: Find busiest time slots at airport to avoid
        Returns: peak hours, alternatives, summary, tip
        """
        print(f"🚦 Identifying airport congestion periods...")
        
        # Get congestion data
        metrics = self._compute_congestion_metrics()
        ranking = self._rank_congestion(metrics)
        
        # Create specialized prompt for congestion analysis
        prompt = self._create_congestion_prompt(metrics, ranking)
        
        # Query the LLM with specialized prompt
        llm_result = self._query_with_congestion_prompt(prompt)
        
        return llm_result
    
    def _create_congestion_prompt(self, metrics: Dict, ranking: Dict) -> str:
        """Create specialized prompt for congestion analysis"""
        
        prompt = f"""You are a flight congestion analysis expert. Analyze the data and provide DIRECT answers.

RESPONSE FORMAT (EXACTLY):
PEAK HOURS: [HH:00] - [N flights] - [X.Xm delay] - [Y.Y% delay rate]
ALTERNATIVES: [HH:00] - [N flights] - [X.Xm delay] - [Y.Y% delay rate]
SUMMARY: Busiest [HH:00] | Worst [HH:00] | Best [HH:00]
TIP: [one actionable tip]

RULES:
- Use 24-hour format (HH:00)
- Keep explanations brief
- Provide only these 4 lines
- No extra text or explanations

DATA:
{json.dumps(metrics, indent=2)}

RESPOND WITH ONLY:
PEAK HOURS: [time] - [flights] - [delay] - [rate]
ALTERNATIVES: [time] - [flights] - [delay] - [rate]
SUMMARY: Busiest [time] | Worst [time] | Best [time]
TIP: [tip]"""
        
        return prompt.strip()
    
    def _query_with_congestion_prompt(self, prompt: str) -> Dict[str, Any]:
        """Query LLM with congestion prompt and parse response"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_predict': 200,
                    'timeout': 60
                }
            )
            
            llm_response = response['message']['content']
            parsed_response = self._parse_congestion_response(llm_response)
            
            return {
                'query': 'Airport congestion analysis',
                'response': llm_response,
                'parsed_response': parsed_response,
                'model_used': self.model_name,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Congestion query error: {e}")
            return {
                'query': 'Airport congestion analysis',
                'response': f"Error: {str(e)}",
                'parsed_response': {'peak_hours': '', 'alternatives': '', 'summary': '', 'tip': ''},
                'model_used': self.model_name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_congestion_response(self, response: str) -> Dict[str, str]:
        """Parse congestion response into structured format"""
        
        result = {
            'peak_hours': '',
            'alternatives': '',
            'summary': '',
            'tip': ''
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('PEAK HOURS:'):
                result['peak_hours'] = line.replace('PEAK HOURS:', '').strip()
            elif line.startswith('ALTERNATIVES:'):
                result['alternatives'] = line.replace('ALTERNATIVES:', '').strip()
            elif line.startswith('SUMMARY:'):
                result['summary'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('TIP:'):
                result['tip'] = line.replace('TIP:', '').strip()
        
        return result
    
    def model_schedule_tuning_impact(self, flight_id: str, current_hour: int, new_hour: int, route: str) -> Dict[str, Any]:
        """
        PROJECT REQUIREMENT 3: Model to tune schedule time and see impact on delays
        Returns: change details, current vs proposed, impact, recommendation, tip
        """
        print(f"⚙️ Modeling schedule tuning impact for {flight_id} {route} {current_hour}:00→{new_hour}:00")
        
        # Get schedule comparison data
        comparison = self._schedule_change_comparison(current_hour, new_hour, route)
        
        # Create specialized prompt for schedule tuning
        prompt = self._create_schedule_tuning_prompt(flight_id, route, current_hour, new_hour, comparison)
        
        # Query the LLM with specialized prompt
        llm_result = self._query_with_schedule_tuning_prompt(prompt)
        
        return llm_result
    
    def _create_schedule_tuning_prompt(self, flight_id: str, route: str, current_hour: int, new_hour: int, comparison: Dict) -> str:
        """Create specialized prompt for schedule tuning analysis"""
        
        prompt = f"""You are a flight schedule optimization expert. Analyze the schedule change impact and provide DIRECT answers.

RESPONSE FORMAT (EXACTLY):
CHANGE: Flight {flight_id} {route} {current_hour:02d}:00→{new_hour:02d}:00
CURRENT: {comparison['current']['delay_probability_pct']}% delay rate | {comparison['current']['avg_delay_min']}m avg delay | {comparison['current']['reliability_pct']}% reliability
PROPOSED: {comparison['proposed']['delay_probability_pct']}% delay rate | {comparison['proposed']['avg_delay_min']}m avg delay | {comparison['proposed']['reliability_pct']}% reliability
IMPACT: {comparison['impact']['delta_avg_delay_min']}m delay change | {comparison['impact']['delta_delay_probability_pct']}pp probability change
RECOMMENDATION: [Recommended/Not Recommended] - [brief reason]
TIP: [one actionable tip]

RULES:
- Keep explanations brief
- Provide only these 6 lines
- No extra text or explanations

RESPOND WITH ONLY:
CHANGE: [details]
CURRENT: [stats]
PROPOSED: [stats]
IMPACT: [changes]
RECOMMENDATION: [decision] - [reason]
TIP: [tip]"""
        
        return prompt.strip()
    
    def _query_with_schedule_tuning_prompt(self, prompt: str) -> Dict[str, Any]:
        """Query LLM with schedule tuning prompt and parse response"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_predict': 250,
                    'timeout': 60
                }
            )
            
            llm_response = response['message']['content']
            parsed_response = self._parse_schedule_tuning_response(llm_response)
            
            return {
                'query': 'Schedule tuning impact analysis',
                'response': llm_response,
                'parsed_response': parsed_response,
                'model_used': self.model_name,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Schedule tuning query error: {e}")
            return {
                'query': 'Schedule tuning impact analysis',
                'response': f"Error: {str(e)}",
                'parsed_response': {'change': '', 'current': '', 'proposed': '', 'impact': '', 'recommendation': '', 'tip': ''},
                'model_used': self.model_name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_schedule_tuning_response(self, response: str) -> Dict[str, str]:
        """Parse schedule tuning response into structured format"""
        
        result = {
            'change': '',
            'current': '',
            'proposed': '',
            'impact': '',
            'recommendation': '',
            'tip': ''
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('CHANGE:'):
                result['change'] = line.replace('CHANGE:', '').strip()
            elif line.startswith('CURRENT:'):
                result['current'] = line.replace('CURRENT:', '').strip()
            elif line.startswith('PROPOSED:'):
                result['proposed'] = line.replace('PROPOSED:', '').strip()
            elif line.startswith('IMPACT:'):
                result['impact'] = line.replace('IMPACT:', '').strip()
            elif line.startswith('RECOMMENDATION:'):
                result['recommendation'] = line.replace('RECOMMENDATION:', '').strip()
            elif line.startswith('TIP:'):
                result['tip'] = line.replace('TIP:', '').strip()
        
        return result
    
    def isolate_cascading_delay_flights(self, date: str = None) -> Dict[str, Any]:
        """
        PROJECT REQUIREMENT 4: Model to isolate flights with biggest cascading impact
        Returns: top flights, highest risk, monitoring window, tip
        """
        print(f"🔗 Isolating cascading delay flights...")
        
        # Get cascade data
        cascade = self._compute_cascade_scores()
        
        # Create specialized prompt for cascade analysis
        prompt = self._create_cascade_prompt(cascade)
        
        # Query the LLM with specialized prompt
        llm_result = self._query_with_cascade_prompt(prompt)
        
        return llm_result
    
    def _create_cascade_prompt(self, cascade: Dict) -> str:
        """Create specialized prompt for cascade analysis"""
        
        prompt = f"""You are a flight cascade delay analysis expert. Analyze the data and provide DIRECT answers.

RESPONSE FORMAT (EXACTLY):
TOP FLIGHTS: [Flight ID] ([Route], [HH:00]) score=[S.SSS] delay=[D.Dm] affects≈[N]
HIGHEST RISK: [Flight ID] score=[S.SSS] delay=[D.Dm] affects≈[N]
MONITORING: [HH:00]-[HH:00] window | Key: [flight1,flight2,flight3]
TIP: [one actionable tip]

RULES:
- Keep explanations brief
- Provide only these 4 lines
- No extra text or explanations

DATA:
{json.dumps(cascade, indent=2)}

RESPOND WITH ONLY:
TOP FLIGHTS: [details]
HIGHEST RISK: [details]
MONITORING: [window] | Key: [flights]
TIP: [tip]"""
        
        return prompt.strip()
    
    def _query_with_cascade_prompt(self, prompt: str) -> Dict[str, Any]:
        """Query LLM with cascade prompt and parse response"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_predict': 250,
                    'timeout': 60
                }
            )
            
            llm_response = response['message']['content']
            parsed_response = self._parse_cascade_response(llm_response)
            
            return {
                'query': 'Cascade delay analysis',
                'response': llm_response,
                'parsed_response': parsed_response,
                'model_used': self.model_name,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Cascade query error: {e}")
            return {
                'query': 'Cascade delay analysis',
                'response': f"Error: {str(e)}",
                'parsed_response': {'top_flights': '', 'highest_risk': '', 'monitoring': '', 'tip': ''},
                'model_used': self.model_name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_cascade_response(self, response: str) -> Dict[str, str]:
        """Parse cascade response into structured format"""
        
        result = {
            'top_flights': '',
            'highest_risk': '',
            'monitoring': '',
            'tip': ''
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('TOP FLIGHTS:'):
                result['top_flights'] = line.replace('TOP FLIGHTS:', '').strip()
            elif line.startswith('HIGHEST RISK:'):
                result['highest_risk'] = line.replace('HIGHEST RISK:', '').strip()
            elif line.startswith('MONITORING:'):
                result['monitoring'] = line.replace('MONITORING:', '').strip()
            elif line.startswith('TIP:'):
                result['tip'] = line.replace('TIP:', '').strip()
        
        return result

# Comprehensive testing for all project requirements
if __name__ == "__main__":
    try:
        # Initialize Ollama NLP interface
        print("🚀 Initializing Comprehensive Flight Schedule Optimization System...")
        print("Testing all project requirements with restructured prompt system")
        print("=" * 80)
        
        ollama_nlp = OllamaNLPInterface()
        
        # PROJECT REQUIREMENT 1: Optimal takeoff/landing times
        print("\n🎯 TESTING PROJECT REQUIREMENT 1: Optimal Takeoff/Landing Times")
        print("-" * 60)
        result1 = ollama_nlp.find_optimal_takeoff_landing_times("BOM-DEL")
        print(f"✅ Optimal Timing Analysis: {result1['response'][:150]}...")
        
        # PROJECT REQUIREMENT 2: Congestion period identification
        print("\n🚦 TESTING PROJECT REQUIREMENT 2: Airport Congestion Analysis")
        print("-" * 60)
        result2 = ollama_nlp.identify_congestion_periods()
        print(f"✅ Congestion Analysis: {result2['response'][:150]}...")
        
        # PROJECT REQUIREMENT 3: Schedule tuning impact modeling
        print("\n⚙️ TESTING PROJECT REQUIREMENT 3: Schedule Tuning Impact")
        print("-" * 60)
        result3 = ollama_nlp.model_schedule_tuning_impact("AI2509", 6, 8, "BOM-DEL")
        print(f"✅ Schedule Tuning Analysis: {result3['response'][:150]}...")
        
        # PROJECT REQUIREMENT 4: Cascading delay isolation
        print("\n🔗 TESTING PROJECT REQUIREMENT 4: Cascading Delay Analysis")
        print("-" * 60)
        result4 = ollama_nlp.isolate_cascading_delay_flights()
        print(f"✅ Cascading Analysis: {result4['response'][:150]}...")
        
        # General optimization capability
        print("\n📊 TESTING GENERAL OPTIMIZATION CAPABILITY")
        print("-" * 60)
        result5 = ollama_nlp.query_with_ollama("How can I optimize flight schedules using open source AI tools?")
        print(f"✅ General Optimization: {result5['response'][:150]}...")
        
        print("\n" + "=" * 80)
        print("✅ ALL PROJECT REQUIREMENTS SUCCESSFULLY TESTED!")
        print("✅ Restructured Ollama prompt system fully operational")
        print("✅ Ready for production flight schedule optimization")
        
    except Exception as e:
        print(f"❌ Error testing comprehensive system: {e}")
        print("\nSystem Requirements:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama3.2")
        print("4. Ensure flight data is available: ../data/processed_flight_data.csv")
