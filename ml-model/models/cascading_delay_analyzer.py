import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
from config import *

class CascadingDelayAnalyzer:
    """
    Analyze and predict cascading effects of flight delays
    """
    
    def __init__(self, data_file=None):
        if data_file is None:
            data_file = PROCESSED_DATA_FILE
        # Accept either a path or a preloaded DataFrame
        if isinstance(data_file, pd.DataFrame):
            self.df = data_file.copy()
        else:
            self.df = pd.read_csv(data_file)
        # Normalize column names (make robust to upper/lower or alternate names)
        self._normalize_columns()
        # Parse Flight_Date safely (may be absent or already parsed)
        if 'Flight_Date' in self.df.columns:
            self.df['Flight_Date'] = pd.to_datetime(self.df['Flight_Date'], errors='coerce')
        elif 'flight_date' in self.df.columns:
            self.df['Flight_Date'] = pd.to_datetime(self.df['flight_date'], errors='coerce')
        else:
            self.df['Flight_Date'] = pd.NaT
        self.aircraft_networks = {}
        self.route_networks = {}
        
        # Prepare data for cascade analysis
        self._prepare_cascade_data()
    
    def _prepare_cascade_data(self):
        """Prepare data for cascading delay analysis"""
        print("Preparing cascading delay analysis data...")
        
        # Build full datetimes for scheduled/actual times by combining Flight_Date + time columns
        # This makes time arithmetic robust and allows cross-midnight handling
        def combine_date_time(row, date_col, time_col):
            try:
                date_part = row.get(date_col)
                time_part = row.get(time_col)
                if pd.isna(date_part) or date_part is None:
                    return pd.NaT
                if pd.isna(time_part) or time_part is None:
                    return pd.NaT
                # Convert both to strings and combine
                date_str = pd.to_datetime(date_part).strftime('%Y-%m-%d')
                # ensure time_part is a parsable string
                t = str(time_part)
                return pd.to_datetime(f"{date_str} {t}", errors='coerce')
            except Exception:
                return pd.NaT

        self.df['STD_dt'] = self.df.apply(lambda r: combine_date_time(r, 'Flight_Date', 'STD_parsed'), axis=1)
        self.df['STA_dt'] = self.df.apply(lambda r: combine_date_time(r, 'Flight_Date', 'STA_parsed'), axis=1)
        self.df['ATD_dt'] = self.df.apply(lambda r: combine_date_time(r, 'Flight_Date', 'ATD_parsed'), axis=1)
        self.df['ATA_dt'] = self.df.apply(lambda r: combine_date_time(r, 'Flight_Date', 'ATA_parsed'), axis=1)

        # When actual times are missing, fall back to scheduled datetimes where sensible
        self.df['ATD_dt'] = self.df['ATD_dt'].fillna(self.df['STD_dt'])
        self.df['ATA_dt'] = self.df['ATA_dt'].fillna(self.df['STA_dt'])

        # Sort by aircraft then datetime for sequence analysis
        self.df = self.df.sort_values(['Aircraft', 'STD_dt'])

        # Create unique flight identifiers using ISO datetimes for stability
        self.df['Flight_ID'] = (
            self.df['Flight_Number'].astype(str) + '_' + 
            self.df['STD_dt'].dt.strftime('%Y%m%dT%H%M%S').fillna('NA')
        )

    def _normalize_columns(self):
        """Create standardized column names used by the analyzer.
        This maps various possible column names from processed CSVs or DB exports
        into the expected names (Flight_Number, Flight_Date, STD_parsed, etc.).
        """
        cols = list(self.df.columns)

        def find_col(candidates):
            # exact match first, then substring match
            for cand in candidates:
                for c in cols:
                    if c == cand:
                        return c
            for cand in candidates:
                for c in cols:
                    if cand.lower() in c.lower():
                        return c
            return None

        mapping = {
            'Flight_Number': ['Flight_Number', 'flight_number', 'FlightNumber', 'Flight Number', 'Flight_Number'],
            'Flight_Date': ['Flight_Date', 'flight_date', 'date', 'Date'],
            'STD_parsed': ['STD_parsed', 'scheduled_departure_time', 'scheduled_departure', 'STD', 'std_parsed'],
            'STA_parsed': ['STA_parsed', 'scheduled_arrival_time', 'scheduled_arrival', 'STA', 'sta_parsed'],
            'ATD_parsed': ['ATD_parsed', 'actual_departure_time', 'actual_departure', 'ATD', 'atd_parsed'],
            'ATA_parsed': ['ATA_parsed', 'actual_arrival_time', 'actual_arrival', 'ATA', 'ata_parsed'],
            'Departure_Delay_Minutes': ['departure_delay_minutes', 'Departure_Delay_Minutes', 'dep_delay', 'Departure Delay'],
            'Arrival_Delay_Minutes': ['arrival_delay_minutes', 'Arrival_Delay_Minutes', 'arr_delay', 'Arrival Delay'],
            'Route': ['Route', 'route'],
            'Aircraft': ['Aircraft', 'aircraft', 'Tail_Number', 'Registration']
        }

        for std_name, candidates in mapping.items():
            col = find_col(candidates)
            if col is not None:
                # copy/alias column to standardized name
                try:
                    self.df[std_name] = self.df[col]
                except Exception:
                    # fallback: create empty column if mapping fails
                    self.df[std_name] = None
            else:
                # create empty standardized column if not present
                self.df[std_name] = None

        # Keep parsed time columns as-is (strings or time-like). We'll build full datetimes later.
        for tcol in ['STD_parsed', 'STA_parsed', 'ATD_parsed', 'ATA_parsed']:
            if tcol in self.df.columns:
                try:
                    self.df[tcol] = self.df[tcol].where(self.df[tcol].notna(), None)
                except Exception:
                    self.df[tcol] = None

        # Normalize aircraft identifiers to a compact canonical form to avoid spurious splits
        def normalize_aircraft(x):
            if pd.isna(x) or x is None:
                return None
            s = str(x).strip()
            # extract text inside parentheses if present (common export formats)
            import re
            m = re.search(r"\(([^)]+)\)", s)
            if m:
                s = m.group(1)
            s = s.replace('REGISTERED', '').replace('REG', '')
            s = s.replace(' ', '').upper()
            return s or None

        self.df['Aircraft'] = self.df['Aircraft'].apply(normalize_aircraft)
        
    def build_aircraft_dependency_network(self):
        """Build network of flights connected by aircraft usage"""
        print("\nBuilding aircraft dependency network...")
        aircraft_groups = self.df.groupby('Aircraft')

        rotation_id = 0

        for aircraft, group in aircraft_groups:
            if pd.isna(aircraft) or aircraft is None:
                continue

            # Ensure rows are sorted by scheduled datetime
            group = group.sort_values('STD_dt')

            # split into rotations when gap between flights is large (e.g., >12 hours)
            rotations = []
            current_rotation = []
            prev_arrival = None

            for _, row in group.iterrows():
                std = row.get('STD_dt')
                ata = row.get('ATA_dt')

                if pd.isna(std):
                    # skip rows without scheduled datetime
                    continue

                if prev_arrival is None:
                    current_rotation = [row]
                else:
                    gap = (std - prev_arrival).total_seconds() / 60.0 if pd.notna(prev_arrival) else None
                    # If gap is too large, start a new rotation
                    if gap is None or gap > 12 * 60:
                        if current_rotation:
                            rotations.append(current_rotation)
                        current_rotation = [row]
                    else:
                        current_rotation.append(row)

                # update prev_arrival using ATA_dt if available else STA_dt
                prev_arrival = ata if pd.notna(ata) else row.get('STA_dt')

            if current_rotation:
                rotations.append(current_rotation)

            # Build dependency networks for each rotation
            for rot in rotations:
                if len(rot) < 2:
                    continue

                # Build graph
                G = nx.DiGraph()

                # Add nodes
                for r in rot:
                    flight = {
                        'flight_id': r['Flight_ID'],
                        'flight_number': r['Flight_Number'],
                        'departure_time': r['STD_dt'],
                        'arrival_time': r['STA_dt'],
                        'actual_departure': r['ATD_dt'],
                        'actual_arrival': r['ATA_dt'],
                        'departure_delay': r['Departure_Delay_Minutes'],
                        'arrival_delay': r['Arrival_Delay_Minutes'],
                        'route': r['Route']
                    }
                    G.add_node(flight['flight_id'], **flight)

                # Add edges between consecutive flights
                for i in range(len(rot) - 1):
                    cur = rot[i]
                    nxt = rot[i + 1]

                    cur_arr = cur.get('ATA_dt') if pd.notna(cur.get('ATA_dt')) else cur.get('STA_dt')
                    nxt_dep = nxt.get('STD_dt')

                    try:
                        if pd.isna(cur_arr) or pd.isna(nxt_dep):
                            raise ValueError('Missing datetime')

                        turnaround_minutes = (nxt_dep - cur_arr).total_seconds() / 60.0
                        # if negative (shouldn't be after combining dates) handle by adding 24h
                        if turnaround_minutes < 0:
                            turnaround_minutes += 24 * 60

                        cascade_risk = self._calculate_cascade_risk(
                            cur.get('Departure_Delay_Minutes'),
                            cur.get('Arrival_Delay_Minutes'),
                            turnaround_minutes
                        )

                        G.add_edge(cur['Flight_ID'], nxt['Flight_ID'], turnaround_minutes=turnaround_minutes, cascade_risk=cascade_risk)
                    except Exception:
                        G.add_edge(cur['Flight_ID'], nxt['Flight_ID'], turnaround_minutes=np.nan, cascade_risk='unknown')

                # store graph keyed by aircraft + rotation index
                self.aircraft_networks[(aircraft, f'rot_{rotation_id}')] = G
                rotation_id += 1
        
        print(f"Built dependency networks for {len(self.aircraft_networks)} aircraft-date combinations")
        
    def _calculate_cascade_risk(self, dep_delay, arr_delay, turnaround_minutes):
        """Calculate the risk of cascading delays"""
        if pd.isna(dep_delay) or pd.isna(arr_delay) or pd.isna(turnaround_minutes):
            return 'unknown'
        
        # Risk factors
        arrival_delay_factor = max(0, arr_delay) / 60  # Hours of arrival delay
        turnaround_factor = max(0, (120 - turnaround_minutes)) / 120  # Risk increases with shorter turnaround
        
        risk_score = arrival_delay_factor * 0.7 + turnaround_factor * 0.3
        
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def identify_high_impact_flights(self):
        """Identify flights that have the biggest cascading impact"""
        print("\nIdentifying high-impact flights...")
        
        if not self.aircraft_networks:
            self.build_aircraft_dependency_network()
        
        high_impact_flights = []
        
        for (aircraft, date), network in self.aircraft_networks.items():
            # Calculate metrics for each flight in the network
            for flight_id in network.nodes():
                flight_data = network.nodes[flight_id]
                
                # Calculate downstream impact
                downstream_flights = list(nx.descendants(network, flight_id))
                downstream_count = len(downstream_flights)
                
                # Calculate total potential delay propagation
                total_cascade_delay = 0
                high_risk_cascades = 0
                
                for successor in network.successors(flight_id):
                    edge_data = network.edges[flight_id, successor]
                    if edge_data['cascade_risk'] == 'high':
                        high_risk_cascades += 1
                        total_cascade_delay += flight_data.get('departure_delay', 0)
                
                # Calculate impact score
                impact_score = (
                    downstream_count * 0.4 +
                    high_risk_cascades * 0.3 +
                    (total_cascade_delay / 60) * 0.3  # Convert to hours
                )
                
                high_impact_flights.append({
                    'flight_id': flight_id,
                    'flight_number': flight_data['flight_number'],
                    'aircraft': aircraft,
                    'date': date,
                    'route': flight_data['route'],
                    'departure_delay': flight_data['departure_delay'],
                    'arrival_delay': flight_data['arrival_delay'],
                    'downstream_flights': downstream_count,
                    'high_risk_cascades': high_risk_cascades,
                    'impact_score': impact_score,
                    'cascade_risk_level': self._categorize_impact_level(impact_score)
                })
        
        # Sort by impact score
        high_impact_flights.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Convert to DataFrame for easier analysis
        impact_df = pd.DataFrame(high_impact_flights)
        
        print(f"Analyzed {len(high_impact_flights)} flights for cascading impact")
        
        if not impact_df.empty:
            print(f"\nTop 10 Highest Impact Flights:")
            top_10 = impact_df.head(10)[['flight_number', 'route', 'departure_delay', 'downstream_flights', 'impact_score', 'cascade_risk_level']]
            print(top_10.to_string(index=False))
        
        return impact_df
    
    def _categorize_impact_level(self, impact_score):
        """Categorize impact level based on score"""
        if impact_score > 2.0:
            return 'Critical'
        elif impact_score > 1.0:
            return 'High'
        elif impact_score > 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def analyze_route_congestion_cascades(self):
        """Analyze how route congestion creates cascading delays"""
        print("\nAnalyzing route congestion cascades...")
        
        # Group by route and time
        route_time_analysis = self.df.groupby(['Route', 'Scheduled_Departure_Hour']).agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': ['mean', 'std', 'max'],
            'Arrival_Delay_Minutes': ['mean', 'std', 'max'],
            'Is_Delayed_Departure': 'mean'
        }).round(2)
        
        route_time_analysis.columns = [
            'Flight_Count', 'Dep_Delay_Mean', 'Dep_Delay_Std', 'Dep_Delay_Max',
            'Arr_Delay_Mean', 'Arr_Delay_Std', 'Arr_Delay_Max', 'Delay_Rate'
        ]
        
        # Calculate congestion cascade score
        route_time_analysis['Cascade_Score'] = (
            route_time_analysis['Flight_Count'] * 0.3 +
            route_time_analysis['Dep_Delay_Mean'] * 0.2 +
            route_time_analysis['Dep_Delay_Max'] * 0.1 +
            route_time_analysis['Delay_Rate'] * 100 * 0.4
        )
        
        # Identify high cascade risk route-time combinations
        high_cascade_risk = route_time_analysis[
            route_time_analysis['Cascade_Score'] > route_time_analysis['Cascade_Score'].quantile(0.8)
        ].sort_values('Cascade_Score', ascending=False)
        
        print(f"High cascade risk route-time combinations:")
        print(high_cascade_risk.head(10))
        
        return route_time_analysis, high_cascade_risk
    
    def simulate_delay_propagation(self, initial_flight_id, initial_delay_minutes):
        """Simulate how a delay propagates through the system"""
        print(f"\nSimulating delay propagation from flight {initial_flight_id} with {initial_delay_minutes} min delay...")
        
        if not self.aircraft_networks:
            self.build_aircraft_dependency_network()
        
        # Find the network containing this flight
        target_network = None
        target_key = None
        
        for key, network in self.aircraft_networks.items():
            if initial_flight_id in network.nodes():
                target_network = network
                target_key = key
                break
        
        if target_network is None:
            print(f"Flight {initial_flight_id} not found in any aircraft network")
            return None
        
        # Simulate propagation
        propagation_results = []
        
        def propagate_delay(flight_id, delay_minutes, depth=0):
            """Recursively propagate delay through network"""
            if depth > 5:  # Prevent infinite recursion
                return
            
            flight_data = target_network.nodes[flight_id]
            
            propagation_results.append({
                'flight_id': flight_id,
                'flight_number': flight_data['flight_number'],
                'route': flight_data['route'],
                'depth': depth,
                'propagated_delay': delay_minutes,
                'original_delay': flight_data.get('departure_delay', 0)
            })
            
            # Propagate to successor flights
            for successor in target_network.successors(flight_id):
                edge_data = target_network.edges[flight_id, successor]
                turnaround_time = edge_data.get('turnaround_minutes', 120)
                
                # Calculate propagated delay (assuming some recovery during turnaround)
                if pd.notna(turnaround_time) and delay_minutes > 0:
                    # Recovery factor based on turnaround time
                    recovery_factor = min(0.8, turnaround_time / 120)  # More time = more recovery
                    propagated_delay = max(0, delay_minutes * (1 - recovery_factor))
                    
                    if propagated_delay > 5:  # Only propagate significant delays
                        propagate_delay(successor, propagated_delay, depth + 1)
        
        # Start propagation
        propagate_delay(initial_flight_id, initial_delay_minutes)
        
        propagation_df = pd.DataFrame(propagation_results)
        
        if not propagation_df.empty:
            print(f"\nDelay Propagation Results:")
            print(propagation_df.to_string(index=False))
            
            total_affected = len(propagation_df) - 1  # Exclude initial flight
            total_propagated_delay = propagation_df['propagated_delay'].sum() - initial_delay_minutes
            
            print(f"\nSummary:")
            print(f"Flights affected downstream: {total_affected}")
            print(f"Total propagated delay: {total_propagated_delay:.1f} minutes")
        
        return propagation_df
    
    def generate_cascade_mitigation_recommendations(self):
        """Generate recommendations to mitigate cascading delays"""
        print("\nGenerating cascade mitigation recommendations...")
        
        # Get high impact flights
        impact_flights = self.identify_high_impact_flights()
        route_cascades, high_risk_routes = self.analyze_route_congestion_cascades()
        
        recommendations = []
        
        # High-impact flight recommendations
        if not impact_flights.empty:
            critical_flights = impact_flights[impact_flights['cascade_risk_level'] == 'Critical']
            
            if not critical_flights.empty:
                recommendations.append({
                    'category': 'Critical Flight Management',
                    'priority': 'HIGH',
                    'description': f'Monitor {len(critical_flights)} critical flights that have high cascading impact',
                    'specific_flights': critical_flights['flight_number'].tolist()[:5],
                    'action': 'Implement priority handling and buffer time for these flights'
                })
        
        # Route-time recommendations
        if not high_risk_routes.empty:
            top_risk_routes = high_risk_routes.head(5)
            recommendations.append({
                'category': 'Route Congestion Management',
                'priority': 'MEDIUM',
                'description': 'Address high cascade risk on specific route-time combinations',
                'details': top_risk_routes.index.tolist(),
                'action': 'Consider redistributing flights from these route-time slots'
            })
        
        # Aircraft utilization recommendations
        high_utilization_aircraft = self._analyze_aircraft_utilization()
        if high_utilization_aircraft:
            recommendations.append({
                'category': 'Aircraft Utilization',
                'priority': 'MEDIUM',
                'description': f'{len(high_utilization_aircraft)} aircraft have high utilization with cascade risk',
                'aircraft': high_utilization_aircraft[:5],
                'action': 'Increase turnaround buffer time for high-utilization aircraft'
            })
        
        # Export recommendations
        self._export_cascade_analysis(impact_flights, recommendations)
        
        return recommendations
    
    def _analyze_aircraft_utilization(self):
        """Analyze aircraft utilization patterns"""
        high_utilization = []
        
        for (aircraft, date), network in self.aircraft_networks.items():
            if len(network.nodes()) >= 3:  # 3+ flights per day
                high_risk_edges = [
                    edge for edge in network.edges(data=True)
                    if edge[2].get('cascade_risk') == 'high'
                ]
                
                if len(high_risk_edges) >= 1:
                    high_utilization.append(aircraft)
        
        return list(set(high_utilization))
    
    def _export_cascade_analysis(self, impact_flights, recommendations):
        """Export cascade analysis results"""
        
        # Export to Excel
        with pd.ExcelWriter('../analysis/cascade_analysis_results.xlsx', engine='openpyxl') as writer:
            if not impact_flights.empty:
                impact_flights.to_excel(writer, sheet_name='High_Impact_Flights', index=False)
            
            # Export recommendations
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        # Export summary to text
        with open('../analysis/cascade_mitigation_plan.txt', 'w') as f:
            f.write("CASCADING DELAY MITIGATION PLAN\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec['category']} (Priority: {rec['priority']})\n")
                f.write(f"   Description: {rec['description']}\n")
                f.write(f"   Action: {rec['action']}\n\n")
        
        print("Cascade analysis results exported to:")
        print("- ../analysis/cascade_analysis_results.xlsx")
        print("- ../analysis/cascade_mitigation_plan.txt")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CascadingDelayAnalyzer()
    
    # Build dependency networks
    analyzer.build_aircraft_dependency_network()
    
    # Analyze high-impact flights
    impact_flights = analyzer.identify_high_impact_flights()
    
    # Analyze route cascades
    route_analysis, high_risk = analyzer.analyze_route_congestion_cascades()
    
    # Generate recommendations
    recommendations = analyzer.generate_cascade_mitigation_recommendations()
    
    print("\n" + "="*50)
    print("CASCADING DELAY ANALYSIS COMPLETE!")
    print("="*50)
