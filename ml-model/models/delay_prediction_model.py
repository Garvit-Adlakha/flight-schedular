import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
from config import *

class FlightDelayPredictor:
    """
    ML Model to predict flight delays and optimize scheduling
    """
    
    def __init__(self, data_file=None):
        if data_file is None:
            data_file = PROCESSED_DATA_FILE
        
        self.df = pd.read_csv(data_file)
        self.delay_regressor = None
        self.delay_classifier = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Prepare data
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare features for ML models"""
        print("Preparing features for ML models...")
        
        # Convert date to datetime
        self.df['Flight_Date'] = pd.to_datetime(self.df['Flight_Date'])
        
        # Create additional time-based features
        self.df['Hour_Sin'] = np.sin(2 * np.pi * self.df['Scheduled_Departure_Hour'] / 24)
        self.df['Hour_Cos'] = np.cos(2 * np.pi * self.df['Scheduled_Departure_Hour'] / 24)
        self.df['Day_of_Year'] = self.df['Flight_Date'].dt.dayofyear
        self.df['Day_Sin'] = np.sin(2 * np.pi * self.df['Day_of_Year'] / 365)
        self.df['Day_Cos'] = np.cos(2 * np.pi * self.df['Day_of_Year'] / 365)
        
        # Encode categorical variables
        categorical_columns = ['Origin_Airport', 'Destination_Airport', 'Route', 'Weekday_Name', 'Time_Period_Category']
        
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle NaN values
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        # Select features for modeling
        self.feature_columns = [
            'Scheduled_Departure_Hour', 'Scheduled_Arrival_Hour',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
            'Route_Avg_Dep_Delay', 'Route_Popularity_Rank',
            'Is_Weekend', 'Is_Peak_Hour'
        ]
        
        # Add encoded categorical features
        for col in categorical_columns:
            if f'{col}_Encoded' in self.df.columns:
                self.feature_columns.append(f'{col}_Encoded')
        
        # Remove any columns that don't exist
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        print(f"Selected {len(self.feature_columns)} features for modeling")
        
    def train_delay_prediction_models(self):
        """Train both regression and classification models for delay prediction"""
        print("\nTraining delay prediction models...")
        
        # Prepare data
        X = self.df[self.feature_columns].fillna(0)
        y_reg = self.df['Departure_Delay_Minutes'].fillna(0)  # Regression target
        y_clf = self.df['Is_Delayed_Departure'].fillna(False)  # Classification target
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regression model (predict delay minutes)
        print("Training regression model...")
        self.delay_regressor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.delay_regressor.fit(X_train_scaled, y_reg_train)
        
        # Train classification model (predict if delayed)
        print("Training classification model...")
        self.delay_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.delay_classifier.fit(X_train_scaled, y_clf_train)
        
        # Evaluate models
        reg_score = self.delay_regressor.score(X_test_scaled, y_reg_test)
        clf_score = self.delay_classifier.score(X_test_scaled, y_clf_test)
        
        reg_mae = mean_absolute_error(y_reg_test, self.delay_regressor.predict(X_test_scaled))
        
        print(f"\nModel Performance:")
        print(f"Regression R² Score: {reg_score:.3f}")
        print(f"Regression MAE: {reg_mae:.2f} minutes")
        print(f"Classification Accuracy: {clf_score:.3f}")
        
        # Feature importance
        self._analyze_feature_importance()
        
        # Save models
        self._save_models()
        
        return {
            'regression_score': reg_score,
            'regression_mae': reg_mae,
            'classification_accuracy': clf_score
        }
    
    def _analyze_feature_importance(self):
        """Analyze and display feature importance"""
        if self.delay_regressor is None:
            return
        
        print("\nFeature Importance Analysis:")
        
        # Get feature importance from regression model
        importance_scores = self.delay_regressor.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def predict_delay(self, flight_features):
        """Predict delay for a given flight"""
        if self.delay_regressor is None or self.delay_classifier is None:
            raise ValueError("Models not trained yet. Call train_delay_prediction_models() first.")
        
        # Prepare features
        features_df = pd.DataFrame([flight_features])
        features_scaled = self.scaler.transform(features_df[self.feature_columns].fillna(0))
        
        # Predict
        delay_minutes = self.delay_regressor.predict(features_scaled)[0]
        delay_probability = self.delay_classifier.predict_proba(features_scaled)[0][1]
        is_delayed = delay_probability > 0.5
        
        return {
            'predicted_delay_minutes': max(0, delay_minutes),  # Can't have negative delays
            'delay_probability': delay_probability,
            'is_delayed': is_delayed
        }
    
    def find_optimal_departure_time(self, route, date, original_hour=None):
        """Find the optimal departure time for a given route and date"""
        print(f"\nFinding optimal departure time for route {route}...")
        
        # Test different hours
        hours_to_test = list(range(5, 24))  # 5 AM to 11 PM
        results = []
        
        base_features = self._get_base_features_for_route(route, date)
        
        for hour in hours_to_test:
            test_features = base_features.copy()
            test_features.update({
                'Scheduled_Departure_Hour': hour,
                'Hour_Sin': np.sin(2 * np.pi * hour / 24),
                'Hour_Cos': np.cos(2 * np.pi * hour / 24),
                'Is_Peak_Hour': hour in [6, 10]  # Based on our analysis
            })
            
            try:
                prediction = self.predict_delay(test_features)
                results.append({
                    'Hour': hour,
                    'Predicted_Delay': prediction['predicted_delay_minutes'],
                    'Delay_Probability': prediction['delay_probability'],
                    'Is_Delayed': prediction['is_delayed']
                })
            except:
                continue
        
        if not results:
            return None
        
        results_df = pd.DataFrame(results)
        
        # Find best options
        best_overall = results_df.loc[results_df['Predicted_Delay'].idxmin()]
        best_reliable = results_df.loc[results_df['Delay_Probability'].idxmin()]
        
        # Sort by different criteria to get comprehensive recommendations
        results_df_sorted_delay = results_df.sort_values('Predicted_Delay')
        results_df_sorted_prob = results_df.sort_values('Delay_Probability')
        
        # Get top 3 best times and worst times
        top_3_by_delay = results_df_sorted_delay.head(3)
        top_3_by_reliability = results_df_sorted_prob.head(3)
        worst_3_times = results_df_sorted_delay.tail(3)
        
        recommendations = {
            'best_time_overall': {
                'hour': int(best_overall['Hour']),
                'time_slot': f"{int(best_overall['Hour']):02d}:00-{int(best_overall['Hour'])+1:02d}:00",
                'predicted_delay': float(best_overall['Predicted_Delay']),
                'delay_probability': float(best_overall['Delay_Probability']),
                'on_time_probability': round((1 - float(best_overall['Delay_Probability'])) * 100, 1),
                'rationale': f"Lowest expected delay of {float(best_overall['Predicted_Delay']):.1f} minutes"
            },
            'most_reliable_time': {
                'hour': int(best_reliable['Hour']),
                'time_slot': f"{int(best_reliable['Hour']):02d}:00-{int(best_reliable['Hour'])+1:02d}:00",
                'predicted_delay': float(best_reliable['Predicted_Delay']),
                'delay_probability': float(best_reliable['Delay_Probability']),
                'on_time_probability': round((1 - float(best_reliable['Delay_Probability'])) * 100, 1),
                'rationale': f"Highest on-time probability of {round((1 - float(best_reliable['Delay_Probability'])) * 100, 1)}%"
            },
            'alternative_times': [
                {
                    'hour': int(row['Hour']),
                    'time_slot': f"{int(row['Hour']):02d}:00-{int(row['Hour'])+1:02d}:00",
                    'predicted_delay': float(row['Predicted_Delay']),
                    'on_time_probability': round((1 - float(row['Delay_Probability'])) * 100, 1),
                    'ranking': idx + 1
                }
                for idx, (_, row) in enumerate(top_3_by_delay.iterrows()) if idx < 3
            ],
            'times_to_avoid': [
                {
                    'hour': int(row['Hour']),
                    'time_slot': f"{int(row['Hour']):02d}:00-{int(row['Hour'])+1:02d}:00",
                    'predicted_delay': float(row['Predicted_Delay']),
                    'delay_probability': float(row['Delay_Probability']),
                    'reason': f"High delay risk: {float(row['Predicted_Delay']):.1f} min expected delay"
                }
                for _, row in worst_3_times.iterrows()
            ],
            'operational_insights': {
                'best_window': f"{int(top_3_by_delay.iloc[0]['Hour']):02d}:00-{int(top_3_by_delay.iloc[2]['Hour'])+1:02d}:00",
                'avg_delay_best_window': float(top_3_by_delay['Predicted_Delay'].mean()),
                'avg_delay_worst_window': float(worst_3_times['Predicted_Delay'].mean()),
                'improvement_potential': float(worst_3_times['Predicted_Delay'].mean() - top_3_by_delay['Predicted_Delay'].mean())
            },
            'all_options': results_df.to_dict('records')
        }
        
        if original_hour:
            original_prediction = results_df[results_df['Hour'] == original_hour]
            if not original_prediction.empty:
                recommendations['original_time'] = {
                    'hour': original_hour,
                    'predicted_delay': float(original_prediction.iloc[0]['Predicted_Delay']),
                    'delay_probability': float(original_prediction.iloc[0]['Delay_Probability'])
                }
        
        return recommendations
    
    def _get_base_features_for_route(self, route, date):
        """Get base features for a route and date"""
        # Get route statistics from existing data
        route_data = self.df[self.df['Route'] == route]
        
        if route_data.empty:
            # Default values if route not found
            return {
                'Route_Avg_Dep_Delay': self.df['Route_Avg_Dep_Delay'].mean(),
                'Route_Popularity_Rank': 50,
                'Scheduled_Arrival_Hour': 12,  # Default
                'Is_Weekend': pd.to_datetime(date).weekday() >= 5,
                'Day_Sin': np.sin(2 * np.pi * pd.to_datetime(date).dayofyear / 365),
                'Day_Cos': np.cos(2 * np.pi * pd.to_datetime(date).dayofyear / 365),
            }
        
        # Use actual route statistics
        sample_flight = route_data.iloc[0]
        
        return {
            'Route_Avg_Dep_Delay': sample_flight['Route_Avg_Dep_Delay'],
            'Route_Popularity_Rank': sample_flight['Route_Popularity_Rank'],
            'Scheduled_Arrival_Hour': sample_flight['Scheduled_Arrival_Hour'],
            'Is_Weekend': pd.to_datetime(date).weekday() >= 5,
            'Day_Sin': np.sin(2 * np.pi * pd.to_datetime(date).dayofyear / 365),
            'Day_Cos': np.cos(2 * np.pi * pd.to_datetime(date).dayofyear / 365),
        }
    
    def identify_busiest_times_to_avoid(self):
        """Identify the busiest time slots that should be avoided"""
        print("\nIdentifying busiest time slots to avoid...")
        
        # Analyze by hour
        hourly_stats = self.df.groupby('Scheduled_Departure_Hour').agg({
            'Flight_Number': 'count',
            'Departure_Delay_Minutes': 'mean',
            'Is_Delayed_Departure': 'mean'
        }).round(2)
        
        hourly_stats.columns = ['Flight_Count', 'Avg_Delay', 'Delay_Rate']
        
        # Calculate congestion score
        hourly_stats['Congestion_Score'] = (
            hourly_stats['Flight_Count'] * 0.4 +
            hourly_stats['Avg_Delay'] * 0.3 +
            hourly_stats['Delay_Rate'] * 100 * 0.3
        )
        
        # Identify times to avoid (top 25% congestion)
        congestion_threshold = hourly_stats['Congestion_Score'].quantile(0.75)
        times_to_avoid = hourly_stats[hourly_stats['Congestion_Score'] >= congestion_threshold]
        
        recommendations = {
            'times_to_avoid': times_to_avoid.index.tolist(),
            'recommended_times': hourly_stats[hourly_stats['Congestion_Score'] < congestion_threshold].index.tolist(),
            'hourly_analysis': hourly_stats.to_dict('index')
        }
        
        print(f"Times to avoid (high congestion): {recommendations['times_to_avoid']}")
        print(f"Recommended times (low congestion): {recommendations['recommended_times']}")
        
        return recommendations
    
    def simulate_schedule_change_impact(self, flight_id, new_hour, route):
        """Simulate the impact of changing a flight's schedule"""
        print(f"\nSimulating schedule change for flight {flight_id} to hour {new_hour}...")
        
        # Get original flight data
        original_flight = self.df[self.df['Flight_Number'] == flight_id]
        
        if original_flight.empty:
            print(f"Flight {flight_id} not found in data")
            return None
        
        original_flight = original_flight.iloc[0]
        original_hour = original_flight['Scheduled_Departure_Hour']
        
        # Predict original delay
        original_features = {col: original_flight[col] for col in self.feature_columns if col in original_flight.index}
        original_prediction = self.predict_delay(original_features)
        
        # Predict new delay
        new_features = original_features.copy()
        new_features.update({
            'Scheduled_Departure_Hour': new_hour,
            'Hour_Sin': np.sin(2 * np.pi * new_hour / 24),
            'Hour_Cos': np.cos(2 * np.pi * new_hour / 24),
            'Is_Peak_Hour': new_hour in [6, 10]
        })
        
        new_prediction = self.predict_delay(new_features)
        
        # Calculate impact
        delay_change = new_prediction['predicted_delay_minutes'] - original_prediction['predicted_delay_minutes']
        probability_change = new_prediction['delay_probability'] - original_prediction['delay_probability']
        
        impact_analysis = {
            'flight_id': flight_id,
            'original_hour': int(original_hour),
            'new_hour': new_hour,
            'original_prediction': original_prediction,
            'new_prediction': new_prediction,
            'delay_change_minutes': delay_change,
            'probability_change': probability_change,
            'recommendation': 'BENEFICIAL' if delay_change < -5 else 'NEUTRAL' if abs(delay_change) <= 5 else 'DETRIMENTAL'
        }
        
        print(f"Schedule change impact:")
        print(f"  Delay change: {delay_change:+.1f} minutes")
        print(f"  Probability change: {probability_change:+.3f}")
        print(f"  Recommendation: {impact_analysis['recommendation']}")
        
        return impact_analysis
    
    def _save_models(self):
        """Save trained models"""
        model_dir = Path("../models")
        model_dir.mkdir(exist_ok=True)
        
        if self.delay_regressor:
            joblib.dump(self.delay_regressor, model_dir / "delay_regressor.pkl")
        if self.delay_classifier:
            joblib.dump(self.delay_classifier, model_dir / "delay_classifier.pkl")
        
        joblib.dump(self.scaler, model_dir / "feature_scaler.pkl")
        joblib.dump(self.label_encoders, model_dir / "label_encoders.pkl")
        joblib.dump(self.feature_columns, model_dir / "feature_columns.pkl")
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        model_dir = Path("../models")
        
        try:
            self.delay_regressor = joblib.load(model_dir / "delay_regressor.pkl")
            self.delay_classifier = joblib.load(model_dir / "delay_classifier.pkl")
            self.scaler = joblib.load(model_dir / "feature_scaler.pkl")
            self.label_encoders = joblib.load(model_dir / "label_encoders.pkl")
            self.feature_columns = joblib.load(model_dir / "feature_columns.pkl")
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved models found. Train models first.")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = FlightDelayPredictor()
    
    # Train models
    performance = predictor.train_delay_prediction_models()
    
    # Test optimal time finding
    optimal_times = predictor.find_optimal_departure_time("BOM-DEL", "2025-07-26")
    if optimal_times:
        print(f"\nOptimal times for BOM-DEL:")
        print(f"Best overall: {optimal_times['best_time_overall']['hour']}:00 "
              f"(predicted delay: {optimal_times['best_time_overall']['predicted_delay']:.1f} min)")
    
    # Identify busy times
    busy_times = predictor.identify_busiest_times_to_avoid()
    
    # Test schedule change simulation
    # impact = predictor.simulate_schedule_change_impact("AI2509", 8, "BOM-IXC")
    
    print("\n" + "="*50)
    print("DELAY PREDICTION MODEL TRAINING COMPLETE!")
    print("="*50)
