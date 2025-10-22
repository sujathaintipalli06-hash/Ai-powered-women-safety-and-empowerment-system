import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta
import random

class ThreatPredictionEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic crime data for training"""
        np.random.seed(42)
        random.seed(42)
        
        data = []
        
        for _ in range(n_samples):
            # Generate random location (simulating a city area)
            latitude = np.random.uniform(28.4, 28.8)  # Delhi area coordinates
            longitude = np.random.uniform(77.0, 77.4)
            
            # Generate time features
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            month = np.random.randint(1, 13)
            
            # Generate contextual features
            population_density = np.random.uniform(100, 10000)
            lighting_quality = np.random.uniform(0, 1)
            police_presence = np.random.uniform(0, 1)
            economic_status = np.random.uniform(0, 1)
            
            # Calculate threat level based on realistic factors
            threat_score = 0
            
            # Time-based factors
            if hour >= 22 or hour <= 6:  # Night time
                threat_score += 0.4
            elif hour >= 18 or hour <= 8:  # Evening/early morning
                threat_score += 0.2
                
            # Day of week (weekends might be riskier)
            if day_of_week in [5, 6]:  # Friday, Saturday
                threat_score += 0.1
                
            # Environmental factors
            threat_score += (1 - lighting_quality) * 0.3
            threat_score += (1 - police_presence) * 0.2
            threat_score += (1 - economic_status) * 0.2
            threat_score -= (population_density / 10000) * 0.1
            
            # Add some randomness
            threat_score += np.random.uniform(-0.1, 0.1)
            
            # Normalize and categorize
            threat_score = max(0, min(1, threat_score))
            
            if threat_score >= 0.7:
                threat_level = 2  # High
            elif threat_score >= 0.4:
                threat_level = 1  # Medium
            else:
                threat_level = 0  # Low
                
            data.append([
                latitude, longitude, hour, day_of_week, month,
                population_density, lighting_quality, police_presence,
                economic_status, threat_level
            ])
        
        columns = [
            'latitude', 'longitude', 'hour', 'day_of_week', 'month',
            'population_density', 'lighting_quality', 'police_presence',
            'economic_status', 'threat_level'
        ]
        
        return pd.DataFrame(data, columns=columns)
    
    def train_model(self):
        """Train the threat prediction model"""
        print("Generating synthetic training data...")
        df = self.generate_synthetic_data()
        
        # Prepare features and target
        feature_columns = [
            'latitude', 'longitude', 'hour', 'day_of_week', 'month',
            'population_density', 'lighting_quality', 'police_presence',
            'economic_status'
        ]
        
        X = df[feature_columns]
        y = df['threat_level']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Low', 'Medium', 'High']))
        
        self.is_trained = True
        
        # Save the model
        self.save_model()
        
        return accuracy
    
    def predict_threat(self, latitude, longitude, hour=None, day_of_week=None, 
                      month=None, population_density=1000, lighting_quality=0.5,
                      police_presence=0.5, economic_status=0.5):
        """Predict threat level for given parameters"""
        if not self.is_trained:
            print("Model not trained. Training now...")
            self.train_model()
        
        # Use current time if not provided
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        if month is None:
            month = datetime.now().month
        
        # Prepare input data
        input_data = np.array([[
            latitude, longitude, hour, day_of_week, month,
            population_density, lighting_quality, police_presence,
            economic_status
        ]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        
        threat_levels = ['Low', 'Medium', 'High']
        
        return {
            'threat_level': threat_levels[prediction],
            'threat_score': prediction,
            'probabilities': {
                'Low': probability[0],
                'Medium': probability[1],
                'High': probability[2]
            }
        }
    
    def save_model(self):
        """Save the trained model and scaler"""
        model_dir = '/home/ubuntu/women_safety_backend/models'
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'threat_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        print(f"Model saved to {model_dir}")
    
    def load_model(self):
        """Load a pre-trained model and scaler"""
        model_dir = '/home/ubuntu/women_safety_backend/models'
        model_path = os.path.join(model_dir, 'threat_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        else:
            print("No pre-trained model found.")
            return False

class AutomaticThreatDetection:
    def __init__(self):
        self.prediction_engine = ThreatPredictionEngine()
        
    def analyze_user_behavior(self, user_locations, time_window_minutes=30):
        """Analyze user behavior patterns for anomaly detection"""
        if len(user_locations) < 2:
            return {'anomaly_detected': False, 'reason': 'Insufficient data'}
        
        # Calculate speed between consecutive locations
        speeds = []
        for i in range(1, len(user_locations)):
            prev_loc = user_locations[i-1]
            curr_loc = user_locations[i]
            
            # Calculate distance (simplified)
            lat_diff = curr_loc['latitude'] - prev_loc['latitude']
            lon_diff = curr_loc['longitude'] - prev_loc['longitude']
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
            
            # Calculate time difference
            time_diff = (curr_loc['timestamp'] - prev_loc['timestamp']).total_seconds() / 3600
            
            if time_diff > 0:
                speed = distance / time_diff  # km/h
                speeds.append(speed)
        
        if not speeds:
            return {'anomaly_detected': False, 'reason': 'No speed data'}
        
        avg_speed = np.mean(speeds)
        max_speed = max(speeds)
        
        # Detect anomalies
        anomalies = []
        
        # Sudden speed increase (possible chase or panic)
        if max_speed > 50:  # km/h
            anomalies.append('Sudden high speed detected')
        
        # Erratic movement pattern
        if len(speeds) > 3 and np.std(speeds) > 20:
            anomalies.append('Erratic movement pattern')
        
        # Stationary for too long in unsafe area
        if avg_speed < 1 and len(user_locations) > 5:
            latest_location = user_locations[-1]
            threat_prediction = self.prediction_engine.predict_threat(
                latest_location['latitude'], 
                latest_location['longitude']
            )
            if threat_prediction['threat_level'] in ['Medium', 'High']:
                anomalies.append('Stationary in potentially unsafe area')
        
        return {
            'anomaly_detected': len(anomalies) > 0,
            'anomalies': anomalies,
            'avg_speed': avg_speed,
            'max_speed': max_speed
        }
    
    def check_safe_zones(self, latitude, longitude):
        """Check if location is in a known safe zone"""
        # Define some safe zones (hospitals, police stations, etc.)
        safe_zones = [
            {'name': 'Police Station', 'lat': 28.6139, 'lon': 77.2090, 'radius': 0.5},
            {'name': 'Hospital', 'lat': 28.6289, 'lon': 77.2065, 'radius': 0.3},
            {'name': 'Shopping Mall', 'lat': 28.5355, 'lon': 77.3910, 'radius': 0.2},
        ]
        
        for zone in safe_zones:
            distance = np.sqrt((latitude - zone['lat'])**2 + (longitude - zone['lon'])**2)
            if distance <= zone['radius'] / 111:  # Convert to degrees
                return {'is_safe_zone': True, 'zone_name': zone['name']}
        
        return {'is_safe_zone': False, 'zone_name': None}

# Initialize the AI engine
ai_engine = ThreatPredictionEngine()
threat_detector = AutomaticThreatDetection()

