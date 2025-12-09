"""
IoT Integration Module
Integrates environmental sensor data for enhanced disease prediction and monitoring.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np


class IoTSensorData:
    """Class to handle IoT sensor data collection and processing."""
    
    def __init__(self, data_file='iot_data.json'):
        """
        Initialize IoT sensor data handler.
        
        Args:
            data_file: Path to store sensor data
        """
        self.data_file = data_file
        self.data_history = []
        self.load_data()
    
    def load_data(self):
        """Load historical sensor data."""
        if Path(self.data_file).exists():
            try:
                with open(self.data_file, 'r') as f:
                    self.data_history = json.load(f)
            except:
                self.data_history = []
    
    def save_data(self):
        """Save sensor data to file."""
        with open(self.data_file, 'w') as f:
            json.dump(self.data_history, f, indent=2)
    
    def record_reading(self, temperature, humidity, soil_moisture, 
                       light_intensity=None, location=None):
        """
        Record a sensor reading.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (%)
            soil_moisture: Soil moisture level (%)
            light_intensity: Light intensity (lux)
            location: GPS coordinates or location name
        """
        reading = {
            'timestamp': datetime.now().isoformat(),
            'temperature': temperature,
            'humidity': humidity,
            'soil_moisture': soil_moisture,
            'light_intensity': light_intensity,
            'location': location
        }
        
        self.data_history.append(reading)
        self.save_data()
        
        return reading
    
    def get_latest_reading(self):
        """Get the most recent sensor reading."""
        if self.data_history:
            return self.data_history[-1]
        return None
    
    def get_statistics(self, hours=24):
        """
        Get statistical summary of recent sensor data.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not self.data_history:
            return None
        
        # Filter recent data
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_data = [
            r for r in self.data_history 
            if datetime.fromisoformat(r['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_data:
            return None
        
        # Calculate statistics
        temps = [r['temperature'] for r in recent_data]
        humidities = [r['humidity'] for r in recent_data]
        moistures = [r['soil_moisture'] for r in recent_data]
        
        stats = {
            'temperature': {
                'avg': np.mean(temps),
                'min': np.min(temps),
                'max': np.max(temps),
                'current': temps[-1]
            },
            'humidity': {
                'avg': np.mean(humidities),
                'min': np.min(humidities),
                'max': np.max(humidities),
                'current': humidities[-1]
            },
            'soil_moisture': {
                'avg': np.mean(moistures),
                'min': np.min(moistures),
                'max': np.max(moistures),
                'current': moistures[-1]
            },
            'period_hours': hours,
            'num_readings': len(recent_data)
        }
        
        return stats


class DiseaseRiskAssessment:
    """Assess disease risk based on environmental conditions."""
    
    # Disease risk thresholds
    RISK_PROFILES = {
        'Early_blight': {
            'temperature': {'optimal': (24, 29), 'favorable': (20, 32)},
            'humidity': {'optimal': (90, 100), 'favorable': (80, 100)},
            'description': 'High humidity and moderate temperatures favor early blight'
        },
        'Late_blight': {
            'temperature': {'optimal': (10, 25), 'favorable': (7, 27)},
            'humidity': {'optimal': (90, 100), 'favorable': (85, 100)},
            'description': 'Cool, wet conditions are ideal for late blight'
        },
        'Powdery_mildew': {
            'temperature': {'optimal': (20, 30), 'favorable': (15, 35)},
            'humidity': {'optimal': (50, 70), 'favorable': (40, 80)},
            'description': 'Moderate humidity and warm temperatures favor powdery mildew'
        },
        'Leaf_spot': {
            'temperature': {'optimal': (25, 30), 'favorable': (20, 35)},
            'humidity': {'optimal': (85, 100), 'favorable': (75, 100)},
            'description': 'Warm, humid conditions promote leaf spot diseases'
        },
        'Rust': {
            'temperature': {'optimal': (15, 25), 'favorable': (10, 30)},
            'humidity': {'optimal': (95, 100), 'favorable': (85, 100)},
            'description': 'High humidity with moderate temperatures favor rust'
        }
    }
    
    @staticmethod
    def assess_risk(temperature, humidity, disease_type=None):
        """
        Assess disease risk based on environmental conditions.
        
        Args:
            temperature: Current temperature (°C)
            humidity: Current humidity (%)
            disease_type: Specific disease to assess (optional)
            
        Returns:
            Risk assessment dictionary
        """
        if disease_type and disease_type in DiseaseRiskAssessment.RISK_PROFILES:
            # Assess specific disease
            profile = DiseaseRiskAssessment.RISK_PROFILES[disease_type]
            risk_level = DiseaseRiskAssessment._calculate_risk(
                temperature, humidity, profile
            )
            
            return {
                'disease': disease_type,
                'risk_level': risk_level,
                'temperature': temperature,
                'humidity': humidity,
                'description': profile['description']
            }
        else:
            # Assess all diseases
            risks = {}
            for disease, profile in DiseaseRiskAssessment.RISK_PROFILES.items():
                risk = DiseaseRiskAssessment._calculate_risk(
                    temperature, humidity, profile
                )
                risks[disease] = risk
            
            # Find highest risk
            max_risk_disease = max(risks, key=risks.get)
            
            return {
                'temperature': temperature,
                'humidity': humidity,
                'risks': risks,
                'highest_risk_disease': max_risk_disease,
                'highest_risk_level': risks[max_risk_disease]
            }
    
    @staticmethod
    def _calculate_risk(temp, humidity, profile):
        """Calculate risk level (0-100) based on conditions."""
        temp_optimal = profile['temperature']['optimal']
        temp_favorable = profile['temperature']['favorable']
        hum_optimal = profile['humidity']['optimal']
        hum_favorable = profile['humidity']['favorable']
        
        # Temperature risk
        if temp_optimal[0] <= temp <= temp_optimal[1]:
            temp_risk = 100
        elif temp_favorable[0] <= temp <= temp_favorable[1]:
            # Calculate proportional risk
            if temp < temp_optimal[0]:
                temp_risk = 50 + 50 * (temp - temp_favorable[0]) / (temp_optimal[0] - temp_favorable[0])
            else:
                temp_risk = 50 + 50 * (temp_favorable[1] - temp) / (temp_favorable[1] - temp_optimal[1])
        else:
            temp_risk = 0
        
        # Humidity risk
        if hum_optimal[0] <= humidity <= hum_optimal[1]:
            hum_risk = 100
        elif hum_favorable[0] <= humidity <= hum_favorable[1]:
            if humidity < hum_optimal[0]:
                hum_risk = 50 + 50 * (humidity - hum_favorable[0]) / (hum_optimal[0] - hum_favorable[0])
            else:
                hum_risk = 50 + 50 * (hum_favorable[1] - humidity) / (hum_favorable[1] - hum_optimal[1])
        else:
            hum_risk = 0
        
        # Combined risk (average)
        risk = (temp_risk + hum_risk) / 2
        
        return round(risk, 2)
    
    @staticmethod
    def get_recommendations(risk_level):
        """
        Get management recommendations based on risk level.
        
        Args:
            risk_level: Risk level (0-100)
            
        Returns:
            List of recommendations
        """
        if risk_level >= 75:
            return [
                "🔴 HIGH RISK - Immediate action required",
                "Apply preventive fungicides",
                "Increase monitoring frequency",
                "Improve air circulation",
                "Reduce overhead irrigation",
                "Scout fields daily for symptoms"
            ]
        elif risk_level >= 50:
            return [
                "🟠 MODERATE RISK - Preventive measures recommended",
                "Monitor environmental conditions closely",
                "Consider preventive treatments",
                "Ensure good plant spacing",
                "Monitor for early symptoms",
                "Prepare spray equipment"
            ]
        elif risk_level >= 25:
            return [
                "🟡 LOW-MODERATE RISK - Maintain vigilance",
                "Continue routine monitoring",
                "Maintain good cultural practices",
                "Keep records of observations",
                "Check weather forecasts regularly"
            ]
        else:
            return [
                "🟢 LOW RISK - Continue normal operations",
                "Routine monitoring sufficient",
                "Maintain plant health",
                "Good growing conditions"
            ]


def integrate_iot_with_prediction(image_prediction, sensor_data):
    """
    Combine image-based prediction with IoT sensor data for enhanced assessment.
    
    Args:
        image_prediction: Prediction result from the model
        sensor_data: Current sensor readings
        
    Returns:
        Enhanced prediction with environmental context
    """
    if not sensor_data:
        return image_prediction
    
    # Get disease from prediction
    predicted_disease = image_prediction.get('predicted_class', '')
    confidence = image_prediction.get('confidence', 0)
    
    # Extract disease name from class
    disease_name = predicted_disease.split('___')[-1] if '___' in predicted_disease else predicted_disease
    
    # Assess environmental risk
    risk_assessment = DiseaseRiskAssessment.assess_risk(
        temperature=sensor_data.get('temperature', 25),
        humidity=sensor_data.get('humidity', 70),
        disease_type=disease_name
    )
    
    # Combine predictions
    enhanced_prediction = {
        **image_prediction,
        'environmental_data': sensor_data,
        'environmental_risk': risk_assessment,
        'integrated_confidence': (confidence + risk_assessment.get('risk_level', 0) / 100) / 2,
        'recommendations': DiseaseRiskAssessment.get_recommendations(
            risk_assessment.get('risk_level', 0)
        )
    }
    
    return enhanced_prediction


# Simulated sensor data generator (for testing without actual sensors)
def generate_simulated_data():
    """Generate simulated sensor data for testing."""
    return {
        'temperature': round(20 + np.random.randn() * 5, 1),
        'humidity': round(max(30, min(100, 70 + np.random.randn() * 15)), 1),
        'soil_moisture': round(max(0, min(100, 50 + np.random.randn() * 20)), 1),
        'light_intensity': round(max(0, 50000 + np.random.randn() * 20000), 0),
        'location': 'Test Farm'
    }


if __name__ == "__main__":
    print("🌐 IoT Integration Module - Smart Agriculture")
    print("=" * 70)
    
    # Test IoT data collection
    print("\n📊 Testing IoT sensor data collection...")
    iot = IoTSensorData()
    
    # Simulate some readings
    for i in range(3):
        data = generate_simulated_data()
        reading = iot.record_reading(**data)
        print(f"   Reading {i+1}: Temp={reading['temperature']}°C, "
              f"Humidity={reading['humidity']}%, "
              f"Moisture={reading['soil_moisture']}%")
        time.sleep(0.5)
    
    # Get statistics
    print("\n📈 24-hour statistics:")
    stats = iot.get_statistics(24)
    if stats:
        print(f"   Temperature: {stats['temperature']['avg']:.1f}°C "
              f"(range: {stats['temperature']['min']:.1f}-{stats['temperature']['max']:.1f})")
        print(f"   Humidity: {stats['humidity']['avg']:.1f}% "
              f"(range: {stats['humidity']['min']:.1f}-{stats['humidity']['max']:.1f})")
    
    # Test risk assessment
    print("\n🎯 Testing disease risk assessment...")
    risk = DiseaseRiskAssessment.assess_risk(28, 95)
    print(f"   Highest risk: {risk['highest_risk_disease']} "
          f"(Risk level: {risk['highest_risk_level']}%)")
    
    # Get recommendations
    recommendations = DiseaseRiskAssessment.get_recommendations(
        risk['highest_risk_level']
    )
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n✅ IoT integration module ready!")
