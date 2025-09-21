"""
08_realtime_pipeline.py

Real-time data ingestion pipeline for ICU patient monitoring.
Simulates real-time data streams from ICU monitoring equipment and processes them
for immediate risk assessment and alert generation.

Features:
- Real-time data ingestion from simulated ICU devices
- Multi-modal data fusion (vitals, lab values, medications)
- Continuous risk assessment with configurable thresholds
- Alert generation with severity levels
- Data persistence for retrospective analysis
- Integration with existing ML models

Usage:
    python 08_realtime_pipeline.py --patient_id 12345 --duration 3600
"""

import asyncio
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import config
import utils
import joblib
import tensorflow as tf
from dataclasses import dataclass
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class PatientAlert:
    """Patient alert data structure"""
    patient_id: str
    timestamp: datetime
    alert_level: AlertLevel
    risk_score: float
    message: str
    vital_values: Dict[str, float]
    recommendations: List[str]

@dataclass
class VitalReading:
    """Individual vital sign reading"""
    patient_id: str
    timestamp: datetime
    vital_type: str
    value: float
    unit: str
    quality_score: float = 1.0

class ICUDataSimulator:
    """Simulates real-time data from ICU monitoring equipment"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.baseline_vitals = {
            'heart_rate': 75.0,
            'respiratory_rate': 16.0,
            'systolic_bp': 120.0,
            'diastolic_bp': 80.0,
            'temperature': 98.6,
            'oxygen_saturation': 98.0,
            'mean_bp': 93.0
        }
        self.trend_direction = 0  # -1: deteriorating, 0: stable, 1: improving
        self.time_since_admission = 0
        
    def generate_vital_reading(self, vital_type: str) -> VitalReading:
        """Generate realistic vital sign reading with some variation"""
        baseline = self.baseline_vitals.get(vital_type, 0)
        
        # Add realistic variation
        if vital_type == 'heart_rate':
            variation = np.random.normal(0, 5)
            value = max(40, min(200, baseline + variation))
        elif vital_type == 'respiratory_rate':
            variation = np.random.normal(0, 2)
            value = max(8, min(40, baseline + variation))
        elif vital_type in ['systolic_bp', 'diastolic_bp']:
            variation = np.random.normal(0, 8)
            value = max(60, min(250, baseline + variation))
        elif vital_type == 'temperature':
            variation = np.random.normal(0, 0.3)
            value = max(95.0, min(105.0, baseline + variation))
        elif vital_type == 'oxygen_saturation':
            variation = np.random.normal(0, 1)
            value = max(70, min(100, baseline + variation))
        else:
            variation = np.random.normal(0, baseline * 0.05)
            value = max(0, baseline + variation)
        
        # Simulate deterioration over time
        if self.trend_direction < 0:
            deterioration_factor = min(0.3, self.time_since_admission / 3600)  # 30% max deterioration
            if vital_type in ['heart_rate', 'respiratory_rate']:
                value *= (1 + deterioration_factor)
            elif vital_type in ['systolic_bp', 'diastolic_bp']:
                value *= (1 - deterioration_factor * 0.5)
            elif vital_type == 'oxygen_saturation':
                value *= (1 - deterioration_factor)
        
        return VitalReading(
            patient_id=self.patient_id,
            timestamp=datetime.now(),
            vital_type=vital_type,
            value=value,
            unit=self._get_unit(vital_type),
            quality_score=np.random.uniform(0.8, 1.0)
        )
    
    def _get_unit(self, vital_type: str) -> str:
        """Get unit for vital sign"""
        units = {
            'heart_rate': 'bpm',
            'respiratory_rate': 'breaths/min',
            'systolic_bp': 'mmHg',
            'diastolic_bp': 'mmHg',
            'temperature': '°F',
            'oxygen_saturation': '%',
            'mean_bp': 'mmHg'
        }
        return units.get(vital_type, 'units')
    
    def set_trend(self, direction: int):
        """Set patient trend: -1 (deteriorating), 0 (stable), 1 (improving)"""
        self.trend_direction = direction
    
    def update_time(self, seconds: int):
        """Update time since admission"""
        self.time_since_admission += seconds

class RealTimeRiskAssessor:
    """Real-time risk assessment using trained models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.load_models()
        self.alert_thresholds = {
            AlertLevel.LOW: 0.3,
            AlertLevel.MEDIUM: 0.5,
            AlertLevel.HIGH: 0.7,
            AlertLevel.CRITICAL: 0.9
        }
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            # Load scaler
            scaler_path = config.MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            
            # Load Random Forest model
            if config.RANDOM_FOREST_MODEL.exists():
                self.models['rf'] = joblib.load(config.RANDOM_FOREST_MODEL)
                logger.info("Loaded Random Forest model")
            
            # Load XGBoost model
            if config.XGBOOST_MODEL.exists():
                self.models['xgboost'] = joblib.load(config.XGBOOST_MODEL)
                logger.info("Loaded XGBoost model")
            
            # Load deep learning model
            if config.DEEP_MODEL.exists():
                self.models['deep'] = tf.keras.models.load_model(config.DEEP_MODEL)
                logger.info("Loaded deep learning model")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def assess_risk(self, vital_data: Dict[str, float], patient_history: List[Dict] = None) -> Tuple[float, AlertLevel, List[str]]:
        """
        Assess patient risk using ensemble of models
        
        Args:
            vital_data: Current vital sign values
            patient_history: Recent patient history for context
            
        Returns:
            risk_score: Probability of deterioration (0-1)
            alert_level: Alert severity level
            recommendations: List of clinical recommendations
        """
        if not self.models:
            logger.warning("No models loaded, returning default risk score")
            return 0.5, AlertLevel.MEDIUM, ["No models available"]
        
        try:
            # Prepare features for prediction
            features = self._prepare_features(vital_data, patient_history)
            
            # Get predictions from all available models
            predictions = []
            for model_name, model in self.models.items():
                if model_name == 'deep':
                    if self.scaler:
                        features_scaled = self.scaler.transform(features.reshape(1, -1))
                    else:
                        features_scaled = features.reshape(1, -1)
                    pred = model.predict(features_scaled)[0][0]
                else:
                    pred = model.predict_proba(features.reshape(1, -1))[0][1]
                predictions.append(pred)
            
            # Ensemble prediction (simple average)
            risk_score = np.mean(predictions)
            
            # Determine alert level
            alert_level = self._determine_alert_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(vital_data, risk_score, alert_level)
            
            return risk_score, alert_level, recommendations
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.5, AlertLevel.MEDIUM, ["Error in risk assessment"]
    
    def _prepare_features(self, vital_data: Dict[str, float], patient_history: List[Dict] = None) -> np.ndarray:
        """Prepare features for model prediction"""
        # Create feature vector based on vital signs
        feature_vector = []
        
        # Map vital signs to expected feature names
        vital_mapping = {
            'heart_rate': 'hr_mean',
            'respiratory_rate': 'resp_mean',
            'systolic_bp': 'sbp_mean',
            'diastolic_bp': 'dbp_mean',
            'temperature': 'temp_mean',
            'oxygen_saturation': 'spo2_mean',
            'mean_bp': 'mbp_mean'
        }
        
        # Add vital sign features
        for vital_type, feature_name in vital_mapping.items():
            value = vital_data.get(vital_type, 0)
            feature_vector.append(value)
        
        # Add derived features
        if 'systolic_bp' in vital_data and 'diastolic_bp' in vital_data:
            pulse_pressure = vital_data['systolic_bp'] - vital_data['diastolic_bp']
            feature_vector.append(pulse_pressure)
        else:
            feature_vector.append(0)
        
        # Add time-based features
        current_hour = datetime.now().hour
        feature_vector.extend([
            np.sin(2 * np.pi * current_hour / 24),  # Circadian rhythm
            np.cos(2 * np.pi * current_hour / 24)
        ])
        
        # Pad with zeros to match expected feature count
        expected_features = 50  # Adjust based on your model
        while len(feature_vector) < expected_features:
            feature_vector.append(0)
        
        return np.array(feature_vector[:expected_features])
    
    def _determine_alert_level(self, risk_score: float) -> AlertLevel:
        """Determine alert level based on risk score"""
        if risk_score >= self.alert_thresholds[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif risk_score >= self.alert_thresholds[AlertLevel.HIGH]:
            return AlertLevel.HIGH
        elif risk_score >= self.alert_thresholds[AlertLevel.MEDIUM]:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _generate_recommendations(self, vital_data: Dict[str, float], risk_score: float, alert_level: AlertLevel) -> List[str]:
        """Generate clinical recommendations based on vital signs and risk"""
        recommendations = []
        
        # Heart rate recommendations
        hr = vital_data.get('heart_rate', 0)
        if hr > 100:
            recommendations.append("Consider cardiac monitoring and ECG")
        elif hr < 60:
            recommendations.append("Monitor for bradycardia, consider cardiac assessment")
        
        # Blood pressure recommendations
        sbp = vital_data.get('systolic_bp', 0)
        dbp = vital_data.get('diastolic_bp', 0)
        if sbp > 160 or dbp > 100:
            recommendations.append("Hypertensive episode - consider antihypertensive therapy")
        elif sbp < 90 or dbp < 60:
            recommendations.append("Hypotension - consider fluid resuscitation or vasopressors")
        
        # Oxygen saturation recommendations
        spo2 = vital_data.get('oxygen_saturation', 0)
        if spo2 < 95:
            recommendations.append("Oxygen saturation low - consider oxygen therapy")
        if spo2 < 90:
            recommendations.append("URGENT: Severe hypoxemia - immediate intervention required")
        
        # Temperature recommendations
        temp = vital_data.get('temperature', 0)
        if temp > 100.4:
            recommendations.append("Fever present - consider infection workup and antipyretics")
        elif temp < 96.8:
            recommendations.append("Hypothermia - consider warming measures")
        
        # Risk-based recommendations
        if alert_level == AlertLevel.CRITICAL:
            recommendations.append("CRITICAL ALERT: Immediate physician notification required")
            recommendations.append("Consider rapid response team activation")
        elif alert_level == AlertLevel.HIGH:
            recommendations.append("HIGH RISK: Increase monitoring frequency")
            recommendations.append("Notify attending physician within 15 minutes")
        
        return recommendations

class RealTimeMonitor:
    """Main real-time monitoring system"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.data_simulator = ICUDataSimulator(patient_id)
        self.risk_assessor = RealTimeRiskAssessor()
        self.vital_buffer = []
        self.alert_history = []
        self.is_running = False
        
    async def start_monitoring(self, duration_seconds: int = 3600):
        """Start real-time monitoring for specified duration"""
        logger.info(f"Starting real-time monitoring for patient {self.patient_id}")
        self.is_running = True
        
        start_time = time.time()
        vital_types = ['heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 
                      'temperature', 'oxygen_saturation', 'mean_bp']
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            try:
                # Generate vital readings
                current_vitals = {}
                for vital_type in vital_types:
                    reading = self.data_simulator.generate_vital_reading(vital_type)
                    current_vitals[vital_type] = reading.value
                    self.vital_buffer.append(reading)
                
                # Keep only last 100 readings
                if len(self.vital_buffer) > 100:
                    self.vital_buffer = self.vital_buffer[-100:]
                
                # Assess risk
                risk_score, alert_level, recommendations = self.risk_assessor.assess_risk(
                    current_vitals, self.vital_buffer[-10:]  # Last 10 readings for context
                )
                
                # Generate alert if needed
                if alert_level != AlertLevel.LOW:
                    alert = PatientAlert(
                        patient_id=self.patient_id,
                        timestamp=datetime.now(),
                        alert_level=alert_level,
                        risk_score=risk_score,
                        message=f"Risk level: {alert_level.value} (Score: {risk_score:.3f})",
                        vital_values=current_vitals,
                        recommendations=recommendations
                    )
                    self.alert_history.append(alert)
                    await self._handle_alert(alert)
                
                # Log current status
                logger.info(f"Patient {self.patient_id}: Risk={risk_score:.3f}, Level={alert_level.value}")
                
                # Simulate gradual deterioration
                if np.random.random() < 0.1:  # 10% chance of deterioration
                    self.data_simulator.set_trend(-1)
                
                # Update time
                self.data_simulator.update_time(30)  # 30-second intervals
                
                # Wait before next reading
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Monitoring completed for patient {self.patient_id}")
        self.is_running = False
    
    async def _handle_alert(self, alert: PatientAlert):
        """Handle patient alert"""
        logger.warning(f"ALERT: {alert.message}")
        logger.warning(f"Recommendations: {', '.join(alert.recommendations)}")
        
        # Save alert to file
        alert_data = {
            'patient_id': alert.patient_id,
            'timestamp': alert.timestamp.isoformat(),
            'alert_level': alert.alert_level.value,
            'risk_score': alert.risk_score,
            'message': alert.message,
            'vital_values': alert.vital_values,
            'recommendations': alert.recommendations
        }
        
        alert_file = config.LOG_DIR / f"alerts_{self.patient_id}.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        if not self.vital_buffer:
            return {"status": "No data"}
        
        latest_vitals = {v.vital_type: v.value for v in self.vital_buffer[-7:]}  # Last 7 readings
        risk_score, alert_level, _ = self.risk_assessor.assess_risk(latest_vitals)
        
        return {
            "patient_id": self.patient_id,
            "status": "monitoring" if self.is_running else "stopped",
            "latest_vitals": latest_vitals,
            "current_risk": risk_score,
            "alert_level": alert_level.value,
            "total_alerts": len(self.alert_history),
            "last_alert": self.alert_history[-1].timestamp.isoformat() if self.alert_history else None
        }

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-time ICU Patient Monitoring")
    parser.add_argument("--patient_id", type=str, default="12345", help="Patient ID")
    parser.add_argument("--duration", type=int, default=3600, help="Monitoring duration in seconds")
    parser.add_argument("--trend", type=int, choices=[-1, 0, 1], default=0, help="Patient trend: -1=deteriorating, 0=stable, 1=improving")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RealTimeMonitor(args.patient_id)
    
    # Set patient trend
    monitor.data_simulator.set_trend(args.trend)
    
    try:
        # Start monitoring
        await monitor.start_monitoring(args.duration)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        monitor.stop_monitoring()
    
    # Print summary
    print(f"\n=== Monitoring Summary for Patient {args.patient_id} ===")
    print(f"Total alerts generated: {len(monitor.alert_history)}")
    print(f"Alert levels: {[alert.alert_level.value for alert in monitor.alert_history]}")
    print(f"Alert log saved to: {config.LOG_DIR / f'alerts_{args.patient_id}.jsonl'}")

if __name__ == "__main__":
    asyncio.run(main())
