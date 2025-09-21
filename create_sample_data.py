#!/usr/bin/env python3
"""
create_sample_data.py

Create sample data for the ICU Patient Monitoring System dashboard.
This script generates synthetic patient data to demonstrate the system capabilities.

Usage:
    python create_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from pathlib import Path

def create_sample_vitals_data():
    """Create sample vital signs data"""
    print("Creating sample vital signs data...")
    
    # Create sample patients
    patients = [10001, 10002, 10003, 10004, 10005]
    
    # Create time series data for each patient
    all_data = []
    
    for patient_id in patients:
        # Generate 24 hours of data (every 15 minutes)
        start_time = datetime.now() - timedelta(hours=24)
        time_points = [start_time + timedelta(minutes=15*i) for i in range(96)]
        
        # Generate realistic vital signs with some variation
        np.random.seed(patient_id)  # For reproducible data
        
        for i, time_point in enumerate(time_points):
            # Add some trend over time
            trend_factor = 1 + (i / 96) * 0.1  # 10% change over 24 hours
            
            # Heart rate (60-120 bpm)
            hr = np.random.normal(75, 10) * trend_factor
            hr = max(60, min(120, hr))
            
            # Respiratory rate (12-20 breaths/min)
            resp = np.random.normal(16, 2) * trend_factor
            resp = max(12, min(20, resp))
            
            # Systolic BP (90-140 mmHg)
            sbp = np.random.normal(120, 15) * trend_factor
            sbp = max(90, min(140, sbp))
            
            # Diastolic BP (60-90 mmHg)
            dbp = np.random.normal(80, 10) * trend_factor
            dbp = max(60, min(90, dbp))
            
            # Temperature (97-99°F)
            temp = np.random.normal(98.6, 0.5) * trend_factor
            temp = max(97, min(99, temp))
            
            # Oxygen saturation (95-100%)
            spo2 = np.random.normal(98, 1.5) * trend_factor
            spo2 = max(95, min(100, spo2))
            
            # Mean BP
            mbp = (sbp + 2 * dbp) / 3
            
            # Create deterioration label (higher risk for some patients)
            risk_score = np.random.random()
            if patient_id in [10002, 10004]:  # Higher risk patients
                risk_score += 0.3
            deterioration_label = 1 if risk_score > 0.7 else 0
            
            all_data.append({
                'subject_id': patient_id,
                'window_start': time_point,
                'hr_mean': hr,
                'resp_mean': resp,
                'sbp_mean': sbp,
                'dbp_mean': dbp,
                'temp_mean': temp,
                'spo2_mean': spo2,
                'mbp_mean': mbp,
                'deterioration_label': deterioration_label
            })
    
    return pd.DataFrame(all_data)

def create_sample_features_data():
    """Create sample features data"""
    print("Creating sample features data...")
    
    # Use the same patients and time points
    patients = [10001, 10002, 10003, 10004, 10005]
    
    all_features = []
    
    for patient_id in patients:
        # Generate 24 hours of data (every hour)
        start_time = datetime.now() - timedelta(hours=24)
        time_points = [start_time + timedelta(hours=i) for i in range(24)]
        
        np.random.seed(patient_id)
        
        for i, time_point in enumerate(time_points):
            # Generate comprehensive features
            features = {
                'subject_id': patient_id,
                'window_start': time_point,
                
                # Vital signs (mean values)
                'hr_mean': np.random.normal(75, 10),
                'resp_mean': np.random.normal(16, 2),
                'sbp_mean': np.random.normal(120, 15),
                'dbp_mean': np.random.normal(80, 10),
                'temp_mean': np.random.normal(98.6, 0.5),
                'spo2_mean': np.random.normal(98, 1.5),
                'mbp_mean': np.random.normal(93, 12),
                
                # Vital signs (std values)
                'hr_std': np.random.uniform(2, 8),
                'resp_std': np.random.uniform(1, 3),
                'sbp_std': np.random.uniform(5, 15),
                'dbp_std': np.random.uniform(3, 10),
                'temp_std': np.random.uniform(0.1, 0.5),
                'spo2_std': np.random.uniform(0.5, 2),
                'mbp_std': np.random.uniform(3, 12),
                
                # Lab values
                'hemoglobin_mean': np.random.normal(12, 2),
                'wbc_mean': np.random.normal(7, 2),
                'platelets_mean': np.random.normal(250, 50),
                'sodium_mean': np.random.normal(140, 3),
                'potassium_mean': np.random.normal(4, 0.5),
                'chloride_mean': np.random.normal(100, 3),
                'bicarbonate_mean': np.random.normal(24, 2),
                'creatinine_mean': np.random.normal(1, 0.3),
                'bun_mean': np.random.normal(15, 5),
                'glucose_mean': np.random.normal(100, 20),
                
                # Derived features
                'pulse_pressure': np.random.normal(40, 8),
                'shock_index': np.random.normal(0.8, 0.2),
                'map_score': np.random.normal(93, 12),
                
                # Time-based features
                'hour_of_day': time_point.hour,
                'day_of_week': time_point.weekday(),
                'is_weekend': 1 if time_point.weekday() >= 5 else 0,
                
                # Patient demographics (static)
                'age': np.random.randint(40, 80),
                'gender': np.random.choice([0, 1]),
                'admission_type': np.random.choice([0, 1, 2]),
                
                # Risk factors
                'has_diabetes': np.random.choice([0, 1], p=[0.7, 0.3]),
                'has_hypertension': np.random.choice([0, 1], p=[0.6, 0.4]),
                'has_copd': np.random.choice([0, 1], p=[0.8, 0.2]),
                'has_heart_disease': np.random.choice([0, 1], p=[0.7, 0.3]),
                
                # ICU stay information
                'icu_days': np.random.randint(1, 10),
                'apache_score': np.random.randint(10, 50),
                'sofa_score': np.random.randint(0, 15),
                
                # Target variable
                'deterioration_label': np.random.choice([0, 1], p=[0.8, 0.2])
            }
            
            # Add some trend for higher risk patients
            if patient_id in [10002, 10004]:
                features['deterioration_label'] = np.random.choice([0, 1], p=[0.6, 0.4])
                features['hr_mean'] *= 1.1
                features['sbp_mean'] *= 0.95
                features['spo2_mean'] *= 0.98
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)

def main():
    """Main function to create sample data"""
    print("🏥 Creating Sample Data for ICU Patient Monitoring System")
    print("=" * 60)
    
    # Create sample vitals data
    vitals_df = create_sample_vitals_data()
    vitals_file = config.PROCESSED_DATA_DIR / "vitals_timeseries.csv"
    vitals_df.to_csv(vitals_file, index=False)
    print(f"✅ Sample vitals data saved to {vitals_file}")
    print(f"   Records: {len(vitals_df)}, Patients: {vitals_df['subject_id'].nunique()}")
    
    # Create sample features data
    features_df = create_sample_features_data()
    features_file = config.FEATURES_FILE
    features_df.to_parquet(features_file, index=False)
    print(f"✅ Sample features data saved to {features_file}")
    print(f"   Records: {len(features_df)}, Patients: {features_df['subject_id'].nunique()}")
    
    # Create sample alerts data
    alerts_data = []
    for patient_id in [10002, 10004]:  # High risk patients
        for i in range(3):  # 3 alerts per high risk patient
            alert_time = datetime.now() - timedelta(hours=np.random.randint(1, 24))
            alerts_data.append({
                'patient_id': patient_id,
                'timestamp': alert_time.isoformat(),
                'alert_level': np.random.choice(['MEDIUM', 'HIGH', 'CRITICAL']),
                'risk_score': np.random.uniform(0.6, 0.95),
                'title': f'Alert {i+1}',
                'description': f'Patient {patient_id} showing signs of deterioration',
                'vital_values': {
                    'heart_rate': np.random.randint(100, 150),
                    'systolic_bp': np.random.randint(80, 180),
                    'oxygen_saturation': np.random.randint(85, 95)
                },
                'recommendations': [
                    'Increase monitoring frequency',
                    'Notify physician',
                    'Consider intervention'
                ]
            })
    
    # Save alerts to JSONL file
    alerts_file = config.LOG_DIR / "alerts_10002.jsonl"
    with open(alerts_file, 'w') as f:
        for alert in alerts_data:
            f.write(str(alert).replace("'", '"') + '\n')
    print(f"✅ Sample alerts data saved to {alerts_file}")
    print(f"   Alerts: {len(alerts_data)}")
    
    print("\n🎉 Sample data creation complete!")
    print("\nYou can now:")
    print("1. View the dashboard at http://localhost:8501")
    print("2. Select different patients from the dropdown")
    print("3. View real-time monitoring and alerts")
    print("4. Explore the different dashboard tabs")

if __name__ == "__main__":
    main()
