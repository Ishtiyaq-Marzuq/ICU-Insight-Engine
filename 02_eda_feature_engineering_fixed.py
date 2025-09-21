#!/usr/bin/env python3
"""
02_eda_feature_engineering_fixed.py

Modified EDA and feature engineering script that works with our sample data format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import utils
import config
from sklearn.preprocessing import StandardScaler

utils.set_seed()

def create_window_features(df: pd.DataFrame, time_col: str = 'window_start', id_col: str = 'subject_id', window_minutes: int = 60) -> pd.DataFrame:
    """
    Create window-based features from the sample data.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Group by patient and create time windows
    df = df.sort_values([id_col, time_col])
    
    # Add time-based features
    df['hour_of_day'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
    
    # Add rolling statistics for each vital sign
    vital_cols = [col for col in df.columns if any(vital in col.lower() 
                 for vital in ['hr', 'resp', 'sbp', 'dbp', 'temp', 'spo2', 'mbp'])]
    
    for col in vital_cols:
        if col in df.columns:
            # Rolling mean over 3 time points
            df[f'{col}_rolling3_mean'] = df.groupby(id_col)[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            # Rolling std over 3 time points
            df[f'{col}_rolling3_std'] = df.groupby(id_col)[col].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
            # Difference from previous value
            df[f'{col}_diff'] = df.groupby(id_col)[col].diff().fillna(0)
            # Trend (slope over last 3 points)
            df[f'{col}_trend'] = df.groupby(id_col)[col].rolling(3, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0).reset_index(level=0, drop=True)
    
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived clinical features.
    """
    # Pulse pressure
    if 'sbp_mean' in df.columns and 'dbp_mean' in df.columns:
        df['pulse_pressure'] = df['sbp_mean'] - df['dbp_mean']
    
    # Shock index (HR/SBP)
    if 'hr_mean' in df.columns and 'sbp_mean' in df.columns:
        df['shock_index'] = df['hr_mean'] / df['sbp_mean']
    
    # MAP (Mean Arterial Pressure) - approximate
    if 'sbp_mean' in df.columns and 'dbp_mean' in df.columns:
        df['map_approx'] = (df['sbp_mean'] + 2 * df['dbp_mean']) / 3
    
    # Temperature in Celsius (if in Fahrenheit)
    if 'temp_mean' in df.columns:
        # Assume temperature is in Fahrenheit, convert to Celsius
        df['temp_celsius'] = (df['temp_mean'] - 32) * 5/9
    
    return df

def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic demographic features for demonstration.
    """
    # Add synthetic age, gender, and other demographics
    np.random.seed(42)
    unique_patients = df['subject_id'].unique()
    
    demographics = []
    for patient_id in unique_patients:
        demographics.append({
            'subject_id': patient_id,
            'age': np.random.randint(40, 80),
            'gender': np.random.choice([0, 1]),
            'admission_type': np.random.choice([0, 1, 2]),  # 0: elective, 1: emergency, 2: urgent
            'has_diabetes': np.random.choice([0, 1], p=[0.7, 0.3]),
            'has_hypertension': np.random.choice([0, 1], p=[0.6, 0.4]),
            'has_copd': np.random.choice([0, 1], p=[0.8, 0.2]),
            'has_heart_disease': np.random.choice([0, 1], p=[0.7, 0.3]),
            'icu_days': np.random.randint(1, 10),
            'apache_score': np.random.randint(10, 50),
            'sofa_score': np.random.randint(0, 15)
        })
    
    demo_df = pd.DataFrame(demographics)
    df = df.merge(demo_df, on='subject_id', how='left')
    
    return df

def main():
    """Main feature engineering pipeline"""
    print("🔬 Starting EDA and Feature Engineering")
    print("=" * 50)
    
    # Load sampled data
    sampled_path = Path(config.SAMPLED_FILE)
    if not sampled_path.exists():
        raise FileNotFoundError(f"Sampled data not found at {sampled_path}. Run 01_data_sampling.py first.")
    
    df = pd.read_parquet(sampled_path)
    print(f"✓ Loaded {len(df)} records for {df['subject_id'].nunique()} patients")
    
    # Basic EDA
    print(f"\n📊 Data Overview:")
    print(f"  Rows: {len(df)}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['window_start'].min()} to {df['window_start'].max()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("✓ No missing values found")
    
    # Create window features
    print("\n🔧 Creating window features...")
    df = create_window_features(df)
    
    # Add derived features
    print("🔧 Adding derived clinical features...")
    df = add_derived_features(df)
    
    # Add demographic features
    print("🔧 Adding demographic features...")
    df = add_demographic_features(df)
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"  Final shape: {df.shape}")
    print(f"  Features: {len(df.columns)}")
    
    # Save features
    df.to_parquet(config.FEATURES_FILE, index=False)
    print(f"✓ Features saved to {config.FEATURES_FILE}")
    
    # Create and save scaler
    print("🔧 Creating feature scaler...")
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'window_start', 'deterioration_label']]
    
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    import joblib
    joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
    print(f"✓ Scaler saved to {config.MODELS_DIR / 'scaler.pkl'}")
    
    # Save feature metadata
    feature_metadata = {
        'total_features': len(feature_cols),
        'vital_signs': [col for col in feature_cols if any(vital in col.lower() 
                     for vital in ['hr', 'resp', 'sbp', 'dbp', 'temp', 'spo2', 'mbp'])],
        'demographic_features': [col for col in feature_cols if col in 
                               ['age', 'gender', 'admission_type', 'has_diabetes', 'has_hypertension']],
        'derived_features': [col for col in feature_cols if col in 
                           ['pulse_pressure', 'shock_index', 'map_approx', 'temp_celsius']],
        'window_features': [col for col in feature_cols if 'rolling' in col or 'diff' in col or 'trend' in col]
    }
    
    utils.save_json(feature_metadata, str(config.FEATURE_METADATA_FILE))
    print(f"✓ Feature metadata saved to {config.FEATURE_METADATA_FILE}")
    
    print("\n🎉 EDA and Feature Engineering Complete!")

if __name__ == "__main__":
    main()

