#!/usr/bin/env python3
"""
train_quick_models.py

Quick model training script for the ICU Patient Monitoring System.
Trains basic models using the sample data for immediate dashboard functionality.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import config
import utils
from pathlib import Path

def prepare_training_data():
    """Prepare data for training"""
    print("Loading sample data...")
    
    # Load features data
    if config.FEATURES_FILE.exists():
        df = pd.read_parquet(config.FEATURES_FILE)
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
    else:
        print("No features file found. Please run create_sample_data.py first.")
        return None, None
    
    # Prepare features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'window_start', 'deterioration_label']]
    
    X = df[feature_cols].fillna(0)
    y = df['deterioration_label']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_models():
    """Train basic models"""
    print("🏥 Training Quick Models for ICU Patient Monitoring")
    print("=" * 60)
    
    # Prepare data
    X, y = prepare_training_data()
    if X is None:
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Random Forest
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    print(f"Random Forest AUC: {rf_auc:.3f}")
    
    # Logistic Regression
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    print(f"Logistic Regression AUC: {lr_auc:.3f}")
    
    # Save models
    print("\nSaving models...")
    
    # Save scaler
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save Random Forest
    rf_path = config.RANDOM_FOREST_MODEL
    joblib.dump(rf_model, rf_path)
    print(f"✓ Random Forest saved to {rf_path}")
    
    # Save Logistic Regression
    lr_path = config.LR_MODEL
    joblib.dump(lr_model, lr_path)
    print(f"✓ Logistic Regression saved to {lr_path}")
    
    # Save evaluation results
    eval_results = {
        'random_forest': {
            'auroc': float(rf_auc),
            'auprc': 0.0,  # Would need to calculate
            'f1': 0.0,
            'precision': 0.0
        },
        'logistic_regression': {
            'auroc': float(lr_auc),
            'auprc': 0.0,
            'f1': 0.0,
            'precision': 0.0
        }
    }
    
    eval_file = config.RESULTS_DIR / "evaluation_metrics.json"
    utils.save_json(eval_results, str(eval_file))
    print(f"✓ Evaluation results saved to {eval_file}")
    
    print("\n🎉 Model training complete!")
    print("The dashboard should now show trained models in the dropdown.")
    
    return True

if __name__ == "__main__":
    success = train_models()
    if success:
        print("\n✅ Models trained successfully!")
        print("You can now refresh the dashboard to see the trained models.")
    else:
        print("\n❌ Model training failed!")


