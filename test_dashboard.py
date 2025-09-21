#!/usr/bin/env python3
"""
test_dashboard.py

Test script to verify dashboard components work correctly.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except Exception as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except Exception as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
    except Exception as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        import config
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        import utils
        print("✓ Utils imported successfully")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if data can be loaded"""
    print("\nTesting data loading...")
    
    try:
        import config
        import pandas as pd
        
        # Test if features file exists and can be loaded
        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
            print(f"✓ Features data loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Patients: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A'}")
            return True
        else:
            print("✗ Features file not found")
            return False
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import config
        import joblib
        import tensorflow as tf
        
        models = {}
        
        # Test scaler
        scaler_path = config.MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            models['scaler'] = joblib.load(scaler_path)
            print("✓ Scaler loaded")
        else:
            print("⚠ Scaler not found")
        
        # Test Random Forest
        if config.RANDOM_FOREST_MODEL.exists():
            models['rf'] = joblib.load(config.RANDOM_FOREST_MODEL)
            print("✓ Random Forest model loaded")
        else:
            print("⚠ Random Forest model not found")
        
        # Test XGBoost
        if config.XGBOOST_MODEL.exists():
            models['xgboost'] = joblib.load(config.XGBOOST_MODEL)
            print("✓ XGBoost model loaded")
        else:
            print("⚠ XGBoost model not found")
        
        # Test Deep Learning
        if config.DEEP_MODEL.exists():
            models['deep'] = tf.keras.models.load_model(config.DEEP_MODEL)
            print("✓ Deep learning model loaded")
        else:
            print("⚠ Deep learning model not found")
        
        print(f"Total models loaded: {len(models)}")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_functions():
    """Test dashboard functions"""
    print("\nTesting dashboard functions...")
    
    try:
        # Import dashboard functions
        from dashboard import load_models, load_patient_data, create_vital_signs_chart, create_risk_gauge
        
        # Test model loading
        models = load_models()
        print(f"✓ Models loaded: {list(models.keys())}")
        
        # Test patient data loading
        patient_data = load_patient_data()
        print(f"✓ Patient data loaded: {len(patient_data)} rows")
        
        # Test chart creation
        if not patient_data.empty and 'subject_id' in patient_data.columns:
            patient_id = patient_data['subject_id'].iloc[0]
            chart = create_vital_signs_chart(patient_data, patient_id)
            if chart:
                print("✓ Vital signs chart created successfully")
            else:
                print("⚠ Vital signs chart creation returned None")
        
        # Test risk gauge
        gauge = create_risk_gauge(0.5)
        if gauge:
            print("✓ Risk gauge created successfully")
        else:
            print("⚠ Risk gauge creation returned None")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard functions failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🏥 ICU Patient Monitoring System - Dashboard Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_loading,
        test_model_loading,
        test_dashboard_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


