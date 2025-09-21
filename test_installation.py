#!/usr/bin/env python3
"""
test_installation.py

Test script to verify the ICU Patient Monitoring System installation.
This script checks if all required packages are installed and can be imported.

Usage:
    python test_installation.py
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 Testing ICU Patient Monitoring System Installation")
    print("=" * 60)
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ Python 3.9+ is required")
        return False
    else:
        print("✅ Python version is compatible")
    
    print("\nTesting core packages...")
    
    # Core data science packages
    core_packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("joblib", "Joblib"),
    ]
    
    # Visualization packages
    viz_packages = [
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
    ]
    
    # ML packages
    ml_packages = [
        ("xgboost", "XGBoost"),
        ("tensorflow", "TensorFlow"),
    ]
    
    # Web framework
    web_packages = [
        ("streamlit", "Streamlit"),
    ]
    
    # Data processing
    data_packages = [
        ("pyarrow", "PyArrow"),
        ("psutil", "PSUtil"),
    ]
    
    # Database packages
    db_packages = [
        ("psycopg2", "PostgreSQL"),
        ("redis", "Redis"),
        ("sqlalchemy", "SQLAlchemy"),
    ]
    
    # API packages
    api_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
    ]
    
    # Utility packages
    util_packages = [
        ("requests", "Requests"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("click", "Click"),
    ]
    
    all_packages = [
        ("Core Data Science", core_packages),
        ("Visualization", viz_packages),
        ("Machine Learning", ml_packages),
        ("Web Framework", web_packages),
        ("Data Processing", data_packages),
        ("Database", db_packages),
        ("API Framework", api_packages),
        ("Utilities", util_packages),
    ]
    
    passed_tests = 0
    total_tests = 0
    
    for category, packages in all_packages:
        print(f"\n{category}:")
        for module, name in packages:
            total_tests += 1
            if test_import(module, name):
                passed_tests += 1
    
    print(f"\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} packages passed")
    
    if passed_tests == total_tests:
        print("🎉 All packages are installed correctly!")
        print("\nNext steps:")
        print("1. Run: python config.py")
        print("2. Run: streamlit run dashboard.py")
        return True
    else:
        print("⚠️  Some packages are missing. Please install them with:")
        print("   pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


