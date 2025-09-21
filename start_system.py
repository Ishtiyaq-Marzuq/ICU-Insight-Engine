#!/usr/bin/env python3
"""
start_system.py

Startup script for the ICU Patient Monitoring System.
This script initializes the system and launches the appropriate components.

Usage:
    python start_system.py [--mode MODE] [--port PORT]
    
Modes:
    - dashboard: Start the Streamlit dashboard (default)
    - pipeline: Start the real-time pipeline
    - test: Run system tests
    - all: Start all components
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import config

def run_command(cmd, background=False):
    """Run a command and return success status"""
    try:
        if background:
            subprocess.Popen(cmd, shell=True)
            return True
        else:
            result = subprocess.run(cmd, shell=True, check=True)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {cmd}")
        print(f"   Error: {e}")
        return False

def start_dashboard(port=8501):
    """Start the Streamlit dashboard"""
    print("🏥 Starting ICU Patient Monitoring Dashboard...")
    cmd = f"streamlit run dashboard.py --server.port {port} --server.address 0.0.0.0"
    return run_command(cmd, background=True)

def start_pipeline():
    """Start the real-time pipeline"""
    print("🔄 Starting Real-time Pipeline...")
    cmd = "python 08_realtime_pipeline.py --patient_id 12345 --duration 3600"
    return run_command(cmd, background=True)

def run_tests():
    """Run system tests"""
    print("🧪 Running System Tests...")
    cmd = "python test_installation.py"
    return run_command(cmd, background=False)

def initialize_system():
    """Initialize the system"""
    print("⚙️  Initializing ICU Patient Monitoring System...")
    
    # Run configuration
    print("   - Setting up configuration...")
    if not run_command("python config.py", background=False):
        print("❌ Failed to initialize configuration")
        return False
    
    # Run data sampling if needed
    if not Path(config.SAMPLED_FILE).exists():
        print("   - Running data sampling...")
        if not run_command("python 01_data_sampling.py", background=False):
            print("❌ Failed to run data sampling")
            return False
    
    # Run feature engineering if needed
    if not Path(config.FEATURES_FILE).exists():
        print("   - Running feature engineering...")
        if not run_command("python 02_eda_feature_engineering.py", background=False):
            print("❌ Failed to run feature engineering")
            return False
    
    print("✅ System initialized successfully")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ICU Patient Monitoring System Startup")
    parser.add_argument("--mode", choices=["dashboard", "pipeline", "test", "all"], 
                       default="dashboard", help="Startup mode")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--init", action="store_true", help="Initialize system first")
    
    args = parser.parse_args()
    
    print("🏥 ICU Patient Monitoring System")
    print("=" * 40)
    
    # Initialize system if requested
    if args.init:
        if not initialize_system():
            print("❌ System initialization failed")
            sys.exit(1)
    
    # Start components based on mode
    if args.mode == "dashboard":
        if start_dashboard(args.port):
            print(f"✅ Dashboard started on port {args.port}")
            print(f"   Open: http://localhost:{args.port}")
        else:
            print("❌ Failed to start dashboard")
            sys.exit(1)
    
    elif args.mode == "pipeline":
        if start_pipeline():
            print("✅ Real-time pipeline started")
        else:
            print("❌ Failed to start pipeline")
            sys.exit(1)
    
    elif args.mode == "test":
        if run_tests():
            print("✅ All tests passed")
        else:
            print("❌ Some tests failed")
            sys.exit(1)
    
    elif args.mode == "all":
        print("🚀 Starting all components...")
        
        # Start pipeline in background
        if start_pipeline():
            print("✅ Real-time pipeline started")
        else:
            print("❌ Failed to start pipeline")
        
        # Wait a moment
        time.sleep(2)
        
        # Start dashboard
        if start_dashboard(args.port):
            print(f"✅ Dashboard started on port {args.port}")
            print(f"   Open: http://localhost:{args.port}")
        else:
            print("❌ Failed to start dashboard")
            sys.exit(1)
    
    print("\n🎉 System startup complete!")
    print("\nAvailable commands:")
    print("  - Dashboard: streamlit run dashboard.py")
    print("  - Pipeline: python 08_realtime_pipeline.py")
    print("  - Tests: python test_installation.py")
    print("  - Help: python start_system.py --help")

if __name__ == "__main__":
    main()


