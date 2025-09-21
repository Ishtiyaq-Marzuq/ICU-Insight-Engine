"""
config.py

Central configuration file storing dataset paths, model paths, constants, and other global variables.
This file contains all data paths for the MIMIC-III clinical database project.

Project layout:
.
├── data/
│   ├── raw/
│   │   └── mimic-iii-clinical-database-demo-1.4/
│   │       ├── ADMISSIONS.csv
│   │       ├── PATIENTS.csv
│   │       ├── CHARTEVENTS.csv
│   │       ├── LABEVENTS.csv
│   │       └── ... (other MIMIC-III files)
│   ├── processed/
│   └── features/
├── models/
├── results/
│   ├── figures/
│   └── explainability/
├── logs/
└── notebooks/
"""

from pathlib import Path
import os

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Raw data directory (MIMIC-III files)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mimic-iii-clinical-database-demo-1.4"

# Processed data directory
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Features directory
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# =============================================================================
# MODEL & ARTIFACTS DIRECTORIES
# =============================================================================

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

# Logs directory
LOG_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# RAW MIMIC-III DATA FILES
# =============================================================================

# Core patient data
RAW_ADMISSIONS_FILE = RAW_DATA_DIR / "ADMISSIONS.csv"
RAW_PATIENTS_FILE = RAW_DATA_DIR / "PATIENTS.csv"
RAW_ICUSTAYS_FILE = RAW_DATA_DIR / "ICUSTAYS.csv"
RAW_TRANSFERS_FILE = RAW_DATA_DIR / "TRANSFERS.csv"

# Clinical events
RAW_CHARTEVENTS_FILE = RAW_DATA_DIR / "CHARTEVENTS.csv"
RAW_DATETIMEEVENTS_FILE = RAW_DATA_DIR / "DATETIMEEVENTS.csv"
RAW_LABEVENTS_FILE = RAW_DATA_DIR / "LABEVENTS.csv"
RAW_MICROBIOLOGYEVENTS_FILE = RAW_DATA_DIR / "MICROBIOLOGYEVENTS.csv"
RAW_NOTEEVENTS_FILE = RAW_DATA_DIR / "NOTEEVENTS.csv"
RAW_OUTPUTEVENTS_FILE = RAW_DATA_DIR / "OUTPUTEVENTS.csv"

# Input events
RAW_INPUTEVENTS_CV_FILE = RAW_DATA_DIR / "INPUTEVENTS_CV.csv"
RAW_INPUTEVENTS_MV_FILE = RAW_DATA_DIR / "INPUTEVENTS_MV.csv"

# Procedure events
RAW_PROCEDUREEVENTS_MV_FILE = RAW_DATA_DIR / "PROCEDUREEVENTS_MV.csv"
RAW_CPTEVENTS_FILE = RAW_DATA_DIR / "CPTEVENTS.csv"

# Diagnosis and procedure codes
RAW_DIAGNOSES_ICD_FILE = RAW_DATA_DIR / "DIAGNOSES_ICD.csv"
RAW_PROCEDURES_ICD_FILE = RAW_DATA_DIR / "PROCEDURES_ICD.csv"
RAW_DRGCODES_FILE = RAW_DATA_DIR / "DRGCODES.csv"

# Prescriptions
RAW_PRESCRIPTIONS_FILE = RAW_DATA_DIR / "PRESCRIPTIONS.csv"

# Services and other
RAW_SERVICES_FILE = RAW_DATA_DIR / "SERVICES.csv"
RAW_CALLOUT_FILE = RAW_DATA_DIR / "CALLOUT.csv"
RAW_CAREGIVERS_FILE = RAW_DATA_DIR / "CAREGIVERS.csv"

# Dictionary tables
RAW_D_ITEMS_FILE = RAW_DATA_DIR / "D_ITEMS.csv"
RAW_D_LABITEMS_FILE = RAW_DATA_DIR / "D_LABITEMS.csv"
RAW_D_ICD_DIAGNOSES_FILE = RAW_DATA_DIR / "D_ICD_DIAGNOSES.csv"
RAW_D_ICD_PROCEDURES_FILE = RAW_DATA_DIR / "D_ICD_PROCEDURES.csv"
RAW_D_CPT_FILE = RAW_DATA_DIR / "D_CPT.csv"

# =============================================================================
# PROCESSED DATA FILES
# =============================================================================

# Consolidated and processed data
RAW_VITALS_FILE = PROCESSED_DATA_DIR / "vitals_timeseries.csv"
SAMPLED_FILE = PROCESSED_DATA_DIR / "sampled_patients.parquet"
CONSOLIDATED_FILE = PROCESSED_DATA_DIR / "consolidated_data.parquet"

# Train/test splits
TRAIN_SPLIT_FILE = PROCESSED_DATA_DIR / "train.npz"
TEST_SPLIT_FILE = PROCESSED_DATA_DIR / "test.npz"
VAL_SPLIT_FILE = PROCESSED_DATA_DIR / "val.npz"

# =============================================================================
# FEATURES FILES
# =============================================================================

# Engineered features
FEATURES_FILE = FEATURES_DIR / "features.parquet"
TRAINING_FEATURES_FILE = FEATURES_DIR / "training_features.parquet"
TEST_FEATURES_FILE = FEATURES_DIR / "test_features.parquet"

# Feature metadata
FEATURE_METADATA_FILE = FEATURES_DIR / "feature_metadata.json"
FEATURE_IMPORTANCE_FILE = FEATURES_DIR / "feature_importance.json"

# =============================================================================
# MODEL FILES
# =============================================================================

# Traditional ML models
RANDOM_FOREST_MODEL = MODELS_DIR / "rf_model.pkl"
XGBOOST_MODEL = MODELS_DIR / "xgb_model.pkl"
LR_MODEL = MODELS_DIR / "lr_model.pkl"
SVM_MODEL = MODELS_DIR / "svm_model.pkl"

# Deep learning models
DEEP_MODEL = MODELS_DIR / "deep_multimodal.h5"
LSTM_MODEL = MODELS_DIR / "lstm_model.h5"
TRANSFORMER_MODEL = MODELS_DIR / "transformer_model.h5"

# Model metadata and configs
MODEL_CONFIGS_DIR = MODELS_DIR / "configs"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# =============================================================================
# RESULTS FILES
# =============================================================================

# Evaluation results
EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
PERFORMANCE_METRICS_FILE = RESULTS_DIR / "performance_metrics.json"
CROSS_VALIDATION_RESULTS_FILE = RESULTS_DIR / "cv_results.json"

# Explainability results
SHAP_VALUES_FILE = EXPLAIN_DIR / "shap_values.pkl"
LIME_EXPLANATIONS_FILE = EXPLAIN_DIR / "lime_explanations.pkl"
FEATURE_ATTRIBUTION_FILE = EXPLAIN_DIR / "feature_attribution.json"

# =============================================================================
# LOG FILES
# =============================================================================

# Pipeline logs
PIPELINE_LOG_FILE = LOG_DIR / "pipeline.log"
TRAINING_LOG_FILE = LOG_DIR / "training.log"
EVALUATION_LOG_FILE = LOG_DIR / "evaluation.log"

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Data sampling
SAMPLE_FRACTION = 0.2  # default fraction to sample from raw dataset
MIN_PATIENTS_PER_ICU_STAY = 1  # minimum patients per ICU stay to include

# Time series parameters
TIME_WINDOW_MINUTES = 60  # window for time series aggregation
PREDICTION_HORIZON_HOURS = 24  # hours ahead to predict deterioration
LOOKBACK_HOURS = 48  # hours of data to look back for prediction

# Target variable configuration
TARGET_COLUMN = "deterioration_label"  # binary label (1 -> deterioration within horizon, 0 -> stable)
TIMESTAMP_COL = "charttime"  # adapt to MIMIC column for vitals times
PATIENT_ID_COL = "subject_id"
ICU_STAY_ID_COL = "icustay_id"
HADM_ID_COL = "hadm_id"

# Feature engineering
VITAL_SIGNS_ITEMS = [
    'Heart Rate', 'Respiratory Rate', 'Temperature', 'Systolic BP', 
    'Diastolic BP', 'Mean BP', 'Oxygen Saturation', 'Glasgow Coma Scale'
]

LAB_VALUES_ITEMS = [
    'Hemoglobin', 'White Blood Cells', 'Platelets', 'Sodium', 
    'Potassium', 'Chloride', 'Bicarbonate', 'Creatinine', 'BUN'
]

# Model parameters
CV_FOLDS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        MODEL_CONFIGS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        EXPLAIN_DIR,
        LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

def validate_data_files():
    """Validate that critical data files exist."""
    critical_files = [
        RAW_ADMISSIONS_FILE,
        RAW_PATIENTS_FILE,
        RAW_CHARTEVENTS_FILE,
        RAW_LABEVENTS_FILE
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("⚠️  Warning: The following critical data files are missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure MIMIC-III data files are properly placed in the raw data directory.")
    else:
        print("✓ All critical data files found.")

# Initialize directories and validate files
if __name__ == "__main__":
    print("Initializing project directories and validating data files...")
    create_directories()
    validate_data_files()
    print("Configuration setup complete!")

