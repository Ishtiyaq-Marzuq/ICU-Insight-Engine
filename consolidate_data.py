import pandas as pd
import os
from pathlib import Path

# --- 1. Define Paths and Create Directories ---
# This mirrors the organized project structure you've defined in config.py
RAW_DATA_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")
PROCESSED_DATA_DIR = Path("data/processed")
RAW_VITALS_FILE = PROCESSED_DATA_DIR / "vitals_timeseries.csv"

# Create necessary directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
print(f"Directory structure created: {PROCESSED_DATA_DIR} and {RAW_DATA_DIR}")

# --- 2. Load the Raw Data ---
# This code assumes your MIMIC-III CSV files are in the same directory as this script.
print("\nLoading CHARTEVENTS.csv and D_ITEMS.csv...")
try:
    chartevents_df = pd.read_csv("CHARTEVENTS.csv", low_memory=False)
    d_items_df = pd.read_csv("D_ITEMS.csv", low_memory=False)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Please make sure 'CHARTEVENTS.csv' and 'D_ITEMS.csv' are in the same directory as this script.")
    exit()

# --- 3. Define the Vital Signs We Need ---
# We use the D_ITEMS table to identify the 'itemid' for common vital signs.
# This is a crucial engineering step: using metadata to understand our raw data.
vital_sign_itemids = [
    211,    # Heart Rate
    618,    # Respiratory Rate
    646,    # SpO2
    51,     # Blood Pressure (Systolic & Diastolic are on separate rows)
    678     # Temperature
]

# --- 4. Merge and Filter the Data ---
print("\nMerging dataframes on 'itemid' to get vital sign labels...")
# We use an 'inner' merge to keep only the rows that exist in both tables.
merged_df = pd.merge(chartevents_df, d_items_df, on='itemid', how='inner')

print("Filtering for only the specified vital signs...")
vitals_df = merged_df[merged_df['itemid'].isin(vital_sign_itemids)].copy()

# --- 5. Clean and Prepare the Data ---
print("Cleaning data and converting 'charttime' to datetime format...")
vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'])
vitals_df = vitals_df.sort_values(['subject_id', 'charttime'])

# Drop unnecessary columns to keep the file clean
vitals_df = vitals_df[['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'itemid', 'valuenum', 'label']]
vitals_df.rename(columns={'valuenum': 'value', 'label': 'vital_name'}, inplace=True)

# --- 6. Save the Processed File ---
print(f"\nSaving consolidated data to {RAW_VITALS_FILE}...")
vitals_df.to_csv(RAW_VITALS_FILE, index=False)
print(f"File '{RAW_VITALS_FILE}' has been created successfully.")

print("\nData consolidation complete. The next step is to run '01_data_sampling.py' on the new file.")