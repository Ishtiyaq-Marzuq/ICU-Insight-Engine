"""
01_data_sampling.py

Randomly sample patients/time-series to reduce data size and reduce computational bias.
Saves a sampled parquet file.

Usage:
    python 01_data_sampling.py --frac 0.2
"""

import argparse
from pathlib import Path
import pandas as pd
import utils
import config
from typing import Any

utils.set_seed()

def main(frac: float = config.SAMPLE_FRACTION) -> None:
    # Example: sample vitals timeseries which contains subject_id and charttime
    raw_vitals = Path(config.RAW_VITALS_FILE)
    if not raw_vitals.exists():
        raise FileNotFoundError(f"{raw_vitals} not found. Place your MIMIC vitals timeseries csv at this path or update config.")
    df = pd.read_csv(raw_vitals, low_memory=False)
    sampled = utils.sample_patients(df, subject_col=config.PATIENT_ID_COL, frac=frac, seed=config.RANDOM_SEED)
    # Save sampled data
    sampled_path = config.SAMPLED_FILE
    sampled.to_parquet(sampled_path, index=False)
    print(f"Sampled data saved to {sampled_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=config.SAMPLE_FRACTION, help="Fraction of patients to sample")
    args = parser.parse_args()
    main(args.frac)
