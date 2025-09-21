"""
utils.py

Utility functions for loading data, preprocessing helpers, metrics, model save/load, plotting, and seeding.
"""

from typing import Tuple, List, Optional, Dict, Any
import os
import random
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import config

# Setup logging
logger = logging.getLogger("icu_monitor")
if not logger.handlers:
    # Ensure log directory exists
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(config.LOG_DIR / "pipeline.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV into DataFrame with some default settings."""
    logger.info(f"Reading CSV: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    logger.info(f"Saving dataframe to parquet: {path}")
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    logger.info(f"Loading parquet: {path}")
    return pd.read_parquet(path)


def sample_patients(df: pd.DataFrame, subject_col: str = config.PATIENT_ID_COL, frac: float = config.SAMPLE_FRACTION, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Randomly sample patient-level data ensuring that entire patient time-series remain intact.
    `df` must contain subject_id column.
    """
    set_seed(seed)
    subjects = df[subject_col].unique()
    sample_n = max(1, int(len(subjects) * frac))
    sampled_subjects = np.random.choice(subjects, size=sample_n, replace=False)
    sampled_df = df[df[subject_col].isin(sampled_subjects)].copy()
    logger.info(f"Sampled {len(sampled_subjects)} subjects ({frac*100:.2f}%) -> rows: {len(sampled_df)}")
    return sampled_df


def save_model(obj: Any, path: str) -> None:
    """Save sklearn/xgboost models with joblib; Keras models handled separately."""
    logger.info(f"Saving model to {path}")
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute AUROC, AUPRC, F1, precision, recall.
    y_pred_proba is probability for positive class.
    """
    auc = float(roc_auc_score(y_true, y_pred_proba))
    auprc = float(average_precision_score(y_true, y_pred_proba))
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred))
    precision, recall, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "auroc": auc,
        "auprc": auprc,
        "f1": f1,
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist()
    }
    logger.info(f"Metrics: {metrics}")
    return metrics


def plot_timeseries(df: pd.DataFrame, time_col: str, value_cols: List[str], subject_id: Optional[int] = None, save_path: Optional[str] = None) -> None:
    """Plot multiple vitals time-series for a single patient or aggregated."""
    if subject_id is not None and config.PATIENT_ID_COL in df.columns:
        df = df[df[config.PATIENT_ID_COL] == subject_id]
    df = df.sort_values(time_col)
    plt.figure(figsize=(12, len(value_cols) * 2.2))
    for i, col in enumerate(value_cols, 1):
        plt.subplot(len(value_cols), 1, i)
        plt.plot(pd.to_datetime(df[time_col]), df[col], marker='.', linewidth=0.7)
        plt.title(col)
        plt.xlabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved json to {path}")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
