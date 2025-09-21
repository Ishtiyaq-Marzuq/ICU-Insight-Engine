"""
06_evaluation.py

Standardized evaluation pipeline to compute metrics and calibration plots for saved models.
Saves metrics JSON to results and calibration figure.

Usage:
    python 06_evaluation.py
"""

import json
from pathlib import Path
import config
import utils
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def main():
    if not Path(config.FEATURES_FILE).exists():
        raise FileNotFoundError("Features file missing. Run feature engineering.")
    df = pd.read_parquet(config.FEATURES_FILE)
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError("Target column missing from features.")
    X = df.drop(columns=[config.TARGET_COLUMN, 'subject_id','window_start'], errors='ignore')
    y = df[config.TARGET_COLUMN].values

    metrics_all = {}
    # Evaluate RF
    if Path(config.RANDOM_FOREST_MODEL).exists():
        rf = joblib.load(config.RANDOM_FOREST_MODEL)
        y_proba = rf.predict_proba(X.fillna(0))[:,1]
        m = utils.compute_classification_metrics(y, y_proba)
        metrics_all['rf'] = m
        # calibration
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],"--")
        plt.title("Calibration: RF")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.savefig(config.FIGURES_DIR / "calibration_rf.png")
        plt.close()

    # Evaluate XGBoost
    if Path(config.XGBOOST_MODEL).exists():
        xgb = joblib.load(config.XGBOOST_MODEL)
        y_proba = xgb.predict_proba(X.fillna(0))[:,1]
        m = utils.compute_classification_metrics(y, y_proba)
        metrics_all['xgboost'] = m
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],"--")
        plt.title("Calibration: XGBoost")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.savefig(config.FIGURES_DIR / "calibration_xgboost.png")
        plt.close()

    # Evaluate Logistic Regression
    if Path(config.LR_MODEL).exists():
        lr = joblib.load(config.LR_MODEL)
        # Load scaler for LR
        scaler_path = config.MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X.fillna(0))
        else:
            X_scaled = X.fillna(0).values
        y_proba = lr.predict_proba(X_scaled)[:,1]
        m = utils.compute_classification_metrics(y, y_proba)
        metrics_all['logistic_regression'] = m
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],"--")
        plt.title("Calibration: Logistic Regression")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.savefig(config.FIGURES_DIR / "calibration_lr.png")
        plt.close()

    # Evaluate Deep (if exists)
    try:
        import tensorflow as tf
        if Path(config.DEEP_MODEL).exists():
            deep = tf.keras.models.load_model(config.DEEP_MODEL)
            # load scaler
            scaler_path = config.MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X.fillna(0))
            else:
                X_scaled = X.fillna(0).values
            y_proba = deep.predict(X_scaled).ravel()
            m = utils.compute_classification_metrics(y, y_proba)
            metrics_all['deep'] = m
            prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0,1],[0,1],"--")
            plt.title("Calibration: Deep")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.savefig(config.FIGURES_DIR / "calibration_deep.png")
            plt.close()
    except ImportError:
        print("TensorFlow not available, skipping deep learning model evaluation")

    # Save metrics
    utils.save_json(metrics_all, config.RESULTS_DIR / "evaluation_metrics.json")
    print("Evaluation complete, metrics saved.")

if __name__ == "__main__":
    main()
