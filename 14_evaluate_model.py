# 


import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import json
from tabulate import tabulate
import config
import utils

def load_data_and_models():
    """Load features, target, and all trained models."""
    print("⏳ Loading data and models...")
    try:
        # Load features and target
        features_path = Path(config.FEATURES_FILE)
        df = pd.read_parquet(features_path)
        
        # Prepare data for all models
        X = df.drop(columns=[config.TARGET_COLUMN, 'subject_id', 'window_start'], errors='ignore').fillna(0)
        y = df[config.TARGET_COLUMN].astype(int)
        groups = df['subject_id'] if 'subject_id' in df.columns else None
        
        # Load scaler and models
        scaler_path = Path(config.MODELS_DIR) / "scaler.pkl"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        models = {}
        if Path(config.LR_MODEL).exists():
            models['Logistic Regression'] = joblib.load(config.LR_MODEL)
        if Path(config.RANDOM_FOREST_MODEL).exists():
            models['Random Forest'] = joblib.load(config.RANDOM_FOREST_MODEL)
        if Path(config.XGBOOST_MODEL).exists():
            models['XGBoost'] = joblib.load(config.XGBOOST_MODEL)
            
        print("✅ Data and models loaded successfully.")
        return X, y, groups, models, scaler
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return key metrics."""
    # Ensure all models have a predict_proba method for a fair comparison
    if not hasattr(model, 'predict_proba'):
        # For models without it, we can't calculate AUROC
        y_proba = np.full(y_test.shape, 0.5)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    # Predict based on a 0.5 threshold
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "AUROC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }

    # Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["True Positives"] = int(tp)
    metrics["False Positives"] = int(fp)
    metrics["True Negatives"] = int(tn)
    metrics["False Negatives"] = int(fn)

    return metrics

def calculate_ensemble_predictions(models, X_test):
    """Calculate ensemble probabilities from a set of models."""
    y_probas = []
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_probas.append(model.predict_proba(X_test)[:, 1])
    
    if not y_probas:
        return None
    
    # Simple averaging ensemble
    y_ensemble_proba = np.mean(y_probas, axis=0)
    return y_ensemble_proba

def main():
    """Main evaluation pipeline."""
    X, y, groups, models, scaler = load_data_and_models()
    if X is None:
        return

    print("\n🔬 Performing a patient-level evaluation...")
    # Patient-level split to avoid data leakage
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(iter(gkf.split(X, y, groups)))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    results_table = []

    # Evaluate each individual model
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results_table.append([name] + [f"{v:.4f}" if isinstance(v, float) else v for v in metrics.values()])
        
    # Evaluate the ensemble model
    y_ensemble_proba = calculate_ensemble_predictions(models, X_test)
    if y_ensemble_proba is not None:
        y_pred_ensemble = (y_ensemble_proba >= 0.5).astype(int)
        ensemble_metrics = {
            "AUROC": roc_auc_score(y_test, y_ensemble_proba),
            "Accuracy": accuracy_score(y_test, y_pred_ensemble),
            "Precision": precision_score(y_test, y_pred_ensemble, zero_division=0),
            "Recall": recall_score(y_test, y_pred_ensemble, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred_ensemble, zero_division=0),
        }
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
        ensemble_metrics["True Positives"] = int(tp)
        ensemble_metrics["False Positives"] = int(fp)
        ensemble_metrics["True Negatives"] = int(tn)
        ensemble_metrics["False Negatives"] = int(fn)
        results_table.append(["Ensemble Model"] + [f"{v:.4f}" if isinstance(v, float) else v for v in ensemble_metrics.values()])

    # Print the results table
    headers = ["Model", "AUROC", "Accuracy", "Precision", "Recall", "F1-Score",
               "True Pos.", "False Pos.", "True Neg.", "False Neg."]
    print("\n🎉 Model Evaluation Results 🎉")
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))
    
    # You can also add permutation importance for key models here if needed.
    
if __name__ == "__main__":
    main()