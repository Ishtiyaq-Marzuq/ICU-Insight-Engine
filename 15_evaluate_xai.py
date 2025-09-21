import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import load_model
import config
from tabulate import tabulate

def load_data_and_models():
    """Load features, target, and all trained models."""
    print("⏳ Loading data and models for XAI evaluation...")
    try:
        # Load features and target
        features_path = Path(config.FEATURES_FILE)
        df = pd.read_parquet(features_path)
        
        X = df.drop(columns=[config.TARGET_COLUMN, 'subject_id', 'window_start'], errors='ignore').fillna(0)
        y = df[config.TARGET_COLUMN].astype(int)
        groups = df['subject_id'] if 'subject_id' in df.columns else None
        
        # Load models (only those with predict_proba for this evaluation)
        models = {}
        if Path(config.LR_MODEL).exists():
            models['Logistic Regression'] = joblib.load(config.LR_MODEL)
        if Path(config.RANDOM_FOREST_MODEL).exists():
            models['Random Forest'] = joblib.load(config.RANDOM_FOREST_MODEL)
        if Path(config.XGBOOST_MODEL).exists():
            models['XGBoost'] = joblib.load(config.XGBOOST_MODEL)
            
        print("✅ Data and models loaded successfully.")
        return X, y, groups, models
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None, None

def evaluate_fidelity(model, X_test, y_test, sorted_features_indices):
    """
    Evaluates fidelity by measuring the change in AUROC when removing the most important features.
    
    A large drop in AUROC means the explanation has high fidelity, as the model's performance
    was heavily reliant on the features identified as important.
    """
    # Baseline AUROC
    baseline_proba = model.predict_proba(X_test)[:, 1]
    baseline_auroc = roc_auc_score(y_test, baseline_proba)
    
    # Remove top features and re-evaluate
    top_5_percent = int(len(sorted_features_indices) * 0.05)
    top_features_to_remove = sorted_features_indices[:top_5_percent]
    
    X_test_perturbed = X_test.copy()
    X_test_perturbed.iloc[:, top_features_to_remove] = 0
    
    perturbed_proba = model.predict_proba(X_test_perturbed)[:, 1]
    perturbed_auroc = roc_auc_score(y_test, perturbed_proba)
    
    auroc_drop = baseline_auroc - perturbed_auroc
    return auroc_drop

def evaluate_comprehensiveness(model, X_test, y_test, sorted_features_indices):
    """
    Evaluates comprehensiveness by measuring the change in AUROC when removing the least important features.
    
    A small change in AUROC means the explanation is comprehensive, as the model's performance
    was not reliant on features identified as unimportant.
    """
    # Baseline AUROC
    baseline_proba = model.predict_proba(X_test)[:, 1]
    baseline_auroc = roc_auc_score(y_test, baseline_proba)
    
    # Remove least important features and re-evaluate
    bottom_5_percent = int(len(sorted_features_indices) * 0.05)
    bottom_features_to_remove = sorted_features_indices[-bottom_5_percent:]
    
    X_test_perturbed = X_test.copy()
    X_test_perturbed.iloc[:, bottom_features_to_remove] = 0
    
    perturbed_proba = model.predict_proba(X_test_perturbed)[:, 1]
    perturbed_auroc = roc_auc_score(y_test, perturbed_proba)
    
    auroc_change = baseline_auroc - perturbed_auroc
    return auroc_change

def main():
    """Main XAI evaluation pipeline"""
    X, y, groups, models = load_data_and_models()
    if X is None:
        return

    # Patient-level split to ensure fair evaluation
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(iter(gkf.split(X, y, groups)))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    xai_results = []
    
    for name, model in models.items():
        print(f"\nEvaluating XAI for {name}...")
        
        # Calculate Permutation Importance to rank features
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=-1)
        sorted_indices = result.importances_mean.argsort()[::-1]
        
        # Calculate Fidelity (drop in AUROC after removing important features)
        fidelity_drop = evaluate_fidelity(model, X_test, y_test, sorted_indices)
        
        # Calculate Comprehensiveness (change in AUROC after removing unimportant features)
        comprehensiveness_change = evaluate_comprehensiveness(model, X_test, y_test, sorted_indices)
        
        xai_results.append([name, f"{fidelity_drop:.4f}", f"{comprehensiveness_change:.4f}"])
        
    # Print results in a table
    headers = ["Model", "Fidelity (AUROC Drop)", "Comprehensiveness (AUROC Change)"]
    print("\n🎉 XAI Evaluation Results 🎉")
    print("---------------------------------------------------------------------")
    print(tabulate(xai_results, headers=headers, tablefmt="fancy_grid"))
    print("\nInterpretation:")
    print("- High Fidelity (large AUROC drop) means the explanation reflects the model's logic.")
    print("- High Comprehensiveness (small AUROC change) means the model doesn't rely on unimportant features.")

if __name__ == "__main__":
    main()