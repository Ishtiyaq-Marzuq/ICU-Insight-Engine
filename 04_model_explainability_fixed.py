#!/usr/bin/env python3
"""
04_model_explainability_fixed.py

Simplified explainability script that works with our current setup.
"""

import os
import joblib
import numpy as np
import pandas as pd
import config
import utils
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance

def explain_tree_model_simple(model_path: str, X_sample: pd.DataFrame, y_sample: pd.Series, save_name: str = "feature_importance.png"):
    """Simple feature importance for tree-based models"""
    model = joblib.load(model_path)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = X_sample.columns
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {model_path.stem}")
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        
        out_path = Path(config.EXPLAIN_DIR) / save_name
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✓ Saved feature importance to {out_path}")
        
        # Save importance data
        importance_data = {
            'feature_names': feature_names.tolist(),
            'importance_scores': importance.tolist(),
            'top_features': [feature_names[i] for i in indices[:10]]
        }
        
        utils.save_json(importance_data, str(config.EXPLAIN_DIR / f"{model_path.stem}_importance.json"))
        print(f"✓ Saved importance data to {config.EXPLAIN_DIR / f'{model_path.stem}_importance.json'}")
    
    # Permutation importance
    try:
        perm_importance = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(perm_importance.importances_mean)[::-1][:20]
        
        plt.title(f"Permutation Importance - {model_path.stem}")
        plt.barh(range(len(indices)), perm_importance.importances_mean[indices])
        plt.yticks(range(len(indices)), [X_sample.columns[i] for i in indices])
        plt.xlabel("Permutation Importance")
        plt.tight_layout()
        
        out_path = Path(config.EXPLAIN_DIR) / f"perm_{save_name}"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✓ Saved permutation importance to {out_path}")
        
    except Exception as e:
        print(f"⚠️  Permutation importance failed: {e}")

def explain_logistic_regression(model_path: str, X_sample: pd.DataFrame, save_name: str = "lr_coefficients.png"):
    """Explain logistic regression with coefficients"""
    model = joblib.load(model_path)
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        feature_names = X_sample.columns
        
        # Sort by absolute coefficient value
        indices = np.argsort(np.abs(coefficients))[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if coef < 0 else 'blue' for coef in coefficients[indices]]
        plt.title(f"Logistic Regression Coefficients - {model_path.stem}")
        plt.barh(range(len(indices)), coefficients[indices], color=colors)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Coefficient Value")
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_path = Path(config.EXPLAIN_DIR) / save_name
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✓ Saved coefficients to {out_path}")
        
        # Save coefficient data
        coef_data = {
            'feature_names': feature_names.tolist(),
            'coefficients': coefficients.tolist(),
            'top_positive': [feature_names[i] for i in indices[:10] if coefficients[i] > 0],
            'top_negative': [feature_names[i] for i in indices[:10] if coefficients[i] < 0]
        }
        
        utils.save_json(coef_data, str(config.EXPLAIN_DIR / f"{model_path.stem}_coefficients.json"))
        print(f"✓ Saved coefficient data to {config.EXPLAIN_DIR / f'{model_path.stem}_coefficients.json'}")

def create_model_comparison(X_sample: pd.DataFrame, y_sample: pd.Series):
    """Create a comparison of all models"""
    models = []
    model_names = []
    
    # Load available models
    if Path(config.RANDOM_FOREST_MODEL).exists():
        models.append(joblib.load(config.RANDOM_FOREST_MODEL))
        model_names.append("Random Forest")
    
    if Path(config.XGBOOST_MODEL).exists():
        models.append(joblib.load(config.XGBOOST_MODEL))
        model_names.append("XGBoost")
    
    if Path(config.LR_MODEL).exists():
        models.append(joblib.load(config.LR_MODEL))
        model_names.append("Logistic Regression")
    
    if not models:
        print("No models found for comparison")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]
            
            axes[i].barh(range(len(indices)), importance[indices])
            axes[i].set_yticks(range(len(indices)))
            axes[i].set_yticklabels([X_sample.columns[j] for j in indices])
            axes[i].set_title(f"{name} - Top 10 Features")
            axes[i].set_xlabel("Importance")
    
    plt.tight_layout()
    out_path = Path(config.EXPLAIN_DIR) / "model_comparison.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved model comparison to {out_path}")

def main():
    """Main explainability pipeline"""
    print("🔍 Starting Model Explainability Analysis")
    print("=" * 50)
    
    # Load dataset
    if not Path(config.FEATURES_FILE).exists():
        raise FileNotFoundError("Features file not found. Run feature engineering first.")
    
    df = pd.read_parquet(config.FEATURES_FILE)
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} features")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'window_start', 'deterioration_label']]
    
    X = df[feature_cols].fillna(0)
    y = df['deterioration_label']
    
    print(f"✓ Prepared {len(feature_cols)} features for explanation")
    
    # Explain Random Forest
    if Path(config.RANDOM_FOREST_MODEL).exists():
        print("\n🌳 Explaining Random Forest...")
        explain_tree_model_simple(config.RANDOM_FOREST_MODEL, X, y, "rf_importance.png")
    
    # Explain XGBoost
    if Path(config.XGBOOST_MODEL).exists():
        print("\n🚀 Explaining XGBoost...")
        explain_tree_model_simple(config.XGBOOST_MODEL, X, y, "xgb_importance.png")
    
    # Explain Logistic Regression
    if Path(config.LR_MODEL).exists():
        print("\n📊 Explaining Logistic Regression...")
        explain_logistic_regression(config.LR_MODEL, X, "lr_coefficients.png")
    
    # Create model comparison
    print("\n📈 Creating model comparison...")
    create_model_comparison(X, y)
    
    print("\n🎉 Explainability analysis complete!")
    print(f"✓ Results saved to {config.EXPLAIN_DIR}")

if __name__ == "__main__":
    main()


  
