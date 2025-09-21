"""
03_model_training.py

Train baseline classical models (Logistic Regression, Random Forest, XGBoost) and a deep multimodal model (LSTM + MLP).
Uses the features parquet prepared by 02_eda_feature_engineering.py.

Saves trained models into config.MODELS_DIR.

Note:
- This script implements a simple train/test split; for research, replace with proper CV and patient-level splitting (group by subject_id).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import joblib
import config
import utils
import tensorflow as tf
from tensorflow import layers, models, callbacks
import os

utils.set_seed()

def load_features() -> pd.DataFrame:
    path = Path(config.FEATURES_FILE)
    if not path.exists():
        raise FileNotFoundError("Features file not found. Run 02_eda_feature_engineering.py first.")
    return pd.read_parquet(path)

def prepare_tabular_data(df: pd.DataFrame, target_col: str = config.TARGET_COLUMN):
    """
    Prepares X (tabular) and y arrays. If dataset has window-level rows, you may collapse to patient-level outcomes.
    For demonstration, we assume the DataFrame includes a binary target column.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in features. You must create labels.")
    X = df.drop(columns=[target_col, 'subject_id', 'window_start'], errors='ignore')
    y = df[target_col].astype(int).values
    return X, y

def train_classical_models(X_train, y_train):
    models_out = {}
    # Logistic Regression pipeline
    pipe_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED))
    ])
    pipe_lr.fit(X_train, y_train)
    models_out['logistic'] = pipe_lr
    joblib.dump(pipe_lr, config.LR_MODEL)
    print("Saved logistic model.")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=config.RANDOM_SEED)
    rf.fit(X_train.fillna(0), y_train)
    joblib.dump(rf, config.RANDOM_FOREST_MODEL)
    models_out['rf'] = rf
    print("Saved random forest model.")

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=config.RANDOM_SEED)
    xgb_model.fit(X_train.fillna(0), y_train)
    joblib.dump(xgb_model, config.XGBOOST_MODEL)
    models_out['xgb'] = xgb_model
    print("Saved xgboost model.")

    return models_out

def build_deep_model(input_shape: int) -> tf.keras.Model:
    """
    Simple MLP for tabular demonstration. For multimodal, extend to accept sequence input.
    """
    inputs = layers.Input(shape=(input_shape,), name='tabular_input')
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def train_deep_model(X_train, y_train, X_val, y_val):
    model = build_deep_model(X_train.shape[1])
    es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=128, callbacks=[es], verbose=2)
    model.save(config.DEEP_MODEL)
    print(f"Saved deep model to {config.DEEP_MODEL}")
    return model

def main():
    df = load_features()
    # For research: ensure to construct target labels. Here we require df to have config.TARGET_COLUMN.
    X, y = prepare_tabular_data(df)
    # Group-level splitting to avoid patient leakage
    groups = df['subject_id'] if 'subject_id' in df.columns else None
    if groups is not None:
        gkf = GroupKFold(n_splits=5)
        # pick one fold as test
        train_idx, test_idx = next(iter(gkf.split(X, y, groups)))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)

    # Train classical models
    classical_models = train_classical_models(X_train, y_train)

    # Evaluate classical models
    for name, m in classical_models.items():
        if hasattr(m, "predict_proba"):
            y_proba = m.predict_proba(X_test.fillna(0))[:, 1]
        else:
            # fallback
            y_proba = m.predict(X_test.fillna(0))
        print(f"{name} AUROC: {roc_auc_score(y_test, y_proba):.4f}")

    # Train deep model on tabular features (for demonstration)
    # Scale using scaler saved earlier
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_train_s = scaler.transform(X_train.fillna(0))
        X_test_s = scaler.transform(X_test.fillna(0))
    else:
        X_train_s = X_train.fillna(0).values
        X_test_s = X_test.fillna(0).values
    deep_model = train_deep_model(X_train_s, y_train, X_test_s, y_test)
    # Evaluate deep
    y_proba = deep_model.predict(X_test_s).ravel()
    print("Deep model AUROC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    main()


