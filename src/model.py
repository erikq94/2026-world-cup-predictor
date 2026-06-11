"""
model.py

Trains an XGBoost classifier to predict match outcomes (Win / Draw / Loss).
Handles:
  - Train / validation / test split
  - Cross-validation
  - Hyperparameter tuning
  - Model persistence (save/load via joblib)
  - Evaluation metrics (accuracy, log-loss, confusion matrix)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PROCESSED = Path('data/processed')
MODELS_DIR = Path('models')

FEATURE_COLS = [
    'elo_diff', 'form_diff', 'attack_diff', 'defense_diff',
    'gk_diff', 'depth_diff', 'pace_diff', 'stamina_diff',
    'passing_diff', 'altitude', 'heat', 'humidity',
    'is_wc', 'neutral'
]
TARGET_COL = 'result'

def load_data():
    df = pd.read_csv(DATA_PROCESSED / 'feature_matrix.csv')

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Encode string labels to integers: Away Win=0, Draw=1, Home Win=2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {le.classes_}")
    print(f"Training samples: {len(X)}")
    print(f"Class distribution:\n{pd.Series(y).value_counts(normalize=True).round(3)}")

    return X, y_encoded, le

def train_baseline(X, y):
    print("\n--- Logistic Regression Baseline ---")
    # Pipeline: scale features first, then fit — fixes convergence warning
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    print("\n--- XGBoost ---")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
    )
    # No class weighting: the simulation samples from predict_proba, so we
    # optimize CALIBRATION (log-loss), not draw recall. Balancing inflated draws.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    ll  = -cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss')
    print(f"CV Accuracy: {acc.mean():.3f} (+/- {acc.std():.3f})")
    print(f"CV Log-Loss: {ll.mean():.3f}  (lower = better calibrated)")
    model.fit(X, y)

    # Calibration check: average predicted draw probability vs actual draw rate
    probs = model.predict_proba(X)
    pred_draw_rate = probs[:, 1].mean()
    true_draw_rate = (y == 1).mean()
    print(f"Predicted draw rate: {pred_draw_rate:.1%}  |  Actual draw rate: {true_draw_rate:.1%}")
    return model

def plot_confusion_matrix(y_true, y_pred, classes, title):
    MODELS_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f'{title.replace(" ", "_")}.png')
    plt.close()
    print(f" Confusion matrix saved to models/")


def save_model(model, le):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / 'xgboost_model.pkl')
    joblib.dump(le,    MODELS_DIR / 'label_encoder.pkl')
    print("  Model saved to models/")

def main():
    X, y, le = load_data()

    baseline = train_baseline(X, y)
    xgb_model = train_xgboost(X, y)

    print("\n--- Final Evaluation (XGBoost on full training data) ---")
    y_pred = xgb_model.predict(X)
    print(classification_report(y, y_pred, target_names=le.classes_))
    plot_confusion_matrix(y, y_pred, le.classes_, 'XGBoost Confusion Matrix')

    save_model(xgb_model, le)
    print("\nDone. Ready to simulate the tournament.")

if __name__ == '__main__':
    main()