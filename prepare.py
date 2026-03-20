"""
prepare.py — READ-ONLY evaluation harness.
DO NOT MODIFY THIS FILE. The agent may only modify train.py.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-race-data")
N_FOLDS = 5
RANDOM_SEED = 42
TARGET_COL = "target_event_flag"
ID_COL = "case_id"

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_train():
    """Load the primary training table."""
    return pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))

def load_test():
    """Load the primary test table."""
    return pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def load_auxiliary(name: str):
    """Load an auxiliary parquet table by filename (without path)."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Auxiliary table not found: {path}")
    return pd.read_parquet(path)

# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(train_fn, X, y, X_test):
    """
    Run 5-fold Stratified K-Fold CV and generate averaged test predictions.

    Args:
        train_fn: callable(X_train, y_train, X_val, X_test)
            Must return (val_preds, test_preds) — both np.ndarray of probabilities.
        X: feature DataFrame (no target, no ID)
        y: target Series
        X_test: test feature DataFrame (no ID)

    Returns:
        dict with keys:
            fold_aucs, oof_auc, mean_fold_auc, std_fold_auc,
            oof_preds, test_preds
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(len(X))
    test_preds_sum = np.zeros(len(X_test))
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        val_preds, test_preds = train_fn(X_train, y_train, X_val, X_test)

        oof_preds[val_idx] = val_preds
        test_preds_sum += test_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}: AUC = {fold_auc:.6f}")

    # Overall OOF AUC
    oof_auc = roc_auc_score(y, oof_preds)

    # Average test predictions across folds
    test_preds_avg = np.clip(test_preds_sum / N_FOLDS, 0, 1)

    print(f"  OOF AUC: {oof_auc:.6f}")
    print(f"  Mean Fold AUC: {np.mean(fold_aucs):.6f} (+/- {np.std(fold_aucs):.6f})")

    return {
        "fold_aucs": fold_aucs,
        "oof_auc": oof_auc,
        "mean_fold_auc": np.mean(fold_aucs),
        "std_fold_auc": np.std(fold_aucs),
        "oof_preds": oof_preds,
        "test_preds": test_preds_avg,
    }

# ─── Submission ──────────────────────────────────────────────────────────────

SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")

def generate_submission(test_ids, predictions, description=""):
    """
    Generate and validate a submission CSV file.
    Saves a timestamped copy to submissions/ and a latest copy for convenience.

    Args:
        test_ids: array-like of case_id values
        predictions: array-like of predicted probabilities (clipped to [0, 1])
        description: short experiment description for the filename
    """
    from datetime import datetime

    predictions = np.clip(predictions, 0, 1)
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: predictions})

    # Validation
    assert len(sub) == len(set(sub[ID_COL])), "ERROR: Duplicate case_ids in submission!"
    assert sub[TARGET_COL].notna().all(), "ERROR: NaN predictions found!"
    assert sub[TARGET_COL].dtype in [np.float64, np.float32, np.int64, np.int32], \
        "ERROR: Predictions must be numeric!"

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # Timestamped submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_desc = description.replace(" ", "_")[:50] if description else "experiment"
    filename = f"submission_{timestamp}_{safe_desc}.csv"
    filepath = os.path.join(SUBMISSIONS_DIR, filename)
    sub.to_csv(filepath, index=False)

    # Latest copy for convenience
    latest_path = os.path.join(SUBMISSIONS_DIR, "submission_latest.csv")
    sub.to_csv(latest_path, index=False)

    print(f"Submission saved: {filepath} ({len(sub)} rows)")
    return sub
