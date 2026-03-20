"""
train.py — ML pipeline for credit risk prediction.
THIS IS THE ONLY FILE THE AGENT MAY MODIFY.
"""

import time
from prepare import (
    load_train, load_test, evaluate, generate_submission,
    TARGET_COL, ID_COL,
)


def build_features(train_df, test_df):
    """Build features. Returns (X, y, X_test, test_ids)."""
    pass


def train_fn(X_train, y_train, X_val, X_test):
    """Train a model. Returns (val_preds, test_preds)."""
    pass


def main():
    start_time = time.time()

    train_df = load_train()
    test_df = load_test()

    X, y, X_test, test_ids = build_features(train_df, test_df)

    results = evaluate(train_fn, X, y, X_test)

    generate_submission(test_ids, results["test_preds"])

    elapsed = time.time() - start_time
    print(f"\nval_auc: {results['oof_auc']:.6f}")
    print(f"val_auc_std: {results['std_fold_auc']:.6f}")
    print(f"n_features: {X.shape[1]}")
    print(f"elapsed_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
