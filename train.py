"""
train.py — ML pipeline for credit risk prediction.
THIS IS THE ONLY FILE THE AGENT MAY MODIFY.
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from prepare import (
    load_train, load_test, evaluate, generate_submission,
    split_lockbox, evaluate_lockbox, get_or_compute_features,
    TARGET_COL, ID_COL,
)


def build_features(train_df, test_df):
    """Build features. Returns (X, y, X_test, test_ids)."""
    y = train_df[TARGET_COL]
    test_ids = test_df[ID_COL]

    drop_cols = [TARGET_COL, ID_COL]
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Identify categorical columns and label-encode them
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    return X, y, X_test, test_ids


def train_fn(X_train, y_train, X_val, X_test):
    """Train a model. Returns (val_preds, test_preds)."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    return val_preds, test_preds


def main():
    start_time = time.time()

    train_df = load_train()
    test_df = load_test()

    cv_pool_df, lockbox_df = split_lockbox(train_df)

    X, y, X_test, test_ids = build_features(cv_pool_df, test_df)

    results = evaluate(train_fn, X, y, X_test)

    generate_submission(test_ids, results["test_preds"])

    elapsed = time.time() - start_time
    print(f"\nval_auc: {results['oof_auc']:.6f}")
    print(f"val_auc_std: {results['std_fold_auc']:.6f}")
    print(f"n_features: {X.shape[1]}")
    print(f"pr_auc: {results['oof_pr_auc']:.6f}")
    print(f"logloss: {results['oof_logloss']:.6f}")
    print(f"ks: {results['oof_ks']:.6f}")
    print(f"fold_auc_min: {results['fold_auc_min']:.6f}")
    print(f"fold_auc_max: {results['fold_auc_max']:.6f}")
    print(f"elapsed_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
