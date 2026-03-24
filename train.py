"""
train.py — ML pipeline for credit risk prediction.
THIS IS THE ONLY FILE THE AGENT MAY MODIFY.
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from prepare import (
    load_train, load_test, evaluate, generate_submission,
    load_auxiliary, TARGET_COL, ID_COL,
)
import hashlib, inspect, os, gc

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")

def split_lockbox(df, fraction=0.15, seed=99):
    from sklearn.model_selection import train_test_split
    os.makedirs(SPLITS_DIR, exist_ok=True)
    dh = hashlib.md5(pd.util.hash_pandas_object(df[[ID_COL, TARGET_COL]]).values.tobytes()).hexdigest()[:8]
    lp = os.path.join(SPLITS_DIR, f"lockbox_indices_{dh}.npy")
    if os.path.exists(lp):
        li = np.load(lp); ci = np.setdiff1d(np.arange(len(df)), li)
    else:
        ci, li = train_test_split(np.arange(len(df)), test_size=fraction, random_state=seed, stratify=df[TARGET_COL].values)
        np.save(lp, li)
    return df.iloc[ci].reset_index(drop=True), df.iloc[li].reset_index(drop=True)

def get_or_compute_features(compute_fn, *args):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    sh = hashlib.md5(inspect.getsource(compute_fn).encode()).hexdigest()[:8]
    fp = hashlib.md5("|".join(f"{a.shape[0]}x{a.shape[1]}" for a in args if isinstance(a, pd.DataFrame)).encode()).hexdigest()[:6]
    cp = os.path.join(FEATURES_DIR, f"{compute_fn.__name__}_{sh}_{fp}.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    r = compute_fn(*args); r.to_parquet(cp, index=False); return r


def compute_bureau_features(train_df, test_df):
    """Aggregate features from external_credit_registry."""
    bureau = load_auxiliary("external_credit_registry.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    # Numeric aggregations
    num_aggs = {
        "days_since_credit_opened": ["count", "mean", "max", "min"],
        "days_credit_overdue": ["mean", "max", "sum"],
        "days_until_credit_end": ["mean", "min"],
        "amount_credit_max_overdue": ["mean", "max"],
        "count_credit_extension": ["sum", "mean"],
        "amount_credit_sum": ["sum", "mean", "max"],
        "amount_credit_sum_debt": ["sum", "mean", "max"],
        "amount_credit_sum_limit": ["sum", "mean"],
        "amount_credit_sum_overdue": ["sum", "mean", "max"],
        "amount_annuity_payment": ["sum", "mean"],
        "days_credit_update": ["mean", "min"],
    }
    agg = bureau.groupby(ID_COL).agg(num_aggs)
    agg.columns = ["bureau_" + "_".join(c) for c in agg.columns]

    # Count by account status
    status_counts = bureau.groupby([ID_COL, "account_status"]).size().unstack(fill_value=0)
    status_counts.columns = ["bureau_status_" + str(c) for c in status_counts.columns]

    # Count by account type (top types only)
    type_counts = bureau.groupby([ID_COL, "account_type"]).size().unstack(fill_value=0)
    type_counts.columns = ["bureau_type_" + str(c) for c in type_counts.columns]

    # Debt ratio
    features = agg.join(status_counts).join(type_counts)
    if "bureau_amount_credit_sum_sum" in features.columns and "bureau_amount_credit_sum_debt_sum" in features.columns:
        features["bureau_debt_credit_ratio"] = (
            features["bureau_amount_credit_sum_debt_sum"] /
            features["bureau_amount_credit_sum_sum"].replace(0, np.nan)
        )
    if "bureau_amount_credit_sum_overdue_sum" in features.columns and "bureau_amount_credit_sum_sum" in features.columns:
        features["bureau_overdue_credit_ratio"] = (
            features["bureau_amount_credit_sum_overdue_sum"] /
            features["bureau_amount_credit_sum_sum"].replace(0, np.nan)
        )

    # Merge back
    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_prev_app_features(train_df, test_df):
    """Aggregate features from historical_applications."""
    prev = load_auxiliary("historical_applications.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    # Numeric aggregations
    num_aggs = {
        "amount_annuity_payment": ["mean", "max", "sum"],
        "amount_application": ["mean", "max", "min"],
        "amount_credit": ["mean", "max", "sum"],
        "amount_down_payment": ["mean", "max"],
        "amount_goods_price": ["mean", "max"],
        "days_decision": ["mean", "max", "min"],
        "count_payment": ["mean", "max", "sum"],
        "rate_down_payment": ["mean"],
        "sellerplace_area": ["mean", "max"],
        "hour_application_start": ["mean"],
    }
    agg = prev.groupby(ID_COL).agg(num_aggs)
    agg.columns = ["prev_" + "_".join(c) for c in agg.columns]

    # Count total prior apps
    cnt = prev.groupby(ID_COL).size().rename("prev_app_count")

    # Count by contract status (approved, refused, etc.)
    status_counts = prev.groupby([ID_COL, "category_contract_status"]).size().unstack(fill_value=0)
    status_counts.columns = ["prev_status_" + str(c) for c in status_counts.columns]

    # Approval rate
    features = agg.join(cnt).join(status_counts)
    if "prev_status_Approved" in features.columns:
        features["prev_approval_rate"] = (
            features["prev_status_Approved"] / features["prev_app_count"].replace(0, np.nan)
        )

    # Count by contract type
    type_counts = prev.groupby([ID_COL, "category_contract_type"]).size().unstack(fill_value=0)
    type_counts.columns = ["prev_type_" + str(c) for c in type_counts.columns]
    features = features.join(type_counts)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_prev_app_enriched_features(train_df, test_df):
    """Enriched historical application features — amount discrepancies, recency, rejection patterns."""
    prev = load_auxiliary("historical_applications.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    # Amount discrepancy: how much credit differs from application
    prev["credit_app_diff"] = prev["amount_credit"] - prev["amount_application"]
    prev["credit_app_ratio"] = prev["amount_credit"] / prev["amount_application"].replace(0, np.nan)

    # Most recent application (smallest days_decision = most recent)
    most_recent = prev.sort_values("days_decision").groupby(ID_COL).first()
    recent_cols = ["days_decision", "amount_credit", "amount_annuity_payment",
                   "category_contract_status", "credit_app_diff"]
    recent_feats = most_recent[recent_cols].copy()
    recent_feats.columns = ["prev_recent_" + c for c in recent_cols]

    # Approved-only stats
    approved = prev[prev["category_contract_status"] == "Approved"]
    approved_agg = approved.groupby(ID_COL).agg({
        "amount_credit": ["mean", "sum"],
        "days_decision": ["min", "max"],
    })
    approved_agg.columns = ["prev_approved_" + "_".join(c) for c in approved_agg.columns]

    # Credit discrepancy stats
    disc_agg = prev.groupby(ID_COL).agg({
        "credit_app_diff": ["mean", "max", "min"],
        "credit_app_ratio": ["mean", "min"],
    })
    disc_agg.columns = ["prev_" + "_".join(c) for c in disc_agg.columns]

    # Yield group distribution
    yield_counts = prev.groupby([ID_COL, "category_yield_group"]).size().unstack(fill_value=0)
    yield_counts.columns = ["prev_yield_" + str(c) for c in yield_counts.columns]

    features = recent_feats.join(approved_agg).join(disc_agg).join(yield_counts)

    # Encode the recent contract status
    if "prev_recent_category_contract_status" in features.columns:
        features["prev_recent_category_contract_status"] = features["prev_recent_category_contract_status"].astype(str)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_installment_features(train_df, test_df):
    """Aggregate features from historical_installment_payments."""
    inst = load_auxiliary("historical_installment_payments.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    # Payment delay: days_entry_payment - days_installment (positive = late)
    inst["payment_delay"] = inst["days_entry_payment"] - inst["days_installment"]
    # Payment difference: amount_payment - amount_installment (negative = underpaid)
    inst["payment_diff"] = inst["amount_payment"] - inst["amount_installment"]
    inst["payment_ratio"] = inst["amount_payment"] / inst["amount_installment"].replace(0, np.nan)
    inst["is_late"] = (inst["payment_delay"] > 0).astype(int)
    inst["is_underpaid"] = (inst["payment_diff"] < -1).astype(int)

    agg = inst.groupby(ID_COL).agg({
        "payment_delay": ["mean", "max", "min", "std", "sum"],
        "payment_diff": ["mean", "max", "min", "sum"],
        "payment_ratio": ["mean", "min"],
        "is_late": ["sum", "mean"],
        "is_underpaid": ["sum", "mean"],
        "amount_installment": ["mean", "max", "sum"],
        "amount_payment": ["mean", "max", "sum"],
        "number_installment_number": ["max"],
        "days_installment": ["min", "max"],
    })
    agg.columns = ["inst_" + "_".join(c) for c in agg.columns]

    # Count of installment records
    cnt = inst.groupby(ID_COL).size().rename("inst_count")
    features = agg.join(cnt)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_bureau_active_features(train_df, test_df):
    """Separate features for active vs closed bureau accounts."""
    bureau = load_auxiliary("external_credit_registry.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    num_cols = ["amount_credit_sum", "amount_credit_sum_debt", "amount_credit_sum_overdue",
                "amount_annuity_payment", "days_since_credit_opened", "days_credit_overdue"]

    features_list = []
    for status_val, prefix in [("Active", "bureau_active"), ("Closed", "bureau_closed")]:
        subset = bureau[bureau["account_status"] == status_val]
        if len(subset) == 0:
            continue
        agg = subset.groupby(ID_COL).agg({c: ["count", "mean", "sum"] for c in num_cols if c in subset.columns})
        agg.columns = [f"{prefix}_{'_'.join(c)}" for c in agg.columns]
        features_list.append(agg)

    features = features_list[0] if features_list else pd.DataFrame(index=all_ids[ID_COL])
    for f in features_list[1:]:
        features = features.join(f, how="outer")

    # Active credit load ratio
    if "bureau_active_amount_credit_sum_sum" in features.columns:
        features["bureau_active_debt_ratio"] = (
            features.get("bureau_active_amount_credit_sum_debt_sum", 0) /
            features["bureau_active_amount_credit_sum_sum"].replace(0, np.nan)
        )

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_installment_recent_features(train_df, test_df):
    """Recent installment payment behavior (last 12 months only)."""
    inst = load_auxiliary("historical_installment_payments.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    # Filter to recent installments (days_installment is negative, closer to 0 = more recent)
    recent = inst[inst["days_installment"] > -365]
    recent = recent.copy()
    recent["payment_delay"] = recent["days_entry_payment"] - recent["days_installment"]
    recent["payment_diff"] = recent["amount_payment"] - recent["amount_installment"]
    recent["is_late"] = (recent["payment_delay"] > 0).astype(int)
    recent["late_days"] = recent["payment_delay"].clip(lower=0)

    agg = recent.groupby(ID_COL).agg({
        "payment_delay": ["mean", "max", "std"],
        "is_late": ["sum", "mean"],
        "late_days": ["sum", "max"],
        "payment_diff": ["mean", "min"],
        "amount_payment": ["sum", "mean"],
    })
    agg.columns = ["inst_recent_" + "_".join(c) for c in agg.columns]
    cnt = recent.groupby(ID_COL).size().rename("inst_recent_count")
    features = agg.join(cnt)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_installment_multiwindow_features(train_df, test_df):
    """Multi-window installment payment features (3mo, 6mo, 24mo)."""
    inst = load_auxiliary("historical_installment_payments.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    inst = inst.copy()
    inst["payment_delay"] = inst["days_entry_payment"] - inst["days_installment"]
    inst["is_late"] = (inst["payment_delay"] > 0).astype(int)
    inst["late_days"] = inst["payment_delay"].clip(lower=0)

    all_features = []
    for window_days, prefix in [(90, "inst_3m"), (180, "inst_6m"), (730, "inst_24m")]:
        window = inst[inst["days_installment"] > -window_days]
        agg = window.groupby(ID_COL).agg({
            "payment_delay": ["mean", "max"],
            "is_late": ["mean", "sum"],
            "late_days": ["max"],
        })
        agg.columns = [f"{prefix}_{'_'.join(c)}" for c in agg.columns]
        cnt = window.groupby(ID_COL).size().rename(f"{prefix}_count")
        wf = agg.join(cnt)
        all_features.append(wf)

    features = all_features[0]
    for f in all_features[1:]:
        features = features.join(f, how="outer")

    # Trend features: recent vs older behavior
    if "inst_3m_is_late_mean" in features.columns and "inst_24m_is_late_mean" in features.columns:
        features["inst_late_trend_3m_vs_24m"] = (
            features["inst_3m_is_late_mean"] - features["inst_24m_is_late_mean"]
        )
    if "inst_6m_payment_delay_mean" in features.columns and "inst_24m_payment_delay_mean" in features.columns:
        features["inst_delay_trend_6m_vs_24m"] = (
            features["inst_6m_payment_delay_mean"] - features["inst_24m_payment_delay_mean"]
        )

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_pos_recent_features(train_df, test_df):
    """Recent POS cash behavior (last 12 months)."""
    pos = load_auxiliary("historical_pos_cash_monthly.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    recent = pos[pos["months_relative"] >= -12].copy()
    recent["has_dpd"] = (recent["days_past_due"] > 0).astype(int)

    agg = recent.groupby(ID_COL).agg({
        "days_past_due": ["mean", "max", "sum"],
        "has_dpd": ["sum", "mean"],
        "count_installment_future": ["mean", "min"],
    })
    agg.columns = ["pos_recent_" + "_".join(c) for c in agg.columns]
    cnt = recent.groupby(ID_COL).size().rename("pos_recent_count")
    features = agg.join(cnt)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_card_recent_features(train_df, test_df):
    """Recent card behavior (last 12 months)."""
    card = load_auxiliary("historical_card_monthly.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    recent = card[card["months_relative"] >= -12].copy()
    recent["utilization"] = recent["amount_balance"] / recent["amount_credit_limit_actual"].replace(0, np.nan)
    recent["has_dpd"] = (recent["days_past_due"] > 0).astype(int)

    agg = recent.groupby(ID_COL).agg({
        "utilization": ["mean", "max", "std"],
        "amount_balance": ["mean", "max"],
        "amount_payment_current": ["mean", "sum"],
        "days_past_due": ["mean", "max"],
        "has_dpd": ["sum", "mean"],
        "amount_drawings_current": ["mean", "sum"],
    })
    agg.columns = ["card_recent_" + "_".join(c) for c in agg.columns]
    cnt = recent.groupby(ID_COL).size().rename("card_recent_count")
    features = agg.join(cnt)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_pos_cash_features(train_df, test_df):
    """Aggregate features from historical_pos_cash_monthly."""
    pos = load_auxiliary("historical_pos_cash_monthly.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    agg = pos.groupby(ID_COL).agg({
        "months_relative": ["count", "max", "min"],
        "count_installment": ["mean", "max"],
        "count_installment_future": ["mean", "min"],
        "days_past_due": ["mean", "max", "sum"],
        "days_past_due_tolerance": ["mean", "max", "sum"],
    })
    agg.columns = ["pos_" + "_".join(c) for c in agg.columns]

    # DPD flags
    pos_dpd = pos[pos["days_past_due"] > 0].groupby(ID_COL).size().rename("pos_dpd_count")
    # Contract status counts
    status = pos.groupby([ID_COL, "category_contract_status"]).size().unstack(fill_value=0)
    status.columns = ["pos_status_" + str(c) for c in status.columns]

    features = agg.join(pos_dpd).join(status)
    features["pos_dpd_count"] = features["pos_dpd_count"].fillna(0)

    result = all_ids.merge(features, on=ID_COL, how="left")
    return result


def compute_card_features(train_df, test_df):
    """Aggregate features from historical_card_monthly."""
    card = load_auxiliary("historical_card_monthly.parquet")
    all_ids = pd.concat([train_df[[ID_COL]], test_df[[ID_COL]]])

    card["utilization"] = card["amount_balance"] / card["amount_credit_limit_actual"].replace(0, np.nan)

    agg = card.groupby(ID_COL).agg({
        "months_relative": ["count", "max", "min"],
        "amount_balance": ["mean", "max", "min"],
        "amount_credit_limit_actual": ["mean", "max"],
        "amount_drawings_atm_current": ["mean", "max", "sum"],
        "amount_drawings_current": ["mean", "max", "sum"],
        "amount_payment_current": ["mean", "max", "sum"],
        "amount_payment_total_current": ["mean", "sum"],
        "amount_receivable_principal": ["mean", "max"],
        "amount_total_receivable": ["mean", "max"],
        "count_drawings_atm_current": ["mean", "sum"],
        "count_drawings_current": ["mean", "sum"],
        "count_installment_mature_cum": ["max"],
        "days_past_due": ["mean", "max", "sum"],
        "days_past_due_tolerance": ["mean", "max"],
        "utilization": ["mean", "max", "min"],
    })
    agg.columns = ["card_" + "_".join(c) for c in agg.columns]

    result = all_ids.merge(agg, on=ID_COL, how="left")
    return result


def build_features(train_df, test_df):
    """Build features. Returns (X, y, X_test, test_ids)."""
    y = train_df[TARGET_COL]
    test_ids = test_df[ID_COL]

    drop_cols = [TARGET_COL, ID_COL]
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Bureau features
    bureau_feats = get_or_compute_features(compute_bureau_features, train_df, test_df)
    n_train = len(X)
    bureau_train = bureau_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    bureau_test = bureau_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X.reset_index(drop=True), bureau_train], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), bureau_test], axis=1)

    # Previous application features
    prev_feats = get_or_compute_features(compute_prev_app_features, train_df, test_df)
    prev_train = prev_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    prev_test = prev_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, prev_train], axis=1)
    X_test = pd.concat([X_test, prev_test], axis=1)

    # Installment payment features
    inst_feats = get_or_compute_features(compute_installment_features, train_df, test_df)
    inst_train = inst_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    inst_test = inst_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, inst_train], axis=1)
    X_test = pd.concat([X_test, inst_test], axis=1)

    # Enriched previous application features
    prev_enr = get_or_compute_features(compute_prev_app_enriched_features, train_df, test_df)
    prev_enr_train = prev_enr.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    prev_enr_test = prev_enr.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, prev_enr_train], axis=1)
    X_test = pd.concat([X_test, prev_enr_test], axis=1)

    # Bureau active/closed segmented features
    bact_feats = get_or_compute_features(compute_bureau_active_features, train_df, test_df)
    bact_train = bact_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    bact_test = bact_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, bact_train], axis=1)
    X_test = pd.concat([X_test, bact_test], axis=1)

    # Recent installment features (last 12 months)
    inst_recent = get_or_compute_features(compute_installment_recent_features, train_df, test_df)
    inst_recent_train = inst_recent.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    inst_recent_test = inst_recent.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, inst_recent_train], axis=1)
    X_test = pd.concat([X_test, inst_recent_test], axis=1)

    # POS cash features
    pos_feats = get_or_compute_features(compute_pos_cash_features, train_df, test_df)
    pos_train = pos_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    pos_test = pos_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, pos_train], axis=1)
    X_test = pd.concat([X_test, pos_test], axis=1)

    # Card monthly features
    card_feats = get_or_compute_features(compute_card_features, train_df, test_df)
    card_train = card_feats.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    card_test = card_feats.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, card_train], axis=1)
    X_test = pd.concat([X_test, card_test], axis=1)

    # Multi-window installment features
    inst_mw = get_or_compute_features(compute_installment_multiwindow_features, train_df, test_df)
    inst_mw_train = inst_mw.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    inst_mw_test = inst_mw.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, inst_mw_train], axis=1)
    X_test = pd.concat([X_test, inst_mw_test], axis=1)

    # Recent POS cash features
    pos_recent = get_or_compute_features(compute_pos_recent_features, train_df, test_df)
    pos_recent_train = pos_recent.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    pos_recent_test = pos_recent.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, pos_recent_train], axis=1)
    X_test = pd.concat([X_test, pos_recent_test], axis=1)

    # Recent card features
    card_recent = get_or_compute_features(compute_card_recent_features, train_df, test_df)
    card_recent_train = card_recent.iloc[:n_train].drop(columns=[ID_COL]).reset_index(drop=True)
    card_recent_test = card_recent.iloc[n_train:].drop(columns=[ID_COL]).reset_index(drop=True)
    X = pd.concat([X, card_recent_train], axis=1)
    X_test = pd.concat([X_test, card_recent_test], axis=1)

    # Domain-specific feature engineering on main table
    for df in [X, X_test]:
        # Financial ratios
        df["credit_income_ratio"] = df["amount_credit"] / df["amount_income_annual"].replace(0, np.nan)
        df["annuity_income_ratio"] = df["amount_annuity_payment"] / df["amount_income_annual"].replace(0, np.nan)
        df["credit_goods_ratio"] = df["amount_credit"] / df["amount_goods_price"].replace(0, np.nan)
        df["credit_annuity_ratio"] = df["amount_credit"] / df["amount_annuity_payment"].replace(0, np.nan)
        df["goods_income_ratio"] = df["amount_goods_price"] / df["amount_income_annual"].replace(0, np.nan)
        # Age-related
        df["age_years"] = -df["days_since_birth"] / 365.25
        df["employment_years"] = -df["days_since_employment_start"] / 365.25
        df["employment_age_ratio"] = df["employment_years"] / df["age_years"].replace(0, np.nan)
        # External source combinations
        df["ext_source_mean"] = df[["external_source_1", "external_source_2", "external_source_3"]].mean(axis=1)
        df["ext_source_prod"] = df["external_source_1"] * df["external_source_2"] * df["external_source_3"]
        df["ext_source_std"] = df[["external_source_1", "external_source_2", "external_source_3"]].std(axis=1)
        # Income per family member
        df["income_per_family"] = df["amount_income_annual"] / df["count_family_members"].replace(0, np.nan)
        df["income_per_child"] = df["amount_income_annual"] / (df["count_children"] + 1)
        # Document count
        doc_cols = [c for c in df.columns if c.startswith("has_document_")]
        df["document_count"] = df[doc_cols].sum(axis=1)
        # Phone/contact score
        df["contact_score"] = (
            df.get("has_mobile_phone", 0) + df.get("has_employer_phone", 0) +
            df.get("has_home_phone_reported", 0) + df.get("has_email_reported", 0)
        )
        # Region mismatch score
        mismatch_cols = [c for c in df.columns if "diff_from" in c]
        df["region_mismatch_count"] = df[mismatch_cols].sum(axis=1)
        # Credit bureau inquiry features
        inq_cols = [c for c in df.columns if c.startswith("amount_req_credit_bureau_")]
        if inq_cols:
            df["total_inquiries"] = df[inq_cols].sum(axis=1)
        # Social circle default rates
        if "observation_30_cnt_social_circle" in df.columns:
            df["social_default_rate_30"] = df["default_30_cnt_social_circle"] / df["observation_30_cnt_social_circle"].replace(0, np.nan)
        if "observation_60_cnt_social_circle" in df.columns:
            df["social_default_rate_60"] = df["default_60_cnt_social_circle"] / df["observation_60_cnt_social_circle"].replace(0, np.nan)

    # Smoothed mean target encoding for high-cardinality categoricals
    # Using global mean as prior with smoothing factor
    global_mean = y.mean()
    smooth_factor = 20  # regularization
    high_card_cols = ["organization_type", "category_occupation_type"]
    for col in high_card_cols:
        if col in X.columns:
            col_str = X[col].astype(str)
            counts = col_str.value_counts()
            target_means = pd.DataFrame({"col": col_str, "target": y.values}).groupby("col")["target"].agg(["mean", "count"])
            smoothed = (target_means["count"] * target_means["mean"] + smooth_factor * global_mean) / (target_means["count"] + smooth_factor)
            X[col + "_target_enc"] = col_str.map(smoothed).astype(float)
            X_test[col + "_target_enc"] = X_test[col].astype(str).map(smoothed).fillna(global_mean).astype(float)

    # Label-encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    X["_case_id"] = train_df[ID_COL].values
    X_test["_case_id"] = test_df[ID_COL].values
    return X, y, X_test, test_ids


def train_fn(X_train, y_train, X_val, X_test):
    """Diverse ensemble with meta-model features, seed averaging, and extra models."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold as InnerKFold

    # Extract case_ids
    train_ids = X_train["_case_id"].values.copy()
    val_ids = X_val["_case_id"].values.copy()
    test_ids_fn = X_test["_case_id"].values.copy()
    X_train = X_train.drop(columns=["_case_id"])
    X_val = X_val.drop(columns=["_case_id"])
    X_test = X_test.drop(columns=["_case_id"])

    # KNN target mean features
    knn_cols = ["ext_source_mean", "credit_annuity_ratio"]
    knn_cols_avail = [c for c in knn_cols if c in X_train.columns]
    if len(knn_cols_avail) >= 2:
        imp = SimpleImputer(strategy="median")
        Xk_tr = imp.fit_transform(X_train[knn_cols_avail])
        Xk_va = imp.transform(X_val[knn_cols_avail])
        Xk_te = imp.transform(X_test[knn_cols_avail])
        k = 200; nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
        nn.fit(Xk_tr); y_arr = y_train.values
        _, vi = nn.kneighbors(Xk_va)
        X_val = X_val.copy(); X_val["knn_target_mean"] = np.array([y_arr[i].mean() for i in vi])
        _, ti = nn.kneighbors(Xk_te)
        X_test = X_test.copy(); X_test["knn_target_mean"] = np.array([y_arr[i].mean() for i in ti])
        nn2 = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree", n_jobs=-1)
        nn2.fit(Xk_tr); _, tri = nn2.kneighbors(Xk_tr)
        X_train = X_train.copy(); X_train["knn_target_mean"] = np.array([y_arr[i[1:]].mean() for i in tri])
        del nn, nn2, Xk_tr, Xk_va, Xk_te

    # === Multi-table meta-model features (Kaggle winning technique) ===
    # Train small LGB on each auxiliary table's records to predict target,
    # then aggregate predictions by case_id. Each table captures different risk dimension.
    tmap = pd.Series(y_train.values, index=train_ids)
    train_id_set = set(train_ids)
    mp = {"objective":"binary","metric":"auc","num_leaves":32,"learning_rate":0.05,
          "feature_fraction":0.6,"bagging_fraction":0.8,"bagging_freq":5,
          "min_child_samples":30,"verbose":-1,"seed":42,"n_jobs":-1}
    ikf = InnerKFold(n_splits=3, shuffle=True, random_state=99)
    tcd = pd.DataFrame({"_id":train_ids,"_t":y_train.values}).drop_duplicates(subset=["_id"])
    X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()

    def build_meta_features(table_name, num_cols, prefix, join_col=ID_COL, nrounds=300):
        """Build meta-model features for one auxiliary table."""
        tbl = load_auxiliary(table_name)
        cols = [c for c in num_cols if c in tbl.columns]
        if not cols: return
        # Inner CV for training fold
        recs = []
        for itr, iva in ikf.split(tcd, tcd["_t"]):
            itr_ids = set(tcd.iloc[itr]["_id"].values)
            iva_ids = set(tcd.iloc[iva]["_id"].values)
            ip = tbl[tbl[join_col].isin(itr_ids)].copy()
            ip["_t"] = ip[join_col].map(tmap); ip = ip.dropna(subset=["_t"])
            if len(ip) < 100: continue
            mm = lgb.train(mp, lgb.Dataset(ip[cols].fillna(-999), label=ip["_t"]),
                           num_boost_round=nrounds, callbacks=[lgb.log_evaluation(period=0)])
            ivp = tbl[tbl[join_col].isin(iva_ids)]
            if len(ivp) > 0:
                t = ivp[[join_col]].copy(); t["p"] = mm.predict(ivp[cols].fillna(-999))
                a = t.groupby(join_col)["p"].agg(["mean","max","std"]).reset_index()
                a.columns = [join_col, f"{prefix}_mean", f"{prefix}_max", f"{prefix}_std"]
                recs.append(a)
            del ip, mm; gc.collect()
        if not recs: return
        tr_df = pd.concat(recs, ignore_index=True)
        # Full model for val/test
        fp = tbl[tbl[join_col].isin(train_id_set)].copy()
        fp["_t"] = fp[join_col].map(tmap); fp = fp.dropna(subset=["_t"])
        fm = lgb.train(mp, lgb.Dataset(fp[cols].fillna(-999), label=fp["_t"]),
                       num_boost_round=nrounds, callbacks=[lgb.log_evaluation(period=0)])
        pt = tbl[[join_col]].copy(); pt["p"] = fm.predict(tbl[cols].fillna(-999))
        full_agg = pt.groupby(join_col)["p"].agg(["mean","max","std"]).reset_index()
        full_agg.columns = [join_col, f"{prefix}_mean", f"{prefix}_max", f"{prefix}_std"]
        del fp, fm, tbl, pt; gc.collect()
        # Add to X_train, X_val, X_test
        fnames = [f"{prefix}_mean", f"{prefix}_max", f"{prefix}_std"]
        tm = pd.DataFrame({join_col: train_ids}).merge(tr_df, on=join_col, how="left")
        for c in fnames: X_train[c] = tm[c].values
        vm = pd.DataFrame({join_col: val_ids}).merge(full_agg, on=join_col, how="left")
        for c in fnames: X_val[c] = vm[c].values
        tsm = pd.DataFrame({join_col: test_ids_fn}).merge(full_agg, on=join_col, how="left")
        for c in fnames: X_test[c] = tsm[c].values

    # 1. Previous applications meta-model
    build_meta_features("historical_applications.parquet",
        ["amount_credit","amount_annuity_payment","amount_application","days_decision",
         "count_payment","amount_down_payment","rate_down_payment","amount_goods_price",
         "hour_application_start","sellerplace_area"],
        "meta_prev", nrounds=300)

    # 2. Bureau (external credit registry) meta-model
    build_meta_features("external_credit_registry.parquet",
        ["days_since_credit_opened","days_credit_overdue","days_until_credit_end",
         "amount_credit_max_overdue","count_credit_extension","amount_credit_sum",
         "amount_credit_sum_debt","amount_credit_sum_limit","amount_credit_sum_overdue",
         "amount_annuity_payment","days_credit_update"],
        "meta_bureau", nrounds=300)

    # 3. Installment payments meta-model
    build_meta_features("historical_installment_payments.parquet",
        ["number_installment_number","days_installment","days_entry_payment",
         "amount_installment","amount_payment"],
        "meta_inst", nrounds=200)

    # 4. POS cash monthly meta-model
    build_meta_features("historical_pos_cash_monthly.parquet",
        ["months_relative","count_installment","count_installment_future",
         "days_past_due","days_past_due_tolerance"],
        "meta_pos", nrounds=200)

    # 5. Card monthly meta-model
    build_meta_features("historical_card_monthly.parquet",
        ["months_relative","amount_balance","amount_credit_limit_actual",
         "amount_drawings_atm_current","amount_drawings_current",
         "amount_payment_current","amount_payment_total_current",
         "amount_receivable_principal","amount_total_receivable",
         "count_drawings_atm_current","count_drawings_current",
         "days_past_due","days_past_due_tolerance"],
        "meta_card", nrounds=200)

    # 6. Bureau monthly meta-model (status → DPD risk)
    try:
        bm = load_auxiliary("external_credit_registry_monthly.parquet")
        # Join to case_id via bureau table
        bureau = load_auxiliary("external_credit_registry.parquet")
        rec2case = bureau[[ID_COL,"external_record_id"]].drop_duplicates()
        bm = bm.merge(rec2case, on="external_record_id", how="left")
        dpd_map = {"C":0,"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"X":0}
        bm["dpd_level"] = bm["status"].map(dpd_map).fillna(0).astype(np.int8)
        bm["is_dpd"] = (bm["dpd_level"] > 0).astype(np.int8)
        # Build features at case_id level for meta-model
        bm_cols = ["months_relative","dpd_level","is_dpd"]
        bm_recs = []
        for itr, iva in ikf.split(tcd, tcd["_t"]):
            itr_ids = set(tcd.iloc[itr]["_id"].values)
            iva_ids = set(tcd.iloc[iva]["_id"].values)
            ip = bm[bm[ID_COL].isin(itr_ids)].copy()
            ip["_t"] = ip[ID_COL].map(tmap); ip = ip.dropna(subset=["_t"])
            if len(ip) < 100: continue
            mm = lgb.train(mp, lgb.Dataset(ip[bm_cols].fillna(-999), label=ip["_t"]),
                           num_boost_round=200, callbacks=[lgb.log_evaluation(period=0)])
            ivp = bm[bm[ID_COL].isin(iva_ids)]
            if len(ivp) > 0:
                t = ivp[[ID_COL]].copy(); t["p"] = mm.predict(ivp[bm_cols].fillna(-999))
                a = t.groupby(ID_COL)["p"].agg(["mean","max","std"]).reset_index()
                a.columns = [ID_COL,"meta_bm_mean","meta_bm_max","meta_bm_std"]
                bm_recs.append(a)
            del ip, mm; gc.collect()
        if bm_recs:
            bm_tr = pd.concat(bm_recs, ignore_index=True)
            fp = bm[bm[ID_COL].isin(train_id_set)].copy()
            fp["_t"] = fp[ID_COL].map(tmap); fp = fp.dropna(subset=["_t"])
            fm = lgb.train(mp, lgb.Dataset(fp[bm_cols].fillna(-999), label=fp["_t"]),
                           num_boost_round=200, callbacks=[lgb.log_evaluation(period=0)])
            pt = bm[[ID_COL]].copy(); pt["p"] = fm.predict(bm[bm_cols].fillna(-999))
            bm_full = pt.groupby(ID_COL)["p"].agg(["mean","max","std"]).reset_index()
            bm_full.columns = [ID_COL,"meta_bm_mean","meta_bm_max","meta_bm_std"]
            for c in ["meta_bm_mean","meta_bm_max","meta_bm_std"]:
                X_train[c] = pd.DataFrame({ID_COL:train_ids}).merge(bm_tr,on=ID_COL,how="left")[c].values
                X_val[c] = pd.DataFrame({ID_COL:val_ids}).merge(bm_full,on=ID_COL,how="left")[c].values
                X_test[c] = pd.DataFrame({ID_COL:test_ids_fn}).merge(bm_full,on=ID_COL,how="left")[c].values
            del fp, fm, bm_tr, bm_full
        del bm, bureau, rec2case; gc.collect()
    except Exception:
        pass  # Skip if OOM

    gc.collect()

    # === Diverse ensemble: 7 model types, LGB seed-averaged ===
    dtrain = lgb.Dataset(X_train, label=y_train)
    all_val, all_test, weights = [], [], []

    # LGB deep: 3 seeds averaged (weight 0.20)
    vv, tt = [], []
    for s in [42, 123, 456]:
        m = lgb.train({"objective":"binary","metric":"auc","boosting_type":"gbdt",
            "learning_rate":0.02,"num_leaves":63,"max_depth":-1,"min_child_samples":30,
            "feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":5,
            "reg_alpha":0.1,"reg_lambda":1.0,"min_gain_to_split":0.01,
            "verbose":-1,"seed":s,"n_jobs":-1}, dtrain, num_boost_round=2000,
            callbacks=[lgb.log_evaluation(period=0)])
        vv.append(m.predict(X_val)); tt.append(m.predict(X_test)); del m
    all_val.append(np.mean(vv,axis=0)); all_test.append(np.mean(tt,axis=0)); weights.append(0.20)

    # LGB shallow: 3 seeds averaged (weight 0.15)
    vv, tt = [], []
    for s in [77, 234, 567]:
        m = lgb.train({"objective":"binary","metric":"auc","boosting_type":"gbdt",
            "learning_rate":0.02,"max_depth":5,"num_leaves":31,"min_child_samples":50,
            "feature_fraction":0.3,"bagging_fraction":0.8,"bagging_freq":5,
            "reg_alpha":0.5,"reg_lambda":5.0,"verbose":-1,"seed":s,"n_jobs":-1},
            dtrain, num_boost_round=3000, callbacks=[lgb.log_evaluation(period=0)])
        vv.append(m.predict(X_val)); tt.append(m.predict(X_test)); del m
    all_val.append(np.mean(vv,axis=0)); all_test.append(np.mean(tt,axis=0)); weights.append(0.15)

    # LGB very shallow + GOSS (weight 0.10 + 0.05)
    m = lgb.train({"objective":"binary","metric":"auc","boosting_type":"gbdt",
        "learning_rate":0.03,"max_depth":3,"num_leaves":8,"min_child_samples":100,
        "feature_fraction":0.4,"bagging_fraction":0.7,"bagging_freq":5,
        "reg_alpha":1.0,"reg_lambda":5.0,"verbose":-1,"seed":555,"n_jobs":-1},
        dtrain, num_boost_round=4000, callbacks=[lgb.log_evaluation(period=0)])
    all_val.append(m.predict(X_val)); all_test.append(m.predict(X_test)); del m; weights.append(0.05)

    m = lgb.train({"objective":"binary","metric":"auc","boosting_type":"goss",
        "learning_rate":0.02,"num_leaves":63,"max_depth":-1,"min_child_samples":30,
        "feature_fraction":0.7,"reg_alpha":0.1,"reg_lambda":1.0,
        "verbose":-1,"seed":99,"n_jobs":-1},
        dtrain, num_boost_round=2000, callbacks=[lgb.log_evaluation(period=0)])
    all_val.append(m.predict(X_val)); all_test.append(m.predict(X_test)); del m; weights.append(0.10)
    del dtrain; gc.collect()

    # XGB: 2 seeds averaged (weight 0.25)
    dxtr = xgb.DMatrix(X_train, label=y_train)
    vv, tt = [], []
    for s in [42, 123]:
        m = xgb.train({"objective":"binary:logistic","eval_metric":"auc",
            "learning_rate":0.02,"max_depth":6,"min_child_weight":30,
            "subsample":0.7,"colsample_bytree":0.7,"reg_alpha":0.1,
            "reg_lambda":1.0,"seed":s,"nthread":-1,"verbosity":0},
            dxtr, num_boost_round=2000)
        vv.append(m.predict(xgb.DMatrix(X_val))); tt.append(m.predict(xgb.DMatrix(X_test))); del m
    all_val.append(np.mean(vv,axis=0)); all_test.append(np.mean(tt,axis=0)); weights.append(0.25)
    del dxtr; gc.collect()

    # CatBoost standard + deep (weight 0.15 + 0.10)
    m = CatBoostClassifier(iterations=2000,learning_rate=0.03,depth=6,
        l2_leaf_reg=3.0,random_seed=42,verbose=0,thread_count=-1)
    m.fit(X_train,y_train)
    all_val.append(m.predict_proba(X_val)[:,1]); all_test.append(m.predict_proba(X_test)[:,1])
    del m; gc.collect(); weights.append(0.15)

    m = CatBoostClassifier(iterations=1500,learning_rate=0.02,depth=8,
        l2_leaf_reg=5.0,random_seed=99,verbose=0,thread_count=-1)
    m.fit(X_train,y_train)
    all_val.append(m.predict_proba(X_val)[:,1]); all_test.append(m.predict_proba(X_test)[:,1])
    del m; gc.collect(); weights.append(0.10)

    val_preds = sum(w*v for w,v in zip(weights, all_val))
    test_preds = sum(w*t for w,t in zip(weights, all_test))
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
    fa = results["fold_aucs"]
    n_feat = X.shape[1] - (1 if "_case_id" in X.columns else 0)
    print(f"\nval_auc: {results['oof_auc']:.6f}")
    print(f"val_auc_std: {results['std_fold_auc']:.6f}")
    print(f"n_features: {n_feat}")
    print(f"fold_auc_min: {min(fa):.6f}")
    print(f"fold_auc_max: {max(fa):.6f}")
    print(f"elapsed_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
