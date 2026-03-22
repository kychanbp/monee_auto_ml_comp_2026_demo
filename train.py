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
    split_lockbox, evaluate_lockbox, get_or_compute_features,
    load_auxiliary, TARGET_COL, ID_COL,
)


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

    return X, y, X_test, test_ids


def train_fn(X_train, y_train, X_val, X_test):
    """Train LightGBM + XGBoost blend. Returns (val_preds, test_preds)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.impute import SimpleImputer

    # KNN target mean features (computed within fold to avoid leakage)
    knn_cols = ["ext_source_mean", "credit_annuity_ratio"]
    knn_cols_avail = [c for c in knn_cols if c in X_train.columns]
    if len(knn_cols_avail) >= 2:
        imp = SimpleImputer(strategy="median")
        X_knn_train = imp.fit_transform(X_train[knn_cols_avail])
        X_knn_val = imp.transform(X_val[knn_cols_avail])
        X_knn_test = imp.transform(X_test[knn_cols_avail])

        k = 200
        nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
        nn.fit(X_knn_train)
        y_arr = y_train.values

        _, val_idx = nn.kneighbors(X_knn_val)
        X_val = X_val.copy()
        X_val["knn_target_mean"] = np.array([y_arr[idx].mean() for idx in val_idx])

        _, test_idx = nn.kneighbors(X_knn_test)
        X_test = X_test.copy()
        X_test["knn_target_mean"] = np.array([y_arr[idx].mean() for idx in test_idx])

        nn_self = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree", n_jobs=-1)
        nn_self.fit(X_knn_train)
        _, train_idx = nn_self.kneighbors(X_knn_train)
        X_train = X_train.copy()
        X_train["knn_target_mean"] = np.array([y_arr[idx[1:]].mean() for idx in train_idx])

    # LightGBM model 1 — original deep config
    lgb_params1 = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
        "min_child_samples": 30, "feature_fraction": 0.7, "bagging_fraction": 0.7,
        "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_gain_to_split": 0.01, "verbose": -1, "seed": 42, "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    lgb1 = lgb.train(lgb_params1, dtrain, num_boost_round=2000,
                     callbacks=[lgb.log_evaluation(period=0)])
    lgb1_val = lgb1.predict(X_val)
    lgb1_test = lgb1.predict(X_test)

    # LightGBM model 2 — shallow config from top Kaggle solution (depth=5, feat_frac=0.3)
    lgb_params2 = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.02, "max_depth": 5, "num_leaves": 31,
        "min_child_samples": 50, "feature_fraction": 0.3, "bagging_fraction": 0.8,
        "bagging_freq": 5, "reg_alpha": 0.5, "reg_lambda": 5.0,
        "verbose": -1, "seed": 77, "n_jobs": -1,
    }
    lgb2 = lgb.train(lgb_params2, dtrain, num_boost_round=3000,
                     callbacks=[lgb.log_evaluation(period=0)])
    lgb2_val = lgb2.predict(X_val)
    lgb2_test = lgb2.predict(X_test)

    lgb_val = 0.6 * lgb1_val + 0.4 * lgb2_val
    lgb_test = 0.6 * lgb1_test + 0.4 * lgb2_test

    # XGBoost
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 30,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "nthread": -1,
        "verbosity": 0,
    }
    dxtr = xgb.DMatrix(X_train, label=y_train)
    xgb_model = xgb.train(xgb_params, dxtr, num_boost_round=2000)
    xgb_val = xgb_model.predict(xgb.DMatrix(X_val))
    xgb_test = xgb_model.predict(xgb.DMatrix(X_test))

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    cb_model.fit(X_train, y_train)
    cb_val = cb_model.predict_proba(X_val)[:, 1]
    cb_test = cb_model.predict_proba(X_test)[:, 1]

    # Level-2 stacking: train a meta-learner on base model predictions
    # Use internal 3-fold CV on training data to generate OOF meta-features
    from sklearn.model_selection import StratifiedKFold as SKF2
    from sklearn.linear_model import LogisticRegression

    base_val = np.column_stack([lgb1_val, lgb2_val, xgb_val, cb_val])
    base_test = np.column_stack([lgb1_test, lgb2_test, xgb_test, cb_test])
    base_train = np.column_stack([
        lgb1.predict(X_train), lgb2.predict(X_train),
        xgb_model.predict(xgb.DMatrix(X_train)),
        cb_model.predict_proba(X_train)[:, 1],
    ])

    # Add key raw features for the stacker to use
    key_feats = ["knn_target_mean", "ext_source_mean", "credit_annuity_ratio"]
    key_avail = [c for c in key_feats if c in X_train.columns]
    if key_avail:
        imp_st = SimpleImputer(strategy="median")
        kf_train = imp_st.fit_transform(X_train[key_avail])
        kf_val = imp_st.transform(X_val[key_avail])
        kf_test = imp_st.transform(X_test[key_avail])
        meta_train = np.hstack([base_train, kf_train])
        meta_val = np.hstack([base_val, kf_val])
        meta_test = np.hstack([base_test, kf_test])
    else:
        meta_train = base_train
        meta_val = base_val
        meta_test = base_test

    # Train OOF stacker to avoid overfitting
    meta_oof = np.zeros(len(meta_train))
    skf2 = SKF2(n_splits=3, shuffle=True, random_state=99)
    for tr_idx, va_idx in skf2.split(meta_train, y_train):
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(meta_train[tr_idx], y_train.values[tr_idx])
        meta_oof[va_idx] = lr.predict_proba(meta_train[va_idx])[:, 1]

    # Final stacker trained on all training meta-features
    lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_final.fit(meta_train, y_train)
    stacker_val = lr_final.predict_proba(meta_val)[:, 1]
    stacker_test = lr_final.predict_proba(meta_test)[:, 1]

    # Blend base models (80%) + stacker (20%)
    base_blend_val = 0.50 * (0.6*lgb1_val+0.4*lgb2_val) + 0.25 * xgb_val + 0.25 * cb_val
    base_blend_test = 0.50 * (0.6*lgb1_test+0.4*lgb2_test) + 0.25 * xgb_test + 0.25 * cb_test
    val_preds = 0.80 * base_blend_val + 0.20 * stacker_val
    test_preds = 0.80 * base_blend_test + 0.20 * stacker_test

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
