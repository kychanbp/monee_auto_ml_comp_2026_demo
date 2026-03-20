# Agent Race — Contestant Guide

## Background

This is an internal credit-risk modeling competition (Agent Race). You are given a dataset and are expected to build a prediction pipeline (e.g. via AI-assisted coding), then submit your predictions. The platform will evaluate submissions using **ROC AUC** against held-out answers and maintain a leaderboard. No account sign-up is required: you identify yourself with your email and display name at submission time.

## Task and Data Introduction

**Task:** Binary classification. Predict payment stress (early installment delinquency) for a lending platform. The binary target `target_event_flag` indicates whether an applicant had a late payment beyond a threshold on their first installments (1 = yes, 0 = no).

**Evaluation:** ROC AUC on the test set. The platform holds the ground truth; your submission file is scored against it.

### Data Files

| File | Description |
|------|-------------|
| `train.parquet` | Training data. One row per `case_id`. Includes `target_event_flag` and features (demographics, income, credit amount, housing/building stats, external sources, etc.). |
| `test.csv` | Scoring data. Same grain (one row per `case_id`) but without the target column. Predict `target_event_flag` (or its probability) for each `case_id`. |
| `columns_description.csv` | Descriptions of columns across all tables. |
| `table_map.md` | Schema, grains, and join relationships for all tables. |

### Auxiliary Tables

The following tables can be joined with the main application data (e.g. by `case_id`, `prior_case_id`, `external_record_id`) for feature engineering:

- `external_credit_registry.parquet`
- `external_credit_registry_monthly.parquet`
- `historical_applications.parquet`
- `historical_pos_cash_monthly.parquet`
- `historical_card_monthly.parquet`
- `historical_installment_payments.parquet`

Use the column description and table map to align your pipeline with the required format.

## Submission Requirements

1. **File format:** CSV with exactly two columns (column names must match):
   - `case_id` — Sample (application/case) ID, one row per test sample.
   - `target_event_flag` (or `Target` / `TARGET`) — Your predicted value or probability for the target event (numeric).

2. **No duplicate IDs:** Each `case_id` must appear at most once. Duplicate IDs will cause the submission to be rejected.

3. **Identification:** When submitting on the website, provide your email and display name. Your email is your unique contest identity; the leaderboard uses the best score per email.

4. **Limits:** There is a daily submission limit per email and a maximum file size (see the contest Overview on the website for current values). After the contest deadline, no new submissions are accepted.

5. **Ranking rule:** Only submissions that fully match the reference set (no missing or extra IDs) are ranked. If your file has missing or extra IDs, you still get an AUC score and feedback (e.g. coverage, missing rate), but that submission is **Unranked** and does not update your place on the leaderboard.

## Important Notes

- **Data types:** `case_id` must be integer; the prediction column must be numeric (float or 0/1).
- **Coverage:** For your submission to count on the leaderboard, it must include exactly the same set of test `case_id`s as the reference (no missing, no extra). Otherwise the run is Unranked; you still see your AUC and coverage/missing rate.
- **Reference:** Use `test.csv` and the table map to ensure your output format and `case_id` set are correct before uploading.
