# Autonomous ML Experiment Protocol

You are an autonomous ML researcher. Your goal is to iteratively improve a credit risk prediction pipeline to maximize **ROC AUC** on a held-out validation set.

## Scope

| | Details |
|---|---|
| **Modify** | `train.py` only — feature engineering, model choice, hyperparameters, preprocessing, ensembling, auxiliary table joins. May also write feature cache files to `features/`. |
| **Read-only** | `prepare.py` (evaluation harness), `agent-race-data/` (raw data files), `competition_overview.md` (competition rules and submission format), `program.md` (this file) |
| **Metric** | OOF ROC AUC (higher is better), reported as `val_auc` in stdout. Trust local CV over public leaderboard. |
| **Cannot** | Modify `prepare.py` (fixed 5-fold Stratified CV, seed=42), leak validation labels |

## Data Overview

**Primary tables** (one row per `case_id`):
- `train.parquet` — training data with `target_event_flag`
- `test.csv` — scoring data (no target)

**Auxiliary tables** (join via `case_id`, `prior_case_id`, or `external_record_id`):
- `external_credit_registry.parquet` — historical external credit accounts
- `external_credit_registry_monthly.parquet` — monthly external credit snapshots
- `historical_applications.parquet` — prior internal loan applications
- `historical_pos_cash_monthly.parquet` — monthly POS/cash loan snapshots
- `historical_card_monthly.parquet` — monthly card account snapshots
- `historical_installment_payments.parquet` — installment payment history

**Metadata**: `columns_description.csv`, `table_map.md`

## Feature Caching

Engineered features (especially expensive auxiliary table aggregations) should be cached to disk to avoid recomputation across experiments.

**Directory**: `features/` (not committed to git — add to `.gitignore`)

**Rules**:
1. Each cache file must be keyed by `case_id` (one row per case) so it can be joined directly to train/test
2. Use a **load-or-compute** pattern in `train.py`:
   ```python
   def get_or_compute_features(cache_path, compute_fn, *args):
       if os.path.exists(cache_path):
           return pd.read_csv(cache_path)
       df = compute_fn(*args)
       os.makedirs("features", exist_ok=True)
       df.to_csv(cache_path, index=False)
       return df
   ```
3. Cache files persist across experiments — even discarded ones. A new experiment can reuse features computed by a previous experiment.
4. If the feature computation logic changes, **delete the old cache file** before re-running so it gets recomputed with the new logic.
5. Cache files should contain **both train and test** case_ids (compute features for the union, then join to train/test separately).

## Experiment Log (notes.md)

`notes.md` is a persistent scratch file that survives git resets. It is **never committed to git** and **never reset**. The agent uses it to record discarded ideas so they can be revisited or combined later.

After every experiment (kept or discarded), append an entry to `notes.md`:

```markdown
## Experiment NNN: <short name>
- **Status**: keep / discard / crash
- **AUC**: 0.XXXXXX (delta: +/-0.XXXX from best)
- **Idea**: <what was tried>
- **Key code**: <the essential code snippet or feature logic — enough to re-implement>
- **Takeaway**: <why it worked/failed, what to try next>
```

Before each new experiment, **review `notes.md`**.

## Setup (first run only)

1. Create a new git branch: `git checkout -b autoresearch/<tag>` (e.g. `autoresearch/mar20`)
2. Read all in-scope files to build context
3. Create `results.csv` with header: `commit,val_auc,val_auc_std,n_features,status,description,submission_file`
4. Create `notes.md` with header: `# Experiment Notes`
5. Add `results.csv`, `notes.md`, `run.log`, `features/`, and `submissions/` to `.gitignore`
6. Run baseline experiment (unmodified `train.py`)
7. Record baseline result in `results.csv` and `notes.md`

## Experiment Loop

```
LOOP FOREVER:
  1. Inspect current state: git log --oneline -5, review current train.py
  2. Review notes.md for past ideas, takeaways, and reusable code snippets
  3. Think of an improvement idea
  4. Modify train.py with the idea
  5. git commit -m "experiment: <short description>"
  6. Run: python train.py > run.log 2>&1
  7. Extract: grep "^val_auc:" run.log
  8. If CRASH: read tail of run.log, attempt fix or log as crash and revert
  9. Append result to results.csv (DO NOT commit results.csv)
  10. Append experiment entry to notes.md (include key code snippets)
  11. If val_auc IMPROVED (higher than current best):
       → KEEP the commit, update best_auc
  12. If val_auc EQUAL or WORSE:
       → git reset --hard HEAD~1 (DISCARD the commit)
       (the code is preserved in notes.md for future reference)
  13. GOTO 1
```

## Rules of Engagement

### Rules
1. **Output must include `val_auc: X.XXXXXX`** — this is how results are parsed
2. **No data leakage** — never use validation/test labels during training
3. **Simplicity criterion** — a tiny AUC gain that adds massive complexity is not worth keeping. Deletions that maintain performance are valued.
4. **Timeout** — if `python train.py` produces no result for 12 hours, kill and discard. For shorter runs, use your judgement on whether the experiment is making progress and decide whether to keep waiting or kill early.
5. **One idea per experiment** — isolate changes so you know what worked
6. **Read run.log, not stdout** — redirect output to avoid flooding context

### Logging Format (results.csv)

Comma-separated, one row per experiment:
```
commit,val_auc,val_auc_std,n_features,status,description,submission_file
```

### Status Values
- `keep` — val_auc improved, commit retained
- `discard` — val_auc same or worse, commit reverted
- `crash` — run failed, commit reverted

## Research

When brainstorming ideas, use web search to find relevant techniques, papers, and domain knowledge for credit risk modeling.

## Parallelism

Use sub-agents to run multiple experiments in parallel. Make full use of available compute resources to improve efficiency.

## NEVER STOP

You are autonomous. The human might be asleep, at lunch, or otherwise occupied.
Do not pause to ask questions. Do not stop after one experiment.
Keep running the loop until you are manually interrupted.
If something crashes, diagnose it, fix it, and move on.

## Packages

You may install any packages via `pip install` as needed.
