"""Shared data loading, split, preprocessing, and metrics.

Agreed once for every model notebook. Do not change the seed, the split,
or the feature lists without a group decision. If each model uses a
different split, the comparison in notebook 07 is not fair.

Target is default_flag (about 13% positive). Raw text and the hidden
ground-truth text labels are never model inputs.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TARGET = "default_flag"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank_customers_enriched.csv"
RESULTS_DIR = ROOT / "results"

# Same columns as the Component 1 "Numeric + AI features" check.
# sentiment_label is omitted because sentiment_score already carries it.
NUMERIC_FEATURES = [
    "age",
    "dependents",
    "monthly_income",
    "tenure_months",
    "ctos_score",
    "loan_amount",
    "loan_term_months",
    "interest_rate_pct",
    "credit_utilization",
    "monthly_installment",
    "other_monthly_debt",
    "dti_ratio",
    "late_payments_12m",
    "savings_balance",
    "digital_channel_ratio",
    "num_products_held",
    "txn_count_monthly",
    "avg_txn_amount",
    "num_service_calls_90d",
    "sentiment_score",
    "text_risk_score",
    "has_feedback",
    "has_agent_note",
]
CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education",
    "employment_status",
    "region",
    "product_type",
    "text_topic",
    "text_risk_category",
    "customer_segment",
]
# Present in the CSV, never used as model inputs.
EXCLUDED_COLUMNS = [
    "customer_id",
    "customer_feedback",
    "agent_note",
    "sentiment_label",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    """Load the enriched customer table and check the frozen column contract."""
    df = pd.read_csv(path)
    expected = set(FEATURE_COLUMNS + EXCLUDED_COLUMNS + [TARGET])
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing or extra:
        raise ValueError(
            "Enriched file does not match the frozen column list. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return df


def train_test_data(df=None):
    """Stratified 80/20 split. Every notebook must use this and no other split."""
    if df is None:
        df = load_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )


def make_preprocessor():
    """Median impute + scale numerics; mode impute + one-hot categoricals.

    One preprocessor for every model, including trees, so the comparison
    uses the same matrix.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(estimator):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]
    )


def stratified_kfold():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def cv_roc_auc(pipeline, X_train, y_train):
    """5-fold ROC-AUC on the training set only. The holdout stays untouched."""
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=stratified_kfold(),
        scoring="roc_auc",
        n_jobs=-1,
    )
    return {
        "n_splits": N_SPLITS,
        "roc_auc_mean": round(float(scores.mean()), 6),
        "roc_auc_std": round(float(scores.std()), 6),
        "roc_auc_folds": [round(float(s), 6) for s in scores],
    }


def holdout_metrics(y_true, y_pred, y_prob):
    """Scores on the frozen test set. Positive class is default_flag = 1."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def split_summary(y_train, y_test):
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": round(float(y_train.mean()), 6),
        "positive_rate_test": round(float(y_test.mean()), 6),
    }


def save_metrics(model_key, cv, holdout, y_train, y_test):
    """Write results/<model_key>_metrics.json for notebook 07 to read."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_key,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "split": split_summary(y_train, y_test),
        "cv": cv,
        "holdout": holdout,
    }
    path = RESULTS_DIR / f"{model_key}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_all_metrics(results_dir=RESULTS_DIR):
    """Read every results/*_metrics.json file. Missing files are skipped."""
    rows = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_data()
    print(split_summary(y_train, y_test))
    print(f"features: {X_train.shape[1]}  numeric: {len(NUMERIC_FEATURES)}  categorical: {len(CATEGORICAL_FEATURES)}")
