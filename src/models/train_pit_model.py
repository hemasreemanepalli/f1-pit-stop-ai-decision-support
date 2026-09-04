"""
Train and compare models for PitNextLap (binary classification).

Split (race/time-aware, never mixes laps from the same race across
splits, per project spec):
    train: seasons 2018-2022
    val:   season 2023
    test:  season 2024

Models compared:
    1. baseline   - predicts the historical PitNextLap rate for every row
                     (DummyClassifier, stratified)
    2. logistic_regression (class_weight='balanced', standardized inputs)
    3. random_forest (class_weight='balanced_subsample')
    4. xgboost (scale_pos_weight set from train class ratio)

Because positives (~3%) are rare, model selection is done on
PR-AUC / F1 on the validation set, NOT accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)
from xgboost import XGBClassifier

from src.features.build_features import (
    NUMERIC_FEATURES, encode_categoricals, get_feature_matrix,
)


def time_aware_split(df: pd.DataFrame, train_seasons, val_season, test_season):
    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"] == val_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, val, test


def evaluate(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": cm,
        "threshold": threshold,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def best_threshold_for_f1(y_true, y_prob) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    # precision_recall_curve returns thresholds of len(n-1)
    if len(thr) == 0:
        return 0.5
    best_idx = np.nanargmax(f1s[:-1])
    return float(thr[best_idx])


def run(processed_dir: str, models_dir: str, reports_dir: str, config: dict):
    df = pd.read_parquet(Path(processed_dir) / "pit_model_dataset.parquet")

    train_seasons = config["split"]["train_seasons"]
    val_season = config["split"]["val_season"]
    test_season = config["split"]["test_season"]

    train, val, test = time_aware_split(df, train_seasons, val_season, test_season)

    train_enc, dummy_cols, cats = encode_categoricals(train)
    val_enc, _, _ = encode_categoricals(val, fit_categories=cats)
    test_enc, _, _ = encode_categoricals(test, fit_categories=cats)

    X_train = get_feature_matrix(train_enc, dummy_cols)
    X_val = get_feature_matrix(val_enc, dummy_cols)
    X_test = get_feature_matrix(test_enc, dummy_cols)
    y_train = train_enc["PitNextLap"].values
    y_val = val_enc["PitNextLap"].values
    y_test = test_enc["PitNextLap"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    random_state = config["pit_model"]["random_state"]
    results = {}
    fitted_models = {}

    # 1. baseline
    base = DummyClassifier(strategy="stratified", random_state=random_state)
    base.fit(X_train, y_train)
    p_val = base.predict_proba(X_val)[:, 1]
    results["baseline"] = evaluate(y_val, p_val)
    fitted_models["baseline"] = base

    # 2. logistic regression
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=random_state
    )
    lr.fit(X_train_s, y_train)
    p_val = lr.predict_proba(X_val_s)[:, 1]
    results["logistic_regression"] = evaluate(y_val, p_val)
    fitted_models["logistic_regression"] = lr

    # 3. random forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_val = rf.predict_proba(X_val)[:, 1]
    results["random_forest"] = evaluate(y_val, p_val)
    fitted_models["random_forest"] = rf

    # 4. xgboost
    pos = y_train.sum()
    neg = len(y_train) - pos
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    p_val = xgb.predict_proba(X_val)[:, 1]
    results["xgboost"] = evaluate(y_val, p_val)
    fitted_models["xgboost"] = xgb

    # model selection on validation PR-AUC (imbalanced target)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = fitted_models[best_name]

    # tune decision threshold for F1 on validation, then evaluate on test
    if best_name in ("logistic_regression",):
        p_val_best = best_model.predict_proba(X_val_s)[:, 1]
        p_test_best = best_model.predict_proba(X_test_s)[:, 1]
    else:
        p_val_best = best_model.predict_proba(X_val)[:, 1]
        p_test_best = best_model.predict_proba(X_test)[:, 1]

    thr = best_threshold_for_f1(y_val, p_val_best)
    test_metrics = evaluate(y_test, p_test_best, threshold=thr)

    # persist
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler if best_name == "logistic_regression" else None,
            "dummy_cols": dummy_cols,
            "fit_categories": cats,
            "numeric_features": NUMERIC_FEATURES,
            "decision_threshold": thr,
        },
        Path(models_dir) / "pit_model.joblib",
    )

    report = {
        "validation_metrics_by_model": results,
        "selected_model": best_name,
        "selection_criterion": "highest PR-AUC on validation season 2023",
        "tuned_decision_threshold_from_val_f1": thr,
        "test_metrics_season_2024": test_metrics,
        "train_seasons": train_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
    }
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir, "pit_model_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    rep = run("data/processed", "models", "reports", cfg)
    print(json.dumps(rep, indent=2))
