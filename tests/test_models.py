import json
from pathlib import Path

import joblib
import pandas as pd
import pytest


def test_pit_model_artifact_exists_and_loads(project_root):
    path = project_root / "models" / "pit_model.joblib"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce model artifacts.")
    bundle = joblib.load(path)
    assert "model" in bundle
    assert "model_name" in bundle
    assert "dummy_cols" in bundle
    assert "decision_threshold" in bundle


def test_pit_model_report_uses_time_aware_split(project_root):
    path = project_root / "reports" / "pit_model_report.json"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    report = json.loads(path.read_text())
    assert report["train_seasons"] == [2018, 2019, 2020, 2021, 2022]
    assert report["val_season"] == 2023
    assert report["test_season"] == 2024
    # models must be evaluated with more than accuracy
    for m, metrics in report["validation_metrics_by_model"].items():
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics


def test_pit_model_beats_baseline_on_pr_auc(project_root):
    path = project_root / "reports" / "pit_model_report.json"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    report = json.loads(path.read_text())
    metrics = report["validation_metrics_by_model"]
    baseline_pr_auc = metrics["baseline"]["pr_auc"]
    best_pr_auc = metrics[report["selected_model"]]["pr_auc"]
    assert best_pr_auc > baseline_pr_auc, (
        "Selected model should beat the stratified-random baseline on PR-AUC."
    )


def test_degradation_model_artifact_exists(project_root):
    path = project_root / "models" / "degradation_model.joblib"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce model artifacts.")
    bundle = joblib.load(path)
    assert "model" in bundle
    assert bundle["model_name"] in ("linear_regression", "random_forest", "xgboost")


def test_degradation_report_labeled_as_proxy(project_root):
    path = project_root / "reports" / "degradation_model_report.json"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    report = json.loads(path.read_text())
    assert "PROXY" in report["note"]
    for split in ("mae", "rmse", "r2"):
        assert split in report["test_metrics_season_2024"]
