import json

import pandas as pd
import pytest


def test_backtest_report_exists_and_labeled(project_root):
    path = project_root / "reports" / "backtest_summary.json"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    summary = json.loads(path.read_text())
    assert summary["n_backtested_driver_races"] > 0
    assert "ACTUAL" in summary["note"]
    assert "SIMULATED" in summary["note"]


def test_backtest_csv_distinguishes_actual_vs_simulated(project_root):
    path = project_root / "reports" / "backtest_results.csv"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    df = pd.read_csv(path)
    assert "historical_pit_lap_ACTUAL" in df.columns
    assert "recommended_pit_lap_SIMULATED" in df.columns
    assert len(df) > 0


def test_shap_report_exists(project_root):
    path = project_root / "reports" / "shap_report.json"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    report = json.loads(path.read_text())
    assert "top_15_features" in report
    assert len(report["top_15_features"]) > 0


def test_shap_figure_exists(project_root):
    path = project_root / "reports" / "figures" / "shap_summary.png"
    if not path.exists():
        pytest.skip("Run `python run_pipeline.py` first to produce reports.")
    assert path.stat().st_size > 0
