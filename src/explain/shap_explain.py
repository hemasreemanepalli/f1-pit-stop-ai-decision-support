"""
SHAP explainability for the pit-stop prediction model.

Provides:
  - global feature importance (mean |SHAP value| across a sample of the
    test set)
  - a single-row explanation for one race situation (waterfall-style
    values)
  - saved summary figure (reports/figures/shap_summary.png)

Uses TreeExplainer when the selected model is tree-based (random_forest /
xgboost); for logistic_regression falls back to shap.LinearExplainer, and
for the DummyClassifier baseline SHAP is not meaningful and is skipped
(documented, not silently faked).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.build_features import NUMERIC_FEATURES, encode_categoricals, get_feature_matrix


def _build_explainer(bundle: dict, X_background: pd.DataFrame):
    model = bundle["model"]
    name = bundle["model_name"]
    if name in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model), "tree"
    elif name == "logistic_regression":
        return shap.LinearExplainer(model, X_background), "linear"
    else:
        return None, "unsupported"


def compute_global_importance(models_dir: str, processed_dir: str, reports_dir: str, figures_dir: str, sample_size: int = 2000):
    bundle = joblib.load(Path(models_dir) / "pit_model.joblib")
    df = pd.read_parquet(Path(processed_dir) / "pit_model_dataset.parquet")
    test = df[df["season"] == 2024].copy()

    enc, _, _ = encode_categoricals(test, fit_categories=bundle["fit_categories"])
    X = get_feature_matrix(enc, bundle["dummy_cols"])

    if bundle.get("scaler") is not None:
        X_model = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)
    else:
        X_model = X

    if len(X_model) > sample_size:
        X_sample = X_model.sample(sample_size, random_state=42)
    else:
        X_sample = X_model

    explainer, kind = _build_explainer(bundle, X_sample)
    if explainer is None:
        report = {"note": f"SHAP not computed: model type '{bundle['model_name']}' unsupported."}
        Path(reports_dir, "shap_report.json").write_text(json.dumps(report, indent=2))
        return report

    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]  # positive class for some tree explainer outputs
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=X_sample.columns)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15)[::-1]
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance - PitNextLap model")
    fig.tight_layout()
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / "shap_summary.png", dpi=120)
    plt.close(fig)

    report = {
        "model_used": bundle["model_name"],
        "explainer_type": kind,
        "sample_size": int(len(X_sample)),
        "top_15_features": importance.head(15).round(5).to_dict(),
    }
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir, "shap_report.json").write_text(json.dumps(report, indent=2))

    # persist explainer + background for on-demand per-row explanations in the dashboard
    joblib.dump({"explainer": explainer, "kind": kind, "feature_names": list(X_sample.columns)},
                Path(models_dir) / "shap_explainer.joblib")
    return report


def explain_row(models_dir: str, row_features: pd.DataFrame) -> dict:
    """row_features: single-row DataFrame already in raw feature form
    (NUMERIC_FEATURES + categorical columns), matching what
    build_features produces. Returns per-feature SHAP contributions for
    that one prediction."""
    bundle = joblib.load(Path(models_dir) / "pit_model.joblib")
    exp_bundle = joblib.load(Path(models_dir) / "shap_explainer.joblib")

    enc, _, _ = encode_categoricals(row_features, fit_categories=bundle["fit_categories"])
    X = get_feature_matrix(enc, bundle["dummy_cols"])
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)

    explainer = exp_bundle["explainer"]
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = sv[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else float(np.atleast_1d(base_value)[0])

    contributions = (
        pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False)
    )
    return {
        "base_value": float(base_value),
        "prediction_proba": float(bundle["model"].predict_proba(X)[0, 1]),
        "top_contributions": contributions.head(10).round(5).to_dict(),
    }


if __name__ == "__main__":
    rep = compute_global_importance("models", "data/processed", "reports", "reports/figures")
    print(json.dumps(rep, indent=2))
