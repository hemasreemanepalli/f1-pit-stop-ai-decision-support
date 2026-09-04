import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_target, encode_categoricals, get_feature_matrix,
    assert_no_target_leakage, FEATURE_COLUMNS,
    EXCLUDED_POST_RACE_COLUMNS, EXCLUDED_FUTURE_OR_TARGET_DERIVED_COLUMNS,
)


def _fake_lap_df(n_laps=10):
    rows = []
    for lap in range(1, n_laps + 1):
        rows.append({
            "lap": lap, "laps": n_laps, "is_pit_lap": lap == 5,
            "tyre_age": lap if lap <= 5 else lap - 5,
            "laps_since_last_pit": lap if lap <= 5 else lap - 5,
            "previous_pit_stops": 0 if lap <= 5 else 1,
            "stint": 1 if lap <= 5 else 2,
            "tire_compound": "SOFT" if lap <= 5 else "HARD",
            "circuit_id": "monza", "constructor_id": "ferrari",
            "air_temp_c": 25.0, "track_temp_c": 35.0, "humidity_pct": 40.0,
            "wind_speed_kmh": 8.0,
        })
    df = pd.DataFrame(rows)
    df["race_progress"] = df["lap"] / df["laps"]
    df["laps_remaining"] = df["laps"] - df["lap"]
    return df


def test_build_target_drops_final_lap():
    df = _fake_lap_df(10)
    out = build_target(df)
    assert out["lap"].max() == 9  # last lap dropped (no "next lap" exists)
    # is_pit_lap is defined AT the pit lap itself (lap 5) -> PitNextLap==1 there
    assert out.loc[out["lap"] == 5, "PitNextLap"].iloc[0] == 1
    assert out.loc[out["lap"] == 4, "PitNextLap"].iloc[0] == 0


def test_no_excluded_columns_in_feature_list():
    for c in EXCLUDED_POST_RACE_COLUMNS + EXCLUDED_FUTURE_OR_TARGET_DERIVED_COLUMNS:
        assert c not in FEATURE_COLUMNS


def test_encode_categoricals_consistent_across_splits():
    df = _fake_lap_df(10)
    df = build_target(df)
    train = df.iloc[:5]
    test = df.iloc[5:]
    train_enc, dummy_cols, cats = encode_categoricals(train)
    test_enc, dummy_cols2, _ = encode_categoricals(test, fit_categories=cats)
    assert dummy_cols == dummy_cols2
    X_train = get_feature_matrix(train_enc, dummy_cols)
    X_test = get_feature_matrix(test_enc, dummy_cols)
    assert list(X_train.columns) == list(X_test.columns)


def test_leakage_check_raises_on_obvious_leak():
    df = _fake_lap_df(20)
    df = build_target(df)
    df["leaky_feature"] = df["PitNextLap"]  # perfectly correlated with target
    with pytest.raises(RuntimeError):
        assert_no_target_leakage(df, ["leaky_feature"])


def test_leakage_check_passes_on_real_features():
    df = _fake_lap_df(20)
    df = build_target(df)
    from src.features.build_features import NUMERIC_FEATURES
    # should not raise
    assert_no_target_leakage(df, NUMERIC_FEATURES)
