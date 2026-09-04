"""
Feature engineering for the pit-stop decision model.

Target
------
PitNextLap = 1 if the driver pits at the end of the CURRENT lap (i.e. the
next lap is on a fresh tyre / new stint), else 0. Built directly from
`is_pit_lap` in the lap dataset, which was derived from the CSV's real
`pit_lap` field (see build_lap_dataset.py) - not fabricated.

The row for the actual final lap of the race is dropped for the
classification task (there is no "next lap" to pit on, and the CSV
naturally never marks the last lap as a pit lap anyway, which would
otherwise inject a structural "always False" pattern for `lap==laps` that
the leakage checks below would rightly flag).

Leakage prevention
-------------------
A feature is only allowed into `FEATURE_COLUMNS` if its value at row
(driver, race, lap=L) is knowable using information from laps 1..L of
THAT race only, plus static pre-race information (weather snapshot,
circuit, compound in use). The following categories are explicitly
EXCLUDED, with the reason:

  - Anything describing lap > L of the same race (future lap times,
    future positions, future stint/compound, any post-pit information).
  - Race-final aggregates that are constant across the whole driver-race
    in the raw CSV (Position, TotalPitStops, AvgPitStopTime, "Lap Time
    Variation", "Total Pit Stops", "Tire Usage Aggression", "Fast Lap
    Attempts", "Position Changes", "Driver Aggression Score") - these are
    computed from the complete race and are unavailable at decision time.
  - `stint_planned_length` / the eventual stint_end lap - this literally
    encodes the pit lap itself for the current stint and would leak the
    target by construction. Only `tyre_age` / `laps_since_last_pit`
    (laps completed SO FAR on this tyre set) are used, not the eventual
    total.
  - Next stint's tyre compound (only the CURRENT compound is used).
  - `lap_time_s`: real per-lap lap times are only available if a live
    Jolpica download was run (see jolpica_client.py). When absent
    (CSV-only mode, the mode actually used to produce the results in
    this project), lap-time-based features are NaN for every row and are
    therefore not usable by the trained model - documented in the model
    report rather than silently imputed with a fabricated value.

A programmatic check (`assert_no_target_leakage`) verifies that none of
the engineered feature columns are perfectly / near-perfectly correlated
with the target in a way that would indicate the target's own
information leaked back in (e.g. AUC-per-feature sanity check on the
training split only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED_POST_RACE_COLUMNS = [
    "position", "totalpitstops", "avgpitstoptime",
    "race_lap_time_variation", "race_total_pit_stops_dup",
    "race_tire_usage_aggression", "race_fast_lap_attempts",
    "race_position_changes", "race_driver_aggression_score",
]

EXCLUDED_FUTURE_OR_TARGET_DERIVED_COLUMNS = [
    "stint_planned_length",  # encodes eventual pit lap for current stint
    "is_pit_lap",            # this IS the raw signal the target is built from
    "lap_time_s",            # only ever populated for the CURRENT lap, not
                              # future ones, but excluded from the trained
                              # feature set in CSV-only mode because it is
                              # ~100% missing (see module docstring) and we
                              # do not want a model silently trained on a
                              # feature that is NaN for real-world use.
]

CATEGORICAL_FEATURES = ["tire_compound", "circuit_id", "constructor_id"]

NUMERIC_FEATURES = [
    "lap",                  # current lap number
    "laps",                 # total race laps (known pre-race)
    "race_progress",        # lap / laps
    "laps_remaining",
    "tyre_age",              # laps completed on current tyre set so far
    "laps_since_last_pit",
    "previous_pit_stops",    # stints completed so far this race
    "stint",                 # current stint number
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "wind_speed_kmh",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_target(lap_df: pd.DataFrame) -> pd.DataFrame:
    df = lap_df.copy()
    df["PitNextLap"] = df["is_pit_lap"].astype(int)
    # drop the very last lap of each race: no "next lap" exists
    df = df[df["lap"] < df["laps"]].copy()
    return df


def encode_categoricals(df: pd.DataFrame, fit_categories: dict | None = None):
    """One-hot encode CATEGORICAL_FEATURES. If `fit_categories` is given
    (dict col -> list of known categories from the TRAIN split), apply the
    same encoding to a different split so train/val/test share columns."""
    df = df.copy()
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("string").fillna("missing")

    if fit_categories is None:
        fit_categories = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES}

    for c in CATEGORICAL_FEATURES:
        df[c] = pd.Categorical(df[c], categories=fit_categories[c])

    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    df = pd.concat([df, dummies], axis=1)
    return df, dummies.columns.tolist(), fit_categories


def get_feature_matrix(df: pd.DataFrame, dummy_cols: list[str]) -> pd.DataFrame:
    cols = NUMERIC_FEATURES + dummy_cols
    X = df[cols].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X


def assert_no_target_leakage(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Single-feature AUC sanity check on numeric features: if any one
    feature alone predicts the target with AUC > 0.97, treat it as a
    likely leakage signal and raise, rather than silently training on it."""
    from sklearn.metrics import roc_auc_score

    suspects = {}
    y = df["PitNextLap"].values
    for c in feature_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().all() or x.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y, x.fillna(x.median()))
            auc = max(auc, 1 - auc)
        except ValueError:
            continue
        if auc > 0.97:
            suspects[c] = float(auc)
    if suspects:
        raise RuntimeError(f"Potential leakage detected (single-feature AUC > 0.97): {suspects}")
    return suspects


def run(processed_dir: str) -> pd.DataFrame:
    lap_df = pd.read_parquet(Path(processed_dir) / "laps_dataset.parquet")
    df = build_target(lap_df)

    # sanity check on the excluded columns list vs what's actually present
    for c in EXCLUDED_POST_RACE_COLUMNS + EXCLUDED_FUTURE_OR_TARGET_DERIVED_COLUMNS:
        assert c not in FEATURE_COLUMNS, f"leakage-prone column {c} must not be in FEATURE_COLUMNS"

    assert_no_target_leakage(df, NUMERIC_FEATURES)

    out_path = Path(processed_dir) / "pit_model_dataset.parquet"
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    df = run("data/processed")
    print(f"pit-model dataset: {df.shape}, PitNextLap rate = {df['PitNextLap'].mean():.4f}")
