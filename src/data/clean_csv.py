"""
Clean and standardize the Kaggle-style F1 pit stop CSV
(data/raw/f1_pitstops_2018_2024.csv).

This module NEVER modifies the original file. It reads it, fixes known
data-quality issues, standardizes identifiers (so the table can later be
joined to Jolpica data on season/round/driver), and writes a cleaned
Parquet file to data/processed/.

Known issues in the raw file (discovered by inspection, see
reports/data_quality_report.md for the full audit):

1. Mojibake in some `Circuit` values (double-encoded UTF-8), e.g.
   "AutÃ£Â³dromo Hermanos RodrÃ£Â­guez" instead of
   "Autódromo Hermanos Rodríguez". Fixed with a latin1->utf8 re-decode.
2. The table is at STINT granularity (one row per driver x race x stint),
   not per-lap. `Stint Length` + `Pit_Lap` let us reconstruct the lap on
   which each stint started and ended.
3. Several race-level columns (Position, TotalPitStops, AvgPitStopTime,
   "Lap Time Variation", "Total Pit Stops", "Tire Usage Aggression",
   "Fast Lap Attempts", "Position Changes", "Driver Aggression Score")
   are constant across all stints of a driver-race: they are POST-RACE
   aggregates. They are flagged here as leakage-prone and excluded later
   in feature engineering (see src/features/build_features.py).
4. 9 races (see report) are missing weather/metadata entirely.
5. ~23% of driver-races have `sum(Stint Length) != Laps` by a small
   margin (mostly ±1, a handful of larger cases for drivers with
   incomplete stint records, e.g. mid-season substitute drivers with
   Stint Length == 0). These rows are flagged, not silently dropped.
6. `Pit_Time` mixes numeric pit-stop durations (seconds, as strings) with
   the literal string "Final Stint" for a driver's last stint.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
import ftfy


RENAME_MAP = {
    "Humidity_%": "humidity_pct",
    "Race Name": "race_name",
    "Lap Time Variation": "race_lap_time_variation",
    "Total Pit Stops": "race_total_pit_stops_dup",  # duplicate of TotalPitStops
    "Tire Usage Aggression": "race_tire_usage_aggression",
    "Fast Lap Attempts": "race_fast_lap_attempts",
    "Position Changes": "race_position_changes",
    "Driver Aggression Score": "race_driver_aggression_score",
    "Tire Compound": "tire_compound",
    "Stint Length": "stint_length",
}

# Columns that are constant per (season, round, driver) and represent
# END-OF-RACE aggregates. They are real information but only become known
# after the race is finished, so they must never be used as pre-decision
# features for the pit-stop classifier. Kept in the cleaned table (useful
# for backtesting / reporting) but flagged.
POST_RACE_AGGREGATE_COLS = [
    "Position",
    "TotalPitStops",
    "AvgPitStopTime",
    "race_lap_time_variation",
    "race_total_pit_stops_dup",
    "race_tire_usage_aggression",
    "race_fast_lap_attempts",
    "race_position_changes",
    "race_driver_aggression_score",
]


def _fix_mojibake(text):
    """Repair multiply-encoded UTF-8 text (e.g. triple-mojibake
    'AutÃƒÂ³dromo' -> 'Autódromo'). The raw CSV has been re-encoded
    through latin1 more than once for some accented circuit names, so a
    single latin1<->utf8 round trip is not enough; ftfy detects and
    reverses an arbitrary number of such round trips."""
    if not isinstance(text, str):
        return text
    if "Ã" not in text:
        return text
    return ftfy.fix_text(text)


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw stint-level dataframe. Returns (clean_df, qa_report_dict)."""
    df = df.copy()
    qa = {}

    # --- 1. fix mojibake in text columns ---
    text_cols = ["Circuit", "Race Name", "Location", "Country", "Driver", "Constructor"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(_fix_mojibake)

    # --- 2. rename to snake_case-ish, stable names ---
    df = df.rename(columns=RENAME_MAP)
    df.columns = [c if c in RENAME_MAP.values() else _slugify(c) for c in df.columns]

    # --- 3. standardized identifiers for later joins ---
    df["driver_id"] = df["driver"].map(_slugify)
    df["circuit_id"] = df["circuit"].map(_slugify)
    df["constructor_id"] = df["constructor"].map(_slugify)
    df["race_key"] = (
        df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    )
    df["driver_race_key"] = df["race_key"] + "_" + df["driver_id"]

    # --- 4. Pit_Time -> numeric pit duration + boolean "is final stint" ---
    df["is_final_stint"] = df["pit_time"].astype(str).str.strip().eq("Final Stint")
    df["pit_stop_duration_s"] = pd.to_numeric(
        df["pit_time"].where(~df["is_final_stint"]), errors="coerce"
    )

    # --- 5. dtypes ---
    numeric_cols = [
        "season", "round", "laps", "position", "totalpitstops",
        "avgpitstoptime", "air_temp_c", "track_temp_c", "humidity_pct",
        "wind_speed_kmh", "stint", "stint_length", "pit_lap",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 6. date parsing ---
    if "date" in df.columns:
        df["race_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

    # --- 7. flag rows with missing race metadata (weather etc.) ---
    meta_missing = df["race_name"].isna()
    qa["races_missing_metadata"] = (
        df.loc[meta_missing, ["season", "round", "circuit"]]
        .drop_duplicates()
        .to_dict("records")
    )
    df["metadata_missing"] = meta_missing

    # --- 8. flag driver-races whose stint lengths don't sum to Laps ---
    sums = df.groupby("driver_race_key")["stint_length"].sum()
    laps = df.groupby("driver_race_key")["laps"].first()
    diff = (sums - laps).rename("stint_length_diff")
    bad = diff[diff.abs() > 0]
    qa["n_driver_races"] = int(laps.shape[0])
    qa["n_driver_races_stint_mismatch"] = int((diff.abs() > 0).sum())
    qa["n_driver_races_zero_stint_data"] = int((sums == 0).sum())
    df = df.merge(diff.reset_index(), on="driver_race_key", how="left")
    df["stint_length_consistent"] = df["stint_length_diff"].abs().le(0)

    # driver-races with ZERO recorded stint length are unusable for the
    # lap-expansion step (no stint boundaries at all) - flag explicitly.
    zero_stint_keys = sums[sums == 0].index
    df["has_stint_data"] = ~df["driver_race_key"].isin(zero_stint_keys)

    # --- 9. tire compound cleanup ---
    df["tire_compound"] = df["tire_compound"].replace({"UNKNOWN": np.nan})

    # --- 10. leakage-flag column list, stored for downstream modules ---
    qa["post_race_aggregate_columns_excluded_from_features"] = POST_RACE_AGGREGATE_COLS

    # --- 11. duplicate rows ---
    n_dupes = df.duplicated(
        subset=["season", "round", "driver_id", "stint"]
    ).sum()
    qa["n_duplicate_stint_rows"] = int(n_dupes)
    df = df.drop_duplicates(subset=["season", "round", "driver_id", "stint"])

    qa["n_rows_out"] = int(len(df))
    qa["n_seasons"] = int(df["season"].nunique())
    qa["seasons"] = sorted(df["season"].dropna().unique().tolist())

    return df, qa


def write_qa_report(qa: dict, out_path: str) -> None:
    lines = ["# Data Quality Report - raw CSV cleaning\n"]
    lines.append(f"- Driver-race groups (stint sets): {qa['n_driver_races']}")
    lines.append(
        f"- Driver-races where sum(stint_length) != total race laps: "
        f"{qa['n_driver_races_stint_mismatch']} "
        f"({qa['n_driver_races_stint_mismatch']/qa['n_driver_races']:.1%})"
    )
    lines.append(
        f"- Driver-races with NO usable stint length data at all (sum==0): "
        f"{qa['n_driver_races_zero_stint_data']}"
    )
    lines.append(f"- Duplicate stint rows removed: {qa['n_duplicate_stint_rows']}")
    lines.append(f"- Races missing weather/metadata entirely: "
                  f"{len(qa['races_missing_metadata'])}")
    for r in qa["races_missing_metadata"]:
        lines.append(f"    - {r}")
    lines.append(f"\n- Rows after cleaning: {qa['n_rows_out']}")
    lines.append(f"- Seasons covered: {qa['seasons']}")
    lines.append(
        "\n- Columns excluded from ML features because they are computed "
        "from the FULL race result (post-race aggregates -> leakage if "
        "used to predict an in-race decision):"
    )
    for c in qa["post_race_aggregate_columns_excluded_from_features"]:
        lines.append(f"    - {c}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def run(csv_path: str, processed_dir: str, reports_dir: str) -> pd.DataFrame:
    df_raw = load_raw(csv_path)
    df_clean, qa = clean(df_raw)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / "stints_clean.parquet"
    df_clean.to_parquet(out_path, index=False)
    write_qa_report(qa, Path(reports_dir) / "data_quality_report.md")
    return df_clean


if __name__ == "__main__":
    run(
        "data/raw/f1_pitstops_2018_2024.csv",
        "data/processed",
        "reports",
    )
