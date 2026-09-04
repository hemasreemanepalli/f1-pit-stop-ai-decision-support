"""
Build the driver x race x lap dataset.

Two possible modes:

1. JOLPICA-ENRICHED (used automatically if cached Jolpica laps/pitstops
   JSON files are present in data/raw/jolpica_cache/, i.e. the user ran
   `python run_pipeline.py --download` on a machine with internet
   access): real per-lap timing and real pit-stop laps from Jolpica are
   joined onto the CSV's race/driver/weather columns after standardizing
   identifiers on both sides (season, round, driver_id built the same
   way as in clean_csv.py). Join quality (match rate) is measured and
   reported, not assumed.

2. CSV-ONLY / STINT-EXPANSION (the mode actually exercised in this
   project, since this sandbox cannot reach the Jolpica API - see
   src/data/jolpica_client.py docstring). The Kaggle CSV is stint-level,
   not lap-level: each row says "this driver ran a stint of N laps on
   compound X". We reconstruct lap-level rows by expanding each stint
   into one row per lap, using the (validated) cumulative stint lengths
   to compute the first/last lap number of each stint. This is a
   deterministic transformation of REAL fields already in the CSV
   (`stint`, `stint_length`, `pit_lap`) - it does not fabricate any new
   information. What it cannot recover is a genuine per-lap lap time
   (the CSV has none), so `lap_time_s` is left as NaN in this mode and
   any model relying on it degrades gracefully / documents the gap (see
   README "Limitations").

Driver-races flagged `has_stint_data == False` in cleaning (109 cases,
mostly single-race substitute drivers with no recorded stints) are
dropped from the lap grid - there is nothing to expand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _expand_driver_race(group: pd.DataFrame) -> pd.DataFrame:
    """Given all stint rows for one driver-race (sorted by stint number),
    return one row per lap of that race.

    Boundary logic (in priority order):
    1. `pit_lap` on stint i is the RECORDED first lap of stint i+1 (verified
       by inspection: e.g. stint1 stint_length=25, pit_lap=26 => stint2
       starts at lap 26). Where present, this is ground truth and is used
       directly for the next stint's start lap - it is more reliable than
       cumulative stint_length, which mismatches the recorded race-lap
       count in ~23% of driver-races (see data_quality_report.md).
    2. If `pit_lap` is missing for a non-final stint (rare), fall back to
       cumulative stint_length.
    3. The last stint always runs to the recorded total race `laps`.
    """
    group = group.sort_values("stint")
    total_laps = int(group["laps"].iloc[0])
    rows = []
    lap_cursor = 1
    n_stints = len(group)
    stint_rows = list(group.iterrows())
    for i, (_, stint_row) in enumerate(stint_rows):
        is_last = i == n_stints - 1
        stint_start = lap_cursor

        if is_last:
            stint_end = total_laps
        else:
            recorded_pit_lap = stint_row["pit_lap"]
            if pd.notna(recorded_pit_lap) and recorded_pit_lap > stint_start:
                stint_end = int(recorded_pit_lap) - 1
            else:
                stint_len = stint_row["stint_length"]
                if pd.isna(stint_len) or stint_len <= 0:
                    continue
                stint_end = stint_start + int(stint_len) - 1

        if stint_end < stint_start:
            continue

        for lap in range(stint_start, stint_end + 1):
            rows.append(
                {
                    "lap": lap,
                    "stint": stint_row["stint"],
                    "tire_compound": stint_row["tire_compound"],
                    "stint_start_lap": stint_start,
                    "stint_planned_length": stint_end - stint_start + 1,
                    "is_pit_lap": (lap == stint_end) and not is_last,
                }
            )
        lap_cursor = stint_end + 1
    return pd.DataFrame(rows)


META_COLS = [
    "season", "round", "race_key", "driver", "driver_id", "constructor",
    "constructor_id", "circuit", "circuit_id", "race_name", "race_date",
    "location", "country", "air_temp_c", "track_temp_c", "humidity_pct",
    "wind_speed_kmh", "laps", "metadata_missing",
]


def expand_stints_to_laps(stints_df: pd.DataFrame) -> pd.DataFrame:
    usable = stints_df[stints_df["has_stint_data"]].copy()
    dropped = stints_df.loc[~stints_df["has_stint_data"], "driver_race_key"].nunique()

    lap_frames = []
    for key, group in usable.groupby("driver_race_key"):
        laps = _expand_driver_race(group)
        if laps.empty:
            continue
        meta = group.iloc[0][META_COLS]
        for c, v in meta.items():
            laps[c] = v
        laps["driver_race_key"] = key
        lap_frames.append(laps)

    if not lap_frames:
        lap_df = pd.DataFrame(columns=[
            "lap", "stint", "tire_compound", "stint_start_lap",
            "stint_planned_length", "is_pit_lap", "driver_race_key",
            *META_COLS,
        ])
    else:
        lap_df = pd.concat(lap_frames, ignore_index=True)

    # laps_remaining, race_progress: only use CURRENT/PAST info -> safe
    lap_df["laps_remaining"] = lap_df["laps"] - lap_df["lap"]
    lap_df["race_progress"] = lap_df["lap"] / lap_df["laps"]

    # tyre age = laps completed on current tyre set, known at decision time
    lap_df["tyre_age"] = lap_df["lap"] - lap_df["stint_start_lap"] + 1

    # laps since previous pit (== tyre age unless mid-stint pit anomalies)
    lap_df["laps_since_last_pit"] = lap_df["tyre_age"]

    # previous pit stops so far this race (stint number - 1), known at
    # decision time (does not use future stints)
    lap_df["previous_pit_stops"] = lap_df["stint"] - 1

    lap_df = lap_df.sort_values(["driver_race_key", "lap"]).reset_index(drop=True)
    return lap_df, dropped


def try_enrich_with_jolpica(lap_df: pd.DataFrame, cache_dir: str) -> tuple[pd.DataFrame, dict]:
    """Join Jolpica lap times using race-results driver-name mapping.

    The CSV driver_id is a full-name slug (e.g. lewis_hamilton), while
    Jolpica lap timings use short IDs (e.g. hamilton). Race results contain
    the given/family name needed to bridge the two identifiers.
    """
    from src.data.jolpica_client import parse_laps_response, parse_results_driver_mapping
    import json

    cache_path = Path(cache_dir)
    lap_files = sorted(cache_path.glob("laps_*.json")) if cache_path.exists() else []
    result_files = sorted(cache_path.glob("results_*.json")) if cache_path.exists() else []
    report = {"jolpica_cache_found": bool(lap_files), "results_cache_found": bool(result_files), "races_enriched": 0}

    if not lap_files:
        report["note"] = "No cached Jolpica lap data found. Run `python run_pipeline.py --download`."
        return lap_df, report
    if not result_files:
        report["note"] = ("Jolpica lap cache exists, but results cache is missing. Run `python run_pipeline.py "
                           "--download` again; results are required to bridge Jolpica driver IDs to the CSV full-name IDs.")
        return lap_df, report

    all_laps, all_results = [], []
    for f in lap_files:
        parsed = parse_laps_response(json.loads(f.read_text()))
        if not parsed.empty: all_laps.append(parsed)
    for f in result_files:
        parsed = parse_results_driver_mapping(json.loads(f.read_text()))
        if not parsed.empty: all_results.append(parsed)

    jolpica_laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    driver_map = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if jolpica_laps.empty:
        report["note"] = "Jolpica cache present but contained no lap rows."; return lap_df, report
    if driver_map.empty:
        report["note"] = "Results cache present but contained no driver mappings."; return lap_df, report

    for df in (jolpica_laps, driver_map):
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    driver_map = driver_map.dropna(subset=["season","round","driver_id","driver_name_slug"]).drop_duplicates(["season","round","driver_id"])
    jolpica_laps = jolpica_laps.merge(
        driver_map,
        on=["season", "round", "driver_id"],
        how="left",
        validate="many_to_one",
    )

    # Protect the downstream one-to-one join from duplicate timing records
    # at the same season/round/lap/driver grain. These can occur at pagination
    # boundaries in cached Jolpica responses. Keep one timing record per key.
    jolpica_laps = (
        jolpica_laps
        .sort_values(["season", "round", "lap", "driver_name_slug"])
        .drop_duplicates(
            subset=["season", "round", "lap", "driver_name_slug"],
            keep="first",
        )
    )

    lap_df = lap_df.copy()
    lap_df["season"] = pd.to_numeric(lap_df["season"], errors="coerce").astype("Int64")
    lap_df["round"] = pd.to_numeric(lap_df["round"], errors="coerce").astype("Int64")
    merged = lap_df.merge(
        jolpica_laps[["season","round","lap","driver_name_slug","lap_time_s"]],
        left_on=["season","round","lap","driver_id"],
        right_on=["season","round","lap","driver_name_slug"],
        how="left", validate="one_to_one",
    ).drop(columns=["driver_name_slug"])
    report["lap_time_match_rate"] = float(merged["lap_time_s"].notna().mean())
    report["races_enriched"] = int(jolpica_laps[["season","round"]].drop_duplicates().shape[0])
    report["lap_rows_input"] = int(len(merged))
    report["lap_rows_joined"] = int(merged["lap_time_s"].notna().sum())
    report["driver_mapping_rows"] = int(len(driver_map))
    report["driver_mapping_match_rate"] = float(jolpica_laps["driver_name_slug"].notna().mean())
    return merged, report


def run(processed_dir: str, jolpica_cache_dir: str, reports_dir: str) -> pd.DataFrame:
    stints_path = Path(processed_dir) / "stints_clean.parquet"
    stints_df = pd.read_parquet(stints_path)

    lap_df, dropped_driver_races = expand_stints_to_laps(stints_df)
    lap_df, jolpica_report = try_enrich_with_jolpica(lap_df, jolpica_cache_dir)

    if "lap_time_s" not in lap_df.columns:
        lap_df["lap_time_s"] = np.nan

    out_path = Path(processed_dir) / "laps_dataset.parquet"
    lap_df.to_parquet(out_path, index=False)

    report_lines = [
        "# Lap-level dataset build report\n",
        f"- Driver-races expanded to laps: "
        f"{lap_df['driver_race_key'].nunique()}",
        f"- Driver-races dropped (no usable stint data): {dropped_driver_races}",
        f"- Total lap-level rows: {len(lap_df)}",
        f"- Jolpica cache found: {jolpica_report['jolpica_cache_found']}",
        f"- Jolpica results cache found: {jolpica_report.get('results_cache_found', False)}",
    ]
    if jolpica_report.get("note"):
        report_lines.append(f"- {jolpica_report['note']}")
    if "lap_time_match_rate" in jolpica_report:
        report_lines.append(
            f"- Jolpica lap-time join match rate: "
            f"{jolpica_report['lap_time_match_rate']:.1%}"
        )
        report_lines.append(
            f"- Jolpica lap rows joined: {jolpica_report.get('lap_rows_joined', 0):,} "
            f"/ {jolpica_report.get('lap_rows_input', 0):,}"
        )
        report_lines.append(
            f"- Jolpica driver-ID bridge match rate: "
            f"{jolpica_report.get('driver_mapping_match_rate', 0):.1%}"
        )
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir, "lap_dataset_report.md").write_text("\n".join(report_lines))

    return lap_df


if __name__ == "__main__":
    run("data/processed", "data/raw/jolpica_cache", "reports")
