"""
Backtest the strategy optimizer on races NOT used for training
(test season 2024, per the time-aware split).

Methodology
-----------
For each driver-race in the test season:
  1. At the driver's actual FIRST pit decision point, take the race state
     as of a few laps before the historical pit lap (using only
     information available up to that lap - the classifier/optimizer are
     never shown the actual outcome).
  2. Ask the optimizer for its recommended pit lap.
  3. Compare the recommended lap to the historical (actual) pit lap:
       lap_difference = recommended_lap - historical_lap
  4. Record whether the historical lap fell inside the optimizer's
     recommended window.

This is compared explicitly labeled as SIMULATED vs ACTUAL: the
"historical pit lap" is the real, recorded decision from the CSV; the
"recommended pit lap" is the model's simulated output, run on data the
model was never trained on (2024). No outcome (finishing position, race
time) is estimated here beyond the documented pit-time-loss proxy,
because no lap-time model is available to estimate on-track time deltas
(see train_degradation_model.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.optimization.strategy_optimizer import StrategyOptimizer, RaceState


def run(processed_dir: str, models_dir: str, reports_dir: str, config: dict, n_races: int | None = None):
    stints = pd.read_parquet(Path(processed_dir) / "stints_clean.parquet")
    laps = pd.read_parquet(Path(processed_dir) / "laps_dataset.parquet")

    test_season = config["split"]["test_season"]
    train_seasons = config["split"]["train_seasons"]
    train_stints = stints[stints["season"].isin(train_seasons)]

    test_stints = stints[
        (stints["season"] == test_season)
        & (stints["stint"] == 1)
        & (stints["has_stint_data"])
        & (stints["is_final_stint"] == False)  # noqa: E712  - driver actually pitted
    ].copy()

    pit_bundle = joblib.load(Path(models_dir) / "pit_model.joblib")
    deg_bundle = joblib.load(Path(models_dir) / "degradation_model.joblib")
    optimizer = StrategyOptimizer(pit_bundle, deg_bundle, train_stints, config)

    results = []
    rows = test_stints.itertuples()
    count = 0
    for r in rows:
        print(f"BACKTESTING {r}", flush=True)
        if n_races is not None and count >= n_races:
            break
        historical_pit_lap = r.pit_lap
        if pd.isna(historical_pit_lap) or historical_pit_lap < 4:
            continue
        decision_lap = int(historical_pit_lap) - 3  # 3 laps before the real stop
        if decision_lap < 1:
            continue

        race_laps = laps[
            (laps["season"] == r.season) & (laps["round"] == r.round)
            & (laps["driver_id"] == r.driver_id) & (laps["lap"] == decision_lap)
        ]
        if race_laps.empty:
            continue
        row = race_laps.iloc[0]

        state = RaceState(
            season=int(r.season), round=int(r.round), driver_id=r.driver_id,
            lap=decision_lap, laps=int(row["laps"]), stint=int(row["stint"]),
            tire_compound=row["tire_compound"], stint_start_lap=int(row["stint_start_lap"]),
            circuit_id=row["circuit_id"],
            air_temp_c=row.get("air_temp_c", np.nan), track_temp_c=row.get("track_temp_c", np.nan),
            humidity_pct=row.get("humidity_pct", np.nan), wind_speed_kmh=row.get("wind_speed_kmh", np.nan),
            previous_pit_stops=int(row["previous_pit_stops"]),
        )
        try:
            rec = optimizer.recommend(state, lookahead_laps=15)
        except Exception as e:
            print(
                f"BACKTEST ERROR | season={r.season} round={r.round} "
                f"driver={r.driver_id} decision_lap={decision_lap}: {e}"
            )
            continue

        lap_diff = rec["recommended_pit_lap"] - int(historical_pit_lap)
        in_window = rec["recommended_pit_window"][0] <= historical_pit_lap <= rec["recommended_pit_window"][1]
        results.append({
            "season": int(r.season), "round": int(r.round), "driver_id": r.driver_id,
            "decision_lap": decision_lap,
            "historical_pit_lap_ACTUAL": int(historical_pit_lap),
            "recommended_pit_lap_SIMULATED": rec["recommended_pit_lap"],
            "recommended_window_SIMULATED": rec["recommended_pit_window"],
            "lap_difference": lap_diff,
            "historical_lap_in_recommended_window": bool(in_window),
        })
        count += 1

    res_df = pd.DataFrame(results)
    summary = {
        "n_backtested_driver_races": int(len(res_df)),
        "mean_abs_lap_difference": float(res_df["lap_difference"].abs().mean()) if len(res_df) else None,
        "median_abs_lap_difference": float(res_df["lap_difference"].abs().median()) if len(res_df) else None,
        "pct_historical_lap_in_recommended_window": float(res_df["historical_lap_in_recommended_window"].mean()) if len(res_df) else None,
        "note": (
            "historical_pit_lap_ACTUAL is the real recorded pit lap from the "
            "CSV. recommended_pit_lap_SIMULATED is the optimizer's output, "
            "computed using only information available up to 3 laps before "
            "that real stop, on the 2024 test season the models were never "
            "trained/validated on."
        ),
    }
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    res_df.to_csv(Path(reports_dir) / "backtest_results.csv", index=False)
    Path(reports_dir, "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, res_df


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    summary, df = run("data/processed", "models", "reports", cfg)
    print(json.dumps(summary, indent=2))
