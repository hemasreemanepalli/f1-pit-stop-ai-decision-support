"""Pit-stop strategy optimizer using the real Jolpica lap-time model.

If a trained lap-time model is present in degradation_model.joblib, candidate
pit laps are evaluated by simulating predicted remaining race time before and
after the stop.  The model predicts lap time from tyre age, compound, circuit,
race progress, driver/constructor and weather.

The previous stint-length proxy remains a fallback for older model bundles.
This is still a decision-support proxy: tyre allocation rules, safety cars,
rival strategies and future weather changes are not simulated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RaceState:
    season: int
    round: int
    driver_id: str
    lap: int
    laps: int
    stint: int
    tire_compound: str
    stint_start_lap: int
    circuit_id: str
    air_temp_c: float
    track_temp_c: float
    humidity_pct: float
    wind_speed_kmh: float
    previous_pit_stops: int
    constructor_id: str = "missing"


def historical_pit_time_loss(train_stints_df: pd.DataFrame, circuit_id: str, default_s: float) -> float:
    sub = train_stints_df[
        (train_stints_df["circuit_id"] == circuit_id)
        & train_stints_df["pit_stop_duration_s"].notna()
    ]
    if len(sub) >= 5:
        return float(sub["pit_stop_duration_s"].median())
    overall = train_stints_df["pit_stop_duration_s"].dropna()
    return float(overall.median()) if len(overall) else default_s


class StrategyOptimizer:
    def __init__(self, pit_model_bundle: dict, degradation_model_bundle: dict, train_stints_df: pd.DataFrame, config: dict):
        self.pit_bundle = pit_model_bundle
        self.deg_bundle = degradation_model_bundle
        self.train_stints_df = train_stints_df
        self.config = config

    def pit_probability_curve(self, state: RaceState, lookahead_laps: int = 15) -> pd.DataFrame:
        from src.features.build_features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        if state.lap >= state.laps:
            raise ValueError("No future pit decision exists at/past the final lap.")
        rows = []
        max_lap = min(state.laps - 1, state.lap + lookahead_laps)
        for lap in range(state.lap, max_lap + 1):
            tyre_age = state.tyre_age if hasattr(state, "tyre_age") else (state.lap - state.stint_start_lap + 1)
            tyre_age += lap - state.lap
            rows.append({
                "lap": lap, "laps": state.laps, "race_progress": lap / state.laps,
                "laps_remaining": state.laps - lap, "tyre_age": tyre_age,
                "laps_since_last_pit": tyre_age, "previous_pit_stops": state.previous_pit_stops,
                "stint": state.stint, "air_temp_c": state.air_temp_c,
                "track_temp_c": state.track_temp_c, "humidity_pct": state.humidity_pct,
                "wind_speed_kmh": state.wind_speed_kmh, "tire_compound": state.tire_compound,
                "circuit_id": state.circuit_id, "constructor_id": state.constructor_id,
            })
        cand = pd.DataFrame(rows)
        cats = self.pit_bundle["fit_categories"]
        for c in CATEGORICAL_FEATURES:
            cand[c] = pd.Categorical(cand[c].astype("string"), categories=cats[c])
        dummies = pd.get_dummies(cand[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
        for col in self.pit_bundle["dummy_cols"]:
            if col not in dummies.columns:
                dummies[col] = False
        dummies = dummies[self.pit_bundle["dummy_cols"]]
        X = pd.concat([cand[NUMERIC_FEATURES], dummies], axis=1)
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        model = self.pit_bundle["model"]
        Xp = self.pit_bundle["scaler"].transform(X) if self.pit_bundle.get("scaler") is not None else X
        cand["pit_probability"] = model.predict_proba(Xp)[:, 1]
        return cand[["lap", "tyre_age", "pit_probability"]]

    def _lap_time_features(self, rows: pd.DataFrame):
        cats = self.deg_bundle["lap_time_fit_categories"]
        for c in self.deg_bundle["lap_time_categorical_features"]:
            rows[c] = pd.Categorical(rows[c].astype("string"), categories=cats[c])
        dummies = pd.get_dummies(rows[self.deg_bundle["lap_time_categorical_features"]], prefix=self.deg_bundle["lap_time_categorical_features"])
        for col in self.deg_bundle["lap_time_dummy_cols"]:
            if col not in dummies.columns:
                dummies[col] = False
        dummies = dummies[self.deg_bundle["lap_time_dummy_cols"]]
        X = pd.concat([rows[self.deg_bundle["lap_time_numeric_features"]], dummies], axis=1)
        nums = self.deg_bundle["lap_time_numeric_features"]
        X[nums] = X[nums].apply(pd.to_numeric, errors="coerce")
        med = self.deg_bundle.get("lap_time_numeric_medians", {})
        X[nums] = X[nums].fillna(pd.Series(med))
        return X

    def _predict_lap_times(self, state: RaceState, laps: list[int], tyre_ages: list[int], compounds: list[str]) -> np.ndarray:
        rows = []
        for lap, age, compound in zip(laps, tyre_ages, compounds):
            rows.append({
                "lap": lap, "laps": state.laps, "race_progress": lap / state.laps,
                "laps_remaining": state.laps - lap, "tyre_age": age,
                "laps_since_last_pit": age, "previous_pit_stops": state.previous_pit_stops,
                "stint": state.stint, "air_temp_c": state.air_temp_c,
                "track_temp_c": state.track_temp_c, "humidity_pct": state.humidity_pct,
                "wind_speed_kmh": state.wind_speed_kmh, "tire_compound": compound,
                "circuit_id": state.circuit_id, "driver_id": state.driver_id,
                "constructor_id": state.constructor_id,
            })
        X = self._lap_time_features(pd.DataFrame(rows))
        pred = self.deg_bundle["lap_time_model"].predict(X)
        return np.maximum(np.asarray(pred, dtype=float), 1.0)

    def expected_stint_length(self, state: RaceState) -> float:
        """Backward-compatible proxy value for the existing dashboard."""
        from src.models.train_degradation_model import PROXY_NUMERIC_FEATURES, PROXY_CATEGORICAL_FEATURES
        row = pd.DataFrame([{
            "stint": state.stint, "stint_start_lap": state.stint_start_lap, "laps": state.laps,
            "air_temp_c": state.air_temp_c, "track_temp_c": state.track_temp_c,
            "humidity_pct": state.humidity_pct, "wind_speed_kmh": state.wind_speed_kmh,
            "tire_compound": state.tire_compound, "circuit_id": state.circuit_id,
        }])
        cats = self.deg_bundle["fit_categories"]
        for c in PROXY_CATEGORICAL_FEATURES:
            row[c] = pd.Categorical(row[c].astype("string"), categories=cats[c])
        dummies = pd.get_dummies(row[PROXY_CATEGORICAL_FEATURES], prefix=PROXY_CATEGORICAL_FEATURES)
        for col in self.deg_bundle["dummy_cols"]:
            if col not in dummies.columns:
                dummies[col] = False
        dummies = dummies[self.deg_bundle["dummy_cols"]]
        X = pd.concat([row[PROXY_NUMERIC_FEATURES], dummies], axis=1)
        X[PROXY_NUMERIC_FEATURES] = X[PROXY_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        return float(self.deg_bundle["model"].predict(X)[0])

    def _recommend_with_lap_model(self, state: RaceState, curve: pd.DataFrame, pit_loss_s: float) -> dict:
        compounds = [c for c in self.deg_bundle["lap_time_fit_categories"]["tire_compound"] if c != "missing"]
        if not compounds:
            compounds = [state.tire_compound]
        candidates = []
        for pit_lap in curve["lap"].astype(int).tolist():
            current_laps = list(range(state.lap, pit_lap + 1))
            current_ages = [state.lap - state.stint_start_lap + 1 + (x - state.lap) for x in current_laps]
            current_compounds = [state.tire_compound] * len(current_laps)
            current_time = float(self._predict_lap_times(state, current_laps, current_ages, current_compounds).sum())

            best_post = float("inf")
            best_compound = None
            if pit_lap < state.laps:
                future_laps = list(range(pit_lap + 1, state.laps + 1))
                for compound in compounds:
                    ages = list(range(1, len(future_laps) + 1))
                    pred = self._predict_lap_times(state, future_laps, ages, [compound] * len(future_laps))
                    total = float(pred.sum())
                    if best_compound is None or total < best_post:
                        best_post = total
                        best_compound = compound
            total_time = current_time + pit_loss_s + best_post
            candidates.append({"lap": pit_lap, "predicted_total_time_s": total_time, "pit_probability": float(curve.loc[curve.lap == pit_lap, "pit_probability"].iloc[0]), "next_compound": best_compound})

        out = pd.DataFrame(candidates)
        # Primary objective is predicted total race time. Pit probability is a tie-breaker.
        min_time = out["predicted_total_time_s"].min()
        near = out[out["predicted_total_time_s"] <= min_time + 0.25]
        best = near.sort_values("pit_probability", ascending=False).iloc[0]
        target = int(best["lap"])
        return {
            "recommended_pit_lap": target,
            "recommended_pit_window": [max(state.lap, target - 1), min(state.laps - 1, target + 1)],
            "expected_stint_length_laps": round(target - state.stint_start_lap, 1),
            "predicted_total_race_time_s": round(float(best["predicted_total_time_s"]), 2),
            "recommended_next_compound": best["next_compound"],
            "candidate_curve": out.to_dict("records"),
        }

    def recommend(self, state: RaceState, lookahead_laps: int = 15) -> dict:
        curve = self.pit_probability_curve(state, lookahead_laps)
        pit_loss_s = historical_pit_time_loss(self.train_stints_df, state.circuit_id, self.config["optimizer"]["pit_time_loss_default_s"])

        if "lap_time_model" in self.deg_bundle:
            rec = self._recommend_with_lap_model(state, curve, pit_loss_s)
            curve_out = curve.merge(pd.DataFrame(rec.pop("candidate_curve")), on=["lap", "pit_probability"], how="left")
            rec["pit_probability_curve"] = curve_out.to_dict("records")
            rec["historical_pit_time_loss_s"] = round(pit_loss_s, 2)
            rec["current_lap"] = state.lap
            rec["assumptions"] = (
                "Primary objective uses the trained real-lap-time model and historical circuit pit-stop loss. "
                "The optimizer does not model tyre allocation/rule constraints, safety cars, rival strategies, "
                "or future weather changes. The next compound is selected by lowest predicted remaining lap time."
            )
            return rec

        # Legacy fallback for an older proxy-only model bundle.
        expected_len = self.expected_stint_length(state)
        target_lap = state.stint_start_lap + expected_len
        curve = curve.copy()
        curve["laps_from_expected_window"] = (curve["lap"] - target_lap).round(1)
        curve["strategy_cost_score"] = curve["laps_from_expected_window"].abs() - curve["pit_probability"] * 5
        best_row = curve.loc[curve["strategy_cost_score"].idxmin()]
        window_lo = max(state.lap, int(round(target_lap - 1)))
        window_hi = int(round(target_lap + 1))
        return {
            "current_lap": state.lap,
            "recommended_pit_lap": int(best_row["lap"]),
            "recommended_pit_window": [window_lo, window_hi],
            "expected_stint_length_laps": round(expected_len, 1),
            "historical_pit_time_loss_s": round(pit_loss_s, 2),
            "pit_probability_curve": curve.to_dict("records"),
            "assumptions": "Legacy proxy mode: no real lap-time model is available in the loaded bundle.",
        }
