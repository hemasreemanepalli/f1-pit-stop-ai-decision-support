"""
Streamlit dashboard: F1 Pit Stop Strategy AI

Loads pre-trained models from models/ (does NOT retrain on startup - run
`python run_pipeline.py` beforehand). Lets the user pick a season, race,
driver, and current lap, then shows the race state, the model's pit
probability, the recommended pit window, a candidate-strategy
comparison, and a SHAP explanation for that specific situation.
"""
import sys
from pathlib import Path

# allow `import src...` when launched as `streamlit run dashboard/app.py`
# from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.optimization.strategy_optimizer import StrategyOptimizer, RaceState
from src.explain.shap_explain import explain_row
from src.features.build_features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(page_title="F1 Pit Stop Strategy AI", layout="wide")


@st.cache_resource
def load_everything():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    laps = pd.read_parquet(ROOT / "data/processed/laps_dataset.parquet")
    stints = pd.read_parquet(ROOT / "data/processed/stints_clean.parquet")
    pit_bundle = joblib.load(ROOT / "models/pit_model.joblib")
    deg_bundle = joblib.load(ROOT / "models/degradation_model.joblib")
    pit_report = None
    deg_report = None
    import json
    pr_path = ROOT / "reports/pit_model_report.json"
    dr_path = ROOT / "reports/degradation_model_report.json"
    if pr_path.exists():
        pit_report = json.loads(pr_path.read_text())
    if dr_path.exists():
        deg_report = json.loads(dr_path.read_text())
    train_stints = stints[stints["season"].isin(cfg["split"]["train_seasons"])]
    optimizer = StrategyOptimizer(pit_bundle, deg_bundle, train_stints, cfg)
    return cfg, laps, stints, pit_bundle, deg_bundle, optimizer, pit_report, deg_report


cfg, laps, stints, pit_bundle, deg_bundle, optimizer, pit_report, deg_report = load_everything()

st.title("🏎️ F1 Pit Stop Strategy AI")
st.caption(
    "Decision support for *when should the driver pit, and why* - built on "
    "a stint-level Kaggle CSV (2018-2024). See the Limitations panel below "
    "before treating any number here as ground truth."
)

with st.sidebar:
    st.header("Race situation")
    seasons = sorted(laps["season"].unique().tolist())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)

    races = (
        laps[laps["season"] == season][["round", "race_name"]]
        .drop_duplicates()
        .sort_values("round")
    )
    race_label = st.selectbox(
        "Race", races.apply(lambda r: f"R{int(r['round'])} - {r['race_name']}", axis=1)
    )
    round_ = int(race_label.split(" - ")[0][1:])

    drivers = sorted(
        laps[(laps["season"] == season) & (laps["round"] == round_)]["driver"].unique().tolist()
    )
    driver_name = st.selectbox("Driver", drivers)

    race_laps_df = laps[
        (laps["season"] == season) & (laps["round"] == round_) & (laps["driver"] == driver_name)
    ].sort_values("lap")

    if race_laps_df.empty:
        st.error("No lap data for this selection.")
        st.stop()

    max_lap = int(race_laps_df["lap"].max())
    current_lap = st.slider("Current lap", 1, max_lap, min(10, max_lap))

row = race_laps_df[race_laps_df["lap"] == current_lap].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stint", int(row["stint"]))
col2.metric("Tyre", row["tire_compound"])
col3.metric("Tyre age (laps)", int(row["tyre_age"]))
col4.metric("Laps remaining", int(row["laps_remaining"]))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Race progress", f"{row['race_progress']*100:.0f}%")
col6.metric("Previous pit stops", int(row["previous_pit_stops"]))
air = row.get("air_temp_c")
track = row.get("track_temp_c")
col7.metric("Air / Track temp", f"{air:.1f}°C / {track:.1f}°C" if pd.notna(air) and pd.notna(track) else "n/a")
col8.metric("Humidity / Wind", (
    f"{row['humidity_pct']:.0f}% / {row['wind_speed_kmh']:.0f} km/h"
    if pd.notna(row.get("humidity_pct")) else "n/a"
))

st.divider()

state = RaceState(
    season=season, round=round_, driver_id=row["driver_id"], lap=current_lap,
    laps=int(row["laps"]), stint=int(row["stint"]), tire_compound=row["tire_compound"],
    stint_start_lap=int(row["stint_start_lap"]), circuit_id=row["circuit_id"],
    air_temp_c=row.get("air_temp_c", np.nan), track_temp_c=row.get("track_temp_c", np.nan),
    humidity_pct=row.get("humidity_pct", np.nan), wind_speed_kmh=row.get("wind_speed_kmh", np.nan),
    previous_pit_stops=int(row["previous_pit_stops"]),
)

lookahead = min(15, max(1, max_lap - current_lap))
if current_lap >= max_lap:
    st.info("This is the final lap of the race - no future pit decision to make.")
else:
    rec = optimizer.recommend(state, lookahead_laps=lookahead)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("🔧 Recommendation")
        st.metric("Recommended pit lap", rec["recommended_pit_lap"])
        st.write(f"Recommended window: laps **{rec['recommended_pit_window'][0]}–{rec['recommended_pit_window'][1]}**")
        st.write(f"Expected stint length (proxy model): **{rec['expected_stint_length_laps']} laps**")
        st.write(f"Historical pit-time loss at this circuit: **{rec['historical_pit_time_loss_s']} s** (training-set median)")
        curr_prob = next((r["pit_probability"] for r in rec["pit_probability_curve"] if r["lap"] == current_lap), None)
        if curr_prob is not None:
            st.metric("Pit probability THIS lap", f"{curr_prob*100:.1f}%")
        with st.expander("Assumptions / how to read this"):
            st.write(rec["assumptions"])

    with right:
        st.subheader("📈 Pit probability across future laps")
        curve_df = pd.DataFrame(rec["pit_probability_curve"])
        st.line_chart(curve_df.set_index("lap")["pit_probability"])

    st.divider()
    st.subheader("🧪 Candidate strategy comparison")
    curve_df_display = curve_df.copy()
    curve_df_display["pit_probability"] = (curve_df_display["pit_probability"] * 100).round(1)
    curve_df_display = curve_df_display.rename(columns={
        "lap": "Candidate pit lap", "tyre_age": "Tyre age at pit",
        "pit_probability": "Pit probability (%)",
        "laps_from_expected_window": "Laps from expected window",
        "strategy_cost_score": "Strategy cost score (lower is better)",
    })
    st.dataframe(curve_df_display, width='stretch', hide_index=True)

    st.divider()
    st.subheader("🔍 Why this recommendation? (SHAP)")
    if (ROOT / "models/shap_explainer.joblib").exists():
        row_features = pd.DataFrame([{
            "lap": current_lap, "laps": int(row["laps"]), "race_progress": row["race_progress"],
            "laps_remaining": row["laps_remaining"], "tyre_age": row["tyre_age"],
            "laps_since_last_pit": row["laps_since_last_pit"], "previous_pit_stops": row["previous_pit_stops"],
            "stint": row["stint"], "air_temp_c": row.get("air_temp_c"), "track_temp_c": row.get("track_temp_c"),
            "humidity_pct": row.get("humidity_pct"), "wind_speed_kmh": row.get("wind_speed_kmh"),
            "tire_compound": row["tire_compound"], "circuit_id": row["circuit_id"],
            "constructor_id": row.get("constructor_id", "missing"),
        }])
        try:
            expl = explain_row("models", row_features)
            st.write(f"Model's predicted pit probability for THIS exact lap: **{expl['prediction_proba']*100:.1f}%** "
                     f"(baseline/expected value: {expl['base_value']*100:.1f}%)")
            contrib = pd.Series(expl["top_contributions"]).sort_values()
            st.bar_chart(contrib)
            st.caption("Positive values push the pit probability up; negative values push it down (SHAP values, this specific row).")
        except Exception as e:
            st.warning(f"Could not compute a per-row SHAP explanation: {e}")
    else:
        st.info("Run `python run_pipeline.py` first to generate the SHAP explainer artifact.")

st.divider()
with st.expander("📊 Model performance (from held-out 2024 test season)"):
    if pit_report:
        st.write("**Pit-stop classifier**")
        st.json(pit_report["test_metrics_season_2024"])
        st.caption(f"Selected model: {pit_report['selected_model']} "
                   f"(chosen by highest PR-AUC on validation season {pit_report['val_season']})")
    if deg_report:
        st.write("**Stint-length proxy regressor**")
        st.json(deg_report["test_metrics_season_2024"])
        st.caption(deg_report["note"])

with st.expander("⚠️ Limitations (please read)"):
    st.markdown("""
- The underlying CSV is **stint-level**, not true per-lap telemetry. Per-lap
  rows in this dashboard are reconstructed from real stint boundaries
  (`stint_length` / `pit_lap`), not fabricated, but there are **no real
  per-lap lap times** in this dataset - so a genuine "tyre degradation in
  seconds" model could not be built. The "expected stint length" model is a
  documented **proxy** (predicts laps-until-pit, not seconds-per-lap loss).
- The pit-time-loss figure is a **historical circuit-level median**, not a
  live pit-lane simulation.
- No safety-car, red-flag, rain-change, or rival-strategy signals are in
  the data, so the optimizer cannot react to them.
- Model metrics are modest (see the performance panel) - this reflects a
  genuinely hard prediction problem with a limited feature set, not a bug.
  Treat recommendations as a data-informed prior, not a guarantee.
- Jolpica-F1 API integration code exists (`src/data/jolpica_client.py`)
  but could not be executed in the sandbox this project was built in
  (no outbound network access to that host). All numbers in this
  dashboard come from the Kaggle CSV only.
""")
