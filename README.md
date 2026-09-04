# F1 Pit Stop Strategy AI

An AI decision-support system that answers: **given the current race
situation, when should the driver pit, and why?**

Built for a university sports-analytics project. Read the
**[Limitations](#limitations)** section before trusting any specific
number — this README documents exactly what the data could and could
not support, with real metrics from an actual end-to-end run, not
illustrative placeholders.

---

## 1. Problem & objective

Formula 1 pit strategy is a real-time optimization problem: stay out too
long and tyres degrade past the point where fresh rubber + pit-lane time
loss would be faster; pit too early and you give up track position and
tyre life. This project builds a decision-support pipeline — not a
generic "predict the race result" model — that:

1. Predicts, lap by lap, the probability a driver pits on the next lap
   (`PitNextLap`).
2. Estimates how long the current tyre stint is likely to last.
3. Combines both into a recommended pit lap / pit window, with an
   explicit, documented cost objective.
4. Explains *why* using SHAP.
5. Backtests the recommendation against real historical decisions on a
   held-out season.

## 2. Data sources

- **`data/raw/f1_pitstops_2018_2024.csv`** (provided): a Kaggle-style
  export covering 2018–2024, **at STINT granularity** (one row per
  driver × race × stint — not per lap). 7,374 rows, 2,821 driver-race
  stint groups, 40 drivers, 31 circuits.
- **Jolpica-F1 API** (`https://api.jolpi.ca/ergast/f1`, the maintained
  successor to Ergast): a real, working client exists at
  `src/data/jolpica_client.py` with local JSON caching and conservative
  rate limiting. **It could not be executed while building this
  project** — the sandboxed dev environment has an outbound network
  allow-list that does not include this host (verified directly: a raw
  request returns HTTP 403 `host_not_allowed`). The client's
  request/parsing logic follows Jolpica's documented schema exactly and
  is unit-tested against mocked responses (`tests/test_jolpica_client.py`),
  but **has not been validated against a live response**. See
  [SETUP.md](SETUP.md) for how to actually run `--download` yourself and
  what to check the first time you do.

**Practical consequence:** every number in this README comes from the
CSV alone (CSV-only / stint-expansion mode). The pipeline is written to
automatically use Jolpica data instead/in addition if you populate the
cache — see `src/data/build_lap_dataset.py`.

## 3. Methodology

```
Kaggle CSV (+ Jolpica if cached)
        ↓  src/data/clean_csv.py
Data Cleaning            (mojibake fix, dtype fixes, ID standardization,
                           QA flags for missing/inconsistent rows)
        ↓  src/data/build_lap_dataset.py
Data Integration         (stint→lap expansion using validated pit_lap
                           boundaries; Jolpica join if cache present)
        ↓  src/features/build_features.py
Feature Engineering      (leakage-safe feature set + PitNextLap target)
        ↓  src/models/train_pit_model.py
Pit-Stop Prediction      (baseline / LogReg / RandomForest / XGBoost)
        ↓  src/models/train_degradation_model.py
Stint-Length Proxy Model (documented substitute for lap-time degradation)
        ↓  src/optimization/strategy_optimizer.py
Strategy Optimization    (candidate pit laps → recommended window)
        ↓  src/explain/shap_explain.py
SHAP Explainability
        ↓  src/data/backtest.py
Historical Backtesting   (2024 test season, actual vs. simulated)
        ↓  dashboard/app.py
Streamlit Dashboard
```

Run the whole thing with `python run_pipeline.py` (see
[SETUP.md](SETUP.md)).

### 3.1 Data cleaning (what was actually found)

- **Mojibake**: several circuit names were multiply UTF-8-re-encoded
  (`AutÃƒÂ³dromo Hermanos RodrÃƒÂ­guez` instead of `Autódromo Hermanos
  Rodríguez`). Fixed with `ftfy`.
- **9 races** are missing weather/metadata entirely (see
  `reports/data_quality_report.md` for the exact list).
- **645 of 2,821 driver-races (22.9%)** have `sum(stint_length) !=
  total race laps`, mostly off by 1. Rather than trust `stint_length`
  blindly, the lap-expansion step uses the CSV's own recorded `pit_lap`
  field as ground truth for stint boundaries where present (verified:
  `pit_lap` on stint *i* equals the first lap of stint *i+1*), falling
  back to cumulative `stint_length` only when `pit_lap` is missing. This
  reduced the mismatch to **1 driver-race out of 2,653** with usable
  stint data.
- **109 driver-races** have *no* usable stint data at all (`stint_length`
  sums to zero — mostly single-race substitute drivers, e.g. Jack
  Aitken / Pietro Fittipaldi at the 2020 Sakhir GP). These are dropped
  from the lap grid, not silently imputed.
- Full report: `reports/data_quality_report.md` (regenerated on every
  pipeline run).

### 3.2 Driver × race × lap dataset

The CSV has no per-lap rows, so one is built by expanding each stint
into one row per lap it covers, using the validated stint boundaries
above. This is a deterministic transformation of real fields already in
the CSV — not fabricated data. What it **cannot** recover is a genuine
per-lap **lap time** (the CSV never had one), which limits the
degradation model (§3.4).

Result: **146,982 lap-level rows** across 2,653 usable driver-races.

### 3.3 Target: `PitNextLap`

`PitNextLap = 1` if the driver pits at the end of the current lap
(built directly from the validated stint-boundary flag), else `0`. The
final lap of each race is dropped (no "next lap" exists).

**Explicitly excluded from features** (leakage):
- `Position`, `TotalPitStops`, `AvgPitStopTime`, `Lap Time Variation`,
  `Total Pit Stops`, `Tire Usage Aggression`, `Fast Lap Attempts`,
  `Position Changes`, `Driver Aggression Score` — all verified constant
  across every stint of a driver-race in the raw CSV, i.e. **post-race
  aggregates**, unavailable at decision time.
- `stint_planned_length` / eventual stint-end lap — would literally
  encode the answer.
- Next stint's tyre compound (only the *current* compound is used).
- Real per-lap `lap_time_s` — only populated when a live Jolpica
  download has been run; in CSV-only mode it is entirely NaN and is
  excluded from the trained feature set rather than silently imputed.

A programmatic check (`assert_no_target_leakage`, run automatically
during feature engineering) flags any single feature whose standalone
AUC against the target exceeds 0.97, as a defense against future
regressions.

**Features actually used** (`src/features/build_features.py`):
`lap`, `laps`, `race_progress`, `laps_remaining`, `tyre_age`,
`laps_since_last_pit`, `previous_pit_stops`, `stint`, `air_temp_c`,
`track_temp_c`, `humidity_pct`, `wind_speed_kmh`, plus one-hot
`tire_compound`, `circuit_id`, `constructor_id`.

### 3.4 Tyre / lap-time "degradation" model — a documented proxy

**This is the biggest limitation in the project, stated plainly:** a
real degradation model needs real per-lap lap times, which this CSV
does not contain (only one race-level "Lap Time Variation" aggregate,
itself a post-race number, unusable even if it were per-lap). Jolpica
would supply real lap times but was not reachable (§2).

Per the project's own rule ("if something cannot be obtained or
implemented correctly: explain the problem, implement the most
defensible alternative, document the limitation"), `src/models/train_degradation_model.py`
instead predicts `stint_length` (laps a stint lasts before pitting)
from tyre compound, circuit, weather, and stint number — a genuine,
non-fabricated target, and a real (if coarse) proxy for "how long does
this tyre/track/weather combination typically last." **It is not a
seconds-of-degradation-per-lap model** and the code/reports/dashboard
all label it explicitly as a proxy.

## 4. Models compared & actual results

All results below are from the real, executed pipeline run
(`reports/*.json`), split **race/time-aware**: train on 2018–2022,
validate on 2023, test on 2024 (never mixes laps from the same race
across splits).

### 4.1 Pit-stop classifier (validation = 2023, n=22,867 laps, positive rate 3.7%)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline (stratified random) | 0.044 | 0.037 | 0.040 | 0.503 | 0.037 |
| Logistic Regression | 0.073 | 0.508 | 0.127 | 0.682 | 0.078 |
| **Random Forest (selected)** | **0.112** | **0.434** | **0.178** | **0.706** | **0.100** |
| XGBoost | 0.109 | 0.395 | 0.171 | 0.667 | 0.089 |

Selected by highest validation PR-AUC (appropriate for a ~3% positive
class — accuracy would be misleading). **Test set (2024, n=25,062,
tuned threshold 0.509): precision 0.099, recall 0.475, F1 0.163,
ROC-AUC 0.757, PR-AUC 0.111.**

Honest read: the model is clearly better than random (ROC-AUC ~0.71–0.76)
and recovers roughly half of actual pit stops, but precision is low —
most flagged laps are false alarms. This is consistent with pit timing
being a genuinely hard problem from stint/weather features alone,
without live tyre-wear telemetry, gap-to-rivals, or strategic
(undercut/overcut) signals, none of which exist in this dataset.

### 4.2 Stint-length proxy regressor (validation = 2023, n=845 stints)

| Model | MAE (laps) | RMSE | R² |
|---|---|---|---|
| Baseline (predict mean) | 8.82 | 11.35 | -0.03 |
| Linear Regression | 8.65 | 11.03 | 0.03 |
| **Random Forest (selected)** | **7.61** | **10.23** | **0.17** |
| XGBoost | 8.51 | 10.97 | 0.04 |

**Test set (2024, n=792 stints): MAE 8.36 laps, RMSE 10.58, R² -0.02.**

Honest read: R² near/below zero on test means the model barely beats
predicting the training mean for unseen 2024 races — tyre-strategy
variance from year to year (rule changes, compound allocations) is
large relative to what circuit/weather/compound alone can explain. This
is reported, not hidden.

### 4.3 SHAP global feature importance (Random Forest classifier, 2024 test sample, n=2000)

Top features by mean |SHAP value|: `tyre_age` (0.0495) ≈
`laps_since_last_pit` (0.0459) ≈ `previous_pit_stops` (0.0443) ≈
`stint` (0.0419) > `tire_compound_HARD` (0.0317) > `race_progress`
(0.0296) > `laps_remaining` (0.0244) > `lap` (0.0164). Weather features
(humidity, wind, track/air temp) matter much less (~0.007–0.009 each).
Full figure: `reports/figures/shap_summary.png`.

This matches domain intuition: how long you've been on this tyre set
dominates the pit decision, with compound and race phase as secondary
signals.

### 4.4 Backtest (2024 test season, 397 driver-races, first stint only)

At 3 laps before each driver's real (recorded) first pit stop, the
optimizer is asked for a recommendation using only information
available up to that point:

- **Median |recommended lap − actual lap|: 3 laps.**
- Mean: 5.0 laps.
- **Only 6.3% of actual pit laps fell inside the recommended window.**

Honest read: this is a modest result, directly downstream of the
modest classifier/regressor performance above — it is reported as-is,
not adjusted to look better. `reports/backtest_results.csv` has every
individual comparison, explicitly labeled `_ACTUAL` vs. `_SIMULATED`.

## 5. Strategy optimizer — what it actually computes

`src/optimization/strategy_optimizer.py` combines the classifier's
pit-probability curve over the next laps with the regressor's expected
stint length into a documented proxy cost function, and returns a
recommended pit lap, a ±1-lap window around it, the full candidate
curve, and the assumptions behind the numbers (re-stated in every
result, and again in the dashboard). It explicitly does **not** claim to
simulate exact race time in seconds — see the module docstring for the
full list of assumptions (historical circuit-level pit-time-loss median;
no per-lap lap-time model; no safety-car / rival-strategy modelling).

## 6. Dashboard

`streamlit run dashboard/app.py` — select season / race / driver /
current lap; see race state, pit probability now and across future
laps, recommended pit lap & window, candidate strategy table, a
per-situation SHAP explanation, and a "Limitations" panel restating
everything above. **Does not retrain on startup** — it loads the
artifacts in `models/`.

## 7. Project structure

```
f1-pit-stop-ai/
├── data/{raw,processed}/
├── src/{data,features,models,optimization,explain}/
├── models/            # trained artifacts (.joblib)
├── reports/{,figures}/ # QA + model + SHAP + backtest reports (JSON/MD/CSV/PNG)
├── dashboard/app.py
├── tests/
├── run_pipeline.py
├── requirements.txt
├── config.yaml
├── README.md / SETUP.md
```
No absolute paths, no Claude-specific tooling — plain Python, portable
to any machine (see [SETUP.md](SETUP.md)).

## 8. How to run

See [SETUP.md](SETUP.md) for full, non-programmer-friendly instructions.
Short version:

```bash
pip install -r requirements.txt
python run_pipeline.py --offline      # CSV-only, what produced the numbers above
# or: python run_pipeline.py --download   # also try fetching Jolpica (needs internet)
streamlit run dashboard/app.py
pytest
```

## 9. Limitations

1. **Jolpica-F1 API was never actually reachable while building this
   project** (sandboxed dev environment; verified 403 on direct
   request). The client code is real and unit-tested against mocked
   responses, but unverified against a live response — treat the first
   `--download` run on your own machine as a verification step, not a
   guarantee (see SETUP.md).
2. **The CSV is stint-level, not lap-level.** The lap grid used
   throughout is a validated reconstruction from real stint-boundary
   fields, not fabricated telemetry — but it means there is **no real
   per-lap lap time anywhere in this project**.
3. **The "degradation" model is a stint-length proxy**, not a
   seconds-per-lap tyre-wear model. This is the single biggest scope
   reduction versus the original spec, made necessary by data
   availability, and labeled everywhere it appears.
4. **Model performance is modest** (PR-AUC ~0.10–0.11, backtest median
   3-lap deviation). This reflects a genuinely hard problem with a
   limited, non-telemetry feature set — not a bug, and not tuned away by
   picking a friendlier metric.
5. **No safety-car, red-flag, rain-onset, or rival-strategy signal**
   exists in the data, so the optimizer cannot react to any of it.
6. Weather features are a single per-race snapshot (not time-varying
   within the race), because that is what the CSV provides.
7. 109 driver-races (mostly one-off substitute drivers) were dropped
   from the lap dataset for lack of usable stint data; 645 driver-races
   had a stint-length/lap-count mismatch that was resolved using the
   more reliable `pit_lap` field, documented in
   `reports/data_quality_report.md`.

## 10. Testing

`pytest` (33 tests, 1 intentionally skipped — a slow full-pipeline
smoke run, already verified manually). Covers: CSV loading/cleaning,
mojibake fix, leakage-column exclusion, Jolpica client parsing/caching/
retry (mocked HTTP), stint→lap expansion (incl. edge cases: no stint
data, empty result), feature engineering + programmatic leakage check,
trained-model artifact/report sanity (time-aware split, PR-AUC beats
baseline), optimizer output shape/bounds/edge-cases (final-lap
`ValueError`, discovered via dashboard interaction testing and fixed),
backtest & SHAP report existence and correct ACTUAL/SIMULATED labeling,
and a static import check of the dashboard. The dashboard was also
exercised interactively with `streamlit.testing.v1.AppTest` across
season/race/driver/lap changes including the final-lap edge case, which
caught and led to fixing a real crash before this was written up.
