import joblib
import pandas as pd
import pytest
import yaml

from src.optimization.strategy_optimizer import StrategyOptimizer, RaceState


@pytest.fixture
def optimizer(project_root):
    pit_path = project_root / "models" / "pit_model.joblib"
    deg_path = project_root / "models" / "degradation_model.joblib"
    if not (pit_path.exists() and deg_path.exists()):
        pytest.skip("Run `python run_pipeline.py` first to produce model artifacts.")
    cfg = yaml.safe_load((project_root / "config.yaml").read_text())
    pit_bundle = joblib.load(pit_path)
    deg_bundle = joblib.load(deg_path)
    stints = pd.read_parquet(project_root / "data/processed/stints_clean.parquet")
    train_stints = stints[stints["season"].isin(cfg["split"]["train_seasons"])]
    return StrategyOptimizer(pit_bundle, deg_bundle, train_stints, cfg)


def _sample_state(**overrides):
    base = dict(
        season=2024, round=5, driver_id="max_verstappen", lap=15, laps=57,
        stint=1, tire_compound="MEDIUM", stint_start_lap=1, circuit_id="silverstone_circuit",
        air_temp_c=22.0, track_temp_c=32.0, humidity_pct=50.0, wind_speed_kmh=12.0,
        previous_pit_stops=0,
    )
    base.update(overrides)
    return RaceState(**base)


def test_recommend_returns_expected_keys(optimizer):
    rec = optimizer.recommend(_sample_state())
    for key in [
        "current_lap", "recommended_pit_lap", "recommended_pit_window",
        "expected_stint_length_laps", "historical_pit_time_loss_s",
        "pit_probability_curve", "assumptions",
    ]:
        assert key in rec


def test_recommended_lap_is_in_future(optimizer):
    state = _sample_state(lap=15)
    rec = optimizer.recommend(state)
    assert rec["recommended_pit_lap"] >= state.lap


def test_probability_curve_is_bounded(optimizer):
    rec = optimizer.recommend(_sample_state())
    for row in rec["pit_probability_curve"]:
        assert 0.0 <= row["pit_probability"] <= 1.0


def test_raises_on_final_lap(optimizer):
    state = _sample_state(lap=57, laps=57)
    with pytest.raises(ValueError):
        optimizer.pit_probability_curve(state)


def test_recommendation_respects_lookahead_bound(optimizer):
    state = _sample_state(lap=50, laps=57)
    rec = optimizer.recommend(state, lookahead_laps=15)
    assert rec["recommended_pit_lap"] <= 56  # laps-1
