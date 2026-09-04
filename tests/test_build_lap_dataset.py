import pandas as pd
from src.data.build_lap_dataset import expand_stints_to_laps, _expand_driver_race


def _mini_stints_df():
    return pd.DataFrame([
        {"driver_race_key": "k1", "stint": 1.0, "tire_compound": "SOFT", "pit_lap": 21.0,
         "stint_length": 20.0, "laps": 40, "has_stint_data": True,
         "season": 2023, "round": 1, "race_key": "2023_01", "driver": "A", "driver_id": "a",
         "constructor": "X", "constructor_id": "x", "circuit": "C", "circuit_id": "c",
         "race_name": "R", "race_date": pd.Timestamp("2023-01-01"), "location": "L",
         "country": "CO", "air_temp_c": 20.0, "track_temp_c": 30.0, "humidity_pct": 50.0,
         "wind_speed_kmh": 5.0, "metadata_missing": False},
        {"driver_race_key": "k1", "stint": 2.0, "tire_compound": "HARD", "pit_lap": None,
         "stint_length": 20.0, "laps": 40, "has_stint_data": True,
         "season": 2023, "round": 1, "race_key": "2023_01", "driver": "A", "driver_id": "a",
         "constructor": "X", "constructor_id": "x", "circuit": "C", "circuit_id": "c",
         "race_name": "R", "race_date": pd.Timestamp("2023-01-01"), "location": "L",
         "country": "CO", "air_temp_c": 20.0, "track_temp_c": 30.0, "humidity_pct": 50.0,
         "wind_speed_kmh": 5.0, "metadata_missing": False},
    ])


def test_expand_driver_race_basic():
    stints = _mini_stints_df()
    laps = _expand_driver_race(stints)
    assert laps["lap"].min() == 1
    assert laps["lap"].max() == 40
    assert len(laps) == 40
    # pit lap should be at lap 20 (pit_lap=21 means stint2 starts at 21 -> stint1 ends at 20)
    pit_rows = laps[laps["is_pit_lap"]]
    assert list(pit_rows["lap"]) == [20]


def test_expand_stints_to_laps_full():
    stints = _mini_stints_df()
    lap_df, dropped = expand_stints_to_laps(stints)
    assert dropped == 0
    assert lap_df["driver_race_key"].nunique() == 1
    assert (lap_df["laps_remaining"] >= 0).all()
    assert (lap_df["race_progress"] <= 1.0).all()
    # tyre age resets after the pit lap
    stint2 = lap_df[lap_df["stint"] == 2.0]
    assert stint2["tyre_age"].min() == 1


def test_driver_race_with_no_stint_data_is_dropped():
    stints = _mini_stints_df()
    stints["has_stint_data"] = False
    lap_df, dropped = expand_stints_to_laps(stints)
    assert dropped == 1
    assert len(lap_df) == 0
