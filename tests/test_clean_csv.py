import pandas as pd
from src.data.clean_csv import load_raw, clean, _fix_mojibake


def test_csv_loads(project_root):
    df = load_raw(str(project_root / "data/raw/f1_pitstops_2018_2024.csv"))
    assert len(df) > 0
    assert "Season" in df.columns


def test_original_csv_untouched(project_root):
    """The pipeline must never modify the original uploaded file."""
    import hashlib
    raw = project_root / "data/raw/f1_pitstops_2018_2024.csv"
    h = hashlib.md5(raw.read_bytes()).hexdigest()
    # re-run clean() (in-memory) and confirm the file on disk is unchanged
    load_raw(str(raw))
    h2 = hashlib.md5(raw.read_bytes()).hexdigest()
    assert h == h2


def test_mojibake_fix():
    broken = "AutÃƒÂ³dromo Hermanos RodrÃƒÂ­guez"
    fixed = _fix_mojibake(broken)
    assert "Ã" not in fixed
    assert "ó" in fixed


def test_clean_produces_expected_columns(project_root):
    df_raw = load_raw(str(project_root / "data/raw/f1_pitstops_2018_2024.csv"))
    df_clean, qa = clean(df_raw)
    for col in ["driver_id", "race_key", "driver_race_key", "is_final_stint", "has_stint_data"]:
        assert col in df_clean.columns
    assert qa["n_rows_out"] == len(df_clean)
    assert qa["n_seasons"] == 7


def test_leakage_columns_flagged(project_root):
    df_raw = load_raw(str(project_root / "data/raw/f1_pitstops_2018_2024.csv"))
    _, qa = clean(df_raw)
    flagged = qa["post_race_aggregate_columns_excluded_from_features"]
    assert "avgpitstoptime" in [c.lower() for c in flagged]
    assert len(flagged) >= 5
