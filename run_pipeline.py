"""
Run the full F1 Pit Stop Strategy AI pipeline end to end.

Usage:
    python run_pipeline.py                # normal run (CSV-only if no
                                            # Jolpica cache exists yet)
    python run_pipeline.py --download      # first, try to download and
                                            # cache Jolpica data (needs
                                            # internet access), then run
                                            # the normal pipeline
    python run_pipeline.py --offline       # skip any network access,
                                            # force CSV-only mode even if
                                            # a Jolpica cache is present
    python run_pipeline.py --skip-shap     # skip the SHAP step (slowest
                                            # step on a full run)

Stages (see README.md "Methodology" for what each one does and why):
    1. clean_csv          - clean/standardize the raw Kaggle CSV
    2. jolpica download    - (optional, --download only) cache Jolpica data
    3. build_lap_dataset   - build the driver x race x lap grid
    4. build_features      - target + leakage-safe feature engineering
    5. train_pit_model     - compare baseline/LogReg/RF/XGBoost classifiers
    6. train_degradation   - stint-length proxy regressor
    7. shap_explain        - global SHAP feature importance + explainer artifact
    8. backtest             - backtest the optimizer on the 2024 test season
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Fetch & cache Jolpica data first (needs internet).")
    parser.add_argument("--offline", action="store_true", help="Force CSV-only mode, ignore any Jolpica cache.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip the SHAP explainability step.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the backtesting step.")
    parser.add_argument("--n-backtest-races", type=int, default=None, help="Limit backtest to N driver-races (for a quick smoke run).")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    processed_dir = str(ROOT / cfg["paths"]["processed_dir"])
    models_dir = str(ROOT / cfg["paths"]["models_dir"])
    reports_dir = str(ROOT / cfg["paths"]["reports_dir"])
    figures_dir = str(ROOT / cfg["paths"]["figures_dir"])
    jolpica_cache_dir = str(ROOT / cfg["paths"]["jolpica_cache_dir"])
    raw_csv = str(ROOT / cfg["paths"]["raw_csv"])

    t0 = time.time()

    # ---- 1. clean CSV -----------------------------------------------------
    log("Stage 1/8: cleaning raw CSV ...")
    from src.data.clean_csv import run as run_clean
    run_clean(raw_csv, processed_dir, reports_dir)
    log("  -> data/processed/stints_clean.parquet, reports/data_quality_report.md")

    # ---- 2. Jolpica download (optional) -----------------------------------
    if args.download and not args.offline:
        log("Stage 2/8: downloading & caching Jolpica data ...")
        try:
            from src.data.jolpica_client import JolpicaClient
            client = JolpicaClient(
                base_url=cfg["jolpica"]["base_url"],
                cache_dir=jolpica_cache_dir,
                requests_per_window=cfg["jolpica"]["requests_per_window"],
                window_seconds=cfg["jolpica"]["window_seconds"],
                timeout_seconds=cfg["jolpica"]["timeout_seconds"],
                max_retries=cfg["jolpica"]["max_retries"],
            )
            seasons = range(cfg["seasons"]["start"], cfg["seasons"]["end"] + 1)
            for season in seasons:
                season_data = client.get_season_races(season)
                races = season_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                for race in races:
                    rnd = int(race["round"])
                    log(f"    caching laps, pitstops & results: {season} round {rnd} ...")
                    client.get_laps(season, rnd)
                    client.get_pitstops(season, rnd)
                    client.get_results(season, rnd)
            log("  -> cached to data/raw/jolpica_cache/")
        except Exception as e:
            log(f"  !! Jolpica download failed ({e}). Continuing in CSV-only mode. "
                f"This is expected in network-restricted environments.")
    elif args.offline:
        log("Stage 2/8: skipped (--offline flag set).")
    else:
        log("Stage 2/8: skipped (pass --download to fetch Jolpica data).")

    # ---- 3. build lap dataset ----------------------------------------------
    log("Stage 3/8: building driver x race x lap dataset ...")
    from src.data.build_lap_dataset import run as run_lapds
    effective_cache_dir = "__no_such_dir__" if args.offline else jolpica_cache_dir
    run_lapds(processed_dir, effective_cache_dir, reports_dir)
    log("  -> data/processed/laps_dataset.parquet, reports/lap_dataset_report.md")

    # ---- 4. feature engineering ---------------------------------------------
    log("Stage 4/8: feature engineering & target creation ...")
    from src.features.build_features import run as run_features
    df = run_features(processed_dir)
    log(f"  -> data/processed/pit_model_dataset.parquet "
        f"({df.shape[0]} rows, PitNextLap rate={df['PitNextLap'].mean():.4f})")

    # ---- 5. pit model --------------------------------------------------------
    log("Stage 5/8: training pit-stop prediction models (baseline/LogReg/RF/XGB) ...")
    from src.models.train_pit_model import run as run_pit_model
    pit_report = run_pit_model(processed_dir, models_dir, reports_dir, cfg)
    log(f"  -> selected model: {pit_report['selected_model']} "
        f"(test PR-AUC context in reports/pit_model_report.json)")

    # ---- 6. degradation / stint-length proxy model ---------------------------
    log("Stage 6/8: training stint-length proxy regressor ...")
    from src.models.train_degradation_model import run as run_deg_model
    deg_report = run_deg_model(processed_dir, models_dir, reports_dir, cfg)
    log(f"  -> selected model: {deg_report['selected_model']} "
        f"(test MAE={deg_report['test_metrics_season_2024']['mae']:.2f} laps)")

    # ---- 7. SHAP ----------------------------------------------------------
    if not args.skip_shap:
        log("Stage 7/8: computing SHAP explainability ...")
        from src.explain.shap_explain import compute_global_importance
        shap_report = compute_global_importance(models_dir, processed_dir, reports_dir, figures_dir)
        log("  -> reports/shap_report.json, reports/figures/shap_summary.png")
    else:
        log("Stage 7/8: skipped (--skip-shap).")

    # ---- 8. backtest --------------------------------------------------------
    if not args.skip_backtest:
        log("Stage 8/8: backtesting strategy optimizer on 2024 test season ...")
        from src.data.backtest import run as run_backtest
        summary, _ = run_backtest(processed_dir, models_dir, reports_dir, cfg, n_races=args.n_backtest_races)
        log(f"  -> {summary['n_backtested_driver_races']} driver-races backtested, "
            f"median |lap diff|={summary['median_abs_lap_difference']}")
    else:
        log("Stage 8/8: skipped (--skip-backtest).")

    dt = time.time() - t0
    log(f"Pipeline complete in {dt:.1f}s. Run `streamlit run dashboard/app.py` to explore results.")


if __name__ == "__main__":
    main()
