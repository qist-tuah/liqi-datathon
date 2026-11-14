"""Project paths and config loading. Every other module imports from here."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
PROFILE_DIR = ROOT / "reports" / "profiling"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_PARQUET = INTERIM_DIR / "transactions_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "transactions_features.parquet"
COLUMN_MAPPING_CSV = PROFILE_DIR / "column_mapping.csv"
PROFILE_JSON = PROFILE_DIR / "profile.json"
PROFILE_CSV = PROFILE_DIR / "profile_summary.csv"
TYPE_PLAN_CSV = PROFILE_DIR / "type_plan.csv"
CLEAN_SUMMARY_CSV = PROFILE_DIR / "clean_summary.csv"
FEATURE_DICTIONARY_CSV = PROFILE_DIR / "feature_dictionary.csv"
DUCKDB_TMP_DIR = INTERIM_DIR / "duckdb_tmp"  # DuckDB spill-to-disk space (gitignored)


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_raw_file(cfg: dict) -> Path:
    name = cfg["raw"]["file"]
    if name:
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (raw.file in config.yaml)")
        return path
    files = sorted(RAW_DIR.glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .csv in {RAW_DIR}, found {len(files)}. "
            "Put the dataset there or set raw.file in config/config.yaml."
        )
    return files[0]
