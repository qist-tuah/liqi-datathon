"""Shared setup for the analysis notebooks (04 onwards): DuckDB over the Parquet files.

Usage in a notebook, after the project root is on sys.path:

    from src import eda
    eda.connect()
    eda.q("SELECT count(*) AS n FROM feat")      # SQL in, small pandas DataFrame out
    eda.save(fig, "R1_banks_per_payment")        # -> reports/figures/R1_banks_per_payment.png

Views created by connect():
    tx    data/processed/transactions_clean.parquet     (21 columns)
    feat  data/processed/transactions_features.parquet  (clean columns + features)

The session time zone is set to UTC, so hours and dates computed in SQL agree
with the utc_* feature columns. DuckDB otherwise uses the computer's time zone.

The database is in memory by default: nothing is locked on disk, so several
notebooks can be open at once. Large queries spill to data/interim/duckdb_tmp.
"""
from pathlib import Path

import duckdb
import pandas as pd

from . import settings as S

con: duckdb.DuckDBPyConnection | None = None


def connect(db_file: Path | str | None = None, memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open DuckDB, set UTC and create the tx / feat views. Safe to call again.

    db_file      None = in-memory (default). Give a path to keep working tables between sessions.
    memory_limit e.g. "3GB" if the laptop runs out of memory. None = DuckDB default.
    """
    global con
    if not S.CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{S.CLEAN_PARQUET} not found - run notebooks/01_preprocessing.ipynb first")
    if con is not None:
        con.close()
    S.DUCKDB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_file) if db_file else ":memory:")
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET temp_directory = '{S.DUCKDB_TMP_DIR.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")

    con.execute(f"CREATE OR REPLACE VIEW tx AS SELECT * FROM read_parquet('{S.CLEAN_PARQUET.as_posix()}')")
    views = ["tx"]
    if S.FEATURES_PARQUET.exists():
        con.execute(f"CREATE OR REPLACE VIEW feat AS SELECT * FROM read_parquet('{S.FEATURES_PARQUET.as_posix()}')")
        views.append("feat")
    for view in views:
        n_rows = con.sql(f"SELECT count(*) FROM {view}").fetchall()[0][0]
        n_cols = len(con.sql(f"SELECT * FROM {view} LIMIT 0").columns)
        print(f"{view:<5}{n_rows:>12,} rows x {n_cols} columns")
    if "feat" not in views:
        print("feat  not available - run notebooks/03_features.ipynb to create it")
    time_zone = con.sql("SELECT current_setting('TimeZone')").fetchall()[0][0]
    print(f"DuckDB {duckdb.__version__} | time zone {time_zone}")
    return con


def q(sql: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a pandas DataFrame."""
    if con is None:
        raise RuntimeError("Call eda.connect() first")
    return con.sql(sql).df()


def require_columns(view: str, columns: list[str]) -> None:
    """Stop with a clear message if a view lacks columns a notebook needs (e.g. an old features file)."""
    if con is None:
        raise RuntimeError("Call eda.connect() first")
    have = set(con.sql(f"SELECT * FROM {view} LIMIT 0").columns)
    missing = [c for c in columns if c not in have]
    if missing:
        raise RuntimeError(
            f"View '{view}' is missing columns {missing}. "
            "Restart the kernel and re-run notebooks/03_features.ipynb to rebuild the features file."
        )


def save(fig, name: str) -> Path:
    """Save a chart to reports/figures/<name>.png for the dashboards and slides."""
    S.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = S.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
